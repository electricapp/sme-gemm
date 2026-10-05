//! f16 GEMM entry points (strided, packed, batched, and fused-epilogue), plus
//! the macro-generated f16 packed/batched epilogue implementations.

use half::f16;

use crate::element::{Accum, Packed};
use crate::epilogue::Epilogue;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::exec::unpack_b16_sme;
use crate::exec::{batched_ep_impl, checked_dim2, packed_ep_impl, sme_worth_it};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::ffi::{
    gemm_sme_f16f16_packb, gemm_sme_f16f16_packed_b_elems, gemm_sme_f16f16_run,
    gemm_sme_f16f16_run_packed, gemm_sme_f16f32_run,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::probe::caps;
use crate::reference;

/// Fallback GEMM for the packed / batched-epilogue paths: [`gemm_f16`] at
/// [`Accum::F32`], i.e. the WIDENING kernel, not the scalar reference.
/// Without `FEAT_SME_F16F16` (an M4) `prepack_f16` stores a plain row-major copy
/// -- exactly what the widening driver takes -- so routing here instead of to
/// scalar is free. Off-SME / below the worth-it floor `gemm_f16` falls through to
/// the reference anyway, so those paths are unchanged.
///
/// TODO: this removes the M4 scalar cliff but not the re-pack -- B is packed per
/// call. Real weight reuse needs a pair-interleaved `packb`/`run_packed` on the
/// widening drivers, which can only be validated on M4 hardware.
#[allow(clippy::too_many_arguments)]
fn fallback_gemm_f16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f16],
    c_rs: usize,
    c_cs: usize,
    a: &[f16],
    a_rs: usize,
    a_cs: usize,
    b: &[f16],
    b_rs: usize,
    b_cs: usize,
    alpha: f16,
    beta: f16,
) {
    gemm_f16(
        m,
        n,
        k,
        c,
        c_rs,
        c_cs,
        a,
        a_rs,
        a_cs,
        b,
        b_rs,
        b_cs,
        alpha,
        beta,
        Accum::F32,
    );
}

packed_ep_impl!(
    f16_packed_ep_impl,
    f16,
    gemm_sme_f16f16_run_packed,
    fallback_gemm_f16,
    |_: &Epilogue<'_, f16>| {}
);

batched_ep_impl!(
    f16_batched_ep_impl,
    f16,
    sme_f16f16,
    gemm_sme_f16f16_batched_ep,
    fallback_gemm_f16,
    |_: &Epilogue<'_, f16>| {}
);

/// Row-major `C = A @ B`, f16. `a` is `m x k`, `b` is `k x n`, `c` is `m x n`,
/// all row-major and contiguous. `accum` is [`Accum::F32`] (widening, M4+) or
/// [`Accum::F16`] (`FEAT_SME_F16F16`, M5).
///
/// # Panics
/// Panics if the slice lengths are inconsistent with `m`, `n`, `k`.
pub fn matmul_f16(a: &[f16], b: &[f16], c: &mut [f16], m: usize, n: usize, k: usize, accum: Accum) {
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    // row-major strides: C[i,j]=c[i*n+j], A[i,l]=a[i*k+l], B[l,j]=b[l*n+j]
    gemm_f16(
        m,
        n,
        k,
        c,
        /*c_row*/ n,
        /*c_col*/ 1,
        a,
        /*a_row*/ k,
        /*a_col*/ 1,
        b,
        /*b_row*/ n,
        /*b_col*/ 1,
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        accum,
    );
}

/// Strided f16 GEMM: `C = alpha*C + beta*(A @ B)`. `accum` is [`Accum::F32`]
/// (widening) or [`Accum::F16`] (native, M5).
///
/// Strides are in elements. `C[i,j] = c[i*c_row_stride + j*c_col_stride]`, and
/// analogously for A (`m x k`) and B (`k x n`). When `alpha == 0` C is treated
/// as write-only (not read).
///
/// # Panics
/// Panics if any of `c`/`a`/`b` is too short for its dims-times-strides
/// footprint (maximum index `(rows-1)*row_stride + (cols-1)*col_stride`). The
/// SME kernels honor strides literally and cannot bounds-check, so this guard
/// is what keeps the entry point safe.
#[allow(clippy::too_many_arguments)]
pub fn gemm_f16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f16],
    c_row_stride: usize,
    c_col_stride: usize,
    a: &[f16],
    a_row_stride: usize,
    a_col_stride: usize,
    b: &[f16],
    b_row_stride: usize,
    b_col_stride: usize,
    alpha: f16,
    beta: f16,
    accum: Accum,
) {
    if m == 0 || n == 0 {
        return;
    }
    crate::exec::check_strided("c", c.len(), m, c_row_stride, n, c_col_stride);
    crate::exec::check_strided("a", a.len(), m, a_row_stride, k, a_col_stride);
    crate::exec::check_strided("b", b.len(), k, b_row_stride, n, b_col_stride);
    let read_dst = alpha != f16::from_f32(0.0);

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if sme_worth_it(m, n, k) {
        let f16f16 = accum.is_f16() && caps().sme_f16f16;
        // SAFETY: strides/lengths describe in-bounds m x k / k x n / m x n
        // regions; the kernels read/write only within them. f16 is a
        // transparent u16; scalars cross as bit patterns.
        let rc = unsafe {
            let run = if f16f16 {
                gemm_sme_f16f16_run
            } else {
                gemm_sme_f16f32_run
            };
            run(
                m,
                n,
                k,
                c.as_mut_ptr().cast::<u16>(),
                c_col_stride as isize,
                c_row_stride as isize,
                i32::from(read_dst),
                a.as_ptr().cast::<u16>(),
                a_col_stride as isize,
                a_row_stride as isize,
                b.as_ptr().cast::<u16>(),
                b_col_stride as isize,
                b_row_stride as isize,
                alpha.to_bits(),
                beta.to_bits(),
            )
        };
        // rc != 0 means an allocation failed and dst is untouched; fall through
        // to the scalar reference for a correct result.
        if rc == 0 {
            return;
        }
    }

    reference::gemm_f16(
        m,
        n,
        k,
        c,
        c_row_stride,
        c_col_stride,
        a,
        a_row_stride,
        a_col_stride,
        b,
        b_row_stride,
        b_col_stride,
        alpha,
        beta,
    );
}

/// Pack row-major f16 weights `b` (`k x n`) for the f16-accumulate path. Reuse
/// the result across many [`matmul_f16_packed`] calls to skip re-packing per GEMM.
///
/// # Panics
/// Panics if `b.len() != k * n`.
#[must_use]
pub fn prepack_f16(b: &[f16], n: usize, k: usize) -> Packed<f16> {
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n (row-major)");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_f16f16 {
        // SAFETY: pure size arithmetic on n/k; touches no memory.
        let elems = unsafe { gemm_sme_f16f16_packed_b_elems(n, k) };
        let mut data = vec![0u16; elems];
        let _busy = crate::warm::busy("gemm_sme_f16f16_packb", 0, n, k);
        // row-major B: rhs_rs = n, rhs_cs = 1
        // SAFETY: `data` was sized by the call above, and `b` is the caller's
        // k*n row-major B (rhs_rs = n, rhs_cs = 1) that packb only reads.
        unsafe {
            gemm_sme_f16f16_packb(
                data.as_mut_ptr(),
                b.as_ptr().cast::<u16>(),
                n,
                k,
                n as isize,
                1,
            );
        }
        return Packed {
            data,
            n,
            k,
            sme: true,
            _t: core::marker::PhantomData,
        };
    }
    Packed {
        data: b.iter().map(|x| x.to_bits()).collect(),
        n,
        k,
        sme: false,
        _t: core::marker::PhantomData,
    }
}

/// Row-major `C = A @ B` (f16 accumulate) with B pre-packed. `a` is `m x k`
/// row-major, `c` is `m x n` row-major. Only A is packed per call.
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the packed `n`, `k`.
pub fn matmul_f16_packed(a: &[f16], packed: &Packed<f16>, c: &mut [f16], m: usize) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if packed.sme {
        let _busy = crate::warm::busy("gemm_sme_f16f16_packb", 0, n, k);
        // SAFETY: `packed.data` was produced by `gemm_sme_f16f16_packb` for
        // these (n, k); a / c are row-major m*k / m*n.
        let rc = unsafe {
            gemm_sme_f16f16_run_packed(
                m,
                n,
                k,
                c.as_mut_ptr().cast::<u16>(),
                1,
                n as isize,
                0,
                a.as_ptr().cast::<u16>(),
                1,
                k as isize,
                packed.data.as_ptr(),
                0,
                f16::from_f32(1.0).to_bits(),
                core::ptr::null(),
            )
        };
        if rc == 0 {
            return;
        }
        // rc != 0: an allocation failed and c is untouched. Unpack B to
        // row-major and rerun through the widening path for a correct result.
        let b = unpack_b16_sme(&packed.data, n, k);
        // SAFETY: f16 is repr(transparent) over u16 and  holds exactly that
        // many u16 bit patterns, so the reinterpret is length- and layout-valid.
        let b: &[f16] = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<f16>(), b.len()) };
        fallback_gemm_f16(
            m,
            n,
            k,
            c,
            n,
            1,
            a,
            k,
            1,
            b,
            n,
            1,
            f16::from_f32(0.0),
            f16::from_f32(1.0),
        );
        return;
    }
    // SAFETY: f16 is repr(transparent) over u16; `packed.data` (sme == false) is
    // the row-major k*n bit pattern, so reinterpreting as &[f16] is sound and
    // shares the prepacked buffer without a copy.
    let b: &[f16] = unsafe {
        core::slice::from_raw_parts(packed.data.as_ptr().cast::<f16>(), packed.data.len())
    };
    // `sme == false` means no f16f16 panel (an M4, or a non-Apple target), but the
    // buffer is exactly the row-major `k x n` the WIDENING kernel takes -- so run
    // that rather than the scalar reference. See `fallback_gemm_f16`.
    fallback_gemm_f16(
        m,
        n,
        k,
        c,
        n,
        1,
        a,
        k,
        1,
        b,
        n,
        1,
        f16::from_f32(0.0),
        f16::from_f32(1.0),
    );
}

/// Batched f16 (f16 accumulate) GEMM: `count` independent row-major `C_i = A_i @ B_i`,
/// all the same `m x n x k` shape, run in a single SME streaming session.
///
/// For small per-item GEMMs the streaming entry/exit dominates, so amortizing
/// it across the batch is the win. `a`/`b`/`c` are the contiguous batches
/// (item strides `m*k` / `k*n` / `m*n`); `count` is inferred from `c.len()`.
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`.
pub fn matmul_f16_batched(a: &[f16], b: &[f16], c: &mut [f16], m: usize, n: usize, k: usize) {
    if m == 0 || n == 0 || c.is_empty() {
        return;
    }
    let count = crate::exec::batched_count(c.len(), m, n);
    assert_eq!(
        a.len(),
        crate::exec::checked_dims(count, m, k),
        "a is count*m*k"
    );
    assert_eq!(
        b.len(),
        crate::exec::checked_dims(count, k, n),
        "b is count*k*n"
    );
    assert_eq!(
        c.len(),
        crate::exec::checked_dims(count, m, n),
        "c is count*m*n"
    );

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_f16f16 {
        let _busy = crate::warm::busy("gemm_sme_f16f16_batched", m, n, k);
        // SAFETY: contiguous row-major batches of count items, item strides
        // m*k / k*n / m*n; the kernel reads a/b and writes c within bounds.
        let rc = unsafe {
            crate::ffi::gemm_sme_f16f16_batched(
                count,
                m,
                n,
                k,
                c.as_mut_ptr().cast::<u16>(),
                a.as_ptr().cast::<u16>(),
                b.as_ptr().cast::<u16>(),
            )
        };
        // rc != 0: an allocation failed and c is untouched; fall through to the
        // per-item scalar reference for a correct result.
        if rc == 0 {
            return;
        }
    }
    reference::batched_gemm_f16(a, b, c, m, n, k);
}

/// Batched f16 (f16 accumulate) GEMM with a fused op-graph epilogue applied to EACH
/// item: `C_i = ep(A_i @ B_i)`, `count` independent row-major GEMMs (same `m x
/// n x k`) in a single streaming session.
///
/// The epilogue `ep` (same nodes/operands for all items -- the common
/// bias+activation case) is applied in-register at the store, just like
/// [`Gemm::run`]. Operand vectors are SHARED across all `count` items:
/// `add_col`/`mul_col` bias is per-N (length `n`), `add_row`/`mul_row` per-M
/// (length `m`), `add_tensor`/`mul_tensor` is the single `m*n` operand reused
/// for every item. `a`/`b`/`c` are contiguous batches (item strides `m*k` /
/// `k*n` / `m*n`); `count` is inferred from `c.len()`.
///
/// [`Gemm::run`]: crate::Gemm::run
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`, or
/// if any epilogue operand length mismatches `m`/`n` (see [`Gemm::run`]).
#[allow(clippy::too_many_arguments)]
pub fn matmul_f16_batched_ep(
    a: &[f16],
    b: &[f16],
    c: &mut [f16],
    count: usize,
    m: usize,
    n: usize,
    k: usize,
    ep: &Epilogue<'_, f16>,
) {
    f16_batched_ep_impl(a, b, c, count, m, n, k, ep);
}

// `prepack_f16` only yields `sme == false` on an M4 (or off Apple), so the
// public API can't reach the fallback arm here -- which is why routing it to the
// widening kernel was deferred as untestable. Build the panel directly instead:
// `gemm_f16(Accum::F32)` runs the same widening driver on M5 as on M4.
// The panel-size helpers must SATURATE, not wrap: they size the caller's buffer
// while `packb` writes the full panel regardless. FFI symbols, so in-crate.
// `not(miri)` because these call the C symbols directly and Miri cannot execute
// foreign functions -- without it `cargo miri test --lib` aborts here.
#[cfg(all(target_os = "macos", target_arch = "aarch64", test, not(miri)))]
mod packed_size_tests {
    use crate::ffi::{gemm_sme_f16f16_packed_b_elems, gemm_sme_i8i32_packed_b_elems};

    #[test]
    fn packed_b_elems_saturate_instead_of_wrapping() {
        // n_tiles_pad * k * 32 (f16) and 2*n_tiles*ceil(k/4)*64 (i8) both blow
        // past usize::MAX here; wrapping would hand back a tiny buffer size.
        let huge = 1usize << 60;
        // SAFETY: pure arithmetic on the arguments; touches no memory.
        unsafe {
            assert_eq!(
                gemm_sme_f16f16_packed_b_elems(1, huge),
                usize::MAX,
                "f16 packed-B size must saturate, not wrap"
            );
            assert_eq!(
                gemm_sme_i8i32_packed_b_elems(1, huge),
                usize::MAX,
                "i8 packed-B size must saturate, not wrap"
            );
            // Ordinary shapes are unaffected: n=64 -> 2 tiles (already even), so
            // n_tiles_pad = 2 and the panel is 2 * k * 32.
            assert_eq!(gemm_sme_f16f16_packed_b_elems(64, 64), 2 * 64 * 32);
            // i8: n=64 -> 2 tiles, k=64 -> kp4=16, so 2*2*16*64.
            assert_eq!(gemm_sme_i8i32_packed_b_elems(64, 64), 2 * 2 * 16 * 64);
        }
    }
}

#[cfg(test)]
mod fallback_tests {
    use super::{fallback_gemm_f16, matmul_f16, matmul_f16_packed};
    use crate::element::Packed;
    use crate::exec::Gemm;
    use half::f16;

    // Straddle the SME worth-it floor (2^18 flops): the small shape must take
    // `gemm_f16`'s scalar arm, the large one its widening-SME arm.
    const SHAPES: &[(usize, usize, usize)] = &[(3, 5, 7), (40, 33, 17), (65, 97, 129)];

    fn rowmajor_panel(b: &[f16], n: usize, k: usize) -> Packed<f16> {
        Packed {
            data: b.iter().map(|x| x.to_bits()).collect(),
            n,
            k,
            sme: false,
            _t: core::marker::PhantomData,
        }
    }

    fn rnd(seed: u64, len: usize) -> Vec<f16> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                f16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            })
            .collect()
    }

    #[test]
    fn non_sme_panel_matches_the_unpacked_widening_path() {
        for &(m, n, k) in SHAPES {
            let a = rnd(0x0f16_0001 ^ (m * 131 + k) as u64, m * k);
            let b = rnd(0x0f16_0002 ^ (n * 17 + k) as u64, k * n);
            let mut got = vec![f16::ZERO; m * n];
            matmul_f16_packed(&a, &rowmajor_panel(&b, n, k), &mut got, m);

            let mut want = vec![f16::ZERO; m * n];
            matmul_f16(&a, &b, &mut want, m, n, k, crate::element::Accum::F32);
            assert_eq!(got, want, "sme==false packed panel diverges at {m}x{n}x{k}");
        }
    }

    #[test]
    fn non_sme_panel_fused_epilogue_matches() {
        for &(m, n, k) in SHAPES {
            let a = rnd(0x0f16_0003 ^ (m * 131 + k) as u64, m * k);
            let b = rnd(0x0f16_0004 ^ (n * 17 + k) as u64, k * n);
            let bias = rnd(0x0f16_0005 ^ n as u64, n);
            let panel = rowmajor_panel(&b, n, k);

            let mut got = vec![f16::ZERO; m * n];
            Gemm::new(&a, &panel, m).add_col(&bias).relu().run(&mut got);

            // The fallback arm is exactly "widening GEMM, then the scalar
            // op-graph over C", so reproduce that literally.
            let mut want = vec![f16::ZERO; m * n];
            fallback_gemm_f16(
                m,
                n,
                k,
                &mut want,
                n,
                1,
                &a,
                k,
                1,
                &b,
                n,
                1,
                f16::ZERO,
                f16::from_f32(1.0),
            );
            for i in 0..m {
                for j in 0..n {
                    let v = want[i * n + j].to_f32() + bias[j].to_f32();
                    want[i * n + j] = f16::from_f32(v.max(0.0));
                }
            }
            assert_eq!(
                got, want,
                "sme==false fused epilogue diverges at {m}x{n}x{k}"
            );
        }
    }
}
