//! bf16 GEMM entry points (strided, packed, batched, and fused-epilogue), plus
//! the macro-generated bf16 packed/batched epilogue implementations.

use half::bf16;

use crate::element::{Accuracy, Packed};
use crate::epilogue::Epilogue;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::exec::unpack_b16_sme;
use crate::exec::{batched_ep_impl, checked_dim2, packed_ep_impl, sme_worth_it};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::ffi::{
    gemm_sme_b16b16_packb, gemm_sme_b16b16_packed_b_elems, gemm_sme_b16b16_run,
    gemm_sme_b16b16_run_packed, gemm_sme_bf16f32_run,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::probe::caps;
use crate::reference;

/// Batched bf16 (`Fast`/B16B16) GEMM with a fused op-graph epilogue applied to
/// EACH item: `C_i = ep(A_i @ B_i)`.
///
/// As [`matmul_f16_batched_ep`], bf16. The GEMM accumulates in bf16 (B16B16
/// MOPA) but the epilogue is computed in f32 (upcast, run the whole op-graph,
/// round to bf16), so it supports the full op set --
/// div/sqrt/gelu/silu/transcendentals included, same as f16/f32.
///
/// [`matmul_f16_batched_ep`]: crate::matmul_f16_batched_ep
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`, or
/// if an epilogue operand length mismatches `m`/`n`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_bf16_batched_ep(
    a: &[bf16],
    b: &[bf16],
    c: &mut [bf16],
    count: usize,
    m: usize,
    n: usize,
    k: usize,
    ep: &Epilogue<'_, bf16>,
) {
    bf16_batched_ep_impl(a, b, c, count, m, n, k, ep);
}

/// bf16 mirror of `fallback_gemm_f16`: [`gemm_bf16`] at [`Accuracy::Accurate`]
/// (the widening BFMOPA kernel) instead of the scalar reference.
#[allow(clippy::too_many_arguments)]
fn fallback_gemm_bf16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [bf16],
    c_rs: usize,
    c_cs: usize,
    a: &[bf16],
    a_rs: usize,
    a_cs: usize,
    b: &[bf16],
    b_rs: usize,
    b_cs: usize,
    alpha: bf16,
    beta: bf16,
) {
    gemm_bf16(
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
        Accuracy::Accurate,
    );
}

batched_ep_impl!(
    bf16_batched_ep_impl,
    bf16,
    sme_b16b16,
    gemm_sme_b16b16_batched_ep,
    fallback_gemm_bf16,
    |_: &Epilogue<'_, bf16>| {}
);

/// Row-major `C = A @ B`, bf16. `mode` selects fp32 (accurate, widening BFMOPA,
/// M4+) vs bf16 (fast, non-widening B16B16, M5) accumulation. See [`matmul_f16`].
///
/// [`matmul_f16`]: crate::matmul_f16
///
/// # Panics
/// Panics if the slice lengths are inconsistent with `m`, `n`, `k`.
pub fn matmul_bf16(
    a: &[bf16],
    b: &[bf16],
    c: &mut [bf16],
    m: usize,
    n: usize,
    k: usize,
    mode: Accuracy,
) {
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    gemm_bf16(
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
        bf16::from_f32(0.0),
        bf16::from_f32(1.0),
        mode,
    );
}

/// Strided bf16 GEMM: `C = alpha*C + beta*(A @ B)`. `mode` selects fp32
/// (accurate) vs bf16 (fast) accumulation. See [`gemm_f16`].
///
/// [`gemm_f16`]: crate::gemm_f16
///
/// # Panics
/// Panics if a slice is too short for its strided footprint; see [`gemm_f16`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [bf16],
    c_row_stride: usize,
    c_col_stride: usize,
    a: &[bf16],
    a_row_stride: usize,
    a_col_stride: usize,
    b: &[bf16],
    b_row_stride: usize,
    b_col_stride: usize,
    alpha: bf16,
    beta: bf16,
    mode: Accuracy,
) {
    if m == 0 || n == 0 {
        return;
    }
    crate::exec::check_strided("c", c.len(), m, c_row_stride, n, c_col_stride);
    crate::exec::check_strided("a", a.len(), m, a_row_stride, k, a_col_stride);
    crate::exec::check_strided("b", b.len(), k, b_row_stride, n, b_col_stride);
    let read_dst = alpha != bf16::from_f32(0.0);

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if sme_worth_it(m, n, k) {
        let b16b16 = mode == Accuracy::Fast && caps().sme_b16b16;
        // SAFETY: see `gemm_f16`. bf16 is a transparent u16.
        let rc = unsafe {
            if b16b16 {
                gemm_sme_b16b16_run(
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
            } else {
                gemm_sme_bf16f32_run(
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
            }
        };
        // rc != 0: an allocation failed and c is untouched; fall through to the
        // scalar reference for a correct result.
        if rc == 0 {
            return;
        }
    }

    reference::gemm_bf16(
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

/// Pack row-major bf16 weights `b` (`k x n`) for the `Fast` (B16B16) path.
/// Reuse across many [`matmul_bf16_packed`] calls.
///
/// # Panics
/// Panics if `b.len() != k * n`.
#[must_use]
pub fn prepack_bf16(b: &[bf16], n: usize, k: usize) -> Packed<bf16> {
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n (row-major)");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_b16b16 {
        // SAFETY: pure size arithmetic on n/k; touches no memory.
        let elems = unsafe { gemm_sme_b16b16_packed_b_elems(n, k) };
        let mut data = vec![0u16; elems];
        // row-major B: rhs_rs = n, rhs_cs = 1
        // SAFETY: `data` was sized by the call above, and `b` is the caller's
        // k*n row-major B (rhs_rs = n, rhs_cs = 1) that packb only reads.
        unsafe {
            gemm_sme_b16b16_packb(
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

/// Row-major `C = A @ B` (bf16 `Fast`) with B pre-packed. `a` is `m x k`
/// row-major, `c` is `m x n` row-major. Only A is packed per call.
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the packed `n`, `k`.
pub fn matmul_bf16_packed(a: &[bf16], packed: &Packed<bf16>, c: &mut [bf16], m: usize) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if packed.sme {
        // SAFETY: `packed.data` was produced by `gemm_sme_b16b16_packb` for
        // these (n, k); a / c are row-major m*k / m*n.
        let rc = unsafe {
            gemm_sme_b16b16_run_packed(
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
                bf16::from_f32(1.0).to_bits(),
                core::ptr::null(),
            )
        };
        if rc == 0 {
            return;
        }
        // rc != 0: an allocation failed and c is untouched. Unpack B to
        // row-major and rerun through the widening path for a correct result.
        let b = unpack_b16_sme(&packed.data, n, k);
        // SAFETY: bf16 is repr(transparent) over u16 and  holds exactly that
        // many u16 bit patterns, so the reinterpret is length- and layout-valid.
        let b: &[bf16] = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<bf16>(), b.len()) };
        fallback_gemm_bf16(
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
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
        );
        return;
    }
    // SAFETY: bf16 is repr(transparent) over u16; `packed.data` (sme == false) is
    // the row-major k*n bit pattern, so reinterpreting as &[bf16] is sound and
    // shares the prepacked buffer without a copy.
    let b: &[bf16] = unsafe {
        core::slice::from_raw_parts(packed.data.as_ptr().cast::<bf16>(), packed.data.len())
    };
    // `sme == false` means no b16b16 panel (an M4, or a non-Apple target), but the
    // buffer is exactly the row-major `k x n` the WIDENING kernel takes -- so run
    // that rather than the scalar reference. See `fallback_gemm_bf16`.
    fallback_gemm_bf16(
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
        bf16::from_f32(0.0),
        bf16::from_f32(1.0),
    );
}

// The bf16 GEMM accumulates in bf16 (B16B16 MOPA), but the fused epilogue runs
// the whole op-graph in f32 (upcast, compute, round to bf16), so it supports the
// full op set (div/sqrt/transcendentals included) -- no guard needed.
packed_ep_impl!(
    bf16_packed_ep_impl,
    bf16,
    gemm_sme_b16b16_run_packed,
    fallback_gemm_bf16,
    |_: &Epilogue<'_, bf16>| {}
);

/// Batched bf16 (`Fast`/B16B16) GEMM: `count` independent row-major
/// `C_i = A_i @ B_i`, one streaming session. See [`matmul_f16_batched`].
///
/// [`matmul_f16_batched`]: crate::matmul_f16_batched
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`.
pub fn matmul_bf16_batched(a: &[bf16], b: &[bf16], c: &mut [bf16], m: usize, n: usize, k: usize) {
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
    if caps().sme_b16b16 {
        // SAFETY: as in `matmul_f16_batched`, bf16 buffers.
        let rc = unsafe {
            crate::ffi::gemm_sme_b16b16_batched(
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
    reference::batched_gemm_bf16(a, b, c, m, n, k);
}
