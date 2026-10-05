//! Quantized `i8 -> i32` GEMM (M4+, base `FEAT_SME`), with the fused-dequant
//! variants that scale into f32.

use crate::element::Packed;
use crate::epilogue::Dequant;
use crate::exec::{checked_dim2, dq_apply_cell, dq_validate, sme_worth_it};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::exec::{resolve_nodes, unpack_b_i8_sme};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::ffi::{
    gemm_sme_i8i32_batched, gemm_sme_i8i32_batched_dequant, gemm_sme_i8i32_packb,
    gemm_sme_i8i32_packed_b_elems, gemm_sme_i8i32_run, gemm_sme_i8i32_run_packed,
    gemm_sme_i8i32_run_packed_dequant,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::probe::has_sme;
use crate::reference;

/// Batched quantized i8 GEMM: `count` independent row-major
/// `C_i(i32) = A_i @ B_i`, one streaming session. See [`matmul_f16_batched`].
///
/// [`matmul_f16_batched`]: crate::matmul_f16_batched
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`.
pub fn matmul_i8_batched(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
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
    if has_sme() {
        let _busy = crate::warm::busy("gemm_sme_i8i32_batched", m, n, k);
        // SAFETY: as in `matmul_f16_batched`, i8 in / i32 out.
        let rc = unsafe {
            gemm_sme_i8i32_batched(count, m, n, k, c.as_mut_ptr(), a.as_ptr(), b.as_ptr())
        };
        // rc != 0: an allocation failed and c is untouched; fall through to the
        // per-item scalar reference for a correct result.
        if rc == 0 {
            return;
        }
    }
    reference::batched_gemm_i8(a, b, c, m, n, k);
}

/// Batched quantized i8 GEMM with fused dequant + op-graph applied to EACH item.
///
/// `D_i(f32) = dq(A_i @ B_i)`, `count` independent row-major GEMMs (same shape)
/// in a single streaming session. The i32 accumulator is dequantized in-register
/// (per-tensor `dq.scale`, or per-N `dq.scale_per_n`) and the f32 op-graph applied
/// -- no separate pass over the i32 output. The dequant operands (scale vector,
/// node ROW/COL/TENSOR operands) are SHARED across all `count` items. `a` is the
/// contiguous i8 batch (item stride `m*k`), `b` the i8 batch (stride `k*n`), `c`
/// the f32 output batch (stride `m*n`); `count` is inferred from `c.len()`.
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`, if
/// `dq.scale_per_n` is set with length other than `n`, or if a dequant op-graph
/// operand length mismatches `m`/`n`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_batched_dequant(
    a: &[i8],
    b: &[i8],
    c: &mut [f32],
    count: usize,
    m: usize,
    n: usize,
    k: usize,
    dq: &Dequant<'_>,
) {
    if let Some(s) = dq.scale_per_n {
        assert_eq!(s.len(), n, "scale_per_n is per-N (length n)");
    }
    dq_validate(dq, m, n);
    if m == 0 || n == 0 || count == 0 {
        return;
    }
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
    if has_sme() {
        let fnodes = resolve_nodes(&dq.nodes, dq.act, n);
        let _busy = crate::warm::busy("gemm_sme_i8i32_batched_dequant", m, n, k);
        // SAFETY: contiguous batches of `count` items, item strides m*k / k*n /
        // m*n; i8 in / f32 out. The shared scale vector and node operands (if set)
        // point at f32 the kernel only reads and outlive the call.
        let rc = unsafe {
            gemm_sme_i8i32_batched_dequant(
                count,
                m,
                n,
                k,
                c.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                dq.scale,
                dq.scale_per_n.map_or(core::ptr::null(), <[f32]>::as_ptr),
                u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
                fnodes.as_ptr(),
            )
        };
        // rc != 0: an allocation failed and c is untouched; fall through to the
        // per-item reference + scalar dequant for a correct result.
        if rc == 0 {
            return;
        }
    }
    let mut acc = vec![0i32; m * n];
    for bi in 0..count {
        acc.fill(0);
        reference::gemm_i8(
            m,
            n,
            k,
            &mut acc,
            n,
            1,
            &a[bi * m * k..(bi + 1) * m * k],
            k,
            1,
            &b[bi * k * n..(bi + 1) * k * n],
            n,
            1,
        );
        for i in 0..m {
            for j in 0..n {
                let s = dq.scale_per_n.map_or(dq.scale, |sn| sn[j]);
                let x = s * acc[i * n + j] as f32;
                c[bi * m * n + i * n + j] = dq_apply_cell(dq, x, i, j, n);
            }
        }
    }
}

/// Row-major quantized `C = A @ B`: `i8 x i8 -> i32` (SMOPA, M4+). `c` is i32,
/// raw accumulation; for fused scale/zero-point see [`matmul_i8_packed_dequant`].
///
/// The i32 accumulation wraps mod 2^32 on overflow -- unreachable below
/// `k ~ 130,000` even at full +-127 magnitude. The SME kernel and the scalar
/// fallback wrap identically.
///
/// # Panics
/// Panics if the slice lengths are inconsistent with `m`, `n`, `k`.
pub fn matmul_i8(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    gemm_i8(m, n, k, c, n, 1, a, k, 1, b, n, 1);
}

/// Strided quantized GEMM: `C(i32) = A(i8) @ B(i8)`. See [`matmul_i8`] (incl.
/// the i32 overflow-wrap note).
///
/// # Panics
/// Panics if a slice is too short for its strided footprint; see
/// [`gemm_f16`](crate::gemm_f16).
#[allow(clippy::too_many_arguments)]
pub fn gemm_i8(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [i32],
    c_row_stride: usize,
    c_col_stride: usize,
    a: &[i8],
    a_row_stride: usize,
    a_col_stride: usize,
    b: &[i8],
    b_row_stride: usize,
    b_col_stride: usize,
) {
    if m == 0 || n == 0 {
        return;
    }
    crate::exec::check_strided("c", c.len(), m, c_row_stride, n, c_col_stride);
    crate::exec::check_strided("a", a.len(), m, a_row_stride, k, a_col_stride);
    crate::exec::check_strided("b", b.len(), k, b_row_stride, n, b_col_stride);
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if sme_worth_it(m, n, k) {
        let _busy = crate::warm::busy("gemm_sme_i8i32_run", m, n, k);
        // SAFETY: see `gemm_f16`; i8/i32 pointers cross directly.
        let rc = unsafe {
            gemm_sme_i8i32_run(
                m,
                n,
                k,
                c.as_mut_ptr(),
                c_col_stride as isize,
                c_row_stride as isize,
                a.as_ptr(),
                a_col_stride as isize,
                a_row_stride as isize,
                b.as_ptr(),
                b_col_stride as isize,
                b_row_stride as isize,
            )
        };
        // rc != 0: an allocation failed and c is untouched; fall through to the
        // scalar reference for a correct result.
        if rc == 0 {
            return;
        }
    }
    reference::gemm_i8(
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
    );
}

/// Pack row-major i8 weights `b` (`k x n`) for the quantized path. Reuse across
/// many [`matmul_i8_packed`] calls.
///
/// # Panics
/// Panics if `b.len() != k * n`.
#[must_use]
pub fn prepack_i8(b: &[i8], n: usize, k: usize) -> Packed<i8> {
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n (row-major)");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if has_sme() {
        // SAFETY: pure size arithmetic on n/k; touches no memory.
        let elems = unsafe { gemm_sme_i8i32_packed_b_elems(n, k) };
        let mut data = vec![0i8; elems];
        let _busy = crate::warm::busy("gemm_sme_i8i32_packb", 0, n, k);
        // row-major B: rhs_rs = n, rhs_cs = 1
        // SAFETY: `data` was sized by the call above; `b` is the caller's k*n
        // row-major B that packb only reads.
        unsafe {
            gemm_sme_i8i32_packb(data.as_mut_ptr(), b.as_ptr(), n, k, n as isize, 1);
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
        data: b.to_vec(),
        n,
        k,
        sme: false,
        _t: core::marker::PhantomData,
    }
}

/// Row-major quantized `C(i32) = A(i8) @ B(i8)` with B pre-packed. `a` is `m x k`
/// row-major, `c` is `m x n` row-major i32. Only A is packed per call.
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the packed `n`, `k`.
pub fn matmul_i8_packed(a: &[i8], packed: &Packed<i8>, c: &mut [i32], m: usize) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if packed.sme {
        let _busy = crate::warm::busy("gemm_sme_i8i32_packb", 0, n, k);
        // SAFETY: `packed.data` was produced by `gemm_sme_i8i32_packb` for
        // these (n, k); a / c are row-major m*k / m*n.
        let rc = unsafe {
            gemm_sme_i8i32_run_packed(
                m,
                n,
                k,
                c.as_mut_ptr(),
                1,
                n as isize,
                a.as_ptr(),
                1,
                k as isize,
                packed.data.as_ptr(),
            )
        };
        if rc == 0 {
            return;
        }
        // rc != 0: an allocation failed and c is untouched. Unpack B to
        // row-major and run the scalar reference for a correct result.
        let b = unpack_b_i8_sme(&packed.data, n, k);
        reference::gemm_i8(m, n, k, c, n, 1, a, k, 1, &b, n, 1);
        return;
    }
    reference::gemm_i8(m, n, k, c, n, 1, a, k, 1, &packed.data, n, 1);
}

/// Row-major quantized GEMM with fused dequant: `D(f32) = act(scale*(A@B)+bias)`
/// with B pre-packed. `a` is `m x k` row-major i8, `c` is `m x n` row-major f32.
///
/// The i32 accumulator is dequantized in-register -- no separate pass over the
/// i32 output. `dq.scale_per_n` / `dq.bias` (if set) must have length `n`.
///
/// # Panics
/// Panics if the slice lengths or `scale_per_n`/`bias`/`bias_row`/`residual`
/// lengths are inconsistent with `m`/`n`.
pub fn matmul_i8_packed_dequant(
    a: &[i8],
    packed: &Packed<i8>,
    c: &mut [f32],
    m: usize,
    dq: &Dequant<'_>,
) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    if let Some(s) = dq.scale_per_n {
        assert_eq!(s.len(), n, "scale_per_n is per-N (length n)");
    }
    dq_validate(dq, m, n);
    if m == 0 || n == 0 {
        return;
    }
    // On the OOM fallback this owns an unpacked row-major B; otherwise the
    // scalar fallback borrows `packed.data` directly (sme == false).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let mut unpacked: Vec<i8> = Vec::new();
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let fnodes = resolve_nodes(&dq.nodes, dq.act, n);
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if packed.sme {
        let _busy = crate::warm::busy("gemm_sme_i8i32_run_packed_dequant", m, n, k);
        // SAFETY: as in `matmul_i8_packed`; scale_per_n / node operands (if set)
        // point at f32 the kernel only reads. Output c is row-major f32. The node
        // array outlives the call (owned by `fnodes`).
        let rc = unsafe {
            gemm_sme_i8i32_run_packed_dequant(
                m,
                n,
                k,
                c.as_mut_ptr(),
                1,
                n as isize,
                a.as_ptr(),
                1,
                k as isize,
                packed.data.as_ptr(),
                dq.scale,
                dq.scale_per_n.map_or(core::ptr::null(), <[f32]>::as_ptr),
                u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
                fnodes.as_ptr(),
            )
        };
        if rc == 0 {
            return;
        }
        // rc != 0: an allocation failed and c is untouched. Unpack B to
        // row-major and drop into the scalar dequant fallback below.
        unpacked = unpack_b_i8_sme(&packed.data, n, k);
    }
    // Fallback: i32 reference GEMM, then dequantize + op-graph in Rust.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let b: &[i8] = if unpacked.is_empty() {
        &packed.data
    } else {
        &unpacked
    };
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let b: &[i8] = &packed.data;
    let mut acc = vec![0i32; m * n];
    reference::gemm_i8(m, n, k, &mut acc, n, 1, a, k, 1, b, n, 1);
    for i in 0..m {
        for j in 0..n {
            let s = dq.scale_per_n.map_or(dq.scale, |sn| sn[j]);
            let x = s * acc[i * n + j] as f32;
            c[i * n + j] = dq_apply_cell(dq, x, i, j, n);
        }
    }
}
