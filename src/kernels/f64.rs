//! f64 GEMM entry points (strided, batched, and fused-epilogue). The fused
//! packed-epilogue path itself lives in [`crate::exec`] (`f64_packed_ep_impl`).

use crate::element::Packed;
use crate::epilogue::Epilogue;
use crate::exec::{batched_ep_f64, checked_dim2, validate_ep};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::exec::{resolve_nodes, sme_worth_it};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::ffi::{
    EpDescF64, gemm_sme_f64_batched, gemm_sme_f64_batched_ep, gemm_sme_f64_packb,
    gemm_sme_f64_packed_b_elems, gemm_sme_f64_run,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::probe::caps;
use crate::reference;

/// Row-major `C = A @ B`, f64 (FMOPA double-precision, M5+ `FEAT_SME_F64F64`).
/// Falls back to the scalar reference on CPUs without the extension.
///
/// # Panics
/// Panics if the slice lengths are inconsistent with `m`, `n`, `k`.
pub fn matmul_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    gemm_f64(m, n, k, c, n, 1, a, k, 1, b, n, 1, 0.0, 1.0);
}

/// Strided f64 GEMM: `C = alpha*C + beta*(A @ B)`. See [`gemm_f32`].
///
/// [`gemm_f32`]: crate::gemm_f32
///
/// # Panics
/// Panics if a slice is too short for its strided footprint; see
/// [`gemm_f16`](crate::gemm_f16).
#[allow(clippy::too_many_arguments)]
pub fn gemm_f64(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f64],
    c_row_stride: usize,
    c_col_stride: usize,
    a: &[f64],
    a_row_stride: usize,
    a_col_stride: usize,
    b: &[f64],
    b_row_stride: usize,
    b_col_stride: usize,
    alpha: f64,
    beta: f64,
) {
    if m == 0 || n == 0 {
        return;
    }
    crate::exec::check_strided("c", c.len(), m, c_row_stride, n, c_col_stride);
    crate::exec::check_strided("a", a.len(), m, a_row_stride, k, a_col_stride);
    crate::exec::check_strided("b", b.len(), k, b_row_stride, n, b_col_stride);
    let read_dst = alpha != 0.0;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_f64f64 && sme_worth_it(m, n, k) {
        // SAFETY: see `gemm_f32`; f64 pointers/scalars cross directly.
        let rc = unsafe {
            gemm_sme_f64_run(
                m,
                n,
                k,
                c.as_mut_ptr(),
                c_col_stride as isize,
                c_row_stride as isize,
                i32::from(read_dst),
                a.as_ptr(),
                a_col_stride as isize,
                a_row_stride as isize,
                b.as_ptr(),
                b_col_stride as isize,
                b_row_stride as isize,
                alpha,
                beta,
                core::ptr::null(),
            )
        };
        // rc != 0: an allocation failed and c is untouched; fall through to the
        // scalar reference for a correct result.
        if rc == 0 {
            return;
        }
    }

    reference::gemm_f64(
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

/// Batched f64 GEMM: `count` independent row-major `C_i = A_i @ B_i`, all the
/// same `m x n x k` shape, run in a single SME streaming session (FMOPA
/// double-precision, M5+ `FEAT_SME_F64F64`).
///
/// `a`/`b`/`c` are the contiguous batches (item strides `m*k` / `k*n` /
/// `m*n`); `count` is inferred from `c.len()`. Falls back to a per-item scalar
/// reference without the extension.
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`.
pub fn matmul_f64_batched(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
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
    if caps().sme_f64f64 {
        // SAFETY: contiguous row-major batches of count items, item strides
        // m*k / k*n / m*n; the kernel reads a/b and writes c within bounds.
        let rc =
            unsafe { gemm_sme_f64_batched(count, m, n, k, c.as_mut_ptr(), a.as_ptr(), b.as_ptr()) };
        // rc != 0: an allocation failed and c is untouched; fall through to the
        // per-item scalar reference for a correct result.
        if rc == 0 {
            return;
        }
    }
    reference::batched_gemm_f64(a, b, c, m, n, k);
}

/// Batched f64 GEMM with a fused op-graph epilogue applied to EACH item.
///
/// `C_i = ep(A_i @ B_i)`, `count` independent row-major GEMMs (same
/// `m x n x k`) in a single streaming session (M5+ `FEAT_SME_F64F64`).
///
/// The epilogue `ep` (same nodes/operands for all items) is applied
/// in-register at the f64 store. Operand vectors are SHARED across all `count`
/// items: `add_col`/`mul_col` is per-N (length `n`), `add_row`/`mul_row` per-M
/// (length `m`), `add_tensor`/`mul_tensor` the single `m*n` operand reused for
/// every item. `a`/`b`/`c` are contiguous batches (item strides `m*k` / `k*n`
/// / `m*n`). The scalar fallback applies the epilogue per item in double
/// precision.
///
/// [`Gemm::run`]: crate::Gemm::run
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`, or
/// if any epilogue operand length mismatches `m`/`n` (see [`Gemm::run`]).
#[allow(clippy::too_many_arguments)]
pub fn matmul_f64_batched_ep(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    count: usize,
    m: usize,
    n: usize,
    k: usize,
    ep: &Epilogue<'_, f64>,
) {
    validate_ep(ep, m, n);
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
    if caps().sme_f64f64 {
        let fnodes = resolve_nodes(&ep.nodes, ep.act, n);
        let desc = EpDescF64 {
            n_nodes: u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
            nodes: fnodes.as_ptr(),
        };
        // SAFETY: contiguous row-major batches of `count` items, item strides
        // m*k / k*n / m*n; the kernel reads a/b and writes c in bounds. The
        // shared node array (and its operand slices) outlive the call. Operands
        // were length-validated against m/n.
        let rc = unsafe {
            gemm_sme_f64_batched_ep(
                count,
                m,
                n,
                k,
                c.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                &raw const desc,
            )
        };
        // rc != 0: an allocation failed and c is untouched; fall through to the
        // per-item reference for a correct result.
        if rc == 0 {
            return;
        }
    }
    batched_ep_f64(a, b, c, count, m, n, k, ep);
}

/// Pack row-major f64 weights `b` (`k x n`) into the kernel's `[2*n_tiles][k,8]`
/// band layout, for the [`Gemm`] fused-epilogue path. Reuse across calls to skip
/// re-packing B per GEMM.
///
/// [`Gemm`]: crate::Gemm
///
/// # Panics
/// Panics if `b.len() != k * n`.
#[must_use]
pub fn prepack_f64(b: &[f64], n: usize, k: usize) -> Packed<f64> {
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n (row-major)");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_f64f64 {
        // SAFETY: pure size arithmetic, then a pack of the caller's k*n row-major
        // B (rhs_rs = n, rhs_cs = 1) into a buffer of exactly that many elements.
        let elems = unsafe { gemm_sme_f64_packed_b_elems(n, k) };
        let mut data = vec![0f64; elems];
        // SAFETY: `data` was sized by the call above; `b` is the caller's k*n
        // row-major B that packb only reads.
        unsafe {
            gemm_sme_f64_packb(data.as_mut_ptr(), b.as_ptr(), n, k, n as isize, 1);
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
