//! Quantized `i16 -> i64` GEMM (M5+, `FEAT_SME_I16I64`), with the fused-dequant
//! variants that scale into f32.

use crate::element::Packed;
use crate::epilogue::Dequant;
use crate::exec::{checked_dim2, dq_apply_cell, dq_validate, sme_worth_it};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::exec::{resolve_nodes, unpack_b_i16_sme};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::ffi::{
    gemm_sme_i16i64_batched, gemm_sme_i16i64_batched_dequant, gemm_sme_i16i64_packb,
    gemm_sme_i16i64_packed_b_elems, gemm_sme_i16i64_run, gemm_sme_i16i64_run_dequant,
    gemm_sme_i16i64_run_packed_impl,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::probe::caps;
use crate::reference;

/// Row-major quantized `C = A @ B`: `i16 x i16 -> i64` (SMOPA, M5+
/// `FEAT_SME_I16I64`). `c` is i64 raw accumulation. Falls back to the scalar
/// reference on CPUs without the extension.
///
/// # Panics
/// Panics if the slice lengths are inconsistent with `m`, `n`, `k`.
pub fn matmul_i16(a: &[i16], b: &[i16], c: &mut [i64], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    gemm_i16(m, n, k, c, n, 1, a, k, 1, b, n, 1);
}

/// Strided quantized GEMM: `C(i64) = A(i16) @ B(i16)`. See [`matmul_i16`].
///
/// # Panics
/// Panics if a slice is too short for its strided footprint; see
/// [`gemm_f16`](crate::gemm_f16).
#[allow(clippy::too_many_arguments)]
pub fn gemm_i16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [i64],
    c_row_stride: usize,
    c_col_stride: usize,
    a: &[i16],
    a_row_stride: usize,
    a_col_stride: usize,
    b: &[i16],
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
    if caps().sme_i16i64 && sme_worth_it(m, n, k) {
        // SAFETY: see `gemm_i8`; i16/i64 pointers cross directly.
        let rc = unsafe {
            gemm_sme_i16i64_run(
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
    reference::gemm_i16(
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

/// Row-major quantized GEMM with fused dequant: `D(f32) =
/// act(scale*(A@B)+bias)` for `i16 x i16 -> i64` (SMOPA, M5+
/// `FEAT_SME_I16I64`).
///
/// `a` is `m x k` row-major i16, `b` is `k x n` row-major i16, `c` is `m x n`
/// row-major f32.
///
/// The i64 accumulator is dequantized in-register at the store -- no separate
/// pass over an i64 output. Per-tensor `dq.scale` or per-N `dq.scale_per_n`
/// (length `n`). Non-packed: B is passed directly. Falls back to a scalar i64
/// accumulation + dequant on CPUs without the extension or on allocation
/// failure. The i64 -> f32 conversion may lose precision for accumulators that
/// exceed the f32 mantissa (expected for quantized inference).
///
/// # Panics
/// Panics if the slice lengths or `scale_per_n`/`bias`/`bias_row`/`residual`
/// lengths are inconsistent with `m`/`n`/`k`.
pub fn matmul_i16_dequant(
    a: &[i16],
    b: &[i16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    dq: &Dequant<'_>,
) {
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k (row-major)");
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n (row-major)");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n (row-major)");
    if let Some(s) = dq.scale_per_n {
        assert_eq!(s.len(), n, "scale_per_n is per-N (length n)");
    }
    dq_validate(dq, m, n);
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_i16i64 && sme_worth_it(m, n, k) {
        let fnodes = resolve_nodes(&dq.nodes, dq.act, n);
        // SAFETY: row-major i16 in (lhs_rs=k, lhs_cs=1; rhs_rs=n, rhs_cs=1),
        // row-major f32 out (dst_rs=n, dst_cs=1). scale_per_n / node operands (if
        // set) point at f32 the kernel only reads; the node array outlives the
        // call (owned by `fnodes`).
        let rc = unsafe {
            gemm_sme_i16i64_run_dequant(
                m,
                n,
                k,
                c.as_mut_ptr(),
                1,
                n as isize,
                a.as_ptr(),
                1,
                k as isize,
                b.as_ptr(),
                1,
                n as isize,
                dq.scale,
                dq.scale_per_n.map_or(core::ptr::null(), <[f32]>::as_ptr),
                u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
                fnodes.as_ptr(),
            )
        };
        // rc == 0: done. rc != 0: an allocation failed and c is untouched; fall
        // through to the scalar reference + dequant for a correct result.
        if rc == 0 {
            return;
        }
    }
    // Fallback: i64 reference GEMM, then dequantize + op-graph in Rust.
    let mut acc = vec![0i64; m * n];
    reference::gemm_i16(m, n, k, &mut acc, n, 1, a, k, 1, b, n, 1);
    for i in 0..m {
        for j in 0..n {
            let s = dq.scale_per_n.map_or(dq.scale, |sn| sn[j]);
            #[allow(clippy::cast_precision_loss)]
            let x = s * acc[i * n + j] as f32;
            c[i * n + j] = dq_apply_cell(dq, x, i, j, n);
        }
    }
}

/// Batched quantized i16 GEMM: `count` independent row-major `C_i(i64) = A_i @
/// B_i` (same `m x n x k`), one streaming session (SMOPA, M5+
/// `FEAT_SME_I16I64`).
///
/// `i16 x i16 -> i64` is exact. `a`/`b`/`c` are the contiguous batches (item
/// strides `m*k` / `k*n` / `m*n`); `count` is inferred from `c.len()`. Falls
/// back to a per-item scalar i64 reference without the extension.
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`.
pub fn matmul_i16_batched(a: &[i16], b: &[i16], c: &mut [i64], m: usize, n: usize, k: usize) {
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
    if caps().sme_i16i64 {
        // SAFETY: contiguous row-major batches of `count` items, item strides
        // m*k / k*n / m*n; i16 in / i64 out, written within bounds.
        let rc = unsafe {
            gemm_sme_i16i64_batched(count, m, n, k, c.as_mut_ptr(), a.as_ptr(), b.as_ptr())
        };
        // rc != 0: an allocation failed and c is untouched; fall through to the
        // per-item scalar reference for a correct result.
        if rc == 0 {
            return;
        }
    }
    reference::batched_gemm_i16(a, b, c, m, n, k);
}

/// Batched quantized i16 GEMM with fused dequant + op-graph applied to EACH item.
///
/// `D_i(f32) = dq(A_i @ B_i)`, `count` independent row-major GEMMs (same shape)
/// in a single streaming session (M5+ `FEAT_SME_I16I64`).
///
/// The i64 accumulator is dequantized in-register (per-tensor `dq.scale`, or
/// per-N `dq.scale_per_n`) and the f32 op-graph applied -- no separate pass
/// over an i64 output. The dequant operands (scale vector, node ROW/COL/TENSOR
/// operands) are SHARED across all `count` items. `a` is the contiguous i16
/// batch (item stride `m*k`), `b` the i16 batch (stride `k*n`), `c` the f32
/// output batch (stride `m*n`). The i64 -> f32 conversion may lose precision
/// for accumulators that exceed the f32 mantissa (expected for quantized
/// inference). The scalar fallback applies the epilogue per item in f32.
///
/// # Panics
/// Panics if the buffer lengths are inconsistent with `m`, `n`, `k`, `count`, if
/// `dq.scale_per_n` is set with length other than `n`, or if a dequant op-graph
/// operand length mismatches `m`/`n`.
#[allow(clippy::too_many_arguments)]
pub fn matmul_i16_batched_dequant(
    a: &[i16],
    b: &[i16],
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
    if caps().sme_i16i64 {
        let fnodes = resolve_nodes(&dq.nodes, dq.act, n);
        // SAFETY: contiguous batches of `count` items, item strides m*k / k*n /
        // m*n; i16 in / f32 out. The shared scale vector and node operands (if set)
        // point at f32 the kernel only reads and outlive the call.
        let rc = unsafe {
            gemm_sme_i16i64_batched_dequant(
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
    let mut acc = vec![0i64; m * n];
    for bi in 0..count {
        acc.fill(0);
        reference::gemm_i16(
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
                #[allow(clippy::cast_precision_loss)]
                let x = s * acc[i * n + j] as f32;
                c[bi * m * n + i * n + j] = dq_apply_cell(dq, x, i, j, n);
            }
        }
    }
}

/// Pack row-major i16 weights `b` (`k x n`) into the kernel's
/// `[2*n_tiles][ceil(k/4), 32]` 4-way-interleaved band layout (M5+
/// `FEAT_SME_I16I64`).
///
/// Reuse across many [`matmul_i16_packed`] / [`matmul_i16_packed_dequant`]
/// calls.
///
/// # Panics
/// Panics if `b.len() != k * n`.
#[must_use]
pub fn prepack_i16(b: &[i16], n: usize, k: usize) -> Packed<i16> {
    assert_eq!(b.len(), checked_dim2(k, n), "b is k*n (row-major)");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_i16i64 {
        // SAFETY: pure size arithmetic, then a pack of the caller's k*n row-major
        // B (rhs_rs = n, rhs_cs = 1) into a buffer of exactly that many elements.
        // SAFETY: pure size arithmetic on n/k; touches no memory.
        let elems = unsafe { gemm_sme_i16i64_packed_b_elems(n, k) };
        let mut data = vec![0i16; elems];
        // SAFETY: `data` was sized by the call above, and `b` is the caller's
        // k*n row-major B (rhs_rs = n, rhs_cs = 1) that packb only reads.
        unsafe {
            gemm_sme_i16i64_packb(data.as_mut_ptr(), b.as_ptr(), n, k, n as isize, 1);
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

/// Row-major quantized `C(i64) = A(i16) @ B(i16)` with B pre-packed. `a` is
/// `m x k` row-major, `c` is `m x n` row-major i64. Only A is packed per call.
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the packed `n`, `k`.
pub fn matmul_i16_packed(a: &[i16], packed: &Packed<i16>, c: &mut [i64], m: usize) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if packed.sme {
        // SAFETY: `packed.data` was produced by `gemm_sme_i16i64_packb` for these
        // (n, k); a / c are row-major m*k / m*n. Null dq => the i64 store.
        let rc = unsafe {
            gemm_sme_i16i64_run_packed_impl(
                m,
                n,
                k,
                c.as_mut_ptr().cast::<core::ffi::c_void>(),
                1,
                n as isize,
                a.as_ptr(),
                1,
                k as isize,
                packed.data.as_ptr(),
                core::ptr::null(),
            )
        };
        if rc == 0 {
            return;
        }
        let b = unpack_b_i16_sme(&packed.data, n, k);
        reference::gemm_i16(m, n, k, c, n, 1, a, k, 1, &b, n, 1);
        return;
    }
    reference::gemm_i16(m, n, k, c, n, 1, a, k, 1, &packed.data, n, 1);
}

/// Row-major quantized GEMM with fused dequant: `D(f32) = dq(A(i16) @ B(i16))`
/// with B pre-packed.
///
/// The i64 accumulator is dequantized in-register at the store -- no separate
/// pass over an i64 output. See [`matmul_i16_dequant`].
///
/// # Panics
/// Panics if the slice lengths or `scale_per_n` / op-graph operand lengths are
/// inconsistent with `m`/`n`.
pub fn matmul_i16_packed_dequant(
    a: &[i16],
    packed: &Packed<i16>,
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
    let unpacked: Vec<i16> = if packed.sme {
        let fnodes = resolve_nodes(&dq.nodes, dq.act, n);
        let desc = crate::ffi::DqF32 {
            scale: dq.scale,
            scale_n: dq.scale_per_n.map_or(core::ptr::null(), <[f32]>::as_ptr),
            n_nodes: u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
            nodes: fnodes.as_ptr(),
        };
        // SAFETY: as in `matmul_i16_packed`, with an f32 output and a dq
        // descriptor whose operand slices are length-validated and outlive it.
        let rc = unsafe {
            gemm_sme_i16i64_run_packed_impl(
                m,
                n,
                k,
                c.as_mut_ptr().cast::<core::ffi::c_void>(),
                1,
                n as isize,
                a.as_ptr(),
                1,
                k as isize,
                packed.data.as_ptr(),
                &raw const desc,
            )
        };
        if rc == 0 {
            return;
        }
        unpack_b_i16_sme(&packed.data, n, k)
    } else {
        Vec::new()
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let b: &[i16] = if unpacked.is_empty() {
        &packed.data
    } else {
        &unpacked
    };
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let b: &[i16] = &packed.data;
    let mut acc = vec![0i64; m * n];
    reference::gemm_i16(m, n, k, &mut acc, n, 1, a, k, 1, b, n, 1);
    for i in 0..m {
        for j in 0..n {
            let s = dq.scale_per_n.map_or(dq.scale, |sn| sn[j]);
            #[allow(clippy::cast_precision_loss)]
            let x = s * acc[i * n + j] as f32;
            c[i * n + j] = dq_apply_cell(dq, x, i, j, n);
        }
    }
}
