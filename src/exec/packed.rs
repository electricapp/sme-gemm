//! The f32/f64 fused packed-epilogue paths, their per-item batched
//! fallbacks, and the row-softmax entry points.

use super::{RowReduce, apply_ep_scalar, apply_ep_scalar_f64, checked_dim2, validate_ep};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use super::{resolve_nodes, sme_worth_it, unpack_b_f32_sme, unpack_b_f64_sme};
use crate::element::Packed;
use crate::epilogue::Epilogue;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::ffi::{
    EpDescF32, EpDescF64, gemm_sme_f32_run, gemm_sme_f32_run_packed, gemm_sme_f32_softmax,
    gemm_sme_f64_run, gemm_sme_f64_run_packed,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::probe::caps;
use crate::reference;

/// f32 fused op-graph epilogue: `D = act(beta*(A@B) + composable bias)` applied
/// in-register at the f32 store. The f32 working domain is already f32, so the
/// op-graph runs with no dtype conversion. B has no dedicated `packb`, so
/// `packed.data` is the row-major copy passed straight to the kernel. On a
/// non-SME target or an allocation failure, falls back to the reference GEMM and
/// the scalar op-graph pass.
pub(crate) fn f32_packed_ep_impl(
    a: &[f32],
    packed: &Packed<f32>,
    c: &mut [f32],
    m: usize,
    ep: &Epilogue<'_, f32>,
    beta: f32,
    col_major: bool,
) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    validate_ep(ep, m, n);
    if m == 0 || n == 0 {
        return;
    }
    let (dst_cs, dst_rs): (isize, isize) = if col_major {
        (m as isize, 1)
    } else {
        (1, n as isize)
    };
    let has_ep = !ep.nodes.is_empty() || ep.act.is_some();

    // On the OOM fallback this owns an unpacked row-major B; otherwise the scalar
    // fallback borrows `packed.data` directly (sme == false).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let mut unpacked: Vec<f32> = Vec::new();
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if packed.sme || sme_worth_it(m, n, k) {
        let fnodes = resolve_nodes(&ep.nodes, ep.act, n);
        let desc = EpDescF32::nodes_only(
            u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
            fnodes.as_ptr(),
        );
        // NULL when no epilogue work, so the unscaled row-major call keeps the
        // single-instruction pure-store fast path.
        let ep_ptr = if has_ep {
            &raw const desc
        } else {
            core::ptr::null()
        };
        let _busy = crate::warm::busy("gemm_sme_f32_run_packed", m, n, k);
        // SAFETY: a is row-major m*k; `packed.data` is either the packb panel for
        // these (n, k) or a row-major k*n copy, matched to the entry point called
        // below; c strides describe the chosen layout; operands were validated.
        let rc = unsafe {
            if packed.sme {
                gemm_sme_f32_run_packed(
                    m,
                    n,
                    k,
                    c.as_mut_ptr(),
                    dst_cs,
                    dst_rs,
                    0,
                    a.as_ptr(),
                    1,
                    k as isize,
                    packed.data.as_ptr(),
                    0.0,
                    beta,
                    ep_ptr,
                )
            } else {
                gemm_sme_f32_run(
                    m,
                    n,
                    k,
                    c.as_mut_ptr(),
                    dst_cs,
                    dst_rs,
                    0,
                    a.as_ptr(),
                    1,
                    k as isize,
                    packed.data.as_ptr(),
                    1,
                    n as isize,
                    0.0,
                    beta,
                    ep_ptr,
                )
            }
        };
        if rc == 0 {
            return;
        }
        // rc != 0: an allocation failed and c is untouched. Unpack B to row-major
        // and drop into the scalar fallback below.
        if packed.sme {
            unpacked = unpack_b_f32_sme(&packed.data, n, k);
        }
    }

    // Fallback: reference GEMM (beta scale, overwrite) at the chosen layout,
    // then the scalar epilogue indexed by the same strides.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let b: &[f32] = if unpacked.is_empty() {
        &packed.data
    } else {
        &unpacked
    };
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let b: &[f32] = &packed.data;
    let (rs, cs) = (dst_rs as usize, dst_cs as usize);
    reference::gemm_f32(m, n, k, c, rs, cs, a, k, 1, b, n, 1, 0.0, beta);
    if has_ep {
        apply_ep_scalar(c, m, n, rs, cs, ep);
    }
}

/// `C = softmax_rows(A @ B)` with B pre-packed, row-major `m x n` output.
///
/// One GEMM whose store also yields the per-row maxima, then [`softmax_rows`]
/// with the max sweep skipped. Saves the caller a max buffer and one pass over
/// `c`; the softmax itself is the same parallel NEON pass either way.
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the packed `n`, `k`.
pub fn softmax_gemm_f32(a: &[f32], packed: &Packed<f32>, c: &mut [f32], m: usize) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if packed.sme {
        let mut row_max = vec![0.0f32; m];
        let _busy = crate::warm::busy("gemm_sme_f32_softmax", m, n, k);
        // SAFETY: a is row-major m*k, c row-major m*n, `packed.data` is the packb
        // panel for these (n, k), and row_max is the length-m scratch the kernel
        // fills before consuming it.
        let rc = unsafe {
            gemm_sme_f32_softmax(
                m,
                n,
                k,
                c.as_mut_ptr(),
                a.as_ptr(),
                1,
                k as isize,
                packed.data.as_ptr(),
                row_max.as_mut_ptr(),
            )
        };
        if rc == 0 {
            return;
        }
    }
    // Everything else (unpacked B, or the fused kernel's allocation failing):
    // the same two steps separately, still skipping the softmax's max sweep.
    let mut row_max = vec![f32::NEG_INFINITY; m];
    f32_packed_ep_reduce(
        a,
        packed,
        c,
        m,
        &Epilogue::new(),
        1.0,
        RowReduce::new().max(&mut row_max),
    );
    softmax_rows_with_max(c, m, n, &row_max);
}

/// f32 fused path with per-row reductions. Off-SME (or on OOM) it computes them
/// from the finished `c`, so the result is identical either way.
pub(crate) fn f32_packed_ep_reduce(
    a: &[f32],
    packed: &Packed<f32>,
    c: &mut [f32],
    m: usize,
    ep: &Epilogue<'_, f32>,
    beta: f32,
    mut red: RowReduce<'_>,
) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    validate_ep(ep, m, n);
    if m == 0 || n == 0 {
        return;
    }
    let has_ep = !ep.nodes.is_empty() || ep.act.is_some();

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if packed.sme || sme_worth_it(m, n, k) {
        let fnodes = resolve_nodes(&ep.nodes, ep.act, n);
        let desc = EpDescF32 {
            n_nodes: u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
            nodes: fnodes.as_ptr(),
            row_sum: red
                .sum
                .as_deref_mut()
                .map_or(core::ptr::null_mut(), <[f32]>::as_mut_ptr),
            row_max: red
                .max
                .as_deref_mut()
                .map_or(core::ptr::null_mut(), <[f32]>::as_mut_ptr),
        };
        let _busy = crate::warm::busy("gemm_sme_f32_run_packed", m, n, k);
        // SAFETY: as `f32_packed_ep_impl`, plus two length-m `&mut [f32]` outputs
        // the kernel writes one row at a time. The driver keeps every N-tile of a
        // row on one thread (its N-parallel branch is disabled when a reduction
        // is set), so those writes are unsynchronized but race-free.
        let rc = unsafe {
            if packed.sme {
                gemm_sme_f32_run_packed(
                    m,
                    n,
                    k,
                    c.as_mut_ptr(),
                    1,
                    n as isize,
                    0,
                    a.as_ptr(),
                    1,
                    k as isize,
                    packed.data.as_ptr(),
                    0.0,
                    beta,
                    &raw const desc,
                )
            } else {
                gemm_sme_f32_run(
                    m,
                    n,
                    k,
                    c.as_mut_ptr(),
                    1,
                    n as isize,
                    0,
                    a.as_ptr(),
                    1,
                    k as isize,
                    packed.data.as_ptr(),
                    1,
                    n as isize,
                    0.0,
                    beta,
                    &raw const desc,
                )
            }
        };
        if rc == 0 {
            return;
        }
    }

    // Fallback: plain fused GEMM, then reduce over the finished rows.
    let _ = has_ep;
    f32_packed_ep_impl(a, packed, c, m, ep, beta, false);
    for (i, row) in c.chunks_exact(n).enumerate() {
        if let Some(s) = red.sum.as_deref_mut() {
            s[i] = row.iter().sum();
        }
        if let Some(mx) = red.max.as_deref_mut() {
            mx[i] = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        }
    }
}

/// Numerically-stable row softmax over an already-computed `m x n` row-major
/// `c`, in place: `c[i][j] = exp(c[i][j] - max_i) / sum_j exp(...)`.
///
/// [`softmax_rows_with_max`] skips the max sweep when the maxima are already
/// known from [`Gemm::run_reduce`](crate::Gemm::run_reduce).
///
/// ```
/// # use sme_gemm::{prepack_f32, Gemm, softmax_rows};
/// # let (m, n, k) = (4, 8, 4);
/// # let (a, b) = (vec![0.1f32; m * k], vec![0.2f32; k * n]);
/// let w = prepack_f32(&b, n, k);
/// let mut c = vec![0.0f32; m * n];
/// Gemm::new(&a, &w, m).run(&mut c);
/// softmax_rows(&mut c, m, n);
/// ```
///
/// # Panics
/// Panics if `c.len() != m * n`.
pub fn softmax_rows(c: &mut [f32], m: usize, n: usize) {
    softmax_rows_inner(c, m, n, None);
}

/// [`softmax_rows`] given per-row maxima already produced by
/// [`Gemm::run_reduce`](crate::Gemm::run_reduce), which skips this function's own max sweep.
///
/// # Panics
/// Panics if `c.len() != m * n` or `row_max.len() != m`.
pub fn softmax_rows_with_max(c: &mut [f32], m: usize, n: usize, row_max: &[f32]) {
    assert_eq!(row_max.len(), m, "row_max is per-M (length m)");
    softmax_rows_inner(c, m, n, Some(row_max));
}

fn softmax_rows_inner(c: &mut [f32], m: usize, n: usize, row_max: Option<&[f32]>) {
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n (row-major)");
    // NEON, not SME -- no capability probe, just the platform gate.
    // SAFETY: `c` is exactly m*n row-major (asserted above) and `row_max`, when
    // present, is length m (asserted by the caller). The pass reads and writes
    // only those ranges.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe {
        crate::ffi::attn_softmax_rows_f32(
            c.as_mut_ptr(),
            m,
            n,
            n as isize,
            row_max.map_or(core::ptr::null(), <[f32]>::as_ptr),
        );
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    for (i, row) in c.chunks_exact_mut(n).enumerate() {
        let mx = row_max.map_or_else(
            || row.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            |r| r[i],
        );
        // An all-(-inf) or empty row has no finite max; leave it rather than
        // producing NaN from inf - inf.
        if !mx.is_finite() {
            continue;
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = (*v - mx).exp();
            *v = e;
            sum += e;
        }
        if sum != 0.0 {
            let inv = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
}

/// Per-item scalar reference fallback for `matmul_f32_batched_ep`: `count`
/// independent row-major `C_i = ep(A_i @ B_i)` (item strides `m*k` / `k*n` /
/// `m*n`, alpha=0/beta=1), each followed by the shared scalar epilogue. Mirrors
/// the inlined fallback loop exactly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batched_ep_f32(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    count: usize,
    m: usize,
    n: usize,
    k: usize,
    ep: &Epilogue<'_, f32>,
) {
    for i in 0..count {
        let ci = &mut c[i * m * n..(i + 1) * m * n];
        reference::gemm_f32(
            m,
            n,
            k,
            ci,
            n,
            1,
            &a[i * m * k..(i + 1) * m * k],
            k,
            1,
            &b[i * k * n..(i + 1) * k * n],
            n,
            1,
            0.0,
            1.0,
        );
        apply_ep_scalar(ci, m, n, n, 1, ep);
    }
}

/// Per-item scalar reference fallback for `matmul_f64_batched_ep` (double
/// precision). See [`batched_ep_f32`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn batched_ep_f64(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    count: usize,
    m: usize,
    n: usize,
    k: usize,
    ep: &Epilogue<'_, f64>,
) {
    for i in 0..count {
        let ci = &mut c[i * m * n..(i + 1) * m * n];
        reference::gemm_f64(
            m,
            n,
            k,
            ci,
            n,
            1,
            &a[i * m * k..(i + 1) * m * k],
            k,
            1,
            &b[i * k * n..(i + 1) * k * n],
            n,
            1,
            0.0,
            1.0,
        );
        apply_ep_scalar_f64(ci, m, n, n, 1, ep);
    }
}

/// f64 fused op-graph epilogue: `D = act(beta*(A@B) + composable bias)` applied
/// in-register at the f64 store (FMOPA double-precision, M5+ `FEAT_SME_F64F64`).
/// The f64 working domain is already f64, so the op-graph runs with no dtype
/// conversion (the node `scalar` field is f32 and widens to f64). B has no
/// dedicated `packb`, so `packed.data` is the row-major copy passed straight to
/// the kernel. On a CPU without the extension or an allocation failure, falls
/// back to the reference GEMM and the f64-precise scalar op-graph pass.
pub(crate) fn f64_packed_ep_impl(
    a: &[f64],
    packed: &Packed<f64>,
    c: &mut [f64],
    m: usize,
    ep: &Epilogue<'_, f64>,
    beta: f64,
    col_major: bool,
) {
    let (n, k) = (packed.n, packed.k);
    assert_eq!(a.len(), checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), checked_dim2(m, n), "c is m*n");
    validate_ep(ep, m, n);
    if m == 0 || n == 0 {
        return;
    }
    let (dst_cs, dst_rs): (isize, isize) = if col_major {
        (m as isize, 1)
    } else {
        (1, n as isize)
    };
    let has_ep = !ep.nodes.is_empty() || ep.act.is_some();

    // On the OOM fallback this owns an unpacked row-major B; otherwise the scalar
    // fallback borrows `packed.data` directly (sme == false).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let mut unpacked: Vec<f64> = Vec::new();
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_f64f64 && (packed.sme || sme_worth_it(m, n, k)) {
        let fnodes = resolve_nodes(&ep.nodes, ep.act, n);
        let desc = EpDescF64 {
            n_nodes: u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
            nodes: fnodes.as_ptr(),
        };
        // NULL when no epilogue work, so the unscaled row-major call keeps the
        // single-instruction pure-store fast path.
        let ep_ptr = if has_ep {
            &raw const desc
        } else {
            core::ptr::null()
        };
        let _busy = crate::warm::busy("gemm_sme_f64_run_packed", m, n, k);
        // SAFETY: a is row-major m*k; `packed.data` is either the packb panel for
        // these (n, k) or a row-major k*n copy, matched to the entry point called
        // below; c strides describe the chosen layout; operands were validated.
        let rc = unsafe {
            if packed.sme {
                gemm_sme_f64_run_packed(
                    m,
                    n,
                    k,
                    c.as_mut_ptr(),
                    dst_cs,
                    dst_rs,
                    0,
                    a.as_ptr(),
                    1,
                    k as isize,
                    packed.data.as_ptr(),
                    0.0,
                    beta,
                    ep_ptr,
                )
            } else {
                gemm_sme_f64_run(
                    m,
                    n,
                    k,
                    c.as_mut_ptr(),
                    dst_cs,
                    dst_rs,
                    0,
                    a.as_ptr(),
                    1,
                    k as isize,
                    packed.data.as_ptr(),
                    1,
                    n as isize,
                    0.0,
                    beta,
                    ep_ptr,
                )
            }
        };
        if rc == 0 {
            return;
        }
        if packed.sme {
            unpacked = unpack_b_f64_sme(&packed.data, n, k);
        }
    }

    // Fallback: reference GEMM (beta scale, overwrite) at the chosen layout,
    // then the f64-precise scalar epilogue indexed by the same strides.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let b: &[f64] = if unpacked.is_empty() {
        &packed.data
    } else {
        &unpacked
    };
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let b: &[f64] = &packed.data;
    let (rs, cs) = (dst_rs as usize, dst_cs as usize);
    reference::gemm_f64(m, n, k, c, rs, cs, a, k, 1, b, n, 1, 0.0, beta);
    if has_ep {
        apply_ep_scalar_f64(c, m, n, rs, cs, ep);
    }
}
