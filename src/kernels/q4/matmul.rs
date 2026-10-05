//! The 4-bit-resident GEMM entry points: `C = A @ dequant(B)` in f16 and bf16,
//! each with a fused-epilogue variant.

use half::{bf16, f16};

use super::Q4Weights;
use crate::epilogue::Epilogue;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::ffi::{gemm_sme_b16b16_q4, gemm_sme_f16f16_q4};
use crate::kernels::bf16::prepack_bf16;
use crate::kernels::f16::prepack_f16;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::probe::caps;

/// [`matmul_q4`] with a fused epilogue: `C = ep(A @ dequant(B))`, the op-graph
/// applied in-register at the f16 store (no second pass over `c`).
///
/// # Panics
/// Panics as [`matmul_q4`], plus if an epilogue operand length mismatches
/// `m`/`n`.
pub fn matmul_q4_ep(a: &[f16], w: &Q4Weights, c: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
    q4_f16_impl(a, w, c, m, ep);
}

/// Row-major `C = A @ dequant(B)` with B kept 4-bit resident (see [`Q4Weights`]);
/// each B-tile is dequantized to f16 on the fly. `a` is `m x k` row-major f16,
/// `c` is `m x n` row-major f16.
///
/// [`matmul_q4_ep`] fuses a bias/activation op-graph into the same store.
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the weights' `n`, `k`.
pub fn matmul_q4(a: &[f16], w: &Q4Weights, c: &mut [f16], m: usize) {
    q4_f16_impl(a, w, c, m, &Epilogue::new());
}

fn q4_f16_impl(a: &[f16], w: &Q4Weights, c: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
    let (n, k) = (w.n, w.k);
    assert_eq!(a.len(), crate::exec::checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), crate::exec::checked_dim2(m, n), "c is m*n");
    crate::exec::validate_ep(ep, m, n);
    if m == 0 || n == 0 {
        return;
    }
    // No flop gate: the fallback has to unpack the whole band-major weight set
    // scalar-wise, which costs O(n*k) no matter how small m is, so gating on
    // m*n*k makes the small shapes it is meant to protect slower, not faster.
    // Same rule as the other pre-packed paths (see `packed_ep_impl`).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_f16f16 {
        let fnodes = crate::exec::resolve_nodes(&ep.nodes, ep.act, n);
        let desc = crate::ffi::EpDesc16 {
            n_nodes: u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
            nodes: fnodes.as_ptr(),
        };
        // NULL when there is no epilogue work, so the plain path keeps its
        // single-instruction ZA-slice store.
        let ep_ptr = if fnodes.is_empty() {
            core::ptr::null()
        } else {
            &raw const desc
        };
        let _busy = crate::warm::busy("gemm_sme_f16f16_q4", m, n, k);
        // SAFETY: a / c are row-major m*k / m*n; nibbles/scales describe the
        // weights' (n, k) in the band-major resident layout the kernel expects;
        // the node operands are length-validated above and outlive the call.
        let rc = unsafe {
            gemm_sme_f16f16_q4(
                m,
                n,
                k,
                c.as_mut_ptr().cast::<u16>(),
                a.as_ptr().cast::<u16>(),
                w.nibbles.as_ptr(),
                w.scales.as_ptr(),
                if w.mins.is_empty() {
                    core::ptr::null()
                } else {
                    w.mins.as_ptr()
                },
                w.params.block,
                ep_ptr,
            )
        };
        // rc != 0 means an allocation failed and c is untouched; fall through to
        // the scalar dequant-then-reference path below for a correct result.
        if rc == 0 {
            return;
        }
    }
    // Fallback: dequantize the band-major weights to row-major f16, then the
    // dense packed path, which carries the same op-graph.
    let b = w.dequant_to_rowmajor();
    crate::kernels::f16::f16_packed_ep_impl(
        a,
        &prepack_f16(&b, n, k),
        c,
        m,
        ep,
        f16::from_f32(1.0),
        false,
    );
}

/// One row of [`matmul_q4_ep`] as a link of a chain ([`Mlp`](crate::Mlp)):
/// raises `done` to each output column once it is stored and waits on `ready`
/// for the depths of `a` (either `None`), with the whole op-graph, activation
/// included, in the kernel. `false` with nothing done when there is no SME f16
/// path, or when the kernel failed (`c` untouched, `done` not raised).
///
/// # Safety
/// `a` holds `w.k()` and `c` `w.n()` values for the whole call; other threads
/// write `a` only below what `ready` has published and read `c` only below
/// what `done` has.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) unsafe fn q4_chained(
    a: *const f16,
    w: &Q4Weights,
    c: *mut f16,
    ep: &Epilogue<'_, f16>,
    done: Option<&core::sync::atomic::AtomicUsize>,
    ready: Option<&core::sync::atomic::AtomicUsize>,
) -> bool {
    let (m, n, k) = (1, w.n, w.k);
    crate::exec::validate_ep(ep, m, n);
    if n == 0 || !caps().sme_f16f16 {
        return false;
    }
    let fnodes = crate::exec::resolve_nodes(&ep.nodes, ep.act, n);
    let desc = crate::ffi::EpDesc16 {
        n_nodes: u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
        nodes: fnodes.as_ptr(),
    };
    let ep_ptr = if fnodes.is_empty() {
        core::ptr::null()
    } else {
        &raw const desc
    };
    let _busy = crate::warm::busy("gemm_sme_f16f16_q4", m, n, k);
    // SAFETY: the caller's contract covers a and c; the weights describe
    // (n, k) as in `q4_f16_impl`; `done` and `ready` outlive the call, and the
    // kernel only stores to the one and loads from the other.
    let rc = unsafe {
        crate::ffi::gemm_sme_f16f16_q4_chained(
            m,
            n,
            k,
            c.cast::<u16>(),
            a.cast::<u16>(),
            w.nibbles.as_ptr(),
            w.scales.as_ptr(),
            if w.mins.is_empty() {
                core::ptr::null()
            } else {
                w.mins.as_ptr()
            },
            w.params.block,
            ep_ptr,
            done.map_or(core::ptr::null(), core::ptr::from_ref),
            ready.map_or(core::ptr::null(), core::ptr::from_ref),
        )
    };
    rc == 0
}

/// [`matmul_q4`] with bf16 activations and output (B16B16 MOPA, M5+
/// `FEAT_SME_B16B16`): row-major `C = A @ dequant(B)`, B kept 4-bit resident.
///
/// The same [`Q4Weights`] serves both this and [`matmul_q4`] -- scales and mins
/// stay f16 (the GGUF `Q4_0`/`Q4_1` on-disk type) and each B-tile is rounded to
/// bf16 as it is dequantized. bf16's 8-bit mantissa is the accuracy floor here,
/// so prefer [`matmul_q4`] unless the surrounding model is already bf16.
///
/// [`matmul_q4_ep`] fuses a bias/activation op-graph into the same store.
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the weights' `n`, `k`.
pub fn matmul_q4_bf16(a: &[bf16], w: &Q4Weights, c: &mut [bf16], m: usize) {
    q4_bf16_impl(a, w, c, m, &Epilogue::new());
}

/// [`matmul_q4_bf16`] with a fused epilogue -- the bf16 twin of
/// [`matmul_q4_ep`].
///
/// # Panics
/// Panics as [`matmul_q4_bf16`], plus if an epilogue operand length mismatches
/// `m`/`n`.
pub fn matmul_q4_bf16_ep(
    a: &[bf16],
    w: &Q4Weights,
    c: &mut [bf16],
    m: usize,
    ep: &Epilogue<'_, bf16>,
) {
    q4_bf16_impl(a, w, c, m, ep);
}

fn q4_bf16_impl(a: &[bf16], w: &Q4Weights, c: &mut [bf16], m: usize, ep: &Epilogue<'_, bf16>) {
    let (n, k) = (w.n, w.k);
    assert_eq!(a.len(), crate::exec::checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), crate::exec::checked_dim2(m, n), "c is m*n");
    crate::exec::validate_ep(ep, m, n);
    if m == 0 || n == 0 {
        return;
    }
    // Unconditional for the same reason as `q4_f16_impl`.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_b16b16 {
        let fnodes = crate::exec::resolve_nodes(&ep.nodes, ep.act, n);
        let desc = crate::ffi::EpDesc16 {
            n_nodes: u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
            nodes: fnodes.as_ptr(),
        };
        let ep_ptr = if fnodes.is_empty() {
            core::ptr::null()
        } else {
            &raw const desc
        };
        // The GEMV kernel (m <= 7, Q4_GEMV_MAXR) folds the scale per K-block in bf16.
        let (sb, mb) = if m <= 7 {
            let (sb, mb) = w.bf16_scales();
            (
                sb.as_ptr(),
                if mb.is_empty() {
                    core::ptr::null()
                } else {
                    mb.as_ptr()
                },
            )
        } else {
            (core::ptr::null(), core::ptr::null())
        };
        let _busy = crate::warm::busy("gemm_sme_b16b16_q4", m, n, k);
        // SAFETY: as `q4_f16_impl`, with bf16 activations/output -- the kernel
        // reads the same band-major nibbles and f16 scales; the bf16 copies have
        // the same shape.
        let rc = unsafe {
            gemm_sme_b16b16_q4(
                m,
                n,
                k,
                c.as_mut_ptr().cast::<u16>(),
                a.as_ptr().cast::<u16>(),
                w.nibbles.as_ptr(),
                w.scales.as_ptr(),
                if w.mins.is_empty() {
                    core::ptr::null()
                } else {
                    w.mins.as_ptr()
                },
                sb,
                mb,
                w.params.block,
                ep_ptr,
            )
        };
        if rc == 0 {
            return;
        }
    }
    let b = w.dequant_to_rowmajor_bf16();
    crate::kernels::bf16::bf16_packed_ep_impl(
        a,
        &prepack_bf16(&b, n, k),
        c,
        m,
        ep,
        bf16::from_f32(1.0),
        false,
    );
}
