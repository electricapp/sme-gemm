//! The 2-bit-resident GEMM: `C = A @ dequant(B)` in f16, with a fused-epilogue
//! variant, and the reference it matches bit for bit.

use half::f16;

use super::Q2Weights;
use crate::epilogue::Epilogue;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::probe::caps;

/// Row-major `C = A @ dequant(B)` with B kept 2-bit resident (see
/// [`Q2Weights`]). `a` is `m x k` row-major f16, `c` is `m x n` row-major f16.
///
/// Each row of `a` is quantized to int8 per K-block inside the call (scale
/// `max|a|/127`, round to nearest even), and the products are exact integer
/// sums folded block by block into f32: activation rounding, about 0.4% of a
/// block's largest value, is the accuracy floor beyond the weights' own. One
/// row runs at the int8 dot-product rate, twice the Q4 GEMV's on M5, and the
/// weights are half Q4's bytes. Every row runs that GEMV, so for many rows
/// the call stays at it rather than reaching the f16 MOPA rate. Results do
/// not depend on `m` or on how the call splits over cores.
///
/// [`matmul_q2_ep`] fuses a bias/activation op-graph into the same store.
///
/// ```
/// use half::f16;
/// use sme_gemm::{Q2Params, Q2Weights, WeightLayout, matmul_q2};
/// let (n, k) = (64, 256);
/// let w: Vec<f32> = (0..n * k).map(|i| [-0.5, 0.0, 0.5][i % 3]).collect();
/// let q = Q2Weights::quantize(&w, WeightLayout::OutIn, n, k, Q2Params::default());
/// let (a, mut c) = (vec![f16::ONE; k], vec![f16::ZERO; n]);
/// matmul_q2(&a, &q, &mut c, 1);
/// ```
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the weights' `n`, `k`.
pub fn matmul_q2(a: &[f16], w: &Q2Weights, c: &mut [f16], m: usize) {
    q2_impl(a, w, c, m, &Epilogue::new());
}

/// [`matmul_q2`] with a fused epilogue: `C = ep(A @ dequant(B))`, applied at
/// the f16 store.
///
/// # Panics
/// Panics as [`matmul_q2`], plus if an epilogue operand length mismatches
/// `m`/`n`.
pub fn matmul_q2_ep(a: &[f16], w: &Q2Weights, c: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
    q2_impl(a, w, c, m, ep);
}

fn q2_impl(a: &[f16], w: &Q2Weights, c: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
    let (n, k) = (w.n, w.k);
    assert_eq!(a.len(), crate::exec::checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), crate::exec::checked_dim2(m, n), "c is m*n");
    crate::exec::validate_ep(ep, m, n);
    if m == 0 || n == 0 {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme2p1 {
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
        let _busy = crate::warm::busy("gemm_sme_f16_q2", m, n, k);
        // SAFETY: a / c are row-major m*k / m*n; codes and scales describe the
        // weights' (n, k) in the band-major layout the kernel expects; the
        // codebook is 4 bytes; the node operands are length-validated above
        // and outlive the call.
        let rc = unsafe {
            crate::ffi::gemm_sme_f16_q2(
                m,
                n,
                k,
                c.as_mut_ptr().cast::<u16>(),
                a.as_ptr().cast::<u16>(),
                w.codes.as_ptr(),
                w.scales.as_ptr(),
                w.params.codebook.as_ptr(),
                w.params.block,
                ep_ptr,
            )
        };
        // rc != 0 means the scratch allocation failed and c is untouched.
        if rc == 0 {
            return;
        }
    }
    reference(a, w, c, m);
    crate::exec::apply_ep_scalar(c, m, n, n, 1, ep);
}

/// One row's int8 activations and per-block scales, as `csrc/gemm_q2.c`
/// quantizes them: `amax` NaN if any value is.
fn quantize_row(a: &[f16], block: usize, q: &mut [i8], s: &mut [f32]) {
    for ((ab, qb), sb) in a.chunks(block).zip(q.chunks_mut(block)).zip(s.iter_mut()) {
        let amax = ab.iter().fold(0f32, |m, x| {
            let x = x.to_f32().abs();
            if x.is_nan() || m.is_nan() {
                f32::NAN
            } else {
                m.max(x)
            }
        });
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        *sb = amax / 127.0;
        for (o, x) in qb.iter_mut().zip(ab) {
            // Saturating, NaN to 0, as the kernel's convert and narrow.
            *o = (x.to_f32() * inv).round_ties_even() as i8;
        }
    }
}

/// The 2-bit product without an epilogue, computed as the kernel does: exact
/// integer block sums, each folded as `total + f32(sum) * (scale * s)`, fused,
/// in K order.
pub(crate) fn reference(a: &[f16], w: &Q2Weights, c: &mut [f16], m: usize) {
    let (n, k, block) = (w.n, w.k, w.params.block);
    let nbk = k.div_ceil(block);
    let cb = w.params.codebook.map(i32::from);
    let (mut q, mut s) = (vec![0i8; k], vec![0f32; nbk]);
    for (ar, cr) in a.chunks_exact(k.max(1)).zip(c.chunks_exact_mut(n)).take(m) {
        quantize_row(&ar[..k], block, &mut q, &mut s);
        for (j, out) in cr.iter_mut().enumerate() {
            let mut total = 0f32;
            for (blk, &sa) in s.iter().enumerate() {
                let (d0, d1) = (blk * block, ((blk + 1) * block).min(k));
                let sum: i32 = (d0..d1).map(|d| i32::from(q[d]) * cb[w.code(d, j)]).sum();
                // An i32 converts as SCVTF does: round to nearest even.
                #[allow(clippy::cast_precision_loss)]
                let f = sum as f32;
                total = f.mul_add(w.scale(blk, j) * sa, total);
            }
            *out = f16::from_f32(total);
        }
    }
    if k == 0 {
        c[..m * n].fill(f16::ZERO);
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::reference;
    use crate::kernels::q2::{Q2Params, Q2Weights};

    fn data(seed: u64, len: usize) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (s >> 40) as u8
            })
            .collect()
    }

    /// The kernel gives the reference's bits for every row count, ragged
    /// shapes, block sizes and codebooks, single-threaded and spread over the
    /// cores (n large enough to dispatch).
    #[test]
    fn kernel_matches_reference_bits() {
        let shapes = [
            (1usize, 64usize, 256usize),
            (1, 65, 37),
            (1, 1, 1),
            (2, 200, 300),
            (3, 64, 16),
            (5, 130, 513),
            (6, 192, 256),
            (7, 100, 64),
            (12, 70, 200),
            (1, 2048, 1024),
            (4, 1536, 512),
            (9, 1024, 384),
        ];
        let books = [[-1i8, 0, 1, 0], [-2, -1, 1, 2], [-128, 127, 3, -7]];
        for (si, &(m, n, k)) in shapes.iter().enumerate() {
            for (bi, &block) in [16usize, 64, 256, 4096].iter().enumerate() {
                let book = books[(si + bi) % books.len()];
                let p = Q2Params::new(block).with_codebook(book);
                let nbk = k.div_ceil(block);
                let codes = data(si as u64 * 31 + bi as u64, (k * n).div_ceil(4));
                let scales: Vec<f32> = data(si as u64 + 7, n * nbk)
                    .iter()
                    .map(|&b| f32::from(b) * 1e-3 - 0.1)
                    .collect();
                let a: Vec<f16> = data(si as u64 + 99, m * k)
                    .iter()
                    .map(|&b| f16::from_f32((f32::from(b) - 128.0) / 37.0))
                    .collect();
                let w = Q2Weights::new(&codes, &scales, n, k, p);
                let mut want = vec![f16::ZERO; m * n];
                reference(&a, &w, &mut want, m);
                let mut got = vec![f16::from_f32(9.0); m * n];
                crate::matmul_q2(&a, &w, &mut got, m);
                for (i, (g, r)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        r.to_bits(),
                        "{m}x{n}x{k} block {block} {book:?}: c[{}][{}] = {g} vs {r}",
                        i / n,
                        i % n
                    );
                }
            }
        }
    }
}
