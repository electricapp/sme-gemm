//! Quantizing f32 weights to the 4-bit resident format.

use half::f16;

use super::{Q4Form, Q4Params, Q4Weights};
use crate::WeightLayout;

impl Q4Weights {
    /// Quantizes f32 weights to 4 bits each and packs them resident.
    ///
    /// `w` holds `n` outputs by `k` inputs in `layout`. Every (output, K-block)
    /// gets its own f16 scale. With [`Q4Form::Scale`] (`Q4_0`) the scale is the
    /// block's largest-magnitude value over -8 and a weight's code is
    /// `round(w / scale)` clamped to `-8..=7`, so that value is exact. With
    /// [`Q4Form::Affine`] (`Q4_1`) the block's `min..=max` spans the 16 codes.
    /// A partial last K-block quantizes over the values it has.
    ///
    /// ```
    /// use sme_gemm::{Q4Params, Q4Weights, WeightLayout};
    /// let (n, k) = (64, 128);
    /// let w: Vec<f32> = (0..n * k).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();
    /// let q = Q4Weights::quantize(&w, WeightLayout::OutIn, n, k, Q4Params::default());
    /// assert_eq!((q.n(), q.k()), (n, k));
    /// ```
    ///
    /// # Panics
    /// Panics if `w.len() != n * k`.
    #[must_use]
    pub fn quantize(w: &[f32], layout: WeightLayout, n: usize, k: usize, p: Q4Params) -> Self {
        assert_eq!(
            w.len(),
            crate::exec::checked_dim2(n, k),
            "w holds n * k weights"
        );
        let block = p.block;
        let nbk = k.div_ceil(block);
        let at = |j: usize, d: usize| match layout {
            WeightLayout::OutIn => w[j * k + d],
            WeightLayout::InOut => w[d * n + j],
        };
        let mut quants = vec![0u8; (k * n).div_ceil(2)];
        let mut scales = vec![f16::ZERO; n * nbk];
        let affine = p.form == Q4Form::Affine;
        let mut mins = if affine {
            vec![f16::ZERO; n * nbk]
        } else {
            Vec::new()
        };
        for j in 0..n {
            for b in 0..nbk {
                let (d0, d1) = (b * block, ((b + 1) * block).min(k));
                let (scale, min) = if affine {
                    let (lo, hi) = (d0..d1)
                        .map(|d| at(j, d))
                        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), x| {
                            (lo.min(x), hi.max(x))
                        });
                    // Codes -8..=7 over lo..=hi: min sits at code 0.
                    let s = f16::from_f32((hi - lo) / 15.0);
                    (s, f16::from_f32(lo + 8.0 * s.to_f32()))
                } else {
                    let m = (d0..d1)
                        .map(|d| at(j, d))
                        .fold(0f32, |a, x| if x.abs() > a.abs() { x } else { a });
                    (f16::from_f32(m / -8.0), f16::ZERO)
                };
                scales[j * nbk + b] = scale;
                if affine {
                    mins[j * nbk + b] = min;
                }
                let (s, m) = (scale.to_f32(), min.to_f32());
                let inv = if s == 0.0 { 0.0 } else { 1.0 / s };
                for d in d0..d1 {
                    let code = ((at(j, d) - m) * inv).round().clamp(-8.0, 7.0) as i8;
                    let idx = d * n + j;
                    quants[idx / 2] |= (code as u8 & 0x0f) << (4 * (idx % 2));
                }
            }
        }
        Self::with_params(&quants, &scales, affine.then_some(&mins[..]), n, k, p)
    }
}

#[cfg(test)]
mod tests {
    use crate::WeightLayout;
    use crate::kernels::q4::{Q4Params, Q4Weights};

    fn weights(n: usize, k: usize) -> Vec<f32> {
        (0..n * k)
            .map(|i| ((i * 2_654_435_761) % 1000) as f32 / 500.0 - 1.0)
            .collect()
    }

    /// Every weight comes back within half a step of its block's scale (a
    /// full step for `Q4_0` values clamped at code 7, opposite the block's
    /// largest magnitude), from either layout, for both forms and a ragged
    /// last block.
    #[test]
    fn quantize_round_trips_within_half_a_step() {
        for &(n, k, block) in &[(33usize, 64usize, 32usize), (40, 100, 32), (8, 96, 64)] {
            let w = weights(n, k);
            let t: Vec<f32> = (0..k * n).map(|i| w[(i % n) * k + i / n]).collect();
            for p in [Q4Params::new(block), Q4Params::new(block).affine()] {
                let a = Q4Weights::quantize(&w, WeightLayout::OutIn, n, k, p);
                let b = Q4Weights::quantize(&t, WeightLayout::InOut, n, k, p);
                let (da, db) = (a.dequant_to_rowmajor(), b.dequant_to_rowmajor());
                assert_eq!(da, db, "layouts disagree at {n}x{k} block {block}");
                for d in 0..k {
                    for j in 0..n {
                        let blk = &w
                            [j * k + (d / block) * block..j * k + ((d / block + 1) * block).min(k)];
                        let (lo, hi) = blk
                            .iter()
                            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| {
                                (l.min(x), h.max(x))
                            });
                        let (step, bound) = if p.form() == crate::Q4Form::Affine {
                            ((hi - lo) / 15.0, 0.5)
                        } else {
                            (lo.abs().max(hi.abs()) / 8.0, 1.0)
                        };
                        let err = (da[d * n + j].to_f32() - w[j * k + d]).abs();
                        // plus the f16 rounding of scale and min
                        assert!(
                            err <= step * bound + 0.01,
                            "{n}x{k} block {block} ({:?}) w[{j}][{d}]: err {err} > step {step}",
                            p.form()
                        );
                    }
                }
            }
        }
    }
}
