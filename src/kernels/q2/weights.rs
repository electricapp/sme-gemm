//! The 2-bit-resident weight set: the band-major pack the kernel reads, from
//! row-major codes or quantized from f32.

use super::Q2Params;
use crate::WeightLayout;

/// Columns in a band (`Q2_BAND` in `csrc/gemm_q2.c`): per four depths, a
/// band's 64 bytes hold its 64 columns' codes, which one LUTI2 expands into
/// one SDOT's int8 operand.
pub(super) const BAND: usize = 64;

/// 2-bit weights kept **2-bit resident** for [`matmul_q2`].
///
/// Codes are stored band-major (64 columns by four depths per 64 bytes) with
/// one f32 scale per (column, K-block); a weight is `scale *
/// codebook[code]` ([`Q2Params`]). That is 2.125 bits a weight at the default
/// 256-value blocks.
///
/// [`matmul_q2`]: crate::matmul_q2
#[derive(Debug)]
pub struct Q2Weights {
    /// The band-major arrays cross the FFI from `matmul`, hence `pub(super)`.
    pub(super) codes: Vec<u8>,
    pub(super) scales: Vec<f32>,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) params: Q2Params,
}

impl Q2Weights {
    /// Packs row-major 2-bit codes resident.
    ///
    /// The weights are logically `k x n` row-major (input-major, as
    /// [`WeightLayout::InOut`]). `codes` holds them four to a byte in that
    /// order, low bits first: weight `idx = d * n + j` is bits `2 * (idx % 4)`
    /// of byte `idx / 4` (`len == (k*n).div_ceil(4)`). `scales` holds one f32
    /// per (column, K-block): `scales[j * ceil(k/block) + d / block]`.
    ///
    /// # Panics
    /// Panics if `codes` or `scales` is inconsistent with `n`, `k` and the
    /// block size.
    #[must_use]
    pub fn new(codes: &[u8], scales: &[f32], n: usize, k: usize, p: Q2Params) -> Self {
        let kn = k
            .checked_mul(n)
            .expect("q2 weight dimension product k*n overflows usize");
        assert_eq!(codes.len(), kn.div_ceil(4), "codes is ceil(k*n/4) bytes");
        let nbk = k.div_ceil(p.block);
        assert_eq!(
            scales.len(),
            n.checked_mul(nbk)
                .expect("q2 scales dimension product overflows usize"),
            "scales is n*ceil(k/block)"
        );
        let (band_bytes, sc_per_band) = Self::band_sizes(k, p.block);
        let n_bands = n.div_ceil(BAND);
        let mut packed = vec![0u8; n_bands * band_bytes];
        let mut sc = vec![0f32; n_bands * sc_per_band];
        for d in 0..k {
            for j in 0..n {
                let idx = d * n + j;
                let code = (codes[idx / 4] >> (2 * (idx % 4))) & 3;
                packed[Self::byte(band_bytes, d, j)] |= code << (2 * (d % 4));
            }
        }
        for j in 0..n {
            for blk in 0..nbk {
                sc[(j / BAND) * sc_per_band + blk * BAND + j % BAND] = scales[j * nbk + blk];
            }
        }
        Self {
            codes: packed,
            scales: sc,
            n,
            k,
            params: p,
        }
    }

    /// Quantizes f32 weights to 2 bits each and packs them resident.
    ///
    /// `w` holds `n` outputs by `k` inputs in `layout`. Each (output, K-block)
    /// gets the scale that maps its largest magnitude onto the codebook's, and
    /// each weight the code whose value is nearest. Weights already on a 2-bit
    /// grid -- a ternary model's `{-t, 0, t}` -- come back exactly; for
    /// full-precision weights this is plain rounding to four levels, no
    /// substitute for training at two bits.
    ///
    /// ```
    /// use sme_gemm::{Q2Params, Q2Weights, WeightLayout};
    /// let (n, k) = (64, 256);
    /// let w: Vec<f32> = (0..n * k).map(|i| [-0.5, 0.0, 0.5][i % 3]).collect();
    /// let q = Q2Weights::quantize(&w, WeightLayout::OutIn, n, k, Q2Params::default());
    /// assert_eq!((q.n(), q.k()), (n, k));
    /// ```
    ///
    /// # Panics
    /// Panics if `w.len() != n * k`, or if the codebook is all zeros.
    #[must_use]
    pub fn quantize(w: &[f32], layout: WeightLayout, n: usize, k: usize, p: Q2Params) -> Self {
        assert_eq!(
            w.len(),
            crate::exec::checked_dim2(n, k),
            "w holds n * k weights"
        );
        let cb = p.codebook.map(f32::from);
        let cmax = cb.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(cmax > 0.0, "q2 codebook is all zeros");
        let at = |j: usize, d: usize| match layout {
            WeightLayout::OutIn => w[j * k + d],
            WeightLayout::InOut => w[d * n + j],
        };
        let nbk = k.div_ceil(p.block);
        let mut codes = vec![0u8; (k * n).div_ceil(4)];
        let mut scales = vec![0f32; n * nbk];
        for j in 0..n {
            for b in 0..nbk {
                let (d0, d1) = (b * p.block, ((b + 1) * p.block).min(k));
                let amax = (d0..d1).fold(0f32, |a, d| a.max(at(j, d).abs()));
                let s = amax / cmax;
                scales[j * nbk + b] = s;
                for d in d0..d1 {
                    let x = at(j, d);
                    // Nearest value; ties go to the lower code.
                    let code = (0..4)
                        .min_by(|&a, &c| (x - s * cb[a]).abs().total_cmp(&(x - s * cb[c]).abs()))
                        .unwrap_or(0);
                    let idx = d * n + j;
                    codes[idx / 4] |= (code as u8) << (2 * (idx % 4));
                }
            }
        }
        Self::new(&codes, &scales, n, k, p)
    }

    /// Bytes of codes and scales per band: k padded to 16 depths at 16 bytes
    /// a depth, and 64 f32 per K-block.
    pub(super) const fn band_sizes(k: usize, block: usize) -> (usize, usize) {
        (k.next_multiple_of(16) * 16, k.div_ceil(block) * BAND)
    }

    /// The packed byte holding column `j`'s code at depth `d`.
    pub(super) const fn byte(band_bytes: usize, d: usize, j: usize) -> usize {
        (j / BAND) * band_bytes + (d / 4) * BAND + j % BAND
    }

    /// Column `j`'s code at depth `d`.
    pub(super) fn code(&self, d: usize, j: usize) -> usize {
        let (band_bytes, _) = Self::band_sizes(self.k, self.params.block);
        usize::from((self.codes[Self::byte(band_bytes, d, j)] >> (2 * (d % 4))) & 3)
    }

    /// Column `j`'s scale for K-block `blk`.
    pub(super) fn scale(&self, blk: usize, j: usize) -> f32 {
        let (_, sc_per_band) = Self::band_sizes(self.k, self.params.block);
        self.scales[(j / BAND) * sc_per_band + blk * BAND + j % BAND]
    }

    /// The quantization layout these weights were packed with.
    #[must_use]
    pub const fn params(&self) -> Q2Params {
        self.params
    }

    /// Columns (`n`) of the weight matrix.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Depth (`k`) of the weight matrix.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::Q2Weights;
    use crate::WeightLayout;
    use crate::kernels::q2::Q2Params;

    /// The band-major pack keeps every row-major code and scale where `code`
    /// and `scale` read them, including ragged n, k and the last block.
    #[test]
    fn pack_round_trips() {
        for &(n, k, block) in &[(1usize, 1usize, 16usize), (65, 37, 16), (130, 300, 256)] {
            let nbk = k.div_ceil(block);
            let codes: Vec<u8> = (0..(k * n).div_ceil(4))
                .map(|i| (i * 37 % 251) as u8)
                .collect();
            let scales: Vec<f32> = (0..n * nbk).map(|i| i as f32 * 0.5).collect();
            let q = Q2Weights::new(&codes, &scales, n, k, Q2Params::new(block));
            for d in 0..k {
                for j in 0..n {
                    let idx = d * n + j;
                    let want = usize::from((codes[idx / 4] >> (2 * (idx % 4))) & 3);
                    assert_eq!(q.code(d, j), want, "{n}x{k} code ({d}, {j})");
                }
            }
            for j in 0..n {
                for b in 0..nbk {
                    assert_eq!(q.scale(b, j).to_bits(), scales[j * nbk + b].to_bits());
                }
            }
        }
    }

    /// Weights already on a ternary or four-level grid come back exactly,
    /// from either layout.
    #[test]
    fn quantize_recovers_grid_weights() {
        let (n, k) = (70usize, 300usize);
        for (cb, levels) in [
            ([-1i8, 0, 1, 0], &[-1.0f32, 0.0, 1.0][..]),
            ([-2, -1, 1, 2], &[-2.0, -1.0, 1.0, 2.0][..]),
        ] {
            let p = Q2Params::new(64).with_codebook(cb);
            let w: Vec<f32> = (0..n * k)
                .map(|i| levels[(i * 7 + i / 5) % levels.len()] * 0.03 * (1 + i / k % 3) as f32)
                .collect();
            let t: Vec<f32> = (0..k * n).map(|i| w[(i % n) * k + i / n]).collect();
            for (q, what) in [
                (
                    Q2Weights::quantize(&w, WeightLayout::OutIn, n, k, p),
                    "OutIn",
                ),
                (
                    Q2Weights::quantize(&t, WeightLayout::InOut, n, k, p),
                    "InOut",
                ),
            ] {
                for j in 0..n {
                    for d in 0..k {
                        let got = q.scale(d / 64, j) * f32::from(cb[q.code(d, j)]);
                        assert_eq!(
                            got.to_bits(),
                            w[j * k + d].to_bits(),
                            "{cb:?} {what} w[{j}][{d}]"
                        );
                    }
                }
            }
        }
    }
}
