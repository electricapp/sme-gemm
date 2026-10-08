//! The 1-bit-resident weight set: signs band-major for the kernel, from packed
//! signs in either layout or quantized from f32.

use half::f16;

use super::q2::scale_index;
use crate::WeightLayout;

/// Columns in a band, as for 2-bit codes: per four depths a band holds 32
/// bytes, one nibble of four signs per column, which one LUTI4 expands into
/// one SDOT's int8 operand.
const BAND: usize = super::q2::BAND;

/// 1-bit weights kept **1-bit resident** for [`matmul_q1`]: each weight is
/// `scale * (+1 or -1)`, with one f16 scale per (column, K-block).
///
/// This is llama.cpp's `Q1_0` (`PrismML`'s 1-bit Bonsai models: 128-value
/// blocks, bit 1 for `+scale`) as it stands -- 1.125 bits a
/// weight at 128-value blocks.
///
/// [`matmul_q1`]: crate::matmul_q1
#[derive(Debug)]
pub struct Q1Weights {
    /// The band-major arrays cross the FFI from `matmul`, hence `pub(super)`.
    pub(super) signs: Vec<u8>,
    pub(super) scales: Vec<f16>,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) block: usize,
}

impl Q1Weights {
    /// Packs 1-bit weights resident.
    ///
    /// `signs` holds one bit per weight, 1 for `+scale`, in `layout` order and
    /// low bit first: weight `idx` (`j * k + d` for [`WeightLayout::OutIn`],
    /// `d * n + j` for [`WeightLayout::InOut`]) is bit `idx % 8` of byte
    /// `idx / 8`. `scales` holds one per (column, K-block):
    /// `scales[j * ceil(k/block) + d / block]`. A `Q1_0` tensor is `OutIn`
    /// with block 128: its signs are each block's 16 bytes in order, its
    /// scales the blocks' f16s.
    ///
    /// # Panics
    /// Panics if `block` is not a power of two from 16 to 65536, or if
    /// `signs` or `scales` is inconsistent with `n`, `k` and `block`.
    #[must_use]
    pub fn new(
        signs: &[u8],
        scales: &[f16],
        layout: WeightLayout,
        n: usize,
        k: usize,
        block: usize,
    ) -> Self {
        assert!(
            block.is_power_of_two() && (16..=1 << 16).contains(&block),
            "q1 block size must be a power of two from 16 to 65536, got {block}"
        );
        let kn = k
            .checked_mul(n)
            .expect("q1 weight dimension product k*n overflows usize");
        assert_eq!(signs.len(), kn.div_ceil(8), "signs is ceil(k*n/8) bytes");
        let nbk = k.div_ceil(block);
        assert_eq!(
            scales.len(),
            n.checked_mul(nbk)
                .expect("q1 scales dimension product overflows usize"),
            "scales is n*ceil(k/block)"
        );
        let (band_bytes, sc_per_band) = Self::band_sizes(k, block);
        let n_bands = n.div_ceil(BAND);
        let mut packed = vec![0u8; n_bands * band_bytes];
        let mut sc = vec![f16::ZERO; n_bands * sc_per_band];
        let whole_bytes = layout == WeightLayout::OutIn && k.is_multiple_of(8);
        for j in 0..n {
            if whole_bytes {
                // A byte of a row is eight depths: two quads' nibbles.
                let (base, shift) = Self::bit(band_bytes, 0, j);
                for (b, &byte) in signs[j * k / 8..(j + 1) * k / 8].iter().enumerate() {
                    packed[base + 2 * b * (BAND / 2)] |= (byte & 0x0f) << shift;
                    packed[base + (2 * b + 1) * (BAND / 2)] |= (byte >> 4) << shift;
                }
            } else {
                for d in 0..k {
                    let idx = match layout {
                        WeightLayout::OutIn => j * k + d,
                        WeightLayout::InOut => d * n + j,
                    };
                    if (signs[idx / 8] >> (idx % 8)) & 1 == 1 {
                        let (byte, bit) = Self::bit(band_bytes, d, j);
                        packed[byte] |= 1 << bit;
                    }
                }
            }
            for blk in 0..nbk {
                sc[scale_index(sc_per_band, blk, j)] = scales[j * nbk + blk];
            }
        }
        Self {
            signs: packed,
            scales: sc,
            n,
            k,
            block,
        }
    }

    /// Quantizes f32 weights to 1 bit each, as llama.cpp's `Q1_0` does: each
    /// (output, K-block) scale is the block's mean magnitude, and a weight is
    /// `+scale` when it is at least zero, else `-scale`. Weights already of
    /// the form `+-t` per block, `t` an f16, come back exactly.
    ///
    /// # Panics
    /// Panics if `w.len() != n * k`, or `block` as [`Q1Weights::new`].
    #[must_use]
    pub fn quantize(w: &[f32], layout: WeightLayout, n: usize, k: usize, block: usize) -> Self {
        assert_eq!(
            w.len(),
            crate::exec::checked_dim2(n, k),
            "w holds n * k weights"
        );
        let at = |j: usize, d: usize| match layout {
            WeightLayout::OutIn => w[j * k + d],
            WeightLayout::InOut => w[d * n + j],
        };
        let nbk = k.div_ceil(block);
        let mut signs = vec![0u8; (k * n).div_ceil(8)];
        let mut scales = vec![f16::ZERO; n * nbk];
        for j in 0..n {
            for b in 0..nbk {
                let (d0, d1) = (b * block, ((b + 1) * block).min(k));
                #[allow(clippy::cast_precision_loss)]
                let mean = (d0..d1).map(|d| at(j, d).abs()).sum::<f32>() / (d1 - d0) as f32;
                scales[j * nbk + b] = f16::from_f32(mean);
                for d in d0..d1 {
                    if at(j, d) >= 0.0 {
                        let idx = j * k + d;
                        signs[idx / 8] |= 1 << (idx % 8);
                    }
                }
            }
        }
        Self::new(&signs, &scales, WeightLayout::OutIn, n, k, block)
    }

    /// `gate` and `up` as one weight set of `2n` columns, interleaved 32 at a
    /// time as [`GatedLinear`](crate::GatedLinear) stores them. Signs and
    /// scales move unchanged, so each column is exactly what it was.
    pub(crate) fn interleave(gate: &Self, up: &Self) -> Self {
        let (n, k, block) = (gate.n, gate.k, gate.block);
        assert!(
            (up.n, up.k, up.block) == (n, k, block),
            "gate and up have the same shape and block size"
        );
        let nbk = k.div_ceil(block);
        let (band_bytes, sc_per_band) = Self::band_sizes(k, block);
        let mut signs = vec![0u8; (2 * n * k).div_ceil(8)];
        let mut scales = Vec::with_capacity(2 * n * nbk);
        let mut row = 0;
        for c0 in (0..n).step_by(32) {
            for w in [gate, up] {
                for j in c0..(c0 + 32).min(n) {
                    if k.is_multiple_of(8) {
                        // Column j's signs back to bytes, two nibbles each.
                        let (base, shift) = Self::bit(band_bytes, 0, j);
                        let out = &mut signs[row * k / 8..(row + 1) * k / 8];
                        for (b, o) in out.iter_mut().enumerate() {
                            let lo = w.signs[base + 2 * b * (BAND / 2)] >> shift & 0x0f;
                            let hi = w.signs[base + (2 * b + 1) * (BAND / 2)] >> shift & 0x0f;
                            *o = lo | hi << 4;
                        }
                    } else {
                        for d in 0..k {
                            if w.sign(d, j) {
                                let idx = row * k + d;
                                signs[idx / 8] |= 1 << (idx % 8);
                            }
                        }
                    }
                    scales.extend((0..nbk).map(|b| w.scales[scale_index(sc_per_band, b, j)]));
                    row += 1;
                }
            }
        }
        Self::new(&signs, &scales, WeightLayout::OutIn, 2 * n, k, block)
    }

    /// Bytes of signs and scales per band: k padded to 16 depths at 8 bytes a
    /// depth, and 64 scales per K-block.
    const fn band_sizes(k: usize, block: usize) -> (usize, usize) {
        (k.next_multiple_of(16) * 8, k.div_ceil(block) * BAND)
    }

    /// The packed byte and bit holding column `j`'s sign at depth `d`.
    const fn bit(band_bytes: usize, d: usize, j: usize) -> (usize, usize) {
        let c = j % BAND;
        (
            (j / BAND) * band_bytes + (d / 4) * (BAND / 2) + c / 2,
            4 * (c % 2) + d % 4,
        )
    }

    /// Whether column `j`'s weight at depth `d` is `+scale`.
    pub(super) fn sign(&self, d: usize, j: usize) -> bool {
        let (band_bytes, _) = Self::band_sizes(self.k, self.block);
        let (byte, bit) = Self::bit(band_bytes, d, j);
        (self.signs[byte] >> bit) & 1 == 1
    }

    /// Column `j`'s scale for K-block `blk`.
    pub(super) fn scale(&self, blk: usize, j: usize) -> f32 {
        let (_, sc_per_band) = Self::band_sizes(self.k, self.block);
        self.scales[scale_index(sc_per_band, blk, j)].to_f32()
    }

    /// The K-block size.
    #[must_use]
    pub const fn block(&self) -> usize {
        self.block
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
    use half::f16;

    use super::Q1Weights;
    use crate::WeightLayout;

    /// The pack keeps every sign and scale where `sign` and `scale` read
    /// them, from either layout, including ragged n, k and the last block.
    #[test]
    fn pack_round_trips() {
        for &(n, k, block) in &[(1usize, 1usize, 16usize), (65, 37, 16), (130, 300, 128)] {
            let nbk = k.div_ceil(block);
            let signs: Vec<u8> = (0..(k * n).div_ceil(8))
                .map(|i| (i * 37 % 251) as u8)
                .collect();
            let scales: Vec<f16> = (0..n * nbk)
                .map(|i| f16::from_f32(i as f32 * 0.5))
                .collect();
            for layout in [WeightLayout::OutIn, WeightLayout::InOut] {
                let q = Q1Weights::new(&signs, &scales, layout, n, k, block);
                for j in 0..n {
                    for d in 0..k {
                        let idx = match layout {
                            WeightLayout::OutIn => j * k + d,
                            WeightLayout::InOut => d * n + j,
                        };
                        let want = (signs[idx / 8] >> (idx % 8)) & 1 == 1;
                        assert_eq!(q.sign(d, j), want, "{n}x{k} {layout:?} ({d}, {j})");
                    }
                    for b in 0..nbk {
                        assert_eq!(
                            q.scale(b, j).to_bits(),
                            scales[j * nbk + b].to_f32().to_bits()
                        );
                    }
                }
            }
        }
    }

    /// Weights of the form `+-t` per block come back exactly, and the
    /// interleave keeps every column.
    #[test]
    fn quantize_and_interleave_keep_weights() {
        let (n, k, block) = (70usize, 300usize, 64usize);
        let w: Vec<f32> = (0..n * k)
            .map(|i| if (i * 7 + i / 5) % 3 == 0 { -1.0 } else { 1.0 } * 0.25 * (1 + i / k % 3) as f32)
            .collect();
        let q = Q1Weights::quantize(&w, WeightLayout::OutIn, n, k, block);
        let value = |q: &Q1Weights, d: usize, j: usize| {
            q.scale(d / block, j) * if q.sign(d, j) { 1.0 } else { -1.0 }
        };
        for j in 0..n {
            for d in 0..k {
                assert_eq!(
                    value(&q, d, j).to_bits(),
                    w[j * k + d].to_bits(),
                    "w[{j}][{d}]"
                );
            }
        }
        let neg: Vec<f32> = w.iter().map(|x| -x).collect();
        let p = Q1Weights::quantize(&neg, WeightLayout::OutIn, n, k, block);
        let both = Q1Weights::interleave(&q, &p);
        // Gate columns c0..c0+w, then up's, for each 32-column chunk.
        let order: Vec<(bool, usize)> = (0..n)
            .step_by(32)
            .flat_map(|c0| {
                let c1 = (c0 + 32).min(n);
                (c0..c1)
                    .map(|j| (false, j))
                    .chain((c0..c1).map(|j| (true, j)))
            })
            .collect();
        for (row, &(is_up, src)) in order.iter().enumerate() {
            let from = if is_up { &p } else { &q };
            for d in 0..k {
                assert_eq!(
                    value(&both, d, row).to_bits(),
                    value(from, d, src).to_bits()
                );
            }
        }
    }
}
