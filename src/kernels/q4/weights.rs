//! The 4-bit-resident weight set: the tile-major pack the kernels read, plus the
//! row-major unpack their fallbacks use.

use std::sync::OnceLock;

use half::{bf16, f16};

use super::{Q4Params, q4_check};

/// 4-bit weights kept **4-bit resident** for on-the-fly dequant ([`matmul_q4`]).
///
/// Unlike [`dequant_q4`] (which dequantizes to f16 up front), this stores the
/// nibbles tile-major plus per-(column, K-block) scales; the kernel dequantizes
/// one tile to f16 at a time, so the resident weight set stays 4x smaller during
/// compute -- the llama.cpp inner-loop approach.
///
/// [`matmul_q4`]: crate::matmul_q4
/// [`dequant_q4`]: crate::dequant_q4
#[derive(Debug)]
pub struct Q4Weights {
    /// The tile-major arrays cross the FFI from `matmul`, hence `pub(super)`.
    pub(super) nibbles: Vec<u8>,
    pub(super) scales: Vec<u16>,
    /// Tile-major offsets, same shape as `scales`; empty for
    /// [`Q4Form::Scale`](super::Q4Form::Scale).
    pub(super) mins: Vec<u16>,
    /// `scales` and `mins` as bf16, built on the first small-m bf16 call.
    pub(super) bf16_scales: OnceLock<(Vec<u16>, Vec<u16>)>,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) params: Q4Params,
}

impl Q4Weights {
    /// Repack row-major 4-bit weights (same input layout as [`dequant_q4`]) into
    /// the tile-major resident format, as `Q4_0`: 32-value K-blocks, scale-only.
    /// `quants` is `ceil(k*n/2)` bytes; `scales` is `n * ceil(k/Q4_BLOCK)`.
    ///
    /// # Panics
    /// Panics if `quants`/`scales` lengths are inconsistent with `n`, `k`.
    ///
    /// [`dequant_q4`]: crate::dequant_q4
    #[must_use]
    pub fn new(quants: &[u8], scales: &[f16], n: usize, k: usize) -> Self {
        Self::with_params(quants, scales, None, n, k, Q4Params::default())
    }

    /// [`Q4Weights::new`] with an explicit block size and code form. `mins` must
    /// be `Some` exactly when `p.form()` is [`Q4Form::Affine`].
    ///
    /// # Panics
    /// Panics if any length is inconsistent with `n`, `k`, `p`, or if `mins`
    /// does not match the form.
    ///
    /// [`Q4Form::Affine`]: super::Q4Form::Affine
    #[must_use]
    pub fn with_params(
        quants: &[u8],
        scales: &[f16],
        mins: Option<&[f16]>,
        n: usize,
        k: usize,
        p: Q4Params,
    ) -> Self {
        let (_, nbk) = q4_check(quants, scales, mins, n, k, p);
        let n_tiles = n.div_ceil(32);
        let nib_per_tile = k
            .checked_mul(16)
            .expect("q4 nibble tile size k*16 overflows usize");
        let sc_per_tile = nbk
            .checked_mul(32)
            .expect("q4 scale tile size nbk*32 overflows usize");
        let nib_len = n_tiles
            .checked_mul(nib_per_tile)
            .expect("q4 nibble buffer size overflows usize");
        let sc_len = n_tiles
            .checked_mul(sc_per_tile)
            .expect("q4 scale buffer size overflows usize");
        let mut nibbles = vec![0u8; nib_len];
        let mut sc = vec![0u16; sc_len];
        let mut mn = if mins.is_some() {
            vec![0u16; sc_len]
        } else {
            Vec::new()
        };
        for t in 0..n_tiles {
            for d in 0..k {
                for c in 0..32 {
                    let j = t * 32 + c;
                    if j >= n {
                        continue;
                    }
                    let idx = d * n + j;
                    let nib = if idx.is_multiple_of(2) {
                        quants[idx / 2] & 0x0f
                    } else {
                        quants[idx / 2] >> 4
                    };
                    let bi = t * nib_per_tile + d * 16 + c / 2;
                    if c.is_multiple_of(2) {
                        nibbles[bi] |= nib & 0x0f;
                    } else {
                        nibbles[bi] |= (nib & 0x0f) << 4;
                    }
                    let src = j * nbk + d / p.block;
                    let dst = t * sc_per_tile + (d / p.block) * 32 + c;
                    sc[dst] = scales[src].to_bits();
                    if let Some(mv) = mins {
                        mn[dst] = mv[src].to_bits();
                    }
                }
            }
        }
        Self {
            nibbles,
            scales: sc,
            mins: mn,
            bf16_scales: OnceLock::new(),
            n,
            k,
            params: p,
        }
    }

    /// Tile-major scales and mins converted to bf16 bits.
    pub(super) fn bf16_scales(&self) -> &(Vec<u16>, Vec<u16>) {
        self.bf16_scales.get_or_init(|| {
            let cvt = |v: &[u16]| -> Vec<u16> {
                v.iter()
                    .map(|&b| bf16::from_f32(f16::from_bits(b).to_f32()).to_bits())
                    .collect()
            };
            (cvt(&self.scales), cvt(&self.mins))
        })
    }

    /// The quantization layout these weights were packed with.
    #[must_use]
    pub const fn params(&self) -> Q4Params {
        self.params
    }

    /// Dequantize the tile-major resident weights back to a row-major `k x n`
    /// f16 buffer (the exact inverse of [`Q4Weights::new`]'s pack). Used by the
    /// [`matmul_q4`] OOM/non-SME fallback; factored out so it has a single
    /// definition and can be unit-tested without an SME machine (the kernel path
    /// would otherwise shadow it on every M5).
    #[must_use]
    pub(crate) fn dequant_to_rowmajor(&self) -> Vec<f16> {
        self.dequant_rowmajor(f16::from_f64)
    }

    /// [`Q4Weights::dequant_to_rowmajor`] rounded to bf16 instead -- the
    /// [`matmul_q4_bf16`] fallback. Rounds through f32, as the bf16 kernel does,
    /// so it is not the f16 result re-rounded.
    #[must_use]
    pub(crate) fn dequant_to_rowmajor_bf16(&self) -> Vec<bf16> {
        #[allow(clippy::cast_possible_truncation)]
        self.dequant_rowmajor(|w| bf16::from_f32(w as f32))
    }

    /// The unpack itself: `scale*code (+ min)`, exact in f64, handed to `round`;
    /// row-major `k x n`.
    fn dequant_rowmajor<T: Copy + Default>(&self, round: impl Fn(f64) -> T) -> Vec<T> {
        let (n, k) = (self.n, self.k);
        let block = self.params.block;
        let nbk = k.div_ceil(block);
        let n_tiles = n.div_ceil(32);
        let nib_per_tile = k * 16;
        let sc_per_tile = nbk * 32;
        let mut b = vec![T::default(); k * n];
        for t in 0..n_tiles {
            for d in 0..k {
                for c in 0..32 {
                    let j = t * 32 + c;
                    if j >= n {
                        continue;
                    }
                    let byte = self.nibbles[t * nib_per_tile + d * 16 + c / 2];
                    let nib = if c.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    let code = i32::from(nib) - if nib < 8 { 0 } else { 16 };
                    let bi = t * sc_per_tile + (d / block) * 32 + c;
                    let mut w = f16::from_bits(self.scales[bi]).to_f64() * f64::from(code);
                    if !self.mins.is_empty() {
                        w += f16::from_bits(self.mins[bi]).to_f64();
                    }
                    b[d * n + j] = round(w);
                }
            }
        }
        b
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
    use super::Q4Weights;
    use crate::kernels::q4::{Q4_BLOCK, Q4Form, Q4Params};
    use half::f16;

    /// Canonical row-major dequant for an arbitrary block size / code form --
    /// the reference the tile-major pack and unpack must both reproduce.
    fn rowmajor_dequant_with(
        quants: &[u8],
        scales: &[f16],
        mins: Option<&[f16]>,
        n: usize,
        k: usize,
        p: Q4Params,
    ) -> Vec<f16> {
        let nbk = k.div_ceil(p.block());
        let mut b = vec![f16::ZERO; k * n];
        for d in 0..k {
            for j in 0..n {
                let idx = d * n + j;
                let nib = if idx.is_multiple_of(2) {
                    quants[idx / 2] & 0x0f
                } else {
                    quants[idx / 2] >> 4
                };
                let code = i32::from(nib) - if nib < 8 { 0 } else { 16 };
                let bi = j * nbk + d / p.block();
                let mut w = scales[bi].to_f64() * f64::from(code);
                if let Some(mv) = mins {
                    w += mv[bi].to_f64();
                }
                b[idx] = f16::from_f64(w);
            }
        }
        b
    }

    /// The tile-major resident round-trip must reproduce the canonical dequant
    /// for every supported block size and for the affine (`Q4_1`) form, not just
    /// the scale-only 32-block default.
    #[test]
    fn q4_roundtrip_across_block_sizes_and_forms() {
        for &(n, k) in &[(33usize, 96usize), (64, 64), (17, 130), (96, 32)] {
            for &block in &[16usize, 32, 64, 128] {
                for affine in [false, true] {
                    let p = if affine {
                        Q4Params::new(block).affine()
                    } else {
                        Q4Params::new(block)
                    };
                    let nbk = k.div_ceil(block);
                    let mut s = 0xa4a4_1111_dead_0001u64
                        ^ ((n * 131 + k * 17 + block * 7 + usize::from(affine)) as u64);
                    let mut byte = || {
                        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                        (s >> 40) as u8
                    };
                    let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
                    let scales: Vec<f16> = (0..n * nbk)
                        .map(|i| f16::from_f32(0.01 + 0.001 * (i % 7) as f32))
                        .collect();
                    let mins: Vec<f16> = (0..n * nbk)
                        .map(|i| f16::from_f32(0.05 * (i % 5) as f32 - 0.1))
                        .collect();
                    let mv = affine.then_some(&mins[..]);

                    let want = rowmajor_dequant_with(&quants, &scales, mv, n, k, p);
                    let got =
                        Q4Weights::with_params(&quants, &scales, mv, n, k, p).dequant_to_rowmajor();
                    assert_eq!(got.len(), want.len());
                    for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert_eq!(
                            g.to_bits(),
                            w.to_bits(),
                            "n={n} k={k} block={block} affine={affine} idx={idx}"
                        );
                    }
                }
            }
        }
    }

    /// `Q4Form::Scale` is exactly `Affine` with all-zero mins -- pins that the
    /// offset plane is applied, not silently dropped.
    #[test]
    fn q4_affine_with_zero_mins_matches_scale_only() {
        let (n, k, block) = (48usize, 64usize, 32usize);
        let nbk = k.div_ceil(block);
        let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|i| (i % 251) as u8).collect();
        let scales: Vec<f16> = (0..n * nbk)
            .map(|i| f16::from_f32(0.02 + 0.001 * (i % 3) as f32))
            .collect();
        let zeros = vec![f16::ZERO; n * nbk];
        let plain = Q4Weights::new(&quants, &scales, n, k).dequant_to_rowmajor();
        let aff = Q4Weights::with_params(
            &quants,
            &scales,
            Some(&zeros),
            n,
            k,
            Q4Params::new(block).affine(),
        )
        .dequant_to_rowmajor();
        assert_eq!(plain, aff);
        // ...and a nonzero min actually shifts the result.
        let ones = vec![f16::from_f32(1.0); n * nbk];
        let shifted = Q4Weights::with_params(
            &quants,
            &scales,
            Some(&ones),
            n,
            k,
            Q4Params::new(block).affine(),
        )
        .dequant_to_rowmajor();
        assert!(
            plain
                .iter()
                .zip(&shifted)
                .all(|(a, b)| (b.to_f32() - a.to_f32() - 1.0).abs() < 1e-2),
            "affine min must add exactly 1.0 to every weight"
        );
        assert_eq!(Q4Params::new(block).affine().form(), Q4Form::Affine);
    }

    /// Canonical row-major Q4 dequant: the unambiguous reference for both the
    /// `Q4Weights::new` pack and the `dequant_to_rowmajor` unpack.
    fn rowmajor_dequant(quants: &[u8], scales: &[f16], n: usize, k: usize) -> Vec<f16> {
        let nbk = k.div_ceil(Q4_BLOCK);
        let mut b = vec![f16::ZERO; k * n];
        for d in 0..k {
            for j in 0..n {
                let idx = d * n + j;
                let nib = if idx.is_multiple_of(2) {
                    quants[idx / 2] & 0x0f
                } else {
                    quants[idx / 2] >> 4
                };
                let code = i32::from(nib) - if nib < 8 { 0 } else { 16 };
                let scale = scales[j * nbk + d / Q4_BLOCK].to_f32();
                b[idx] = f16::from_f32(scale * code as f32);
            }
        }
        b
    }

    /// The tile-major resident round-trip (`Q4Weights::new` then
    /// `dequant_to_rowmajor`) must reproduce the canonical row-major dequant
    /// bit-for-bit. This exercises the `matmul_q4` OOM/non-SME fallback's unpack
    /// on every platform -- the integration tests skip it whenever the SME
    /// kernel path is taken (i.e. on every M5), so without this it is untested.
    #[test]
    fn q4_resident_roundtrip_matches_rowmajor() {
        // Shapes spanning: n below/at/above a 32-tile, odd n (pad columns), odd
        // k*n (partial final byte), and k below/at/above a 32-value scale block.
        let shapes = [
            (1usize, 1usize),
            (3, 5),
            (32, 32),
            (33, 31),
            (48, 1),
            (17, 64),
            (64, 65),
            (31, 96),
        ];
        for (n, k) in shapes {
            let mut s = 0xa4a4_5151_dead_0001u64 ^ ((n * 131 + k * 17) as u64);
            let mut byte = || {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (s >> 40) as u8
            };
            let nbk = k.div_ceil(Q4_BLOCK);
            let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
            let scales: Vec<f16> = (0..n * nbk)
                .map(|i| f16::from_f32(0.01 + 0.001 * (i % 7) as f32))
                .collect();

            let want = rowmajor_dequant(&quants, &scales, n, k);
            let got = Q4Weights::new(&quants, &scales, n, k).dequant_to_rowmajor();

            assert_eq!(got.len(), want.len(), "len mismatch n={n} k={k}");
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "n={n} k={k} idx={idx}: resident round-trip diverges from row-major dequant"
                );
            }
        }
    }
}
