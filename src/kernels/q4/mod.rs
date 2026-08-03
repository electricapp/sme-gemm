//! 4-bit block-quantized weights: eager dequant to f16/bf16 ([`dequant_q4`],
//! [`dequant_q4_bf16`]) and the 4-bit-resident on-the-fly dequant path
//! ([`Q4Weights`] / [`matmul_q4`] / [`matmul_q4_bf16`]).
//!
//! Alone among the dtypes, Q4 has no `_batched` variant, because the two things
//! that would motivate one are mutually exclusive. Batching pays by amortizing
//! the streaming entry, which a Q4 kernel could only do by dequantizing every
//! item up front (the dequant cannot run inside a streaming region) -- i.e. by
//! becoming `dequant_q4` + [`matmul_f16_batched`], which already composes and,
//! measured on M5, beats a loop of [`matmul_q4`] by 1.4-3.0x on the small
//! per-item shapes where batching helps at all. At the sizes where 4-bit
//! residency earns its keep the ordering inverts -- 4 items of 2048x2048 run
//! 4.2x faster as a [`matmul_q4`] loop, the dequantized f16 having left cache --
//! and there the streaming entry is already noise.
//!
//! [`matmul_f16_batched`]: crate::matmul_f16_batched
use half::{bf16, f16};

use crate::element::Packed;
use crate::kernels::bf16::prepack_bf16;
use crate::kernels::f16::prepack_f16;

mod matmul;
mod weights;

pub use matmul::{matmul_q4, matmul_q4_bf16, matmul_q4_bf16_ep, matmul_q4_ep};
pub use weights::Q4Weights;

/// Default block size (along K): one scale per 32-value block, as llama.cpp's
/// `Q4_0`/`Q4_1`. [`Q4Params`] takes any power of two.
pub const Q4_BLOCK: usize = 32;

/// How a 4-bit code maps back to a weight.
///
/// `Scale` is the `Q4_0` form `w = scale * code` (codes signed `-8..=7`).
/// `Affine` is the `Q4_1` form `w = scale * code + min`, which needs a second
/// per-(column, K-block) array and is what the `*_1` / K-quant families use --
/// they cannot be expressed by a scale alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Q4Form {
    /// `w = scale * code`. No offsets array.
    #[default]
    Scale,
    /// `w = scale * code + min`. Requires the `mins` array.
    Affine,
}

/// Quantization layout for the 4-bit paths: K-block size plus the code form.
///
/// The block size must be a power of two (the kernel indexes the block with a
/// shift). `Q4Params::default()` is `Q4_0`: 32-value blocks, scale-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q4Params {
    pub(crate) block: usize,
    pub(crate) form: Q4Form,
}

impl Default for Q4Params {
    fn default() -> Self {
        Self {
            block: Q4_BLOCK,
            form: Q4Form::Scale,
        }
    }
}

impl Q4Params {
    /// Params with the given K-block size, scale-only (`Q4_0`).
    ///
    /// # Panics
    /// Panics if `block` is zero or not a power of two.
    #[must_use]
    pub fn new(block: usize) -> Self {
        assert!(
            block != 0 && block.is_power_of_two(),
            "q4 block size must be a power of two, got {block}"
        );
        Self {
            block,
            form: Q4Form::Scale,
        }
    }
    /// Switch to the affine form `w = scale*code + min` (`Q4_1`); the `mins`
    /// array then becomes required and must match `scales` in length.
    #[must_use]
    pub const fn affine(mut self) -> Self {
        self.form = Q4Form::Affine;
        self
    }
    /// The K-block size.
    #[must_use]
    pub const fn block(&self) -> usize {
        self.block
    }
    /// The code form.
    #[must_use]
    pub const fn form(&self) -> Q4Form {
        self.form
    }
}

/// Shared validation: returns `(kn, nbk)` after checking every length.
fn q4_check(
    quants: &[u8],
    scales: &[f16],
    mins: Option<&[f16]>,
    n: usize,
    k: usize,
    p: Q4Params,
) -> (usize, usize) {
    assert!(
        p.block != 0 && p.block.is_power_of_two(),
        "q4 block size must be a power of two, got {}",
        p.block
    );
    let nbk = k.div_ceil(p.block);
    // Checked products so a wrapped k*n / n*nbk can never match a short slice and
    // let the loops below index out of bounds (overflow-checks covers this crate
    // but not downstream consumers -- see `checked_dims` in exec.rs). Validating
    // k*n also keeps the inner `d*n + j` index non-wrapping.
    let kn = k
        .checked_mul(n)
        .expect("q4 weight dimension product k*n overflows usize");
    assert_eq!(quants.len(), kn.div_ceil(2), "quants is ceil(k*n/2) bytes");
    let want = n
        .checked_mul(nbk)
        .expect("q4 scales dimension product n*ceil(k/block) overflows usize");
    assert_eq!(scales.len(), want, "scales is n*ceil(k/block)");
    match (p.form, mins) {
        (Q4Form::Affine, Some(mv)) => assert_eq!(mv.len(), want, "mins is n*ceil(k/block)"),
        (Q4Form::Affine, None) => panic!("Q4Form::Affine requires a mins array"),
        (Q4Form::Scale, Some(_)) => panic!("Q4Form::Scale takes no mins array"),
        (Q4Form::Scale, None) => {}
    }
    (kn, nbk)
}

/// Eagerly dequantize 4-bit block-quantized weights to an f16 [`Packed`],
/// ready for [`matmul_f16_packed`].
///
/// The whole weight matrix is dequantized to f16 up front: weights are stored
/// 4-bit (4x smaller than f16 on disk / at rest), fully expanded to f16 here
/// once, then matmul runs on the fast f16 kernel.
///
/// (Storage is Q4; the resident compute set is full f16. To keep weights 4-bit
/// resident and dequantize on the fly instead, use [`Q4Weights`] / [`matmul_q4`].)
///
/// [`matmul_f16_packed`]: crate::matmul_f16_packed
///
/// Layout: weights are logically `k x n` row-major. `quants` holds the signed
/// 4-bit codes (`-8..=7`) packed two per byte in that row-major order
/// (`len == (k*n).div_ceil(2)`). `scales` holds one f16 per (column, K-block):
/// `scales[j * ceil(k/Q4_BLOCK) + d / Q4_BLOCK]`, and the dequantized weight is
/// `scale * code`.
///
/// # Panics
/// Panics if `quants`/`scales` lengths are inconsistent with `n`, `k`.
#[must_use]
pub fn dequant_q4(quants: &[u8], scales: &[f16], n: usize, k: usize) -> Packed<f16> {
    dequant_q4_with(quants, scales, None, n, k, Q4Params::default())
}

/// [`dequant_q4`] with an explicit block size and code form -- the entry point
/// for `Q4_1`-style affine weights (`w = scale*code + min`) and for block
/// sizes other than 32.
///
/// `mins` must be `Some` exactly when `p.form()` is [`Q4Form::Affine`], and
/// then matches `scales` in length.
///
/// # Panics
/// Panics if any length is inconsistent with `n`, `k`, `p`, or if `mins` does
/// not match the form.
#[must_use]
pub fn dequant_q4_with(
    quants: &[u8],
    scales: &[f16],
    mins: Option<&[f16]>,
    n: usize,
    k: usize,
    p: Q4Params,
) -> Packed<f16> {
    let b: Vec<f16> = q4_rowmajor_f32(quants, scales, mins, n, k, p)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    prepack_f16(&b, n, k)
}

/// Row-major `k x n` dequant of the caller's Q4 arrays, in f32 before the output
/// rounding. Shared by the eager f16 and bf16 entry points; validates lengths.
fn q4_rowmajor_f32(
    quants: &[u8],
    scales: &[f16],
    mins: Option<&[f16]>,
    n: usize,
    k: usize,
    p: Q4Params,
) -> Vec<f32> {
    let (kn, nbk) = q4_check(quants, scales, mins, n, k, p);
    let mut b = vec![0.0f32; kn];
    for d in 0..k {
        for j in 0..n {
            let idx = d * n + j;
            let byte = quants[idx / 2];
            let nib = if idx.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            // sign-extend the 4-bit code to -8..=7
            let code = i32::from(nib) - if nib < 8 { 0 } else { 16 };
            let bi = j * nbk + d / p.block;
            let mut w = scales[bi].to_f32() * code as f32;
            if let Some(mv) = mins {
                w += mv[bi].to_f32();
            }
            b[idx] = w;
        }
    }
    b
}

/// Eagerly dequantize 4-bit block-quantized weights to a bf16 [`Packed`] --
/// [`dequant_q4`]'s bf16 twin, for [`matmul_bf16_packed`].
///
/// [`matmul_bf16_packed`]: crate::matmul_bf16_packed
///
/// # Panics
/// Panics if `quants`/`scales` lengths are inconsistent with `n`, `k`.
#[must_use]
pub fn dequant_q4_bf16(quants: &[u8], scales: &[f16], n: usize, k: usize) -> Packed<bf16> {
    dequant_q4_bf16_with(quants, scales, None, n, k, Q4Params::default())
}

/// [`dequant_q4_bf16`] with an explicit block size and code form. `mins` must be
/// `Some` exactly when `p.form()` is [`Q4Form::Affine`].
///
/// # Panics
/// Panics if any length is inconsistent with `n`, `k`, `p`, or if `mins` does
/// not match the form.
#[must_use]
pub fn dequant_q4_bf16_with(
    quants: &[u8],
    scales: &[f16],
    mins: Option<&[f16]>,
    n: usize,
    k: usize,
    p: Q4Params,
) -> Packed<bf16> {
    let b: Vec<bf16> = q4_rowmajor_f32(quants, scales, mins, n, k, p)
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    prepack_bf16(&b, n, k)
}
