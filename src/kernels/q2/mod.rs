//! 2-bit weights kept 2-bit resident ([`Q2Weights`]) and their GEMM
//! ([`matmul_q2`]): codes expand to int8 and multiply int8-quantized
//! activations, so a one-row call runs at the int8 dot-product rate.
//!
//! The format fits models trained at two bits -- ternary `BitNet b1.58` style
//! weights, llama.cpp's `TQ2_0` -- whose activations are int8 by design.
//! Rounding an f32 model's weights to four levels after training is a
//! different and much lossier thing; [`Q2Weights::quantize`] does it, but
//! mainly to re-encode weights that are already on a 2-bit grid.

mod matmul;
mod weights;

pub use matmul::{matmul_q2, matmul_q2_ep};
pub use weights::Q2Weights;

/// Default K-block size: one scale per 256-value block of a column, as
/// llama.cpp's `TQ2_0`. [`Q2Params`] takes any power of two from 16.
pub const Q2_BLOCK: usize = 256;

/// The ternary codebook: code 0 is -1, 1 is 0, 2 is +1 (3 is also 0), as
/// `TQ2_0` and `BitNet b1.58` store their weights.
pub const Q2_TERNARY: [i8; 4] = [-1, 0, 1, 0];

/// Quantization layout for the 2-bit path: K-block size plus the four int8
/// values the codes stand for.
///
/// A weight is `scale * codebook[code]`, with one f32 scale per (column,
/// K-block). `Q2Params::default()` is ternary with 256-value blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q2Params {
    pub(crate) block: usize,
    pub(crate) codebook: [i8; 4],
}

impl Default for Q2Params {
    fn default() -> Self {
        Self {
            block: Q2_BLOCK,
            codebook: Q2_TERNARY,
        }
    }
}

impl Q2Params {
    /// Ternary params with the given K-block size.
    ///
    /// The kernel folds each block's integer sums into the f32 result, so
    /// small blocks cost time: against one scale per column (a block of `k`
    /// or more), a one-row 4096 x 4096 call on M5 takes about 10% longer with
    /// 256-value blocks and 40% with 64.
    ///
    /// # Panics
    /// Panics if `block` is not a power of two from 16 to 65536 (an int8
    /// block sum must fit an i32).
    #[must_use]
    pub fn new(block: usize) -> Self {
        assert!(
            block.is_power_of_two() && (16..=1 << 16).contains(&block),
            "q2 block size must be a power of two from 16 to 65536, got {block}"
        );
        Self {
            block,
            codebook: Q2_TERNARY,
        }
    }

    /// These params with codes standing for `values` instead: code `c` is
    /// `values[c]`, e.g. `[-2, -1, 1, 2]` for a symmetric four-level grid.
    #[must_use]
    pub const fn with_codebook(mut self, values: [i8; 4]) -> Self {
        self.codebook = values;
        self
    }

    /// The K-block size.
    #[must_use]
    pub const fn block(&self) -> usize {
        self.block
    }

    /// The four values the codes stand for.
    #[must_use]
    pub const fn codebook(&self) -> [i8; 4] {
        self.codebook
    }
}
