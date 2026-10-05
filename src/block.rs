//! [`Block`]: a pre-norm transformer layer over an f32 residual stream.

use std::cell::RefCell;

use half::f16;

use crate::nn::Norm;
use crate::{KvCache, Mlp, SelfAttention};

thread_local! {
    /// The normalized rows, f16 for the next matmul.
    static NORMED: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
}

/// A pre-norm transformer layer: `x += attn(norm1(x))`, then
/// `x += mlp(norm2(x))`, over an f32 residual stream.
///
/// GPT-2's block is `Block::new(ln_1, attn, ln_2, mlp)` with [`Norm::layer`]
/// and [`Mlp::new`]; [`Norm::rms`] and [`Mlp::gated`] give the `RMSNorm` and
/// `SwiGLU` variants. [`SelfAttention`] applies no rotary position embedding, so
/// this fits models with learned or added positions. Layers of another shape
/// (attention and MLP in parallel, norms after the residual add) compose the
/// same parts directly.
///
/// ```
/// use sme_gemm::{nn::Norm, Block, Gate, Linear, Mlp, SelfAttention, WeightLayout};
/// let (d, heads, hd) = (128, 2, 64);
/// let q = |n: usize, k: usize| Linear::quantize(&vec![0.01f32; n * k], WeightLayout::OutIn, n, k);
/// let block = Block::new(
///     Norm::layer(vec![1.0; d], None, 1e-5),
///     SelfAttention::new(q(3 * d, d), q(d, d), heads, heads, hd),
///     Norm::layer(vec![1.0; d], None, 1e-5),
///     Mlp::new(q(4 * d, d), Gate::Gelu, q(d, 4 * d)),
/// );
/// let mut cache = block.cache(256);
/// let mut x = vec![0.5f32; d]; // one position's residual stream
/// block.accumulate(&mut x, &mut cache, 1);
/// assert_eq!(cache.len(), 1);
/// ```
#[derive(Debug)]
pub struct Block {
    norm1: Norm,
    attn: SelfAttention,
    norm2: Norm,
    mlp: Mlp,
}

impl Block {
    /// The layer from its parts.
    ///
    /// # Panics
    /// Panics if the parts do not all take and give the same width.
    #[must_use]
    pub fn new(norm1: Norm, attn: SelfAttention, norm2: Norm, mlp: Mlp) -> Self {
        let d = attn.width();
        assert!(
            norm1.width() == d
                && attn.out_width() == d
                && norm2.width() == d
                && mlp.k() == d
                && mlp.n() == d,
            "the norms, attention and MLP share the residual's width"
        );
        Self {
            norm1,
            attn,
            norm2,
            mlp,
        }
    }

    /// The residual stream's width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.attn.width()
    }

    /// An empty cache for this layer's attention, `capacity` positions long.
    #[must_use]
    pub fn cache(&self, capacity: usize) -> KvCache {
        self.attn.cache(capacity)
    }

    /// Runs the layer on `rows` rows of `x` at the cache's next positions,
    /// which this appends, adding both halves into `x`.
    ///
    /// # Panics
    /// Panics if `x` is not `rows` rows of [`Block::width`], the cache has
    /// other head counts, or the rows do not fit in it.
    pub fn accumulate(&self, x: &mut [f32], cache: &mut KvCache, rows: usize) {
        assert_eq!(
            x.len(),
            crate::exec::checked_dim2(rows, self.width()),
            "x is rows * width"
        );
        NORMED.with_borrow_mut(|h| {
            if h.len() < x.len() {
                h.resize(x.len(), f16::ZERO);
            }
            let h = &mut h[..x.len()];
            self.norm1.forward(&*x, &mut *h);
            self.attn.accumulate(&*h, cache, x, rows);
            self.norm2.forward(&*x, &mut *h);
            self.mlp.accumulate(&*h, x, rows);
        });
    }
}
