//! Weight layouts, and packing weights given in either one.

use half::{bf16, f16};

use crate::element::{Element, Packed};

/// How a weight matrix is laid out in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightLayout {
    /// `[out][in]`: one row per output, `n x k` row-major. `PyTorch` stores
    /// `nn.Linear.weight` this way, as do most checkpoints.
    OutIn,
    /// `[in][out]`: `k x n` row-major, the `B` the GEMMs multiply by.
    InOut,
}

impl WeightLayout {
    /// `w` (`n` outputs by `k` inputs in this layout) as `k x n` row-major:
    /// borrowed for [`WeightLayout::InOut`], transposed for
    /// [`WeightLayout::OutIn`].
    ///
    /// # Panics
    /// Panics if `w.len() != n * k`.
    #[must_use]
    pub fn to_in_out<T: Copy>(self, w: &[T], n: usize, k: usize) -> std::borrow::Cow<'_, [T]>
    where
        [T]: ToOwned<Owned = Vec<T>>,
    {
        assert_eq!(
            w.len(),
            crate::exec::checked_dim2(n, k),
            "w holds n * k weights"
        );
        match self {
            Self::InOut => std::borrow::Cow::Borrowed(w),
            Self::OutIn => {
                let mut t = Vec::with_capacity(w.len());
                for d in 0..k {
                    t.extend((0..n).map(|j| w[j * k + d]));
                }
                std::borrow::Cow::Owned(t)
            }
        }
    }
}

impl WeightLayout {
    /// Multiplies every input (`k`) of `w` (`n` outputs by `k` inputs, in this
    /// layout) by `scale[k]`: `W' = diag(scale) @ W`. Folding a norm's weight
    /// into the layer that follows it this way leaves only the per-row scalar
    /// to apply at run time ([`Linear::rms_norm_input`]).
    ///
    /// # Panics
    /// Panics if `w.len() != n * k` or `scale.len() != k`.
    ///
    /// [`Linear::rms_norm_input`]: crate::Linear::rms_norm_input
    pub fn scale_inputs(self, w: &mut [f32], n: usize, k: usize, scale: &[f32]) {
        assert_eq!(
            w.len(),
            crate::exec::checked_dim2(n, k),
            "w holds n * k weights"
        );
        assert_eq!(scale.len(), k, "scale holds k values");
        match self {
            Self::OutIn => {
                for row in w.chunks_exact_mut(k) {
                    for (v, s) in row.iter_mut().zip(scale) {
                        *v *= s;
                    }
                }
            }
            Self::InOut => {
                for (row, s) in w.chunks_exact_mut(n).zip(scale) {
                    for v in row {
                        *v *= s;
                    }
                }
            }
        }
    }
}

/// Element types with a packed weight panel ([`prepack`]).
pub trait Prepack: Element {
    /// Packs `k x n` row-major weights.
    #[doc(hidden)]
    fn prepack_in_out(b: &[Self], n: usize, k: usize) -> Packed<Self>;
}

macro_rules! prepack_impl {
    ($T:ty, $f:path) => {
        impl Prepack for $T {
            fn prepack_in_out(b: &[Self], n: usize, k: usize) -> Packed<Self> {
                $f(b, n, k)
            }
        }
    };
}
prepack_impl!(f16, crate::prepack_f16);
prepack_impl!(bf16, crate::prepack_bf16);
prepack_impl!(f32, crate::prepack_f32);
prepack_impl!(f64, crate::prepack_f64);
prepack_impl!(i8, crate::prepack_i8);
prepack_impl!(i16, crate::prepack_i16);

/// Packs weights (`n` outputs by `k` inputs, in `layout`) for the `*_packed`
/// entry points and [`Gemm`](crate::Gemm): [`prepack_f16`](crate::prepack_f16)
/// and friends, for weights stored either way round.
///
/// ```
/// use half::f16;
/// use sme_gemm::{WeightLayout, prepack};
/// let (n, k) = (32, 16);
/// let w = vec![f16::from_f32(0.5); n * k]; // [out][in], as PyTorch stores it
/// let packed = prepack(&w, WeightLayout::OutIn, n, k);
/// assert_eq!((packed.n(), packed.k()), (n, k));
/// ```
///
/// # Panics
/// Panics if `w.len() != n * k`.
#[must_use]
pub fn prepack<T: Prepack>(w: &[T], layout: WeightLayout, n: usize, k: usize) -> Packed<T> {
    T::prepack_in_out(&layout.to_in_out(w, n, k), n, k)
}
