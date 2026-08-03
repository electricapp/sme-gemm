//! Element-type machinery: the [`Accuracy`] mode, the sealed [`Element`] trait
//! and its dtype impls, and the reusable [`Packed`] weight panel.

use half::{bf16, f16};

/// f16 accumulation strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Accuracy {
    /// fp32 accumulation (widening MOPA). Accurate; runs on M4+. Default.
    #[default]
    Accurate,
    /// fp16 accumulation (non-widening MOPA, M5 `FEAT_SME_F16F16`). ~2x faster,
    /// lower precision -- error grows ~sqrt(K). Falls back to `Accurate` if
    /// the CPU lacks the extension.
    Fast,
}

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Element types this crate's GEMMs accept. Sealed: implemented only for the
/// supported scalars (`f16`, `bf16`, `f32`, `f64`, `i8`, `i16`).
pub trait Element: Copy + core::fmt::Debug + sealed::Sealed {
    /// Storage element of a packed panel: the 16-bit floats pack as their `u16`
    /// bit pattern, `f32`/`i8` pack as themselves.
    type Store: Copy + core::fmt::Debug;
}
impl sealed::Sealed for f16 {}
impl Element for f16 {
    type Store = u16;
}
impl sealed::Sealed for bf16 {}
impl Element for bf16 {
    type Store = u16;
}
impl sealed::Sealed for f32 {}
impl Element for f32 {
    type Store = Self;
}
impl sealed::Sealed for f64 {}
impl Element for f64 {
    type Store = Self;
}
impl sealed::Sealed for i8 {}
impl Element for i8 {
    type Store = Self;
}
impl sealed::Sealed for i16 {}
impl Element for i16 {
    type Store = Self;
}

/// Pre-packed right-hand side (weights), reusable across many GEMMs.
///
/// Pack once with `prepack_f16` / `prepack_bf16` / `prepack_i8` /
/// `prepack_f32`, then reuse across calls to skip re-packing the weights on
/// every GEMM.
///
/// One type for every dtype; the concrete layout is chosen by `prepack_*`.
#[derive(Debug)]
pub struct Packed<T: Element> {
    pub(crate) data: Vec<T::Store>,
    pub(crate) n: usize,
    pub(crate) k: usize,
    /// Whether `data` is in the SME kernel's packed layout (vs a raw row-major
    /// copy used by the scalar fallback / the f32 path that has no packb yet).
    pub(crate) sme: bool,
    pub(crate) _t: core::marker::PhantomData<T>,
}

impl<T: Element> Packed<T> {
    /// Columns (`n`) of the packed weight matrix.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }
    /// Depth (`k`) of the packed weight matrix.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }
}
