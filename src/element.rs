//! Element-type machinery: the [`Accum`] dtype, the sealed [`Element`] trait
//! and its dtype impls, and the reusable [`Packed`] weight panel.

use half::{bf16, f16};

/// Accumulator dtype for half-precision GEMM.
///
/// Storage dtype is the function you called (`matmul_f16`, `matmul_bf16`, …).
/// This picks the **compute** type:
///
/// - [`Accum::F32`] — widening MOPA, M4+. Default.
/// - [`Accum::F16`] — non-widening `FEAT_SME_F16F16`, M5. f16 kernels only.
/// - [`Accum::Bf16`] — non-widening `FEAT_SME_B16B16`, M5. bf16 kernels only.
///
/// A mismatched 16-bit choice (e.g. [`Accum::Bf16`] on an f16 kernel), or a
/// missing SME feature, falls back to [`Accum::F32`]. Native 16-bit accumulate
/// is ~2× the widening rate; error grows ~√K.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Accum {
    /// f32 accumulate (widening MOPA). M4+. Default.
    #[default]
    F32,
    /// f16 accumulate (non-widening `FEAT_SME_F16F16`). M5.
    F16,
    /// bf16 accumulate (non-widening `FEAT_SME_B16B16`). M5.
    Bf16,
}

impl Accum {
    /// `true` when this is the non-widening f16 path.
    #[must_use]
    pub const fn is_f16(self) -> bool {
        matches!(self, Self::F16)
    }

    /// `true` when this is the non-widening bf16 path.
    #[must_use]
    pub const fn is_bf16(self) -> bool {
        matches!(self, Self::Bf16)
    }
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
