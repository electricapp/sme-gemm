//! The [`Gemm`] builder, the [`PackedEpilogue`] / [`MapElem`] dispatch traits,
//! [`RowReduce`], and the non-fused [`epilogue_map`] escape hatch.

use half::{bf16, f16};

use super::{
    checked_dim2, f32_packed_ep_impl, f32_packed_ep_reduce, f64_packed_ep_impl,
    softmax_rows_with_max,
};
use crate::element::{Element, Packed};
use crate::epilogue::{Activation, Dequant, Epilogue};

/// Sealed dispatch for the packed fused-epilogue path.
///
/// Implemented for the element types whose kernels carry an in-store epilogue
/// (f16, bf16); lets the [`Gemm`] builder stay generic without exposing the
/// per-dtype entry points.
pub trait PackedEpilogue: Element {
    /// Multiplicative identity for the element type (the default `beta`).
    #[doc(hidden)]
    fn one() -> Self;
    #[doc(hidden)]
    fn ep_to_f32(self) -> f32;
    #[doc(hidden)]
    fn ep_from_f32(x: f32) -> Self;
    #[doc(hidden)]
    fn run_packed_ep(
        a: &[Self],
        packed: &Packed<Self>,
        c: &mut [Self],
        m: usize,
        ep: &Epilogue<'_, Self>,
        beta: Self,
        col_major: bool,
    );
}
impl PackedEpilogue for f16 {
    fn one() -> Self {
        Self::from_f32(1.0)
    }
    fn ep_to_f32(self) -> f32 {
        self.to_f32()
    }
    fn ep_from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }
    fn run_packed_ep(
        a: &[Self],
        packed: &Packed<Self>,
        c: &mut [Self],
        m: usize,
        ep: &Epilogue<'_, Self>,
        beta: Self,
        col_major: bool,
    ) {
        crate::kernels::f16::f16_packed_ep_impl(a, packed, c, m, ep, beta, col_major);
    }
}
impl PackedEpilogue for bf16 {
    fn one() -> Self {
        Self::from_f32(1.0)
    }
    fn ep_to_f32(self) -> f32 {
        self.to_f32()
    }
    fn ep_from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }
    fn run_packed_ep(
        a: &[Self],
        packed: &Packed<Self>,
        c: &mut [Self],
        m: usize,
        ep: &Epilogue<'_, Self>,
        beta: Self,
        col_major: bool,
    ) {
        crate::kernels::bf16::bf16_packed_ep_impl(a, packed, c, m, ep, beta, col_major);
    }
}
impl PackedEpilogue for f32 {
    fn one() -> Self {
        1.0
    }
    fn ep_to_f32(self) -> f32 {
        self
    }
    fn ep_from_f32(x: f32) -> Self {
        x
    }
    fn run_packed_ep(
        a: &[Self],
        packed: &Packed<Self>,
        c: &mut [Self],
        m: usize,
        ep: &Epilogue<'_, Self>,
        beta: Self,
        col_major: bool,
    ) {
        f32_packed_ep_impl(a, packed, c, m, ep, beta, col_major);
    }
}
impl PackedEpilogue for f64 {
    fn one() -> Self {
        1.0
    }
    fn ep_to_f32(self) -> f32 {
        self as f32
    }
    fn ep_from_f32(x: f32) -> Self {
        Self::from(x)
    }
    fn run_packed_ep(
        a: &[Self],
        packed: &Packed<Self>,
        c: &mut [Self],
        m: usize,
        ep: &Epilogue<'_, Self>,
        beta: Self,
        col_major: bool,
    ) {
        f64_packed_ep_impl(a, packed, c, m, ep, beta, col_major);
    }
}

/// Float output dtypes (`f16`, `bf16`, `f32`) that round-trip through `f32`.
///
/// Sealed; the conversion uses the same bit-faithful `to_f32`/`from_f32` the
/// kernels do (`half::f16`/`half::bf16` for the 16-bit types, identity for f32).
pub trait MapElem: Element {
    /// Widen the stored value to its `f32` working representation.
    #[doc(hidden)]
    fn to_f32(self) -> f32;
    /// Narrow an `f32` working value back to the stored dtype.
    #[doc(hidden)]
    fn from_f32(x: f32) -> Self;
}
impl MapElem for f16 {
    fn to_f32(self) -> f32 {
        Self::to_f32(self)
    }
    fn from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }
}
impl MapElem for bf16 {
    fn to_f32(self) -> f32 {
        Self::to_f32(self)
    }
    fn from_f32(x: f32) -> Self {
        Self::from_f32(x)
    }
}
impl MapElem for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f32(x: f32) -> Self {
        x
    }
}

/// Non-fused arbitrary-closure epilogue: applies `f` to every element of the
/// row-major `m x n` output `c`, in place, in `f32` working precision.
///
/// This is the escape hatch for epilogue logic that can't be expressed as the
/// fused op-graph nodes ([`Epilogue`]).
///
/// Each element becomes `c[i*n + j] = T::from_f32(f(c[i*n + j] as f32, i, j))`,
/// where `i`/`j` are the row/column indices. Supported for the float output
/// dtypes `f16`, `bf16`, `f32`.
///
/// Unlike the fused [`Epilogue`] op-graph (applied in-register at the store,
/// zero extra passes), this runs as one extra full pass over `c` after the GEMM.
///
/// ```
/// # use sme_gemm::{matmul_f32, epilogue_map};
/// let mut c = vec![0.0_f32; 4]; // 2x2, already holds A@B
/// epilogue_map(&mut c, 2, 2, |x, i, j| x.tanh() * 1.5 + (i + j) as f32);
/// ```
///
/// # Panics
/// Panics if `c.len() != m * n`.
pub fn epilogue_map<T, F>(c: &mut [T], m: usize, n: usize, mut f: F)
where
    T: MapElem,
    F: FnMut(f32, usize, usize) -> f32,
{
    assert!(c.len() == checked_dim2(m, n), "c.len() must equal m * n");
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            c[idx] = T::from_f32(f(c[idx].to_f32(), i, j));
        }
    }
}

/// Layer-2 builder for a packed GEMM with a fused epilogue: `C = act(A@B +
/// bias)` with pre-packed weights, the epilogue applied in-register at the
/// store.
///
/// This is the entry point for the packed fused-epilogue kernels (f16, bf16,
/// f32, f64); [`Epilogue`] is the same op-graph in standalone form, and the
/// batched analogs are `matmul_{f16,bf16,f32,f64}_batched_ep`:
///
/// ```no_run
/// # use sme_gemm::{prepack_f16, Gemm};
/// # use half::f16;
/// # let (m, n, k) = (32, 32, 32);
/// # let a = vec![f16::ZERO; m * k];
/// # let b = vec![f16::ZERO; k * n];
/// # let bias = vec![f16::ZERO; n];
/// # let mut c = vec![f16::ZERO; m * n];
/// let w = prepack_f16(&b, n, k);
/// Gemm::new(&a, &w, m).add_col(&bias).gelu().run(&mut c);
/// ```
#[derive(Debug)]
pub struct Gemm<'a, T: PackedEpilogue> {
    a: &'a [T],
    packed: &'a Packed<T>,
    m: usize,
    beta: T,
    col_major: bool,
    epilogue: Epilogue<'a, T>,
}

/// Sealed dispatch for the packed fused-**dequant** path: [`PackedEpilogue`]'s
/// quantized counterpart, implemented for `i8` and `i16`.
///
/// These kernels are not element-preserving the way the float ones are -- i8/i16
/// in, f32 out, with the op-graph operands in f32 rather than in the input type
/// -- so they cannot share [`PackedEpilogue`] (whose `c: &mut [Self]` and
/// `beta: Self` both assume the output is the input type). [`Dequant::run_packed`]
/// is the entry point; [`Dequant`] itself is the builder.
pub trait PackedQuant: Element {
    #[doc(hidden)]
    fn run_packed_dequant(
        a: &[Self],
        packed: &Packed<Self>,
        c: &mut [f32],
        m: usize,
        dq: &Dequant<'_>,
    );
}
impl PackedQuant for i8 {
    fn run_packed_dequant(
        a: &[Self],
        packed: &Packed<Self>,
        c: &mut [f32],
        m: usize,
        dq: &Dequant<'_>,
    ) {
        crate::kernels::int::matmul_i8_packed_dequant(a, packed, c, m, dq);
    }
}
impl PackedQuant for i16 {
    fn run_packed_dequant(
        a: &[Self],
        packed: &Packed<Self>,
        c: &mut [f32],
        m: usize,
        dq: &Dequant<'_>,
    ) {
        crate::kernels::int::matmul_i16_packed_dequant(a, packed, c, m, dq);
    }
}

impl Dequant<'_> {
    /// Run the quantized GEMM this dequant describes against pre-packed weights:
    /// `D(f32) = act(scale*(A@B) + composable bias)`, row-major `m x n` output.
    ///
    /// The quantized analog of [`Gemm::run`]: [`Dequant`] is already the op-graph
    /// builder, so it doubles as the builder here rather than there being a
    /// second `Gemm`-shaped type.
    ///
    /// ```
    /// # use sme_gemm::{Dequant, prepack_i8};
    /// # let (m, n, k) = (4, 8, 4);
    /// # let (a, b) = (vec![1i8; m * k], vec![2i8; k * n]);
    /// # let bias = vec![0.0f32; n];
    /// let w = prepack_i8(&b, n, k);
    /// let mut c = vec![0.0f32; m * n];
    /// Dequant::new(0.01).add_col(&bias).relu().run_packed(&a, &w, &mut c, m);
    /// ```
    ///
    /// # Panics
    /// Panics if `a`/`c` lengths are inconsistent with `m` and the packed
    /// `n`, `k`, or if an operand length mismatches `m`/`n`.
    pub fn run_packed<T: PackedQuant>(&self, a: &[T], packed: &Packed<T>, c: &mut [f32], m: usize) {
        T::run_packed_dequant(a, packed, c, m, self);
    }
}

/// Per-row (per-M) reductions over the post-epilogue output, computed as each
/// M-tile finishes, while its rows are still L1 resident. Slices are length `m`
/// and fully overwritten. Costs ~4% of the GEMM.
#[derive(Debug, Default)]
pub struct RowReduce<'r> {
    /// Sum of each output row.
    pub sum: Option<&'r mut [f32]>,
    /// Maximum of each output row.
    pub max: Option<&'r mut [f32]>,
}

impl<'r> RowReduce<'r> {
    /// No reductions.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
    /// Also compute the per-row sum into `out` (length `m`).
    #[must_use]
    pub const fn sum(mut self, out: &'r mut [f32]) -> Self {
        self.sum = Some(out);
        self
    }
    /// Also compute the per-row maximum into `out` (length `m`).
    #[must_use]
    pub const fn max(mut self, out: &'r mut [f32]) -> Self {
        self.max = Some(out);
        self
    }
}

impl Gemm<'_, f32> {
    /// Run into `c`, also producing the [`RowReduce`] outputs. Row-major only.
    ///
    /// # Panics
    /// Panics as [`Gemm::run`], plus if a reduction slice is not length `m` or
    /// `col_major_output()` was set.
    pub fn run_reduce(self, c: &mut [f32], red: RowReduce<'_>) {
        assert!(
            !self.col_major,
            "row reductions require a row-major output (drop col_major_output)"
        );
        for o in [red.sum.as_deref(), red.max.as_deref()]
            .into_iter()
            .flatten()
        {
            assert_eq!(o.len(), self.m, "reduction output is per-M (length m)");
        }
        f32_packed_ep_reduce(
            self.a,
            self.packed,
            c,
            self.m,
            &self.epilogue,
            self.beta,
            red,
        );
    }

    /// Run into `c`, then row-softmax it in place. Row-major only.
    ///
    /// The GEMM's store yields the per-row maxima as a by-product (~4% of the
    /// GEMM), so the softmax skips its own max sweep: two passes over `c`
    /// instead of three. Otherwise identical to [`Gemm::run`] followed by
    /// [`softmax_rows`](crate::softmax_rows).
    ///
    /// Unlike [`softmax_gemm_f32`](crate::softmax_gemm_f32), which fuses both
    /// into one kernel but takes no epilogue, this applies the configured
    /// epilogue before the softmax.
    ///
    /// ```
    /// # use sme_gemm::{prepack_f32, Gemm};
    /// # let (m, n, k) = (4, 8, 4);
    /// # let (a, b, bias) = (vec![0.1f32; m * k], vec![0.2f32; k * n], vec![0.0f32; n]);
    /// let w = prepack_f32(&b, n, k);
    /// let mut c = vec![0.0f32; m * n];
    /// Gemm::new(&a, &w, m).add_col(&bias).run_softmax(&mut c);
    /// ```
    ///
    /// # Panics
    /// Panics as [`Gemm::run`], plus if `col_major_output()` was set.
    pub fn run_softmax(self, c: &mut [f32]) {
        assert!(
            !self.col_major,
            "row softmax requires a row-major output (drop col_major_output)"
        );
        let (m, n) = (self.m, self.packed.n);
        let mut row_max = vec![f32::NEG_INFINITY; m];
        f32_packed_ep_reduce(
            self.a,
            self.packed,
            c,
            m,
            &self.epilogue,
            self.beta,
            RowReduce::new().max(&mut row_max),
        );
        softmax_rows_with_max(c, m, n, &row_max);
    }
}

impl<'a, T: PackedEpilogue> Gemm<'a, T> {
    /// Start a GEMM of `a` (`m x k` row-major) against the pre-packed weights.
    /// Output is row-major, `beta = 1`, no epilogue, until configured.
    #[must_use]
    pub fn new(a: &'a [T], packed: &'a Packed<T>, m: usize) -> Self {
        Self {
            a,
            packed,
            m,
            beta: T::one(),
            col_major: false,
            epilogue: Epilogue::new(),
        }
    }
    #[inline]
    #[must_use]
    fn activation(mut self, act: Activation) -> Self {
        self.epilogue = self.epilogue.activation(act);
        self
    }
    /// Shorthand for `Relu`.
    #[must_use]
    pub fn relu(self) -> Self {
        self.activation(Activation::Relu)
    }
    /// Shorthand for `Gelu`.
    #[must_use]
    pub fn gelu(self) -> Self {
        self.activation(Activation::Gelu)
    }
    /// Shorthand for `Silu`.
    #[must_use]
    pub fn silu(self) -> Self {
        self.activation(Activation::Silu)
    }
    /// Append `x = tanh(x)`.
    #[must_use]
    pub fn tanh(mut self) -> Self {
        self.epilogue = self.epilogue.tanh();
        self
    }
    /// Append `x = sigmoid(x)`.
    #[must_use]
    pub fn sigmoid(mut self) -> Self {
        self.epilogue = self.epilogue.sigmoid();
        self
    }
    /// Append `x += scalar`.
    #[must_use]
    pub fn add_scalar(mut self, v: T) -> Self {
        self.epilogue = self.epilogue.add_scalar(v);
        self
    }
    /// Append `x *= scalar`.
    #[must_use]
    pub fn mul_scalar(mut self, v: T) -> Self {
        self.epilogue = self.epilogue.mul_scalar(v);
        self
    }
    /// Append `x += row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn add_row(mut self, row: &'a [T]) -> Self {
        self.epilogue = self.epilogue.add_row(row);
        self
    }
    /// Append `x *= row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn mul_row(mut self, row: &'a [T]) -> Self {
        self.epilogue = self.epilogue.mul_row(row);
        self
    }
    /// Append `x += col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn add_col(mut self, col: &'a [T]) -> Self {
        self.epilogue = self.epilogue.add_col(col);
        self
    }
    /// Append `x *= col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn mul_col(mut self, col: &'a [T]) -> Self {
        self.epilogue = self.epilogue.mul_col(col);
        self
    }
    /// Append `x += tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn add_tensor(mut self, tensor: &'a [T]) -> Self {
        self.epilogue = self.epilogue.add_tensor(tensor);
        self
    }
    /// Append `x += tensor[i*ld + j]` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn add_tensor_strided(mut self, tensor: &'a [T], ld: usize) -> Self {
        self.epilogue = self.epilogue.add_tensor_strided(tensor, ld);
        self
    }
    /// Append `x *= tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn mul_tensor(mut self, tensor: &'a [T]) -> Self {
        self.epilogue = self.epilogue.mul_tensor(tensor);
        self
    }
    /// Append `x *= tensor[i*ld + j]` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn mul_tensor_strided(mut self, tensor: &'a [T], ld: usize) -> Self {
        self.epilogue = self.epilogue.mul_tensor_strided(tensor, ld);
        self
    }
    /// Append `x = max(x, lo)`.
    #[must_use]
    pub fn max(mut self, lo: T) -> Self {
        self.epilogue = self.epilogue.max(lo);
        self
    }
    /// Append `x = min(x, hi)`.
    #[must_use]
    pub fn min(mut self, hi: T) -> Self {
        self.epilogue = self.epilogue.min(hi);
        self
    }
    /// Append `x = clamp(x, lo, hi)`.
    #[must_use]
    pub fn clamp(mut self, lo: T, hi: T) -> Self {
        self.epilogue = self.epilogue.clamp(lo, hi);
        self
    }

    /// Append `x -= scalar`.
    #[must_use]
    pub fn sub_scalar(mut self, v: T) -> Self {
        self.epilogue = self.epilogue.sub_scalar(v);
        self
    }
    /// Append `x /= scalar`.
    #[must_use]
    pub fn div_scalar(mut self, v: T) -> Self {
        self.epilogue = self.epilogue.div_scalar(v);
        self
    }
    /// Append `x -= row[i]` (per-M, length `m`).
    #[must_use]
    pub fn sub_row(mut self, row: &'a [T]) -> Self {
        self.epilogue = self.epilogue.sub_row(row);
        self
    }
    /// Append `x /= row[i]`.
    #[must_use]
    pub fn div_row(mut self, row: &'a [T]) -> Self {
        self.epilogue = self.epilogue.div_row(row);
        self
    }
    /// Append `x = max(x, row[i])`.
    #[must_use]
    pub fn max_row(mut self, row: &'a [T]) -> Self {
        self.epilogue = self.epilogue.max_row(row);
        self
    }
    /// Append `x = min(x, row[i])`.
    #[must_use]
    pub fn min_row(mut self, row: &'a [T]) -> Self {
        self.epilogue = self.epilogue.min_row(row);
        self
    }
    /// Append `x -= col[j]` (per-N, length `n`).
    #[must_use]
    pub fn sub_col(mut self, col: &'a [T]) -> Self {
        self.epilogue = self.epilogue.sub_col(col);
        self
    }
    /// Append `x /= col[j]`.
    #[must_use]
    pub fn div_col(mut self, col: &'a [T]) -> Self {
        self.epilogue = self.epilogue.div_col(col);
        self
    }
    /// Append `x = max(x, col[j])`.
    #[must_use]
    pub fn max_col(mut self, col: &'a [T]) -> Self {
        self.epilogue = self.epilogue.max_col(col);
        self
    }
    /// Append `x = min(x, col[j])`.
    #[must_use]
    pub fn min_col(mut self, col: &'a [T]) -> Self {
        self.epilogue = self.epilogue.min_col(col);
        self
    }
    /// Append `x -= tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn sub_tensor(mut self, tensor: &'a [T]) -> Self {
        self.epilogue = self.epilogue.sub_tensor(tensor);
        self
    }
    /// Append `x -= tensor[i*ld + j]` (row-major, stride `ld`).
    #[must_use]
    pub fn sub_tensor_strided(mut self, tensor: &'a [T], ld: usize) -> Self {
        self.epilogue = self.epilogue.sub_tensor_strided(tensor, ld);
        self
    }
    /// Append `x /= tensor[i*n + j]`.
    #[must_use]
    pub fn div_tensor(mut self, tensor: &'a [T]) -> Self {
        self.epilogue = self.epilogue.div_tensor(tensor);
        self
    }
    /// Append `x /= tensor[i*ld + j]` (row-major, stride `ld`).
    #[must_use]
    pub fn div_tensor_strided(mut self, tensor: &'a [T], ld: usize) -> Self {
        self.epilogue = self.epilogue.div_tensor_strided(tensor, ld);
        self
    }
    /// Append `x = max(x, tensor[i*n + j])` (full `m*n` row-major).
    #[must_use]
    pub fn max_tensor(mut self, tensor: &'a [T]) -> Self {
        self.epilogue = self.epilogue.max_tensor(tensor);
        self
    }
    /// Append `x = max(x, tensor[i*ld + j])` (row-major, stride `ld`).
    #[must_use]
    pub fn max_tensor_strided(mut self, tensor: &'a [T], ld: usize) -> Self {
        self.epilogue = self.epilogue.max_tensor_strided(tensor, ld);
        self
    }
    /// Append `x = min(x, tensor[i*n + j])` (full `m*n` row-major).
    #[must_use]
    pub fn min_tensor(mut self, tensor: &'a [T]) -> Self {
        self.epilogue = self.epilogue.min_tensor(tensor);
        self
    }
    /// Append `x = min(x, tensor[i*ld + j])` (row-major, stride `ld`).
    #[must_use]
    pub fn min_tensor_strided(mut self, tensor: &'a [T], ld: usize) -> Self {
        self.epilogue = self.epilogue.min_tensor_strided(tensor, ld);
        self
    }

    /// Append leaky `ReLU`: `x = x>=0 ? x : alpha*x`.
    #[must_use]
    pub fn leaky_relu(mut self, alpha: f32) -> Self {
        self.epilogue = self.epilogue.leaky_relu(alpha);
        self
    }
    /// Append `ReLU6`: `clamp(x, 0, 6)`.
    #[must_use]
    pub fn relu6(mut self) -> Self {
        self.epilogue = self.epilogue.relu6();
        self
    }
    /// Append hard sigmoid: `clamp(x/6 + 0.5, 0, 1)`.
    #[must_use]
    pub fn hardsigmoid(mut self) -> Self {
        self.epilogue = self.epilogue.hardsigmoid();
        self
    }
    /// Append hard swish: `x * clamp(x/6 + 0.5, 0, 1)`.
    #[must_use]
    pub fn hardswish(mut self) -> Self {
        self.epilogue = self.epilogue.hardswish();
        self
    }
    /// Append `x = |x|`.
    #[must_use]
    pub fn abs(mut self) -> Self {
        self.epilogue = self.epilogue.abs();
        self
    }
    /// Append `x = -x`.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(mut self) -> Self {
        self.epilogue = self.epilogue.neg();
        self
    }
    /// Append `x = x*x`.
    #[must_use]
    pub fn square(mut self) -> Self {
        self.epilogue = self.epilogue.square();
        self
    }
    /// Append `x = sign(x)`.
    #[must_use]
    pub fn sign(mut self) -> Self {
        self.epilogue = self.epilogue.sign();
        self
    }
    /// Append `x = sqrt(x)`.
    #[must_use]
    pub fn sqrt(mut self) -> Self {
        self.epilogue = self.epilogue.sqrt();
        self
    }
    /// Append softsign: `x / (1 + |x|)`.
    #[must_use]
    pub fn softsign(mut self) -> Self {
        self.epilogue = self.epilogue.softsign();
        self
    }
    /// Append `x = 1/x`.
    #[must_use]
    pub fn recip(mut self) -> Self {
        self.epilogue = self.epilogue.recip();
        self
    }
    /// Append `x = 1/sqrt(x)`.
    #[must_use]
    pub fn rsqrt(mut self) -> Self {
        self.epilogue = self.epilogue.rsqrt();
        self
    }
    /// Append `x = exp(x)`.
    #[must_use]
    pub fn exp(mut self) -> Self {
        self.epilogue = self.epilogue.exp();
        self
    }
    /// Append `x = log(x)`.
    #[must_use]
    pub fn log(mut self) -> Self {
        self.epilogue = self.epilogue.log();
        self
    }
    /// Append `ELU`: `x = x>=0 ? x : alpha*(exp(x)-1)`.
    #[must_use]
    pub fn elu(mut self, alpha: f32) -> Self {
        self.epilogue = self.epilogue.elu(alpha);
        self
    }
    /// Append `SELU`.
    #[must_use]
    pub fn selu(mut self) -> Self {
        self.epilogue = self.epilogue.selu();
        self
    }
    /// Append softplus: `log1p(exp(x))`.
    #[must_use]
    pub fn softplus(mut self) -> Self {
        self.epilogue = self.epilogue.softplus();
        self
    }
    /// Append Mish: `x * tanh(softplus(x))`.
    #[must_use]
    pub fn mish(mut self) -> Self {
        self.epilogue = self.epilogue.mish();
        self
    }
    /// Append exact `GELU` via erf (distinct from tanh-approx [`Self::gelu`]).
    #[must_use]
    pub fn gelu_exact(mut self) -> Self {
        self.epilogue = self.epilogue.gelu_exact();
        self
    }
    /// Scale the product: `D = act(beta*(A@B) + bias)`. Default `1`.
    #[must_use]
    pub const fn beta(mut self, beta: T) -> Self {
        self.beta = beta;
        self
    }
    /// Write the output column-major (`C[i,j] = c[i + j*m]`) instead of the
    /// default row-major. The C kernels store either layout natively.
    #[must_use]
    pub const fn col_major_output(mut self) -> Self {
        self.col_major = true;
        self
    }
    /// Run into `c` (`m x n`, row- or column-major per [`Self::col_major_output`]).
    ///
    /// # Panics
    /// Panics if `a`/`c` lengths are inconsistent with `m` and the packed `n`/`k`,
    /// or if any epilogue component length mismatches: `bias` (per-N) must be
    /// length `n`, `bias_row` (per-M) length `m`, and `residual` must hold `m*n`
    /// elements at its row stride.
    pub fn run(self, c: &mut [T]) {
        T::run_packed_ep(
            self.a,
            self.packed,
            c,
            self.m,
            &self.epilogue,
            self.beta,
            self.col_major,
        );
    }
}
