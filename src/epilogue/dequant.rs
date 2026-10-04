//! The quantized (i8/i16 -> f32) fused-dequant op-graph builder.

use super::{
    Activation, EP_ACT_ABS, EP_ACT_ELU, EP_ACT_EXP, EP_ACT_GELU_EXACT, EP_ACT_HARDSIGMOID,
    EP_ACT_HARDSWISH, EP_ACT_LEAKY_RELU, EP_ACT_LOG, EP_ACT_MISH, EP_ACT_NEG, EP_ACT_RECIP,
    EP_ACT_RELU6, EP_ACT_RSQRT, EP_ACT_SELU, EP_ACT_SIGMOID, EP_ACT_SIGN, EP_ACT_SOFTPLUS,
    EP_ACT_SOFTSIGN, EP_ACT_SQRT, EP_ACT_SQUARE, EP_ACT_TANH, EpNode, EpOp,
};

/// Fused i8 dequantization epilogue.
///
/// The i32 accumulator is scaled into the real (f32) domain, then an ORDERED
/// OP-GRAPH (the runtime analog of a CUTLASS epilogue visitor tree) is applied
/// in f32, all in-register (no separate dequant pass over the i32 output):
///
/// ```text
/// x = scale*acc[i,j]
/// for node in nodes: x = node(x)   // add/mul scalar|row|col|tensor, act, min/max
/// out[i,j] = x
/// ```
///
/// `scale` is a per-tensor scalar; `scale_per_n` overrides it with a
/// per-output-channel vector (length `n`). All op-graph operands are f32 (the
/// post-dequant domain). Order of method calls = order of nodes.
///
/// The op-graph is a read-only view of its operand slices, so it is
/// `Send + Sync`: one dequant graph can be shared by concurrent threads.
#[derive(Clone, Debug, Default)]
pub struct Dequant<'a> {
    pub(crate) scale: f32,
    pub(crate) scale_per_n: Option<&'a [f32]>,
    pub(crate) nodes: Vec<EpNode>,
    pub(crate) lens: Vec<usize>,
    /// Trailing activation (applied last), see `Epilogue::act`.
    pub(crate) act: Option<u32>,
    _ops: core::marker::PhantomData<&'a [f32]>,
}

// SAFETY: same argument as `Epilogue` above -- the node pointers all come from
// caller-provided `&'a [f32]` operands and are read-only on both sides of the
// FFI, so sharing a `Dequant` is sharing `&'a [f32]`. This lets one dequant
// graph (scales/biases) be shared by concurrent threads.
// SAFETY: see the argument above -- read-only views of caller f32 slices.
unsafe impl Send for Dequant<'_> {}
// SAFETY: as the Send impl above.
unsafe impl Sync for Dequant<'_> {}

impl<'a> Dequant<'a> {
    /// Dequant with a single per-tensor `scale`.
    #[must_use]
    pub const fn new(scale: f32) -> Self {
        Self {
            scale,
            scale_per_n: None,
            nodes: Vec::new(),
            lens: Vec::new(),
            act: None,
            _ops: core::marker::PhantomData,
        }
    }
    /// Use a per-output-channel scale vector (length `n`) instead of the scalar.
    #[must_use]
    pub const fn scale_per_n(mut self, scale: &'a [f32]) -> Self {
        self.scale_per_n = Some(scale);
        self
    }

    #[inline]
    fn push_scalar(mut self, op: EpOp, v: f32) -> Self {
        self.nodes.push(EpNode {
            op: op as u32,
            aux: 0,
            scalar: v,
            ptr: core::ptr::null(),
            ld: 0,
        });
        self.lens.push(0);
        self
    }
    #[inline]
    fn push_vec(mut self, op: EpOp, v: &'a [f32], ld: usize) -> Self {
        self.nodes.push(EpNode {
            op: op as u32,
            aux: 0,
            scalar: 0.0,
            ptr: v.as_ptr().cast::<core::ffi::c_void>(),
            ld,
        });
        self.lens.push(v.len());
        self
    }
    #[inline]
    fn push_act(mut self, kind: u32, alpha: f32) -> Self {
        self.nodes.push(EpNode {
            op: EpOp::Act as u32,
            aux: kind,
            scalar: alpha,
            ptr: core::ptr::null(),
            ld: 0,
        });
        self.lens.push(0);
        self
    }
    /// `x += scalar`.
    #[must_use]
    pub fn add_scalar(self, v: f32) -> Self {
        self.push_scalar(EpOp::AddScalar, v)
    }
    /// `x *= scalar`.
    #[must_use]
    pub fn mul_scalar(self, v: f32) -> Self {
        self.push_scalar(EpOp::MulScalar, v)
    }
    /// `x += row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn add_row(self, row: &'a [f32]) -> Self {
        self.push_vec(EpOp::AddRow, row, 0)
    }
    /// `x *= row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn mul_row(self, row: &'a [f32]) -> Self {
        self.push_vec(EpOp::MulRow, row, 0)
    }
    /// `x += col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn add_col(self, col: &'a [f32]) -> Self {
        self.push_vec(EpOp::AddCol, col, 0)
    }
    /// `x *= col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn mul_col(self, col: &'a [f32]) -> Self {
        self.push_vec(EpOp::MulCol, col, 0)
    }
    /// `x += tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn add_tensor(self, tensor: &'a [f32]) -> Self {
        self.push_vec(EpOp::AddTensor, tensor, 0)
    }
    /// `x += tensor[i*ld + j]` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn add_tensor_strided(self, tensor: &'a [f32], ld: usize) -> Self {
        self.push_vec(EpOp::AddTensor, tensor, ld)
    }
    /// `x *= tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn mul_tensor(self, tensor: &'a [f32]) -> Self {
        self.push_vec(EpOp::MulTensor, tensor, 0)
    }
    /// `x *= tensor[i*ld + j]` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn mul_tensor_strided(self, tensor: &'a [f32], ld: usize) -> Self {
        self.push_vec(EpOp::MulTensor, tensor, ld)
    }
    #[inline]
    #[must_use]
    fn activation(mut self, act: Activation) -> Self {
        self.act = if act == Activation::None {
            None
        } else {
            Some(act.to_c() as u32)
        };
        self
    }
    /// Trailing `x = max(x, 0)` (`Relu`).
    #[must_use]
    pub fn relu(self) -> Self {
        self.activation(Activation::Relu)
    }
    /// Trailing `x = gelu(x)` (tanh approximation).
    #[must_use]
    pub fn gelu(self) -> Self {
        self.activation(Activation::Gelu)
    }
    /// Trailing `x = silu(x)` (swish).
    #[must_use]
    pub fn silu(self) -> Self {
        self.activation(Activation::Silu)
    }
    /// Trailing `x = tanh(x)`. On the SME path this shares the gelu/silu
    /// in-register rational approximation (~4e-2 abs error vs libm, directly
    /// exposed here since there is no envelope); the scalar fallback is exact.
    #[must_use]
    pub const fn tanh(mut self) -> Self {
        self.act = Some(EP_ACT_TANH);
        self
    }
    /// Trailing `x = sigmoid(x)`. On the SME path this shares the gelu/silu
    /// in-register rational approximation (~4e-2 abs error vs libm, directly
    /// exposed here since there is no envelope); the scalar fallback is exact.
    #[must_use]
    pub const fn sigmoid(mut self) -> Self {
        self.act = Some(EP_ACT_SIGMOID);
        self
    }
    /// `x = max(x, lo)`.
    #[must_use]
    pub fn max(self, lo: f32) -> Self {
        self.push_scalar(EpOp::MaxScalar, lo)
    }
    /// `x = min(x, hi)`.
    #[must_use]
    pub fn min(self, hi: f32) -> Self {
        self.push_scalar(EpOp::MinScalar, hi)
    }
    /// `x = clamp(x, lo, hi)` (`max(lo)` then `min(hi)`).
    #[must_use]
    pub fn clamp(self, lo: f32, hi: f32) -> Self {
        self.max(lo).min(hi)
    }

    // --- elementwise binary against scalar/row/col/tensor operands -----------
    /// `x -= scalar`.
    #[must_use]
    pub fn sub_scalar(self, v: f32) -> Self {
        self.push_scalar(EpOp::SubScalar, v)
    }
    /// `x /= scalar`.
    #[must_use]
    pub fn div_scalar(self, v: f32) -> Self {
        self.push_scalar(EpOp::DivScalar, v)
    }
    /// `x -= row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn sub_row(self, row: &'a [f32]) -> Self {
        self.push_vec(EpOp::SubRow, row, 0)
    }
    /// `x /= row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn div_row(self, row: &'a [f32]) -> Self {
        self.push_vec(EpOp::DivRow, row, 0)
    }
    /// `x = max(x, row[i])` (per-M vector, length `m`).
    #[must_use]
    pub fn max_row(self, row: &'a [f32]) -> Self {
        self.push_vec(EpOp::MaxRow, row, 0)
    }
    /// `x = min(x, row[i])` (per-M vector, length `m`).
    #[must_use]
    pub fn min_row(self, row: &'a [f32]) -> Self {
        self.push_vec(EpOp::MinRow, row, 0)
    }
    /// `x -= col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn sub_col(self, col: &'a [f32]) -> Self {
        self.push_vec(EpOp::SubCol, col, 0)
    }
    /// `x /= col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn div_col(self, col: &'a [f32]) -> Self {
        self.push_vec(EpOp::DivCol, col, 0)
    }
    /// `x = max(x, col[j])` (per-N vector, length `n`).
    #[must_use]
    pub fn max_col(self, col: &'a [f32]) -> Self {
        self.push_vec(EpOp::MaxCol, col, 0)
    }
    /// `x = min(x, col[j])` (per-N vector, length `n`).
    #[must_use]
    pub fn min_col(self, col: &'a [f32]) -> Self {
        self.push_vec(EpOp::MinCol, col, 0)
    }
    /// `x -= tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn sub_tensor(self, tensor: &'a [f32]) -> Self {
        self.push_vec(EpOp::SubTensor, tensor, 0)
    }
    /// `x -= tensor[i*ld + j]` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn sub_tensor_strided(self, tensor: &'a [f32], ld: usize) -> Self {
        self.push_vec(EpOp::SubTensor, tensor, ld)
    }
    /// `x /= tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn div_tensor(self, tensor: &'a [f32]) -> Self {
        self.push_vec(EpOp::DivTensor, tensor, 0)
    }
    /// `x /= tensor[i*ld + j]` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn div_tensor_strided(self, tensor: &'a [f32], ld: usize) -> Self {
        self.push_vec(EpOp::DivTensor, tensor, ld)
    }
    /// `x = max(x, tensor[i*n + j])` (full `m*n` row-major).
    #[must_use]
    pub fn max_tensor(self, tensor: &'a [f32]) -> Self {
        self.push_vec(EpOp::MaxTensor, tensor, 0)
    }
    /// `x = max(x, tensor[i*ld + j])` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn max_tensor_strided(self, tensor: &'a [f32], ld: usize) -> Self {
        self.push_vec(EpOp::MaxTensor, tensor, ld)
    }
    /// `x = min(x, tensor[i*n + j])` (full `m*n` row-major).
    #[must_use]
    pub fn min_tensor(self, tensor: &'a [f32]) -> Self {
        self.push_vec(EpOp::MinTensor, tensor, 0)
    }
    /// `x = min(x, tensor[i*ld + j])` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn min_tensor_strided(self, tensor: &'a [f32], ld: usize) -> Self {
        self.push_vec(EpOp::MinTensor, tensor, ld)
    }

    // --- Group A activations -------------------------------------------------
    /// `x = x >= 0 ? x : alpha*x` (leaky `ReLU`).
    #[must_use]
    pub fn leaky_relu(self, alpha: f32) -> Self {
        self.push_act(EP_ACT_LEAKY_RELU, alpha)
    }
    /// `x = clamp(x, 0, 6)` (`ReLU6`).
    #[must_use]
    pub fn relu6(self) -> Self {
        self.push_act(EP_ACT_RELU6, 0.0)
    }
    /// `x = clamp(x/6 + 0.5, 0, 1)` (hard sigmoid).
    #[must_use]
    pub fn hardsigmoid(self) -> Self {
        self.push_act(EP_ACT_HARDSIGMOID, 0.0)
    }
    /// `x = x * clamp(x/6 + 0.5, 0, 1)` (hard swish).
    #[must_use]
    pub fn hardswish(self) -> Self {
        self.push_act(EP_ACT_HARDSWISH, 0.0)
    }
    /// `x = |x|`.
    #[must_use]
    pub fn abs(self) -> Self {
        self.push_act(EP_ACT_ABS, 0.0)
    }
    /// `x = -x`.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        self.push_act(EP_ACT_NEG, 0.0)
    }
    /// `x = x*x`.
    #[must_use]
    pub fn square(self) -> Self {
        self.push_act(EP_ACT_SQUARE, 0.0)
    }
    /// `x = sign(x)` in `{-1, 0, 1}`.
    #[must_use]
    pub fn sign(self) -> Self {
        self.push_act(EP_ACT_SIGN, 0.0)
    }
    /// `x = sqrt(x)`.
    #[must_use]
    pub fn sqrt(self) -> Self {
        self.push_act(EP_ACT_SQRT, 0.0)
    }
    /// `x = x / (1 + |x|)` (softsign).
    #[must_use]
    pub fn softsign(self) -> Self {
        self.push_act(EP_ACT_SOFTSIGN, 0.0)
    }
    /// `x = 1/x`.
    #[must_use]
    pub fn recip(self) -> Self {
        self.push_act(EP_ACT_RECIP, 0.0)
    }
    /// `x = 1/sqrt(x)`.
    #[must_use]
    pub fn rsqrt(self) -> Self {
        self.push_act(EP_ACT_RSQRT, 0.0)
    }

    // --- Group B activations (need exp/log/erf) -----------------------------
    /// `x = exp(x)`.
    #[must_use]
    pub fn exp(self) -> Self {
        self.push_act(EP_ACT_EXP, 0.0)
    }
    /// `x = log(x)`.
    #[must_use]
    pub fn log(self) -> Self {
        self.push_act(EP_ACT_LOG, 0.0)
    }
    /// `x = x >= 0 ? x : alpha*(exp(x)-1)` (`ELU`).
    #[must_use]
    pub fn elu(self, alpha: f32) -> Self {
        self.push_act(EP_ACT_ELU, alpha)
    }
    /// `x = SELU(x)` (standard constants).
    #[must_use]
    pub fn selu(self) -> Self {
        self.push_act(EP_ACT_SELU, 0.0)
    }
    /// `x = log1p(exp(x))` (softplus, numerically stable).
    #[must_use]
    pub fn softplus(self) -> Self {
        self.push_act(EP_ACT_SOFTPLUS, 0.0)
    }
    /// `x = x * tanh(softplus(x))` (Mish).
    #[must_use]
    pub fn mish(self) -> Self {
        self.push_act(EP_ACT_MISH, 0.0)
    }
    /// `x = 0.5*x*(1 + erf(x/sqrt(2)))` (exact `GELU` via erf).
    #[must_use]
    pub fn gelu_exact(self) -> Self {
        self.push_act(EP_ACT_GELU_EXACT, 0.0)
    }
}
