//! The element-typed fused-epilogue op-graph builder.

use super::{
    Activation, EP_ACT_ABS, EP_ACT_ELU, EP_ACT_EXP, EP_ACT_GELU_EXACT, EP_ACT_HARDSIGMOID,
    EP_ACT_HARDSWISH, EP_ACT_LEAKY_RELU, EP_ACT_LOG, EP_ACT_MISH, EP_ACT_NEG, EP_ACT_RECIP,
    EP_ACT_RELU6, EP_ACT_RSQRT, EP_ACT_SELU, EP_ACT_SIGMOID, EP_ACT_SIGN, EP_ACT_SOFTPLUS,
    EP_ACT_SOFTSIGN, EP_ACT_SQRT, EP_ACT_SQUARE, EP_ACT_TANH, EpNode, EpOp,
};
use crate::element::Element;
use crate::exec::PackedEpilogue;

/// A fused GEMM epilogue as an ORDERED OP-GRAPH (the runtime analog of a
/// CUTLASS epilogue visitor tree), applied in-register at the store (no
/// separate pass over C).
///
/// The running value starts as the post-beta accumulator `beta*(A@B)[i,j]`;
/// each node mutates it in the order the builder methods were called:
///
/// ```text
/// x = beta*(A@B)[i,j]
/// for node in nodes: x = node(x)   // add/mul scalar|row|col|tensor, act, min/max
/// out[i,j] = x
/// ```
///
/// Build with [`Epilogue::new`] then chain `.add_scalar`/`.add_col`/`.mul_row`/
/// `.add_tensor`/`.relu`/`.clamp` etc. Generic over the element type so operands
/// are type-checked against the GEMM dtype. The borrowed operand slices are held
/// for lifetime safety; the raw `EpNode` array is materialized at `.run()`.
///
/// The op-graph is a read-only view of its operand slices, so it is
/// `Send + Sync`: one instance can be shared by concurrent GEMM calls.
#[derive(Clone, Debug)]
pub struct Epilogue<'a, T: Element = half::f16> {
    pub(crate) nodes: Vec<EpNode>,
    /// Per-node operand slice length (0 for scalar/min/max nodes). Kept for
    /// length validation and the scalar fallback; not sent across FFI.
    pub(crate) lens: Vec<usize>,
    /// Trailing activation (applied last, CUTLASS-style). The activation methods
    /// (`.relu`/`.gelu`/`.silu`/`.tanh`/`.sigmoid`) set this rather
    /// than pushing an in-order node, so they are order-independent of the
    /// additive/scaling ops -- preserving the original fixed-epilogue semantics.
    pub(crate) act: Option<u32>,
    _ops: core::marker::PhantomData<&'a [T]>,
}

// SAFETY: an `Epilogue` is plain data (op/aux/scalar/ld) plus raw pointers
// that all originate from caller-provided `&'a [T]` operand slices
// (`push_vec`) and are only ever read (both by the scalar fallback and the C
// kernels -- every op kind in csrc/epilogue.h reads `ptr`, none write).
// Sending or sharing it therefore has exactly the semantics of `&'a [T]`,
// which is `Send + Sync` iff `T: Sync`. The raw-pointer fields opt the type
// out of the auto impls; restore them so one epilogue can be shared by
// concurrent GEMM calls.
// SAFETY: see the argument above -- read-only views of caller slices.
unsafe impl<T: Element + Sync> Send for Epilogue<'_, T> {}
// SAFETY: as the Send impl above.
unsafe impl<T: Element + Sync> Sync for Epilogue<'_, T> {}

// The bf16 fused epilogue computes the whole op-graph in f32 (the bf16
// accumulator and operands are upcast, the graph runs in f32, and the result is
// rounded back to bf16), so it supports the SAME full op set as f16/f32 -- bf16
// carries no divide/transcendental restriction or guard.

impl<T: Element + PackedEpilogue> Default for Epilogue<'_, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: Element + PackedEpilogue> Epilogue<'a, T> {
    /// An empty epilogue (identity). Chain ops to build the graph.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            lens: Vec::new(),
            act: None,
            _ops: core::marker::PhantomData,
        }
    }

    #[inline]
    fn push_scalar(mut self, op: EpOp, v: T) -> Self {
        self.nodes.push(EpNode {
            op: op as u32,
            aux: 0,
            scalar: v.ep_to_f32(),
            ptr: core::ptr::null(),
            ld: 0,
        });
        self.lens.push(0);
        self
    }
    #[inline]
    fn push_vec(mut self, op: EpOp, v: &'a [T], ld: usize) -> Self {
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
    /// Push an in-order activation node (`aux` = kind, `scalar` = alpha param for
    /// `leaky_relu`/`elu`, else unused).
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
    pub fn add_scalar(self, v: T) -> Self {
        self.push_scalar(EpOp::AddScalar, v)
    }
    /// `x *= scalar`.
    #[must_use]
    pub fn mul_scalar(self, v: T) -> Self {
        self.push_scalar(EpOp::MulScalar, v)
    }
    /// `x += row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn add_row(self, row: &'a [T]) -> Self {
        self.push_vec(EpOp::AddRow, row, 0)
    }
    /// `x *= row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn mul_row(self, row: &'a [T]) -> Self {
        self.push_vec(EpOp::MulRow, row, 0)
    }
    /// `x += col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn add_col(self, col: &'a [T]) -> Self {
        self.push_vec(EpOp::AddCol, col, 0)
    }
    /// `x *= col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn mul_col(self, col: &'a [T]) -> Self {
        self.push_vec(EpOp::MulCol, col, 0)
    }
    /// `x += tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn add_tensor(self, tensor: &'a [T]) -> Self {
        self.push_vec(EpOp::AddTensor, tensor, 0)
    }
    /// `x += tensor[i*ld + j]` (full row-major, explicit row stride `ld`).
    #[must_use]
    pub fn add_tensor_strided(self, tensor: &'a [T], ld: usize) -> Self {
        self.push_vec(EpOp::AddTensor, tensor, ld)
    }
    /// `x *= tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn mul_tensor(self, tensor: &'a [T]) -> Self {
        self.push_vec(EpOp::MulTensor, tensor, 0)
    }
    /// `x *= tensor[i*ld + j]` (full row-major, explicit row stride `ld`).
    #[must_use]
    pub fn mul_tensor_strided(self, tensor: &'a [T], ld: usize) -> Self {
        self.push_vec(EpOp::MulTensor, tensor, ld)
    }
    #[inline]
    #[must_use]
    pub(crate) fn activation(mut self, act: Activation) -> Self {
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
    pub fn max(self, lo: T) -> Self {
        self.push_scalar(EpOp::MaxScalar, lo)
    }
    /// `x = min(x, hi)`.
    #[must_use]
    pub fn min(self, hi: T) -> Self {
        self.push_scalar(EpOp::MinScalar, hi)
    }
    /// `x = clamp(x, lo, hi)` (`max(lo)` then `min(hi)`).
    #[must_use]
    pub fn clamp(self, lo: T, hi: T) -> Self {
        self.max(lo).min(hi)
    }

    // --- elementwise binary against scalar/row/col/tensor operands -----------
    /// `x -= scalar`.
    #[must_use]
    pub fn sub_scalar(self, v: T) -> Self {
        self.push_scalar(EpOp::SubScalar, v)
    }
    /// `x /= scalar`.
    #[must_use]
    pub fn div_scalar(self, v: T) -> Self {
        self.push_scalar(EpOp::DivScalar, v)
    }
    /// `x -= row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn sub_row(self, row: &'a [T]) -> Self {
        self.push_vec(EpOp::SubRow, row, 0)
    }
    /// `x /= row[i]` (per-M vector, length `m`).
    #[must_use]
    pub fn div_row(self, row: &'a [T]) -> Self {
        self.push_vec(EpOp::DivRow, row, 0)
    }
    /// `x = max(x, row[i])` (per-M vector, length `m`).
    #[must_use]
    pub fn max_row(self, row: &'a [T]) -> Self {
        self.push_vec(EpOp::MaxRow, row, 0)
    }
    /// `x = min(x, row[i])` (per-M vector, length `m`).
    #[must_use]
    pub fn min_row(self, row: &'a [T]) -> Self {
        self.push_vec(EpOp::MinRow, row, 0)
    }
    /// `x -= col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn sub_col(self, col: &'a [T]) -> Self {
        self.push_vec(EpOp::SubCol, col, 0)
    }
    /// `x /= col[j]` (per-N vector, length `n`).
    #[must_use]
    pub fn div_col(self, col: &'a [T]) -> Self {
        self.push_vec(EpOp::DivCol, col, 0)
    }
    /// `x = max(x, col[j])` (per-N vector, length `n`).
    #[must_use]
    pub fn max_col(self, col: &'a [T]) -> Self {
        self.push_vec(EpOp::MaxCol, col, 0)
    }
    /// `x = min(x, col[j])` (per-N vector, length `n`).
    #[must_use]
    pub fn min_col(self, col: &'a [T]) -> Self {
        self.push_vec(EpOp::MinCol, col, 0)
    }
    /// `x -= tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn sub_tensor(self, tensor: &'a [T]) -> Self {
        self.push_vec(EpOp::SubTensor, tensor, 0)
    }
    /// `x -= tensor[i*ld + j]` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn sub_tensor_strided(self, tensor: &'a [T], ld: usize) -> Self {
        self.push_vec(EpOp::SubTensor, tensor, ld)
    }
    /// `x /= tensor[i*n + j]` (full `m*n` row-major).
    #[must_use]
    pub fn div_tensor(self, tensor: &'a [T]) -> Self {
        self.push_vec(EpOp::DivTensor, tensor, 0)
    }
    /// `x /= tensor[i*ld + j]` (row-major stride `ld`).
    #[must_use]
    pub fn div_tensor_strided(self, tensor: &'a [T], ld: usize) -> Self {
        self.push_vec(EpOp::DivTensor, tensor, ld)
    }
    /// `x = max(x, tensor[i*n + j])` (full `m*n` row-major).
    #[must_use]
    pub fn max_tensor(self, tensor: &'a [T]) -> Self {
        self.push_vec(EpOp::MaxTensor, tensor, 0)
    }
    /// `x = max(x, tensor[i*ld + j])` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn max_tensor_strided(self, tensor: &'a [T], ld: usize) -> Self {
        self.push_vec(EpOp::MaxTensor, tensor, ld)
    }
    /// `x = min(x, tensor[i*n + j])` (full `m*n` row-major).
    #[must_use]
    pub fn min_tensor(self, tensor: &'a [T]) -> Self {
        self.push_vec(EpOp::MinTensor, tensor, 0)
    }
    /// `x = min(x, tensor[i*ld + j])` (row-major, explicit row stride `ld`).
    #[must_use]
    pub fn min_tensor_strided(self, tensor: &'a [T], ld: usize) -> Self {
        self.push_vec(EpOp::MinTensor, tensor, ld)
    }

    // --- Group A activations (exact / no transcendental) --------------------
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
    /// `x = SELU(x)` (standard `1.0507`/`1.6733` constants).
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
    /// `x = 0.5*x*(1 + erf(x/sqrt(2)))` (exact `GELU` via erf, distinct from the
    /// tanh-approx [`Self::gelu`]).
    #[must_use]
    pub fn gelu_exact(self) -> Self {
        self.push_act(EP_ACT_GELU_EXACT, 0.0)
    }
}
