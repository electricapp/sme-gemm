//! The fused-epilogue builder API.
//!
//! This module holds the pieces both graphs share -- the opcodes
//! ([`EpOp`]), the FFI node ([`EpNode`]), the activation table and its
//! scalar evaluators. The builders themselves are one per file: the
//! element-typed [`Epilogue`] in `graph.rs`, the quantized [`Dequant`] in
//! `dequant.rs`.

/// Internal activation kind backing the `.relu`/`.gelu`/`.silu` shortcut methods
/// and the bf16/b16b16 ReLU-only kernel guard. Not part of the public API; the
/// public surface exposes named shortcut methods instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum Activation {
    #[default]
    None,
    Relu,
    Gelu,
    Silu,
}

impl Activation {
    #[inline]
    pub(crate) const fn to_c(self) -> i32 {
        match self {
            Self::None => 0,
            Self::Relu => 1,
            Self::Gelu => 2,
            Self::Silu => 3,
        }
    }
}

/// Exact f64 evaluation of an activation kind (mirrors `enum ep_act`), in f64
/// working precision; the f32 fallback widens through this. `alpha` is the
/// `LEAKY_RELU`/`ELU` parameter (ignored by the other kinds). Used by the scalar
/// (non-SME / strided) fallback, so it computes the true value (libm), not the
/// vector approximation; the SME path uses the in-register C approximations.
#[inline]
#[allow(clippy::many_single_char_names)]
pub(crate) fn act_apply_f64_a(kind: u32, x: f64, alpha: f64) -> f64 {
    match kind {
        1 => x.max(0.0),
        2 => 0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh()),
        3 => x / (1.0 + (-x).exp()),
        4 => x.tanh(),
        5 => 1.0 / (1.0 + (-x).exp()),
        EP_ACT_LEAKY_RELU => {
            if x >= 0.0 {
                x
            } else {
                alpha * x
            }
        }
        EP_ACT_RELU6 => x.clamp(0.0, 6.0),
        EP_ACT_HARDSIGMOID => (x / 6.0 + 0.5).clamp(0.0, 1.0),
        EP_ACT_HARDSWISH => x * (x / 6.0 + 0.5).clamp(0.0, 1.0),
        EP_ACT_ABS => x.abs(),
        EP_ACT_NEG => -x,
        EP_ACT_SQUARE => x * x,
        EP_ACT_SIGN => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        EP_ACT_SQRT => x.sqrt(),
        EP_ACT_SOFTSIGN => x / (1.0 + x.abs()),
        EP_ACT_RECIP => 1.0 / x,
        EP_ACT_RSQRT => 1.0 / x.sqrt(),
        EP_ACT_EXP => x.exp(),
        EP_ACT_LOG => x.ln(),
        EP_ACT_ELU => {
            if x >= 0.0 {
                x
            } else {
                alpha * x.exp_m1()
            }
        }
        EP_ACT_SELU => SELU_LAMBDA * if x >= 0.0 { x } else { SELU_ALPHA * x.exp_m1() },
        EP_ACT_SOFTPLUS => x.max(0.0) + (-x.abs()).exp().ln_1p(),
        EP_ACT_MISH => x * (x.max(0.0) + (-x.abs()).exp().ln_1p()).tanh(),
        EP_ACT_GELU_EXACT => 0.5 * x * (1.0 + erf_approx(x * core::f64::consts::FRAC_1_SQRT_2)),
        _ => x,
    }
}

/// `erf` via Abramowitz-Stegun 7.1.26 (rational, ~1.5e-7 max abs error). Used by
/// the f64 scalar `gelu_exact` fallback so the crate stays dependency-free; the
/// SME path uses the in-register `ep_erf_*` approximation.
#[inline]
fn erf_approx(x: f64) -> f64 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    let e = 1.0 - poly * (-ax * ax).exp();
    if x < 0.0 { -e } else { e }
}

/// f32 activation fallback: widen to f64, evaluate, narrow. `alpha` is the
/// `LEAKY_RELU`/`ELU` parameter.
#[inline]
pub(crate) fn act_apply_f32_a(kind: u32, x: f32, alpha: f32) -> f32 {
    act_apply_f64_a(kind, f64::from(x), f64::from(alpha)) as f32
}

/// Op-graph node opcodes. Mirror of `enum ep_op` in `csrc/epilogue.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum EpOp {
    AddScalar = 0,
    MulScalar = 1,
    AddRow = 2,
    MulRow = 3,
    AddCol = 4,
    MulCol = 5,
    AddTensor = 6,
    MulTensor = 7,
    Act = 8,
    MaxScalar = 9,
    MinScalar = 10,
    SubScalar = 11,
    SubRow = 12,
    SubCol = 13,
    SubTensor = 14,
    DivScalar = 15,
    DivRow = 16,
    DivCol = 17,
    DivTensor = 18,
    MaxRow = 19,
    MaxCol = 20,
    MaxTensor = 21,
    MinRow = 22,
    MinCol = 23,
    MinTensor = 24,
}

/// C-ABI mirror of `EpNode` in `csrc/epilogue.h`. One ordered op-graph node:
/// `op` selects the operation, `aux` carries the activation kind for `Act`,
/// `scalar` is the f32 operand for the `*_SCALAR` ops, `ptr` is the operand base
/// for ROW/COL/TENSOR ops (raw, borrowed from the builder's slices), `ld` is the
/// TENSOR row stride. Field order/layout must match the C struct exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct EpNode {
    pub(crate) op: u32,
    pub(crate) aux: u32,
    pub(crate) scalar: f32,
    pub(crate) ptr: *const core::ffi::c_void,
    pub(crate) ld: usize,
}

// Pin the FFI layout: the C `EpNode` (csrc/epilogue.h) is 32 bytes on arm64
// with a 4-byte hole before `ptr`. Layout drift would be silent UB at the FFI
// boundary, so fail the build instead (the C side has matching
// _Static_asserts).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const _: () = {
    assert!(size_of::<EpNode>() == 32, "size_of::<EpNode>() == 32");
    assert!(
        core::mem::offset_of!(EpNode, op) == 0,
        "core::mem::offset_of!(EpNode, op) == 0"
    );
    assert!(
        core::mem::offset_of!(EpNode, aux) == 4,
        "core::mem::offset_of!(EpNode, aux) == 4"
    );
    assert!(
        core::mem::offset_of!(EpNode, scalar) == 8,
        "core::mem::offset_of!(EpNode, scalar) == 8"
    );
    assert!(
        core::mem::offset_of!(EpNode, ptr) == 16,
        "core::mem::offset_of!(EpNode, ptr) == 16"
    );
    assert!(
        core::mem::offset_of!(EpNode, ld) == 24,
        "core::mem::offset_of!(EpNode, ld) == 24"
    );
};

/// Activation kind for the `Act` node aux field (mirrors `enum ep_act`).
pub(crate) const EP_ACT_TANH: u32 = 4;
/// Activation kind for the `Act` node aux field (mirrors `enum ep_act`).
pub(crate) const EP_ACT_SIGMOID: u32 = 5;
// Group A (exact / no transcendental). LEAKY_RELU/ELU carry their alpha in the
// node `scalar` field; the others ignore it.
pub(crate) const EP_ACT_LEAKY_RELU: u32 = 6;
pub(crate) const EP_ACT_RELU6: u32 = 7;
pub(crate) const EP_ACT_HARDSIGMOID: u32 = 8;
pub(crate) const EP_ACT_HARDSWISH: u32 = 9;
pub(crate) const EP_ACT_ABS: u32 = 10;
pub(crate) const EP_ACT_NEG: u32 = 11;
pub(crate) const EP_ACT_SQUARE: u32 = 12;
pub(crate) const EP_ACT_SIGN: u32 = 13;
pub(crate) const EP_ACT_SQRT: u32 = 14;
pub(crate) const EP_ACT_SOFTSIGN: u32 = 15;
pub(crate) const EP_ACT_RECIP: u32 = 16;
pub(crate) const EP_ACT_RSQRT: u32 = 17;
// Group B (need exp/log/erf).
pub(crate) const EP_ACT_EXP: u32 = 18;
pub(crate) const EP_ACT_LOG: u32 = 19;
pub(crate) const EP_ACT_ELU: u32 = 20;
pub(crate) const EP_ACT_SELU: u32 = 21;
pub(crate) const EP_ACT_SOFTPLUS: u32 = 22;
pub(crate) const EP_ACT_MISH: u32 = 23;
pub(crate) const EP_ACT_GELU_EXACT: u32 = 24;

/// SELU constant: the negative-branch scale (standard).
pub(crate) const SELU_ALPHA: f64 = 1.673_263_242_354_377_2;
/// SELU constant: the overall scale (standard).
pub(crate) const SELU_LAMBDA: f64 = 1.050_700_987_355_480_5;

mod dequant;
mod graph;

pub use dequant::Dequant;
pub use graph::Epilogue;
