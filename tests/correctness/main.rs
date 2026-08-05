//! Correctness tests: the SME kernels checked against an independent f64/i32
//! oracle across a range of small shapes and every dtype/epilogue path.

// Under Miri the SME kernels are unreachable (no FFI) and these shapes are far
// too large for the interpreter; tests/proptest.rs covers the pure-Rust
// fallback path under Miri with small shapes instead.
#![cfg(not(miri))]

use half::f16;
use sme_gemm::{Accuracy, matmul_f16};

/// Test-local activation discriminator (the library's `Activation` enum is
/// internal; the public API exposes the named `.relu()/.gelu()/.silu()`
/// shortcut methods instead).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    None,
    Relu,
    Gelu,
    Silu,
}

/// Apply an [`Act`] to any epilogue/Gemm builder via its canonical shortcut
/// methods. `Act::None` leaves the builder unchanged.
macro_rules! apply_act {
    ($b:expr, $act:expr) => {
        match $act {
            Act::None => $b,
            Act::Relu => $b.relu(),
            Act::Gelu => $b.gelu(),
            Act::Silu => $b.silu(),
        }
    };
}

fn fill(seed: &mut u64, len: usize) -> Vec<f16> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (*seed >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
            f16::from_f32(u - 0.5)
        })
        .collect()
}

// independent f64 oracle, row-major
fn oracle(a: &[f16], b: &[f16], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn max_rel(got: &[f16], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs() / (1.0 + w.abs()))
        .fold(0.0, f64::max)
}

fn check(m: usize, n: usize, k: usize, mode: Accuracy, tol: f64) {
    let mut s = 0x1234_5678_9abc_def0 ^ ((m * 131 + n * 17 + k) as u64);
    let a = fill(&mut s, m * k);
    let b = fill(&mut s, k * n);
    let mut c = vec![f16::ZERO; m * n];
    matmul_f16(&a, &b, &mut c, m, n, k, mode);
    let mr = max_rel(&c, &oracle(&a, &b, m, n, k));
    assert!(mr < tol, "{m}x{n}x{k} {mode:?}: max_rel={mr} >= {tol}");
}

const SIZES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (7, 5, 3),
    (31, 31, 31),
    (32, 32, 32),
    (33, 17, 9),
    (64, 96, 48),
    (128, 64, 80),
    (40, 8, 7),
    (7, 50, 200),
    (96, 96, 256),
];

// Shapes that BOTH clear the SME threshold AND have non-multiple-of-32 tails in
// m/n/k, so the real SME kernels run and their M/N/K-tail predicate handling is
// exercised at a size where the parallel paths engage. (Several of the default
// `SIZES` now clear the tiled floor of 2^12 and reach the kernels too, but they
// are small enough to stay on the serial arm.)
//   65*257*257 = 4_293_185 ; 33*512*257 = 4_342_272
//   97*300*131 = 3_812_100 ; 7*400*800  = 2_240_000   (all >= 262_144)
//
// (33, 161, 401) is there for a dispatch reason, not a tail one: m_tiles=2 and
// n_tiles_pad=6 is the one flat-M band that splits exactly two ways, which the
// f16f16/b16b16 packed paths deliberately run serially (see gemm_f16f16.c).
// Nothing else here lands in it, so without it that branch is never taken.
const TAIL_SIZES: &[(usize, usize, usize)] = &[
    (65, 257, 257),
    (33, 512, 257),
    (97, 300, 131),
    (7, 400, 800),
    (33, 161, 401),
];

// ---- Strided gemm_* API (alpha/beta accumulate, transposed/strided operands,
// column-major output). Sizes clear 2^18 so the real SME kernel runs. ----

// f64 oracle for C = alpha*C_initial + beta*(A @ B), row-major A/B.
#[allow(clippy::too_many_arguments)]
fn gemm_oracle(
    a: &[f64],
    b: &[f64],
    c_init: &[f64],
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            out[i * n + j] = alpha * c_init[i * n + j] + beta * acc;
        }
    }
    out
}

// ---- Fused op-graph epilogue (ordered EpNode list) -------------------------

fn gelu_f64(x: f64) -> f64 {
    0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh())
}

fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

// f32 op-graph helpers: random fill and an f64 product oracle (row-major).
fn fill_f32(seed: &mut u64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn oracle_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
            }
            c[i * n + j] = acc;
        }
    }
    c
}

// Sizes that clear the SME f32 threshold so the real kernel runs.
const F32_EP_SIZES: &[(usize, usize, usize)] =
    &[(32, 32, 32), (33, 17, 9), (64, 96, 48), (96, 96, 256)];

fn fill_f64(seed: &mut u64, len: usize) -> Vec<f64> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*seed >> 40) as f64 / (1u64 << 24) as f64 - 0.5
        })
        .collect()
}

fn oracle_f64(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

// Sizes that clear the SME f64 threshold so the real kernel runs (when present).
const F64_EP_SIZES: &[(usize, usize, usize)] =
    &[(16, 16, 16), (17, 9, 5), (32, 48, 24), (48, 48, 128)];

// --- NaN parity: MAX/MIN nodes and ReLU are maxNum on every path ------------
// (SIMD used FMAX/FMIN, which propagate NaN, while the scalar arms used maxNum
// -- so the answer depended on the output layout. The activation clamps still
// propagate NaN on purpose, matching Rust's `clamp`; the last test pins that.)

/// A@B whose every cell is `inf + (-inf)` = NaN, above the SME worth-it floor.
/// The infinities are in the INPUTS, so the f64-accumulating reference reaches
/// the same NaN and these run as real cross-path tests on non-SME CI too.
fn nan_accumulator_operands(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let a = vec![1.0f32; m * k];
    let b: Vec<f32> = (0..k * n)
        .map(|idx| {
            if (idx / n).is_multiple_of(2) {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    (a, b)
}

mod attention;
mod basic;
mod batched;
mod epilogue_half;
mod epilogue_wide;
mod exhaustive;
mod nan;
mod narrow;
mod q4;
mod quant;
mod reduce;
mod strided;
