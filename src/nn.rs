//! The row-wise passes that sit between matmuls in a model, writing f32 or
//! straight to f16 for the next [`Linear`](crate::Linear).
//!
//! Plain Rust over eight-lane accumulators, which LLVM vectorizes to NEON on
//! the calling thread: one row is a few hundred nanoseconds of work, far below
//! what spreading it across cores or entering a streaming region would cost.

use std::cell::RefCell;

use crate::ModelFloat;

thread_local! {
    static ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static XS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Sum of `f(x)` over a row, eight lanes at a time.
fn lane_sum(x: &[f32], f: impl Fn(f32) -> f32) -> f32 {
    let mut acc = [0f32; 8];
    let mut chunks = x.chunks_exact(8);
    for c in &mut chunks {
        for (a, &v) in acc.iter_mut().zip(c) {
            *a += f(v);
        }
    }
    let tail: f32 = chunks.remainder().iter().map(|&v| f(v)).sum();
    acc.iter().sum::<f32>() + tail
}

/// Runs `norm(row_in, row_out)` over every `dim`-wide row of `x` into `y`.
fn rows<X: ModelFloat, Y: ModelFloat>(
    x: &[X],
    dim: usize,
    y: &mut [Y],
    norm: impl Fn(&[f32], &mut [f32]),
) {
    assert!(dim > 0, "a norm needs at least one feature");
    assert!(
        x.len().is_multiple_of(dim),
        "x holds whole rows of weight.len()"
    );
    assert_eq!(y.len(), x.len(), "y is the same shape as x");
    XS.with_borrow_mut(|xs| {
        ROW.with_borrow_mut(|row| {
            row.resize(dim, 0.0);
            let x = X::as_f32(x, xs);
            for (xr, yr) in x.chunks_exact(dim).zip(y.chunks_exact_mut(dim)) {
                norm(xr, row);
                Y::store_f32(row, yr);
            }
        });
    });
}

/// Layer normalization of each `weight.len()`-wide row of `x`:
/// `(x - mean) / sqrt(var + eps) * weight + bias`, into `y` (f32 or f16, same
/// shape). Mean and variance are two passes in f32.
///
/// ```
/// use half::f16;
/// let (x, w) = (vec![1.0f32, 2.0, 3.0, 4.0], vec![1.0f32; 4]);
/// let mut y = vec![f16::ZERO; 4]; // f16, ready for the next matmul
/// sme_gemm::nn::layer_norm(&x, &w, None, 1e-5, &mut y);
/// ```
///
/// # Panics
/// Panics if `x.len()` is not a multiple of `weight.len()`, `y` is not the
/// shape of `x`, or `bias` is not `weight.len()` long.
pub fn layer_norm<X: ModelFloat, Y: ModelFloat>(
    x: &[X],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    y: &mut [Y],
) {
    let dim = weight.len();
    if let Some(b) = bias {
        assert_eq!(b.len(), dim, "bias holds weight.len() values");
    }
    rows(x, dim, y, |x, out| {
        let mean = lane_sum(x, |v| v) / dim as f32;
        let var = lane_sum(x, |v| (v - mean) * (v - mean)) / dim as f32;
        let r = 1.0 / (var + eps).sqrt();
        match bias {
            Some(b) => {
                for (((o, &v), &w), &b) in out.iter_mut().zip(x).zip(weight).zip(b) {
                    *o = (v - mean) * r * w + b;
                }
            }
            None => {
                for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
                    *o = (v - mean) * r * w;
                }
            }
        }
    });
}

/// RMS normalization of each `weight.len()`-wide row of `x`:
/// `x / sqrt(mean(x^2) + eps) * weight`, into `y` (f32 or f16, same shape).
///
/// # Panics
/// Panics if `x.len()` is not a multiple of `weight.len()` or `y` is not the
/// shape of `x`.
pub fn rms_norm<X: ModelFloat, Y: ModelFloat>(x: &[X], weight: &[f32], eps: f32, y: &mut [Y]) {
    let dim = weight.len();
    rows(x, dim, y, |x, out| {
        let r = 1.0 / (lane_sum(x, |v| v * v) / dim as f32 + eps).sqrt();
        for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
            *o = v * r * w;
        }
    });
}

/// `out[i] = 1 / sqrt(mean(x_i^2) + eps)` for each `k`-wide row of `x`: the
/// run-time half of an `RMSNorm` whose weight was folded into the next layer.
pub(crate) fn rms_scales(x: &[half::f16], k: usize, eps: f32, out: &mut [half::f16]) {
    for (row, o) in x.chunks_exact(k).zip(out.iter_mut()) {
        let mut acc = [0f32; 8];
        let mut chunks = row.chunks_exact(8);
        for c in &mut chunks {
            for (a, v) in acc.iter_mut().zip(c) {
                let v = v.to_f32();
                *a += v * v;
            }
        }
        let tail: f32 = chunks
            .remainder()
            .iter()
            .map(|v| v.to_f32() * v.to_f32())
            .sum();
        let ms = (acc.iter().sum::<f32>() + tail) / k as f32;
        *o = half::f16::from_f32(1.0 / (ms + eps).sqrt());
    }
}
