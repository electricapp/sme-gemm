//! The row-wise passes that sit between matmuls in a model.
//!
//! The norms, as functions and as the [`Norm`] layer that holds its
//! parameters, write f32 or straight to f16 for the next
//! [`Linear`](crate::Linear).
//!
//! NEON on the calling thread (`csrc/neon_ops.c`; plain Rust elsewhere): one
//! row is tens of nanoseconds of work, far below what spreading it across
//! cores or entering a streaming region would cost.

use std::cell::RefCell;

use crate::ModelFloat;

thread_local! {
    static ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static XS: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Sum of `f(x)` over a row, eight lanes at a time.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
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
    if let Some(b) = bias {
        assert_eq!(b.len(), weight.len(), "bias holds weight.len() values");
    }
    norm(x, weight, bias, eps, false, y);
}

/// [`layer_norm`] (`rms` false) or [`rms_norm`] (`rms` true) over every row.
fn norm<X: ModelFloat, Y: ModelFloat>(
    x: &[X],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    rms: bool,
    y: &mut [Y],
) {
    let dim = weight.len();
    // f32 in, f16 out (the usual step before a matmul): one pass a row,
    // rounding and storing as it goes, with no f32 row in between.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let (Some(xf), Some(yh)) = (X::as_f32_slice(x), Y::as_f16_slice_mut(y)) {
        assert!(dim > 0, "a norm needs at least one feature");
        assert!(
            xf.len().is_multiple_of(dim),
            "x holds whole rows of weight.len()"
        );
        assert_eq!(yh.len(), xf.len(), "y is the same shape as x");
        for (xr, yr) in xf.chunks_exact(dim).zip(yh.chunks_exact_mut(dim)) {
            // SAFETY: each row and weight (and bias) hold dim values, checked
            // above and by the callers; the pass only reads x/weight/bias.
            unsafe {
                crate::ffi::neon_norm_row_f16(
                    yr.as_mut_ptr().cast::<u16>(),
                    xr.as_ptr(),
                    weight.as_ptr(),
                    bias.map_or(core::ptr::null(), <[f32]>::as_ptr),
                    dim,
                    eps,
                    i32::from(rms),
                );
            }
        }
        return;
    }
    rows(x, dim, y, |x, out| norm_row(out, x, weight, bias, eps, rms));
}

/// One row of [`layer_norm`] (`rms` false) or [`rms_norm`] (`rms` true).
fn norm_row(out: &mut [f32], x: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32, rms: bool) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    // SAFETY: out, x, weight (and bias) all hold weight.len() values (`rows`
    // and the callers check it); the pass only reads x/weight/bias.
    unsafe {
        crate::ffi::neon_norm_row_f32(
            out.as_mut_ptr(),
            x.as_ptr(),
            weight.as_ptr(),
            bias.map_or(core::ptr::null(), <[f32]>::as_ptr),
            weight.len(),
            eps,
            i32::from(rms),
        );
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let dim = weight.len() as f32;
        let mean = if rms { 0.0 } else { lane_sum(x, |v| v) / dim };
        let r = 1.0 / (lane_sum(x, |v| (v - mean) * (v - mean)) / dim + eps).sqrt();
        for (i, ((o, &v), &w)) in out.iter_mut().zip(x).zip(weight).enumerate() {
            *o = (v - mean) * r * w + bias.map_or(0.0, |b| b[i]);
        }
    }
}

/// RMS normalization of each `weight.len()`-wide row of `x`:
/// `x / sqrt(mean(x^2) + eps) * weight`, into `y` (f32 or f16, same shape).
///
/// # Panics
/// Panics if `x.len()` is not a multiple of `weight.len()` or `y` is not the
/// shape of `x`.
pub fn rms_norm<X: ModelFloat, Y: ModelFloat>(x: &[X], weight: &[f32], eps: f32, y: &mut [Y]) {
    norm(x, weight, None, eps, true, y);
}

/// A norm layer with its parameters: [`layer_norm`] ([`Norm::layer`], GPT-2's
/// `LayerNorm`) or [`rms_norm`] ([`Norm::rms`], Llama's `RMSNorm`).
///
/// ```
/// use half::f16;
/// use sme_gemm::nn::Norm;
/// let ln = Norm::layer(vec![1.0; 4], None, 1e-5);
/// let mut y = vec![f16::ZERO; 4];
/// ln.forward(&[1.0f32, 2.0, 3.0, 4.0], &mut y);
/// ```
#[derive(Clone, Debug)]
pub struct Norm {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    eps: f32,
    rms: bool,
}

impl Norm {
    /// `(x - mean) / sqrt(var + eps) * weight + bias`.
    ///
    /// # Panics
    /// Panics if `weight` is empty or `bias` is not `weight.len()` long.
    #[must_use]
    pub fn layer(weight: Vec<f32>, bias: Option<Vec<f32>>, eps: f32) -> Self {
        assert!(!weight.is_empty(), "a norm needs at least one feature");
        if let Some(b) = &bias {
            assert_eq!(b.len(), weight.len(), "bias holds weight.len() values");
        }
        Self {
            weight,
            bias,
            eps,
            rms: false,
        }
    }

    /// `x / sqrt(mean(x^2) + eps) * weight`.
    ///
    /// # Panics
    /// Panics if `weight` is empty.
    #[must_use]
    pub fn rms(weight: Vec<f32>, eps: f32) -> Self {
        assert!(!weight.is_empty(), "a norm needs at least one feature");
        Self {
            weight,
            bias: None,
            eps,
            rms: true,
        }
    }

    /// Features per row.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.weight.len()
    }

    /// Normalizes each [`Norm::width`]-wide row of `x` into `y` (f32 or f16,
    /// same shape).
    ///
    /// # Panics
    /// Panics if `x.len()` is not a multiple of the width or `y` is not the
    /// shape of `x`.
    pub fn forward<X: ModelFloat, Y: ModelFloat>(&self, x: &[X], y: &mut [Y]) {
        norm(x, &self.weight, self.bias.as_deref(), self.eps, self.rms, y);
    }
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
