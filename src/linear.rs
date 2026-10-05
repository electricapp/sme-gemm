//! [`Linear`]: a linear layer's weights, ready for any number of rows, and the
//! [`ModelFloat`] element types the model-level helpers take.

use std::cell::RefCell;
use std::sync::OnceLock;

use half::f16;

use crate::element::sealed;
use crate::{
    Epilogue, Packed, Q4Params, Q4Weights, WeightLayout, caps, matmul_q4_ep, prepack, prepack_f16,
};

/// The float types the model-level helpers take and give: `f32` and `f16`.
///
/// [`Linear`], [`KvCache`] and [`nn`](crate::nn) convert at their edges, so a
/// model can keep its residual stream in f32 and hand f16 to the kernels without
/// conversion code of its own.
///
/// [`KvCache`]: crate::KvCache
pub trait ModelFloat: Copy + Default + sealed::Sealed {
    /// `x` as f16, borrowed or converted into `scratch`.
    #[doc(hidden)]
    fn as_f16<'a>(x: &'a [Self], scratch: &'a mut Vec<f16>) -> &'a [f16];
    /// `x` as f32, borrowed or converted into `scratch`.
    #[doc(hidden)]
    fn as_f32<'a>(x: &'a [Self], scratch: &'a mut Vec<f32>) -> &'a [f32];
    /// `dst = src`, rounding as needed.
    #[doc(hidden)]
    fn store_f32(src: &[f32], dst: &mut [Self]);
    /// `x` itself when `Self` is f32.
    #[doc(hidden)]
    fn as_f32_slice(x: &[Self]) -> Option<&[f32]>;
    /// `x` itself when `Self` is f16.
    #[doc(hidden)]
    fn as_f16_slice_mut(x: &mut [Self]) -> Option<&mut [f16]>;
}

impl ModelFloat for f16 {
    fn as_f16<'a>(x: &'a [Self], _: &'a mut Vec<f16>) -> &'a [f16] {
        x
    }
    fn as_f32<'a>(x: &'a [Self], scratch: &'a mut Vec<f32>) -> &'a [f32] {
        scratch.resize(x.len(), 0.0);
        crate::convert::to_f32(x, scratch);
        scratch
    }
    fn store_f32(src: &[f32], dst: &mut [Self]) {
        crate::convert::to_f16(src, dst);
    }
    fn as_f32_slice(_: &[Self]) -> Option<&[f32]> {
        None
    }
    fn as_f16_slice_mut(x: &mut [Self]) -> Option<&mut [f16]> {
        Some(x)
    }
}

impl ModelFloat for f32 {
    fn as_f16<'a>(x: &'a [Self], scratch: &'a mut Vec<f16>) -> &'a [f16] {
        scratch.resize(x.len(), f16::ZERO);
        crate::convert::to_f16(x, scratch);
        scratch
    }
    fn as_f32<'a>(x: &'a [Self], _: &'a mut Vec<f32>) -> &'a [f32] {
        x
    }
    fn store_f32(src: &[f32], dst: &mut [Self]) {
        dst.copy_from_slice(src);
    }
    fn as_f32_slice(x: &[Self]) -> Option<&[f32]> {
        Some(x)
    }
    fn as_f16_slice_mut(_: &mut [Self]) -> Option<&mut [f16]> {
        None
    }
}

thread_local! {
    /// Input converted to f16, and f16 output headed for an f32 destination.
    static X16: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
    static Y16: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
    /// Per-row RMS scales, and a gated layer's interleaved gate/up output.
    static R16: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
    static G16: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
}

/// A linear layer `y = x @ W (+ bias)`, `n` outputs by `k` inputs, that picks its
/// kernel by the number of rows.
///
/// Weights are 4-bit ([`Linear::quantize`]) or f16 ([`Linear::f16`]), from f32
/// or f16 in either [`WeightLayout`]; compute is f16. 4-bit weights stay
/// resident as [`Q4Weights`]: with few rows reading them is the whole cost, and
/// the Q4 GEMV reads a quarter of f16's bytes. From [`Linear::PANEL_ROWS`] rows
/// the multiply is compute-bound, and the Q4 MOPA path would unpack every weight
/// block again on each call, so `Linear` unpacks them once into a packed f16
/// panel (built on first use, 4x the Q4 bytes) and runs the dense f16 GEMM on
/// it. The two give the same products: the Q4 MOPA path is bit for bit the
/// eager dequant plus the dense GEMM.
///
/// Inputs are f16 or f32 ([`ModelFloat`]); outputs f16 ([`forward`]), f32
/// ([`forward_f32`]), or added into an f32 residual stream ([`accumulate`]). A
/// bias ([`with_bias`]) leads any epilogue a call adds.
///
/// ```
/// use half::f16;
/// use sme_gemm::{Epilogue, Linear, WeightLayout};
/// let (n, k) = (96, 64);
/// let w: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.01).collect();
/// let b = vec![0.1f32; n];
/// let fc = Linear::quantize(&w, WeightLayout::OutIn, n, k).with_bias(&b); // PyTorch layout
/// let x = vec![0.5f32; 2 * k]; // two rows, f32
/// let mut h = vec![f16::ZERO; 2 * n];
/// fc.forward_ep(&x, &mut h, 2, &Epilogue::new().gelu()); // h = gelu(x @ W + b)
/// let mut residual = vec![0.0f32; 2 * n];
/// fc.accumulate(&x, &mut residual, 2); // residual += x @ W + b
/// ```
///
/// [`forward`]: Linear::forward
/// [`forward_f32`]: Linear::forward_f32
/// [`accumulate`]: Linear::accumulate
/// [`with_bias`]: Linear::with_bias
#[derive(Debug)]
pub struct Linear {
    w: Weights,
    bias: Option<Vec<f16>>,
    rms_eps: Option<f32>,
}

#[derive(Debug)]
enum Weights {
    Q4 {
        q4: Q4Weights,
        panel: OnceLock<Packed<f16>>,
    },
    F16(Packed<f16>),
}

impl Linear {
    /// Rows at or above which 4-bit weights run on their f16 panel; below, the
    /// Q4 GEMV.
    pub const PANEL_ROWS: usize = 8;

    /// Quantizes f32 weights to `Q4_0` (32-deep blocks); see
    /// [`Q4Weights::quantize`].
    ///
    /// # Panics
    /// Panics if `w.len() != n * k`.
    #[must_use]
    pub fn quantize(w: &[f32], layout: WeightLayout, n: usize, k: usize) -> Self {
        Self::quantize_with(w, layout, n, k, Q4Params::default())
    }

    /// [`Linear::quantize`] with an explicit block size and code form.
    ///
    /// # Panics
    /// Panics if `w.len() != n * k`.
    #[must_use]
    pub fn quantize_with(w: &[f32], layout: WeightLayout, n: usize, k: usize, p: Q4Params) -> Self {
        Self::from_q4(Q4Weights::quantize(w, layout, n, k, p))
    }

    /// Wraps weights that are already 4-bit packed.
    #[must_use]
    pub const fn from_q4(q4: Q4Weights) -> Self {
        Self {
            w: Weights::Q4 {
                q4,
                panel: OnceLock::new(),
            },
            bias: None,
            rms_eps: None,
        }
    }

    /// f16 weights (from f32 or f16) in `layout`, packed once.
    ///
    /// # Panics
    /// Panics if `w.len() != n * k`.
    #[must_use]
    pub fn f16<T: ModelFloat>(w: &[T], layout: WeightLayout, n: usize, k: usize) -> Self {
        let mut s = Vec::new();
        Self::from_packed(prepack(T::as_f16(w, &mut s), layout, n, k))
    }

    /// Wraps an f16 panel that is already packed.
    #[must_use]
    pub const fn from_packed(p: Packed<f16>) -> Self {
        Self {
            w: Weights::F16(p),
            bias: None,
            rms_eps: None,
        }
    }

    /// Adds `bias` (`n` values, f32 or f16) to every output row.
    ///
    /// # Panics
    /// Panics if `bias.len() != n`.
    #[must_use]
    pub fn with_bias<T: ModelFloat>(mut self, bias: &[T]) -> Self {
        assert_eq!(bias.len(), self.n(), "bias holds n values");
        let mut s = Vec::new();
        self.bias = Some(T::as_f16(bias, &mut s).to_vec());
        self
    }

    /// Normalizes each input row with `RMSNorm` inside the call: the layer
    /// computes `rms_norm(x) @ W (+ bias)` from the raw `x`. The norm's weight
    /// must already be folded into `W` ([`WeightLayout::scale_inputs`]); at run
    /// time only the per-row `1/sqrt(mean(x^2) + eps)` is left, applied as the
    /// epilogue's first step, so no normalized copy of `x` is written.
    ///
    /// ```
    /// use sme_gemm::{Linear, WeightLayout};
    /// let (n, k) = (64, 32);
    /// let (mut w, gamma) = (vec![0.02f32; n * k], vec![1.5f32; k]);
    /// WeightLayout::OutIn.scale_inputs(&mut w, n, k, &gamma);
    /// let q = Linear::quantize(&w, WeightLayout::OutIn, n, k).rms_norm_input(1e-6);
    /// let (x, mut y) = (vec![2.0f32; k], vec![half::f16::ZERO; n]);
    /// q.forward(&x, &mut y, 1); // = rms_norm(x, gamma) @ W
    /// ```
    #[must_use]
    pub const fn rms_norm_input(mut self, eps: f32) -> Self {
        self.rms_eps = Some(eps);
        self
    }

    /// Outputs (`n`).
    #[must_use]
    pub const fn n(&self) -> usize {
        match &self.w {
            Weights::Q4 { q4, .. } => q4.n(),
            Weights::F16(p) => p.n(),
        }
    }

    /// Inputs (`k`).
    #[must_use]
    pub const fn k(&self) -> usize {
        match &self.w {
            Weights::Q4 { q4, .. } => q4.k(),
            Weights::F16(p) => p.k(),
        }
    }

    /// The 4-bit weights, if the layer is quantized.
    #[must_use]
    pub const fn q4(&self) -> Option<&Q4Weights> {
        match &self.w {
            Weights::Q4 { q4, .. } => Some(q4),
            Weights::F16(_) => None,
        }
    }

    /// The bias, if any, as f16.
    #[must_use]
    pub fn bias(&self) -> Option<&[f16]> {
        self.bias.as_deref()
    }

    /// Builds the f16 panel of 4-bit weights now rather than on the first call
    /// with [`Linear::PANEL_ROWS`] or more rows. A no-op for f16 weights.
    pub fn build_panel(&self) {
        if let Weights::Q4 { q4, panel } = &self.w {
            let _ = Self::panel(q4, panel);
        }
    }

    fn panel<'p>(q4: &Q4Weights, panel: &'p OnceLock<Packed<f16>>) -> &'p Packed<f16> {
        panel.get_or_init(|| prepack_f16(&q4.dequant_to_rowmajor_par(), q4.n(), q4.k()))
    }

    /// `y = x @ W (+ bias)` for `m` rows: `x` is `m x k`, `y` is `m x n` f16.
    ///
    /// # Panics
    /// Panics if `x.len() != m * k` or `y.len() != m * n`.
    pub fn forward<X: ModelFloat>(&self, x: &[X], y: &mut [f16], m: usize) {
        self.forward_ep(x, y, m, &Epilogue::new());
    }

    /// [`Linear::forward`] with a fused epilogue (activation, scaling, ...) on
    /// the f16 output, after the bias.
    ///
    /// # Panics
    /// As [`Linear::forward`], or if an epilogue operand's length does not match.
    pub fn forward_ep<X: ModelFloat>(
        &self,
        x: &[X],
        y: &mut [f16],
        m: usize,
        ep: &Epilogue<'_, f16>,
    ) {
        X16.with_borrow_mut(|s| self.run(X::as_f16(x, s), y, m, ep));
    }

    /// `y = x @ W (+ bias)` with an f32 output.
    ///
    /// # Panics
    /// As [`Linear::forward`].
    pub fn forward_f32<X: ModelFloat>(&self, x: &[X], y: &mut [f32], m: usize) {
        self.forward_f32_ep(x, y, m, &Epilogue::new());
    }

    /// [`Linear::forward_f32`] with a fused epilogue, applied before the f32
    /// conversion.
    ///
    /// # Panics
    /// As [`Linear::forward_ep`].
    pub fn forward_f32_ep<X: ModelFloat>(
        &self,
        x: &[X],
        y: &mut [f32],
        m: usize,
        ep: &Epilogue<'_, f16>,
    ) {
        assert_eq!(y.len(), crate::exec::checked_dim2(m, self.n()), "y is m*n");
        Y16.with_borrow_mut(|out| {
            out.resize(y.len(), f16::ZERO);
            self.forward_ep(x, out, m, ep);
            crate::convert::to_f32(out, y);
        });
    }

    /// `y += x @ W (+ bias)` in f32: the residual-stream update after a
    /// projection.
    ///
    /// # Panics
    /// As [`Linear::forward`].
    pub fn accumulate<X: ModelFloat>(&self, x: &[X], y: &mut [f32], m: usize) {
        assert_eq!(y.len(), crate::exec::checked_dim2(m, self.n()), "y is m*n");
        Y16.with_borrow_mut(|out| {
            out.resize(y.len(), f16::ZERO);
            self.forward(x, out, m);
            crate::convert::add_f16(out, y);
        });
    }

    fn run(&self, x: &[f16], y: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
        self.with_lead(x, m, ep, |ep| self.run_ep(x, y, m, ep));
    }

    /// `f` given `ep` behind this layer's own leading ops (the RMS row scale of
    /// `x`, then the bias), if it has any.
    fn with_lead<R>(
        &self,
        x: &[f16],
        m: usize,
        ep: &Epilogue<'_, f16>,
        f: impl FnOnce(&Epilogue<'_, f16>) -> R,
    ) -> R {
        if self.rms_eps.is_none() && self.bias.is_none() {
            return f(ep);
        }
        R16.with_borrow_mut(|r| {
            let mut lead = Epilogue::new();
            if let Some(eps) = self.rms_eps {
                assert_eq!(x.len(), crate::exec::checked_dim2(m, self.k()), "x is m*k");
                r.resize(m, f16::ZERO);
                crate::nn::rms_scales(x, self.k(), eps, r);
                lead = lead.mul_row(r);
            }
            if let Some(b) = self.bias.as_deref() {
                lead = lead.add_col(b);
            }
            f(&ep.with_leading(lead))
        })
    }

    /// This layer with output `j` taken from output `perm[j]`, exactly; `None`
    /// for f16 weights. The f16 panel is rebuilt if this layer had one.
    pub(crate) fn permuted_outputs(&self, perm: &[usize]) -> Option<Self> {
        let Weights::Q4 { q4, panel } = &self.w else {
            return None;
        };
        let out = Self {
            w: Weights::Q4 {
                q4: q4.permute_columns(perm),
                panel: OnceLock::new(),
            },
            bias: self
                .bias
                .as_ref()
                .map(|b| perm.iter().map(|&i| b[i]).collect()),
            rms_eps: self.rms_eps,
        };
        if panel.get().is_some() {
            out.build_panel();
        }
        Some(out)
    }

    /// Whether [`Linear::run_chained`] runs this layer for one row as
    /// [`Linear::forward`] would, bit for bit: 4-bit weights on SME f16 small
    /// enough that the plain call is also a single GEMV on this thread (larger
    /// ones spread over the cores, where a chain would be slower and band its
    /// passes differently), and with `gated` input (written while the call
    /// runs) no RMS norm, which would need all of it up front.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn chainable(&self, gated: bool) -> bool {
        matches!(self.w, Weights::Q4 { .. })
            && caps().sme_f16f16
            && !(gated && self.rms_eps.is_some())
            // SAFETY: a pure function of the shape.
            && unsafe { crate::ffi::gemm_sme_f16f16_q4_single(1, self.n(), self.k()) } != 0
    }

    /// `y = x @ W (+ bias)` for one row as a link of a chain ([`crate::Mlp`]):
    /// raises `done` to each output column once stored and waits on `ready`
    /// for `x`'s depths (either `None`). `false`, `done` not raised, when the
    /// kernel could not run.
    ///
    /// # Safety
    /// `x` holds `k` and `y` `n` values for the whole call. Other threads may
    /// write `x` only below what they have published in `ready` (all of it
    /// before the call when `ready` is `None`), and read `y` only below what
    /// `done` has published. The layer is [`Linear::chainable`].
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) unsafe fn run_chained(
        &self,
        x: *const f16,
        y: *mut f16,
        done: Option<&std::sync::atomic::AtomicUsize>,
        ready: Option<&std::sync::atomic::AtomicUsize>,
    ) -> bool {
        let Weights::Q4 { q4, .. } = &self.w else {
            return false;
        };
        // SAFETY: an RMS norm (the only reader of x here) implies `ready` is
        // None (chainable), so x is complete; otherwise the slice is not read.
        let xs =
            unsafe { core::slice::from_raw_parts(x, if ready.is_some() { 0 } else { self.k() }) };
        self.with_lead(xs, 1, &Epilogue::new(), |ep| {
            // SAFETY: forwarded from the caller.
            unsafe { crate::kernels::q4::q4_chained(x, q4, y, ep, done, ready) }
        })
    }

    fn run_ep(&self, x: &[f16], y: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
        let packed = |p: &Packed<f16>, y: &mut [f16]| {
            assert_eq!(x.len(), crate::exec::checked_dim2(m, p.k()), "x is m*k");
            crate::kernels::f16::f16_packed_ep_impl(x, p, y, m, ep, f16::ONE, false);
        };
        match &self.w {
            Weights::Q4 { q4, panel } if m >= Self::PANEL_ROWS && caps().sme_f16f16 => {
                packed(Self::panel(q4, panel), y);
            }
            Weights::Q4 { q4, .. } => matmul_q4_ep(x, q4, y, m, ep),
            Weights::F16(p) => packed(p, y),
        }
    }
}

/// An MLP's activation: what a [`GatedLinear`] applies to its gate, or an
/// [`Mlp`](crate::Mlp) to its hidden layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    /// `silu(x)`; gated, `silu(gate) * up`: `SwiGLU` (Llama, Mistral, Qwen).
    Silu,
    /// `gelu(x)`, tanh form; gated, `gelu(gate) * up`: `GeGLU` (Gemma), or
    /// GPT-2's plain MLP.
    Gelu,
}

impl Gate {
    pub(crate) const fn act(self) -> crate::epilogue::Activation {
        match self {
            Self::Silu => crate::epilogue::Activation::Silu,
            Self::Gelu => crate::epilogue::Activation::Gelu,
        }
    }
}

/// A gated linear unit, `act(x @ W_gate) * (x @ W_up)`: the first half of a
/// `SwiGLU` or `GeGLU` MLP, `n` outputs by `k` inputs, as one matmul.
///
/// Gate and up are stored interleaved 32 columns at a time as one [`Linear`]
/// of `2n` outputs, so a call runs one GEMV (or GEMM) over both instead of two,
/// then one NEON pass applies the activation to the gate and multiplies by up.
/// The same bytes of weights are read either way; what goes is a call, its
/// gap, and a separate elementwise pass. Biases, an `RMSNorm` folded into the
/// input ([`GatedLinear::rms_norm_input`]) and f32 or f16 inputs work as on
/// [`Linear`].
///
/// ```
/// use half::f16;
/// use sme_gemm::{Gate, GatedLinear, WeightLayout};
/// let (d, f) = (64, 160);
/// let (w_gate, w_up) = (vec![0.01f32; f * d], vec![0.02f32; f * d]);
/// let mlp_in = GatedLinear::quantize(&w_gate, &w_up, WeightLayout::OutIn, f, d, Gate::Silu);
/// let (x, mut h) = (vec![0.5f32; d], vec![f16::ZERO; f]);
/// mlp_in.forward(&x, &mut h, 1); // h = silu(x @ W_gate) * (x @ W_up)
/// ```
#[derive(Debug)]
pub struct GatedLinear {
    inner: Linear,
    gate: Gate,
}

/// `[gate | up]` as `k x 2n` row-major, interleaved 32 columns at a time (the
/// last chunk `w < 32` wide: `w` gate columns then `w` up columns).
fn interleave<T: Copy>(g: &[T], u: &[T], layout: WeightLayout, n: usize, k: usize) -> Vec<T>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    let (g, u) = (layout.to_in_out(g, n, k), layout.to_in_out(u, n, k));
    let mut out = Vec::with_capacity(2 * n * k);
    for d in 0..k {
        for c0 in (0..n).step_by(32) {
            let c1 = (c0 + 32).min(n);
            out.extend_from_slice(&g[d * n + c0..d * n + c1]);
            out.extend_from_slice(&u[d * n + c0..d * n + c1]);
        }
    }
    out
}

impl GatedLinear {
    /// Quantizes the gate and up weights (`n` outputs by `k` inputs each, in
    /// `layout`) to `Q4_0`.
    ///
    /// # Panics
    /// Panics if either holds other than `n * k` weights.
    #[must_use]
    pub fn quantize(
        w_gate: &[f32],
        w_up: &[f32],
        layout: WeightLayout,
        n: usize,
        k: usize,
        gate: Gate,
    ) -> Self {
        Self::quantize_with(w_gate, w_up, layout, n, k, gate, Q4Params::default())
    }

    /// [`GatedLinear::quantize`] with an explicit block size and code form.
    ///
    /// # Panics
    /// As [`GatedLinear::quantize`].
    #[must_use]
    pub fn quantize_with(
        w_gate: &[f32],
        w_up: &[f32],
        layout: WeightLayout,
        n: usize,
        k: usize,
        gate: Gate,
        p: Q4Params,
    ) -> Self {
        let w = interleave(w_gate, w_up, layout, n, k);
        Self {
            inner: Linear::quantize_with(&w, WeightLayout::InOut, 2 * n, k, p),
            gate,
        }
    }

    /// f16 gate and up weights (from f32 or f16), packed once.
    ///
    /// # Panics
    /// As [`GatedLinear::quantize`].
    #[must_use]
    pub fn f16<T: ModelFloat>(
        w_gate: &[T],
        w_up: &[T],
        layout: WeightLayout,
        n: usize,
        k: usize,
        gate: Gate,
    ) -> Self
    where
        [T]: ToOwned<Owned = Vec<T>>,
    {
        let w = interleave(w_gate, w_up, layout, n, k);
        Self {
            inner: Linear::f16(&w, WeightLayout::InOut, 2 * n, k),
            gate,
        }
    }

    /// Adds `gate` and `up` biases (`n` values each) before the activation.
    ///
    /// # Panics
    /// Panics if either is not `n` long.
    #[must_use]
    pub fn with_bias<T: ModelFloat>(mut self, gate: &[T], up: &[T]) -> Self
    where
        [T]: ToOwned<Owned = Vec<T>>,
    {
        let n = self.n();
        assert!(
            gate.len() == n && up.len() == n,
            "gate and up biases hold n values each"
        );
        self.inner = self
            .inner
            .with_bias(&interleave(gate, up, WeightLayout::InOut, n, 1));
        self
    }

    /// An `RMSNorm` on the input, folded as for [`Linear::rms_norm_input`]:
    /// scale both `W_gate` and `W_up` by the norm's weight first.
    #[must_use]
    pub const fn rms_norm_input(mut self, eps: f32) -> Self {
        self.inner.rms_eps = Some(eps);
        self
    }

    /// Outputs (`n`).
    #[must_use]
    pub const fn n(&self) -> usize {
        self.inner.n() / 2
    }

    /// Inputs (`k`).
    #[must_use]
    pub const fn k(&self) -> usize {
        self.inner.k()
    }

    /// The activation on the gate.
    #[must_use]
    pub const fn gate(&self) -> Gate {
        self.gate
    }

    /// The `2n`-output layer behind it, gate and up interleaved.
    pub(crate) const fn inner(&self) -> &Linear {
        &self.inner
    }

    /// `y = act(x @ W_gate) * (x @ W_up)` for `m` rows: `x` is `m x k`, `y` is
    /// `m x n` f16.
    ///
    /// # Panics
    /// Panics if `x.len() != m * k` or `y.len() != m * n`.
    pub fn forward<X: ModelFloat>(&self, x: &[X], y: &mut [f16], m: usize) {
        let n = self.n();
        assert_eq!(y.len(), crate::exec::checked_dim2(m, n), "y is m*n");
        G16.with_borrow_mut(|both| {
            both.resize(2 * m * n, f16::ZERO);
            self.inner.forward(x, both, m);
            glu(y, both, m, n, self.gate);
        });
    }

    /// [`GatedLinear::forward`] with an f32 output.
    ///
    /// # Panics
    /// As [`GatedLinear::forward`].
    pub fn forward_f32<X: ModelFloat>(&self, x: &[X], y: &mut [f32], m: usize) {
        assert_eq!(y.len(), crate::exec::checked_dim2(m, self.n()), "y is m*n");
        Y16.with_borrow_mut(|out| {
            out.resize(y.len(), f16::ZERO);
            self.forward(x, out, m);
            crate::convert::to_f32(out, y);
        });
    }
}

/// `y = act(gate) * up` over `m` rows of interleaved gate/up.
fn glu(y: &mut [f16], both: &[f16], m: usize, n: usize, gate: Gate) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // SAFETY: `both` holds m rows of 2n and `y` m rows of n (sized by the
        // caller); f16 is repr(transparent) over the u16 the pass reads.
        unsafe {
            crate::ffi::neon_glu_f16(
                y.as_mut_ptr().cast::<u16>(),
                both.as_ptr().cast::<u16>(),
                m,
                n,
                gate.act().to_c() as u32,
            );
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    for (yr, br) in y.chunks_exact_mut(n).zip(both.chunks_exact(2 * n)) {
        for c0 in (0..n).step_by(32) {
            let w = (n - c0).min(32);
            for j in 0..w {
                let (g, u) = (br[2 * c0 + j].to_f32(), br[2 * c0 + w + j].to_f32());
                let a = match gate {
                    Gate::Silu => g / (1.0 + (-g).exp()),
                    Gate::Gelu => {
                        0.5 * g * (1.0 + (0.797_884_6 * (g + 0.044_715 * g * g * g)).tanh())
                    }
                };
                yr[c0 + j] = f16::from_f32(a * u);
            }
        }
    }
}
