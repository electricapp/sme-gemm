//! [`Mlp`]: a transformer MLP block, `y += down(act(up(x)))`, whose activation
//! pass runs alongside the matmuls on either side of it.

use std::cell::RefCell;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::atomic::{AtomicUsize, Ordering};

use half::f16;

use crate::{Epilogue, Gate, GatedLinear, Linear, ModelFloat};

/// Per-thread buffers: the input as f16, `up`'s output, a gated block's
/// hidden layer, and `down`'s output.
#[derive(Default)]
struct Scratch {
    x: Vec<f16>,
    raw: Vec<f16>,
    hid: Vec<f16>,
    out: Vec<f16>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

#[derive(Debug)]
enum Up {
    Plain(Linear, Gate),
    Gated(GatedLinear),
}

/// A transformer MLP block added into a residual stream: `y += down(act(up(x)))`
/// ([`Mlp::new`]), or `y += down(act(x @ W_gate) * (x @ W_up))` ([`Mlp::gated`]).
///
/// Run as three calls -- the `up` GEMV, the NEON activation pass, the `down`
/// GEMV -- the activation sits between two SME calls with the unit idle: for
/// GPT-2's 1536-wide hidden layer that is ~0.65 us of a ~3.5 us block on M5.
/// Here, for one row with 4-bit weights and a [`HotPool`](crate::HotPool)
/// alive, a pool worker applies the activation to each pass of `up`'s outputs
/// as the kernel stores them, while the kernel goes on to the next pass, and
/// `down` starts as soon as `up` returns, waiting before each 32-deep block
/// only until the worker has reached it. The activation's time leaves the
/// critical path. Otherwise (more rows, f16 weights, no pool) the block runs as
/// the three calls. The results are the same bits either way.
///
/// ```
/// use sme_gemm::{Gate, Linear, Mlp, WeightLayout};
/// let (d, f) = (64, 256);
/// let fc = Linear::quantize(&vec![0.01f32; f * d], WeightLayout::OutIn, f, d);
/// let proj = Linear::quantize(&vec![0.02f32; d * f], WeightLayout::OutIn, d, f);
/// let mlp = Mlp::new(fc, Gate::Gelu, proj);
/// let (x, mut residual) = (vec![0.5f32; d], vec![0.0f32; d]);
/// mlp.accumulate(&x, &mut residual, 1); // residual += proj(gelu(fc(x)))
/// ```
#[derive(Debug)]
pub struct Mlp {
    up: Up,
    down: Linear,
}

impl Mlp {
    /// `y += down(act(up(x)))`: GPT-2's MLP is `Mlp::new(c_fc, Gate::Gelu, c_proj)`.
    ///
    /// # Panics
    /// Panics if `down` does not take `up`'s outputs (`down.k() != up.n()`).
    #[must_use]
    pub fn new(up: Linear, act: Gate, down: Linear) -> Self {
        assert_eq!(down.k(), up.n(), "down takes up's outputs");
        Self {
            up: Up::Plain(up, act),
            down,
        }
    }

    /// `y += down(act(x @ W_gate) * (x @ W_up))`: a `SwiGLU` or `GeGLU` MLP.
    ///
    /// # Panics
    /// Panics if `down` does not take `up`'s outputs (`down.k() != up.n()`).
    #[must_use]
    pub fn gated(up: GatedLinear, down: Linear) -> Self {
        assert_eq!(down.k(), up.n(), "down takes up's outputs");
        Self {
            up: Up::Gated(up),
            down,
        }
    }

    /// Inputs (`k`).
    #[must_use]
    pub const fn k(&self) -> usize {
        match &self.up {
            Up::Plain(l, _) => l.k(),
            Up::Gated(g) => g.k(),
        }
    }

    /// Outputs (`n`), the residual's width.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.down.n()
    }

    /// Width of the hidden layer.
    #[must_use]
    pub const fn hidden(&self) -> usize {
        self.down.k()
    }

    /// `y += down(act(up(x)))` for `m` rows: `x` is `m x k`, `y` is `m x n` f32.
    ///
    /// # Panics
    /// Panics if `x.len() != m * k` or `y.len() != m * n`.
    pub fn accumulate<X: ModelFloat>(&self, x: &[X], y: &mut [f32], m: usize) {
        let (k, n, h) = (self.k(), self.n(), self.hidden());
        assert_eq!(x.len(), crate::exec::checked_dim2(m, k), "x is m*k");
        assert_eq!(y.len(), crate::exec::checked_dim2(m, n), "y is m*n");
        SCRATCH.with_borrow_mut(|s| {
            let Scratch {
                x: xs,
                raw,
                hid,
                out,
            } = s;
            let x16 = X::as_f16(x, xs);
            let raw_w = match &self.up {
                Up::Plain(l, _) => l.n(),
                Up::Gated(g) => 2 * g.n(),
            };
            raw.resize(m * raw_w, f16::ZERO);
            hid.resize(m * h, f16::ZERO);
            out.resize(m * n, f16::ZERO);
            if !(m == 1 && self.chained(x16, raw, hid, out)) {
                match &self.up {
                    Up::Plain(l, g) => {
                        l.forward_ep(x16, raw, m, &act_ep(*g));
                        self.down.forward(&raw[..], out, m);
                    }
                    Up::Gated(g) => {
                        g.forward(x16, hid, m);
                        self.down.forward(&hid[..], out, m);
                    }
                }
            }
            crate::convert::add_f16(out, y);
        });
    }

    /// One row with the activation pass on a pool worker, overlapping both
    /// GEMVs; `false` (nothing written to `out`) when that path is unavailable.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn chained(&self, x16: &[f16], raw: &mut [f16], hid: &mut [f16], out: &mut [f16]) -> bool {
        let (up, gate, gated) = match &self.up {
            Up::Plain(l, g) => (l, *g, false),
            Up::Gated(g) => (g.inner(), g.gate(), true),
        };
        if !up.chainable(false) || !self.down.chainable(true) || !crate::pool::live() {
            return false;
        }
        let (n_raw, n_hid) = (up.n(), self.hidden());
        let act = gate.act().to_c() as u32;
        let (done, ready) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let raw_p = crate::pool::Shared(raw.as_mut_ptr());
        let hid_p = crate::pool::Shared(hid.as_mut_ptr());
        // The activation, pass by pass as `up` publishes its outputs, written
        // to `hid` rather than back over `raw`: a line the worker has just read
        // stays in its cache even when rewritten with STNP, and `down`'s reads
        // of such lines wait on that core (0.3-0.5 us over 1536 values on M5).
        // A gated output column c needs gate and up from raw's 64-column chunk
        // c / 32.
        let pass = |_: usize| {
            // Whatever happens, publish everything so `down` never waits forever.
            let _release = ReleaseAll(&ready, n_hid);
            let mut h0 = 0;
            while h0 < n_hid {
                let d = done.load(Ordering::Acquire);
                let h1 = if !gated {
                    d
                } else if d >= n_raw {
                    n_hid
                } else {
                    d / 64 * 32
                };
                if h1 <= h0 {
                    std::hint::spin_loop();
                    continue;
                }
                // SAFETY: up has stored raw[..d] and will not write it again;
                // hid[h0..h1) is written by this item alone, and `down` reads
                // it only once `ready` covers it.
                unsafe {
                    if gated {
                        crate::ffi::neon_glu_f16(
                            hid_p.ptr().add(h0).cast::<u16>(),
                            raw_p.ptr().add(2 * h0).cast::<u16>(),
                            1,
                            h1 - h0,
                            act,
                        );
                    } else {
                        crate::ffi::neon_act_f16(
                            hid_p.ptr().add(h0).cast::<u16>(),
                            raw_p.ptr().add(h0).cast::<u16>(),
                            h1 - h0,
                            act,
                        );
                    }
                }
                ready.store(h1, Ordering::Release);
                h0 = h1;
            }
        };
        let ok = crate::pool::alongside(1, &pass, |posted| {
            // SAFETY: x16 is complete; raw is written by up and read by the
            // pass only below what up has published in `done`.
            let up_ok = unsafe { up.run_chained(x16.as_ptr(), raw_p.ptr(), Some(&done), None) };
            if !up_ok {
                done.store(n_raw, Ordering::Release);
            }
            // The pass, here, if no worker has picked it up (one may be asleep).
            // Progress on `ready` means a worker has it, and `down` needs that
            // line next anyway.
            if ready.load(Ordering::Relaxed) == 0 {
                posted.drain();
            }
            // SAFETY: the hidden layer is written below `ready` only (all of it
            // when the pass ran here); out is ours alone.
            up_ok
                && unsafe {
                    self.down
                        .run_chained(hid_p.ptr(), out.as_mut_ptr(), None, Some(&ready))
                }
        });
        ok == Some(true)
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    const fn chained(&self, _: &[f16], _: &mut [f16], _: &mut [f16], _: &mut [f16]) -> bool {
        false
    }
}

/// Stores `n` to the counter when dropped.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct ReleaseAll<'a>(&'a AtomicUsize, usize);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for ReleaseAll<'_> {
    fn drop(&mut self) {
        self.0.store(self.1, Ordering::Release);
    }
}

/// The epilogue applying `g` to a plain block's hidden layer.
fn act_ep(g: Gate) -> Epilogue<'static, f16> {
    match g {
        Gate::Gelu => Epilogue::new().gelu(),
        Gate::Silu => Epilogue::new().silu(),
    }
}
