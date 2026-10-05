//! [`SelfAttention`]: a transformer attention block added into a residual
//! stream, whose attention runs on pool workers alongside its two GEMVs.

use std::cell::RefCell;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use half::f16;

use crate::{KvCache, Linear, ModelFloat};

/// Per-thread buffers: the input as f16, the projected q/k/v, a query
/// gathered for the cache, the attention output, and the output projection's.
#[derive(Default)]
struct Scratch {
    x: Vec<f16>,
    qkv: Vec<f16>,
    q: Vec<f16>,
    kv: Vec<f16>,
    att: Vec<f32>,
    att16: Vec<f16>,
    out: Vec<f16>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
    /// A pool item's query, attention output and score row, f32.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    static ITEM: RefCell<(Vec<f32>, Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new(), Vec::new())) };
}

/// A transformer self-attention block added into a residual stream: the fused
/// `qkv` projection, the new position's key and value appended to a
/// [`KvCache`], attention over the cache, and the `out` projection,
/// `y += out(attention(q, K, V))`.
///
/// `qkv` has `(heads + 2 * kv_heads) * head_dim` outputs, `[q | k | v]` in
/// head order (GPT-2's `c_attn`; a Llama-style `q_proj`/`k_proj`/`v_proj`
/// stacked). Positional rotations (`RoPE`) are not applied: this fits models
/// with learned or added position embeddings.
///
/// Run as separate steps, attention sits between two SME calls with the unit
/// idle. Here, for one row with 4-bit weights and a [`HotPool`](crate::HotPool)
/// alive, `qkv`'s outputs are kept grouped by KV head (each group's queries,
/// key and value adjacent, a column permutation made once at construction), so
/// the kernel's first pass already holds whole groups: pool workers append each
/// group's key and value and attend for it as soon as its columns are stored,
/// while the kernel goes on, and the `out` GEMV starts right after, waiting
/// before each 32-deep block only for the groups it reaches. Otherwise (more
/// rows, f16 weights, no pool) the block runs as the separate steps. The
/// results are the same bits either way.
///
/// ```
/// use sme_gemm::{KvCache, Linear, SelfAttention, WeightLayout};
/// let (d, heads, hd) = (128, 2, 64);
/// let qkv = Linear::quantize(&vec![0.01f32; 3 * d * d], WeightLayout::OutIn, 3 * d, d);
/// let out = Linear::quantize(&vec![0.02f32; d * d], WeightLayout::OutIn, d, d);
/// let attn = SelfAttention::new(qkv, out, heads, heads, hd);
/// let mut cache = KvCache::new(heads, heads, hd, 64);
/// let (x, mut residual) = (vec![0.5f32; d], vec![0.0f32; d]);
/// attn.accumulate(&x, &mut cache, &mut residual, 1); // one position
/// assert_eq!(cache.len(), 1);
/// ```
#[derive(Debug)]
pub struct SelfAttention {
    /// `qkv`; when 4-bit, with outputs grouped per KV head (`grouped`): group
    /// j's queries, key and value at `j * (heads/kv_heads + 2) * head_dim`.
    /// Both paths use it, so its passes, and bits, are the same either way.
    qkv: Linear,
    grouped: bool,
    out: Linear,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    /// The block from its two projections.
    ///
    /// # Panics
    /// Panics if `qkv` does not have `(heads + 2 * kv_heads) * head_dim`
    /// outputs, `out` does not take `heads * head_dim` inputs, or the head
    /// counts are inconsistent (as for [`KvCache::new`]).
    #[must_use]
    pub fn new(qkv: Linear, out: Linear, heads: usize, kv_heads: usize, head_dim: usize) -> Self {
        assert!(
            kv_heads > 0 && heads > 0 && heads.is_multiple_of(kv_heads),
            "heads must be a positive multiple of kv_heads"
        );
        assert_eq!(
            qkv.n(),
            (heads + 2 * kv_heads) * head_dim,
            "qkv has (heads + 2 kv_heads) * head_dim outputs"
        );
        assert_eq!(
            out.k(),
            heads * head_dim,
            "out takes heads * head_dim inputs"
        );
        let g = heads / kv_heads;
        let (q_len, kv_len) = (heads * head_dim, kv_heads * head_dim);
        // Group j's columns: its g query heads, its key head, its value head.
        let perm: Vec<usize> = (0..kv_heads)
            .flat_map(|j| {
                let q = (j * g * head_dim)..((j + 1) * g * head_dim);
                let k = (q_len + j * head_dim)..(q_len + (j + 1) * head_dim);
                let v = (q_len + kv_len + j * head_dim)..(q_len + kv_len + (j + 1) * head_dim);
                q.chain(k).chain(v)
            })
            .collect();
        let (qkv, grouped) = match qkv.permuted_outputs(&perm) {
            Some(p) => (p, true),
            None => (qkv, false),
        };
        Self {
            qkv,
            grouped,
            out,
            heads,
            kv_heads,
            head_dim,
        }
    }

    /// Inputs and outputs (the residual's width).
    #[must_use]
    pub const fn width(&self) -> usize {
        self.qkv.k()
    }

    /// An empty cache for this block's heads, `capacity` positions long.
    #[must_use]
    pub fn cache(&self, capacity: usize) -> KvCache {
        KvCache::new(self.heads, self.kv_heads, self.head_dim, capacity)
    }

    /// `y += out(attention(qkv(x)))` for `m` rows at the cache's next
    /// positions, which this appends.
    ///
    /// # Panics
    /// Panics if `x` or `y` is not `m` rows of [`SelfAttention::width`], the
    /// cache has other head counts, or the rows do not fit in it.
    pub fn accumulate<X: ModelFloat>(&self, x: &[X], cache: &mut KvCache, y: &mut [f32], m: usize) {
        let d = self.width();
        assert_eq!(x.len(), crate::exec::checked_dim2(m, d), "x is m * width");
        assert_eq!(
            y.len(),
            crate::exec::checked_dim2(m, self.out.n()),
            "y is m * out.n()"
        );
        assert!(
            cache.heads() == self.heads
                && cache.kv_heads() == self.kv_heads
                && cache.head_dim() == self.head_dim,
            "the cache's heads match the block's"
        );
        assert!(
            cache.len() + m <= cache.capacity(),
            "{m} rows do not fit in the cache"
        );
        SCRATCH.with_borrow_mut(|s| {
            let Scratch {
                x: xs,
                qkv,
                q,
                kv,
                att,
                att16,
                out,
            } = s;
            let x16 = X::as_f16(x, xs);
            let (q_len, kv_len) = (self.heads * self.head_dim, self.kv_heads * self.head_dim);
            qkv.resize(m * (q_len + 2 * kv_len), f16::ZERO);
            att16.resize(m * q_len, f16::ZERO);
            out.resize(m * self.out.n(), f16::ZERO);
            if !(m == 1 && self.chained(x16, cache, qkv, att16, out)) {
                self.qkv.forward(x16, qkv, m);
                self.attend_rows(qkv, cache, q, kv, att, m);
                self.out.forward(&att[..], out, m);
            }
            crate::convert::add_f16(out, y);
        });
    }

    /// The attention step for `m` rows of `[q | k | v]`: append k and v,
    /// attend causally, into `att` (f32).
    fn attend_rows(
        &self,
        qkv: &[f16],
        cache: &mut KvCache,
        q: &mut Vec<f16>,
        kv: &mut Vec<f16>,
        att: &mut Vec<f32>,
        m: usize,
    ) {
        let (hd, g) = (self.head_dim, self.heads / self.kv_heads);
        let (q_len, kv_len, gw) = (self.heads * hd, self.kv_heads * hd, (g + 2) * hd);
        let w = q_len + 2 * kv_len;
        q.resize(m * q_len, f16::ZERO);
        kv.resize(2 * m * kv_len, f16::ZERO);
        let (ks, vs) = kv.split_at_mut(m * kv_len);
        for r in 0..m {
            let row = &qkv[r * w..(r + 1) * w];
            let (qr, kr, vr) = (
                &mut q[r * q_len..(r + 1) * q_len],
                &mut ks[r * kv_len..(r + 1) * kv_len],
                &mut vs[r * kv_len..(r + 1) * kv_len],
            );
            if self.grouped {
                for (j, col) in row.chunks_exact(gw).enumerate() {
                    qr[j * g * hd..(j + 1) * g * hd].copy_from_slice(&col[..g * hd]);
                    kr[j * hd..(j + 1) * hd].copy_from_slice(&col[g * hd..(g + 1) * hd]);
                    vr[j * hd..(j + 1) * hd].copy_from_slice(&col[(g + 1) * hd..]);
                }
            } else {
                qr.copy_from_slice(&row[..q_len]);
                kr.copy_from_slice(&row[q_len..q_len + kv_len]);
                vr.copy_from_slice(&row[q_len + kv_len..]);
            }
        }
        cache.extend(&*ks, &*vs);
        att.resize(m * q_len, 0.0);
        if m == 1 {
            cache.attend(&q[..], att);
        } else {
            cache.attend_causal(&q[..], m, att);
        }
    }

    /// One row with each KV-head group's append and attention as a pool item
    /// overlapping both GEMVs; `false` (cache and `out` untouched) when that
    /// path is unavailable.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn chained(
        &self,
        x16: &[f16],
        cache: &mut KvCache,
        qkv: &mut [f16],
        att16: &mut [f16],
        out: &mut [f16],
    ) -> bool {
        // Up to 64 groups: their completion is one bit mask.
        let grouped = &self.qkv;
        if !self.grouped
            || self.kv_heads > 64
            || !grouped.chainable(false)
            || !self.out.chainable(true)
            || !crate::pool::live()
        {
            return false;
        }
        let (hd, kvh) = (self.head_dim, self.kv_heads);
        let g = self.heads / kvh;
        let (gw, q_len, ld) = ((g + 2) * hd, self.heads * hd, kvh * hd);
        let (pos, scale) = (cache.len(), cache.scale());
        let (kc, vc) = cache.storage();
        let (kc, vc) = (crate::pool::Shared(kc), crate::pool::Shared(vc));
        let (done, ready, complete) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicU64::new(0));
        let qkv_p = crate::pool::Shared(qkv.as_mut_ptr());
        let att_p = crate::pool::Shared(att16.as_mut_ptr());
        // Group j: once qkv has stored its columns, append its key and value
        // to the cache's new row, attend its g query heads over pos + 1 rows,
        // and store the output f16 for `out`. `ready` advances over the
        // leading run of finished groups.
        let item = |j: usize| {
            let _release = ReleaseOnPanic(&ready, q_len);
            while done.load(Ordering::Acquire) < (j + 1) * gw {
                std::hint::spin_loop();
            }
            ITEM.with_borrow_mut(|(q32, o32, scores)| {
                q32.resize(g * hd, 0.0);
                o32.resize(g * hd, 0.0);
                scores.resize(pos + 1, 0.0);
                // SAFETY: qkv has stored group j's columns (done covers them)
                // and does not write them again; this item alone writes head j
                // of the cache's row `pos` (in bounds: accumulate checked the
                // capacity) and att16[j*g*hd..(j+1)*g*hd); rows before `pos`
                // are complete and read-only; `out` reads att16 only below
                // `ready`.
                unsafe {
                    let col = qkv_p.ptr().add(j * gw);
                    let qs = core::slice::from_raw_parts(col, g * hd);
                    crate::convert::to_f32(qs, q32);
                    let at = pos * ld + j * hd;
                    core::ptr::copy_nonoverlapping(col.add(g * hd), kc.ptr().add(at), hd);
                    core::ptr::copy_nonoverlapping(col.add((g + 1) * hd), vc.ptr().add(at), hd);
                    crate::ffi::attn_kv_f16(
                        o32.as_mut_ptr(),
                        q32.as_ptr(),
                        kc.ptr().add(j * hd).cast::<u16>(),
                        vc.ptr().add(j * hd).cast::<u16>(),
                        pos + 1,
                        ld,
                        g,
                        1,
                        hd,
                        scale,
                        scores.as_mut_ptr(),
                    );
                    let dst = core::slice::from_raw_parts_mut(att_p.ptr().add(j * g * hd), g * hd);
                    crate::convert::to_f16(o32, dst);
                }
            });
            let mask = complete.fetch_or(1 << j, Ordering::AcqRel) | (1 << j);
            ready.fetch_max(mask.trailing_ones() as usize * g * hd, Ordering::Release);
        };
        let ok = crate::pool::alongside(kvh, &item, |posted| {
            // SAFETY: x16 is complete; qkv is written by the kernel and read
            // by the items only below what it has published in `done`.
            let up_ok =
                unsafe { grouped.run_chained(x16.as_ptr(), qkv_p.ptr(), Some(&done), None) };
            if !up_ok {
                done.store(usize::MAX, Ordering::Release);
            }
            // Items here if no worker has picked them up (one may be asleep).
            if ready.load(Ordering::Relaxed) == 0 {
                posted.drain();
            }
            // SAFETY: att16 is written below `ready` only; out is ours alone.
            up_ok
                && unsafe {
                    self.out
                        .run_chained(att_p.ptr(), out.as_mut_ptr(), None, Some(&ready))
                }
        });
        match ok {
            Some(true) => {
                cache.grow();
                true
            }
            // The items may have written the cache row past its length; it
            // is not counted, so the sequential rerun overwrites it.
            _ => false,
        }
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    const fn chained(
        &self,
        _: &[f16],
        _: &mut KvCache,
        _: &mut [f16],
        _: &mut [f16],
        _: &mut [f16],
    ) -> bool {
        false
    }
}

/// Stores `n` to the counter if dropped while unwinding, so the `out` GEMV,
/// waiting on it, can finish before the panic is raised.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct ReleaseOnPanic<'a>(&'a AtomicUsize, usize);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for ReleaseOnPanic<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(self.1, Ordering::Release);
        }
    }
}
