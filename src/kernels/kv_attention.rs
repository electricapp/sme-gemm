//! Attention against an f16 KV cache: one query row of heads against every
//! cached key, or a block of query rows with causal masking.

use std::cell::RefCell;

use half::f16;

use crate::ModelFloat;

/// `softmax(scale * q @ K^T) @ V` for one query row, per head, against `len`
/// cached keys.
///
/// `q` is `[heads][head_dim]` f32. `k` and `v` are `len` rows of f16 with
/// `kv_heads * head_dim` elements each (KV head `j` at `j * head_dim`), so
/// appending a key/value row to the cache is one row copy. Query head `h` attends with
/// KV head `h / (heads / kv_heads)` (grouped-query attention; `kv_heads ==
/// heads` is plain multi-head). `out` is `[heads][head_dim]` f32.
///
/// Scores accumulate in f32 from f16 products (`q` is rounded to f16 once),
/// probabilities are f32, and values accumulate in f32. On Apple silicon this
/// is NEON on the calling thread: at one row per head it is too small to
/// spread across cores and has nothing for SME to do.
///
/// # Panics
/// Panics if `len == 0`, `head_dim` is not a multiple of 8 or exceeds 256,
/// `heads` is not a positive multiple of `kv_heads`, or a slice is shorter than
/// its shape.
#[allow(clippy::too_many_arguments)]
pub fn attention_kv_f16(
    q: &[f32],
    k: &[f16],
    v: &[f16],
    len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    assert!(len > 0, "attention_kv_f16 needs at least one cached key");
    assert!(
        head_dim.is_multiple_of(8) && (8..=256).contains(&head_dim),
        "head_dim must be a multiple of 8 in 8..=256"
    );
    assert!(
        kv_heads > 0 && heads > 0 && heads.is_multiple_of(kv_heads),
        "heads must be a positive multiple of kv_heads"
    );
    let ld = crate::exec::checked_dim2(kv_heads, head_dim);
    let rows = crate::exec::checked_dim2(len, ld);
    let qd = crate::exec::checked_dim2(heads, head_dim);
    assert!(k.len() >= rows, "k holds len * kv_heads * head_dim");
    assert!(v.len() >= rows, "v holds len * kv_heads * head_dim");
    assert_eq!(q.len(), qd, "q is heads * head_dim");
    assert_eq!(out.len(), qd, "out is heads * head_dim");

    // With a HotPool alive and enough work, one item per KV-head group.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if kv_heads >= 2
        && len * heads >= PAR_MIN_HEAD_KEYS
        && groups_in_parallel(out, q, k, v, len, ld, heads, kv_heads, head_dim, scale)
    {
        return;
    }
    SCORES.with_borrow_mut(|scores| {
        if scores.len() < len {
            scores.resize(len, 0.0);
        }
        one_row(
            out, q, k, v, len, ld, heads, kv_heads, head_dim, scale, scores,
        );
    });
}

thread_local! {
    /// Each thread's score row.
    static SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Head-keys (`len * heads`) from which a call splits across a
/// [`HotPool`](crate::HotPool): at ~3.5 ns each, below this the work is about
/// what the hand-off costs.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const PAR_MIN_HEAD_KEYS: usize = 384;

/// [`attention_kv_f16`] with each KV-head group (its `heads / kv_heads` query
/// heads) as one item on the [`HotPool`](crate::HotPool); `false` (nothing
/// done) without a pool. Shapes were checked by the caller.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn groups_in_parallel(
    out: &mut [f32],
    q: &[f32],
    k: &[f16],
    v: &[f16],
    len: usize,
    ld: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> bool {
    let g = heads / kv_heads;
    let out = crate::pool::Shared(out.as_mut_ptr());
    crate::pool::parallel(kv_heads, &|kh| {
        SCORES.with_borrow_mut(|scores| {
            if scores.len() < len {
                scores.resize(len, 0.0);
            }
            let (qo, ko) = (kh * g * head_dim, kh * head_dim);
            // SAFETY: group kh reads q[qo..qo + g*head_dim] and the kh-th
            // head_dim of each k/v row (all in bounds per the caller's checks)
            // and writes only out[qo..qo + g*head_dim], which no other group
            // touches; out outlives the pool call, which waits for every item.
            unsafe {
                crate::ffi::attn_kv_f16(
                    out.ptr().add(qo),
                    q.as_ptr().add(qo),
                    k.as_ptr().cast::<u16>().add(ko),
                    v.as_ptr().cast::<u16>().add(ko),
                    len,
                    ld,
                    g,
                    1,
                    head_dim,
                    scale,
                    scores.as_mut_ptr(),
                );
            }
        });
    })
}

/// Causal attention for `rows` query rows at positions `start..start + rows`.
///
/// The f16 KV cache has the layout of [`attention_kv_f16`]. Query row `i`
/// attends to keys `0..=start + i`, so the cache must already hold the block's
/// own keys and values (`start + rows` rows).
///
/// `q` and `out` are `[rows][heads][head_dim]` f32. Rows are independent, so a
/// large block spreads across cores; each row matches [`attention_kv_f16`]
/// over its prefix exactly.
///
/// # Panics
/// As [`attention_kv_f16`], with the caches holding `start + rows` rows.
#[allow(clippy::too_many_arguments)]
pub fn attention_kv_causal_f16(
    q: &[f32],
    k: &[f16],
    v: &[f16],
    start: usize,
    rows: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    assert!(
        rows > 0,
        "attention_kv_causal_f16 needs at least one query row"
    );
    assert!(
        head_dim.is_multiple_of(8) && (8..=256).contains(&head_dim),
        "head_dim must be a multiple of 8 in 8..=256"
    );
    assert!(
        kv_heads > 0 && heads > 0 && heads.is_multiple_of(kv_heads),
        "heads must be a positive multiple of kv_heads"
    );
    let ld = crate::exec::checked_dim2(kv_heads, head_dim);
    let keys = start
        .checked_add(rows)
        .expect("start + rows overflows usize");
    let cache = crate::exec::checked_dim2(keys, ld);
    let qd = crate::exec::checked_dims(rows, heads, head_dim);
    assert!(
        k.len() >= cache,
        "k holds (start + rows) * kv_heads * head_dim"
    );
    assert!(
        v.len() >= cache,
        "v holds (start + rows) * kv_heads * head_dim"
    );
    assert_eq!(q.len(), qd, "q is rows * heads * head_dim");
    assert_eq!(out.len(), qd, "out is rows * heads * head_dim");

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // SAFETY: shapes checked above: k/v cover start+rows rows of ld, q and
        // out cover rows*heads*head_dim; f16 is repr(transparent) over u16.
        let rc = unsafe {
            crate::ffi::attn_kv_causal_f16(
                out.as_mut_ptr(),
                q.as_ptr(),
                k.as_ptr().cast::<u16>(),
                v.as_ptr().cast::<u16>(),
                start,
                rows,
                ld,
                heads,
                kv_heads,
                head_dim,
                scale,
            )
        };
        if rc == 0 {
            return;
        }
    }
    // Portable path, and the fallback when a scratch allocation failed.
    let qr = heads * head_dim;
    let mut scores = vec![0.0; keys];
    for i in 0..rows {
        one_row(
            &mut out[i * qr..(i + 1) * qr],
            &q[i * qr..(i + 1) * qr],
            k,
            v,
            start + i + 1,
            ld,
            heads,
            kv_heads,
            head_dim,
            scale,
            &mut scores,
        );
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn one_row(
    out: &mut [f32],
    q: &[f32],
    k: &[f16],
    v: &[f16],
    len: usize,
    ld: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    // SAFETY: shapes were checked by the caller: k/v cover len rows of ld, q and
    // out cover heads*head_dim, scores covers len; f16 is repr(transparent)
    // over the u16 the kernel reads as __fp16.
    unsafe {
        crate::ffi::attn_kv_f16(
            out.as_mut_ptr(),
            q.as_ptr(),
            k.as_ptr().cast::<u16>(),
            v.as_ptr().cast::<u16>(),
            len,
            ld,
            heads,
            kv_heads,
            head_dim,
            scale,
            scores.as_mut_ptr(),
        );
    }
}

/// The portable equivalent, with the same roundings (q to f16, f32 sums).
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[allow(clippy::too_many_arguments)]
fn one_row(
    out: &mut [f32],
    q: &[f32],
    k: &[f16],
    v: &[f16],
    len: usize,
    ld: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    scores: &mut [f32],
) {
    let group = heads / kv_heads;
    for h in 0..heads {
        let off = (h / group) * head_dim;
        let qh: Vec<f32> = q[h * head_dim..(h + 1) * head_dim]
            .iter()
            .map(|x| f16::from_f32(x * scale).to_f32())
            .collect();
        let mut mx = f32::NEG_INFINITY;
        for (t, s) in scores[..len].iter_mut().enumerate() {
            let row = &k[t * ld + off..t * ld + off + head_dim];
            *s = qh.iter().zip(row).map(|(a, b)| a * b.to_f32()).sum();
            mx = mx.max(*s);
        }
        let mut sum = 0.0;
        for s in &mut scores[..len] {
            *s = (*s - mx).exp();
            sum += *s;
        }
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        oh.fill(0.0);
        for (t, p) in scores[..len].iter().enumerate() {
            let row = &v[t * ld + off..t * ld + off + head_dim];
            for (o, x) in oh.iter_mut().zip(row) {
                *o += p * x.to_f32();
            }
        }
        for o in oh {
            *o /= sum;
        }
    }
}

thread_local! {
    /// A query converted to f32, and key/value rows converted to f16.
    static Q32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static K16: RefCell<Vec<f16>> = const { RefCell::new(Vec::new()) };
}

/// An f16 key/value cache for one attention layer, with the attention against
/// it: [`attention_kv_f16`] and [`attention_kv_causal_f16`] without the shape
/// arguments.
///
/// Rows are `kv_heads * head_dim` f16 values, one per position, appended in
/// order. Queries have `heads` heads (a multiple of `kv_heads`: grouped-query
/// attention; equal is plain multi-head) and come as f32 or f16. The score
/// scale defaults to `1/sqrt(head_dim)`.
///
/// ```
/// use half::f16;
/// use sme_gemm::KvCache;
/// let (heads, head_dim) = (4, 64);
/// let mut cache = KvCache::new(heads, heads, head_dim, 256);
/// let row = vec![f16::from_f32(0.1); heads * head_dim];
/// cache.push(&row, &row); // this position's key and value
/// let q = vec![0.2f32; heads * head_dim];
/// let mut out = vec![0.0f32; heads * head_dim];
/// cache.attend(&q, &mut out); // attends to every cached position
/// ```
#[derive(Clone, Debug)]
pub struct KvCache {
    k: Vec<f16>,
    v: Vec<f16>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    capacity: usize,
    len: usize,
    scale: f32,
}

impl KvCache {
    /// An empty cache for `capacity` positions.
    ///
    /// # Panics
    /// Panics if `head_dim` is not a multiple of 8 in `8..=256`, `heads` is not
    /// a positive multiple of `kv_heads`, or the storage size overflows.
    #[must_use]
    pub fn new(heads: usize, kv_heads: usize, head_dim: usize, capacity: usize) -> Self {
        assert!(
            head_dim.is_multiple_of(8) && (8..=256).contains(&head_dim),
            "head_dim must be a multiple of 8 in 8..=256"
        );
        assert!(
            kv_heads > 0 && heads > 0 && heads.is_multiple_of(kv_heads),
            "heads must be a positive multiple of kv_heads"
        );
        let size = crate::exec::checked_dims(capacity, kv_heads, head_dim);
        Self {
            k: vec![f16::ZERO; size],
            v: vec![f16::ZERO; size],
            heads,
            kv_heads,
            head_dim,
            capacity,
            len: 0,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Uses `scale` for the scores instead of `1/sqrt(head_dim)`.
    #[must_use]
    pub const fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Cached positions.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether no position is cached.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions the cache can hold.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Values per key (or value) row: `kv_heads * head_dim`.
    #[must_use]
    pub const fn row_len(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Values per query row: `heads * head_dim`.
    #[must_use]
    pub const fn query_len(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Query heads.
    #[must_use]
    pub const fn heads(&self) -> usize {
        self.heads
    }

    /// Key/value heads.
    #[must_use]
    pub const fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    /// Values per head.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// The score scale.
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    /// The key and value storage, `capacity` rows each, for writers that fill
    /// a row's heads from several threads ([`crate::SelfAttention`]); the row
    /// counts once [`KvCache::grow`] says so.
    pub(crate) const fn storage(&mut self) -> (*mut f16, *mut f16) {
        (self.k.as_mut_ptr(), self.v.as_mut_ptr())
    }

    /// Counts one more row, written through [`KvCache::storage`].
    ///
    /// # Panics
    /// Panics if the cache is full.
    pub(crate) fn grow(&mut self) {
        assert!(
            self.len < self.capacity,
            "KvCache full at {}",
            self.capacity
        );
        self.len += 1;
    }

    /// Forgets every position.
    pub const fn clear(&mut self) {
        self.len = 0;
    }

    /// Keeps the first `len` positions (no-op if there are fewer).
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    /// Appends one position's key and value rows (f32 or f16).
    ///
    /// # Panics
    /// Panics if the cache is full or a row is not [`KvCache::row_len`] long.
    pub fn push<T: ModelFloat>(&mut self, k: &[T], v: &[T]) {
        self.extend(k, v);
    }

    /// Appends whole rows: `k` and `v` hold the same number of
    /// [`KvCache::row_len`]-long rows.
    ///
    /// # Panics
    /// Panics if the rows do not fit, or `k`/`v` are not the same whole number
    /// of rows.
    pub fn extend<T: ModelFloat>(&mut self, k: &[T], v: &[T]) {
        let row = self.row_len();
        assert!(
            k.len() == v.len() && k.len().is_multiple_of(row) && !k.is_empty(),
            "k and v hold the same whole number of kv_heads * head_dim rows"
        );
        let rows = k.len() / row;
        assert!(
            self.len + rows <= self.capacity,
            "KvCache full: {} + {rows} rows over capacity {}",
            self.len,
            self.capacity
        );
        let at = self.len * row..(self.len + rows) * row;
        K16.with_borrow_mut(|f| {
            self.k[at.clone()].copy_from_slice(T::as_f16(k, f));
            self.v[at].copy_from_slice(T::as_f16(v, f));
        });
        self.len += rows;
    }

    /// The cached keys, `len()` rows of [`KvCache::row_len`].
    #[must_use]
    pub fn keys(&self) -> &[f16] {
        &self.k[..self.len * self.row_len()]
    }

    /// The cached values, `len()` rows of [`KvCache::row_len`].
    #[must_use]
    pub fn values(&self) -> &[f16] {
        &self.v[..self.len * self.row_len()]
    }

    /// One query row (`heads * head_dim`) against every cached position; `out`
    /// is `heads * head_dim` f32.
    ///
    /// # Panics
    /// Panics if the cache is empty or a slice has the wrong length.
    pub fn attend<Q: ModelFloat>(&self, q: &[Q], out: &mut [f32]) {
        Q32.with_borrow_mut(|s| {
            attention_kv_f16(
                Q::as_f32(q, s),
                self.keys(),
                self.values(),
                self.len,
                self.heads,
                self.kv_heads,
                self.head_dim,
                self.scale,
                out,
            );
        });
    }

    /// `rows` query rows that are the last `rows` cached positions, each
    /// attending causally (to itself and everything before it); `q` and `out`
    /// are `rows * heads * head_dim`. Push the block's keys and values first.
    ///
    /// # Panics
    /// Panics if fewer than `rows` positions are cached or a slice has the
    /// wrong length.
    pub fn attend_causal<Q: ModelFloat>(&self, q: &[Q], rows: usize, out: &mut [f32]) {
        assert!(
            rows <= self.len,
            "attend_causal: {rows} rows but {} cached",
            self.len
        );
        Q32.with_borrow_mut(|s| {
            attention_kv_causal_f16(
                Q::as_f32(q, s),
                self.keys(),
                self.values(),
                self.len - rows,
                rows,
                self.heads,
                self.kv_heads,
                self.head_dim,
                self.scale,
                out,
            );
        });
    }
}
