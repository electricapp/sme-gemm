//! Fused scaled-dot-product attention: `O = softmax(scale * Q @ K^T) @ V`,
//! computed a key-block at a time so the `m x n` score matrix is never
//! materialized.

use std::cell::Cell;

use crate::exec::{checked_dim2, checked_dims};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::ffi::{attn_flash_block_f32, attn_flash_finish_f32};
use crate::kernels::f32::gemm_f32;

/// Element cap on a reusable buffer, matching [`FlashParams::auto`]'s block
/// budget.
const SCRATCH_CAP: usize = 4 << 20;

thread_local! {
    static F32_SCRATCH: Cell<Vec<f32>> = const { Cell::new(Vec::new()) };
}

/// Borrow a thread-local buffer of at least `need` elements, or allocate one.
///
/// The score block and accumulators are fully written before they are read on
/// every key block, so the stale contents of a reused buffer are never
/// observable and it never needs clearing. Requests above [`SCRATCH_CAP`] --
/// only reachable through the explicit `_with` entry points -- allocate locally,
/// so a caller cannot pin an arbitrarily large buffer to a thread for the
/// process lifetime. Taking the buffer out of the cell (rather than borrowing
/// it) means a re-entrant call just allocates instead of panicking.
pub(crate) fn take<T: Copy + Default + 'static>(
    slot: &'static std::thread::LocalKey<Cell<Vec<T>>>,
    need: usize,
) -> Vec<T> {
    if need > SCRATCH_CAP {
        return vec![T::default(); need];
    }
    let mut v = slot.with(Cell::take);
    if v.len() < need {
        v.resize(need, T::default());
    }
    // Debug builds poison the buffer (all-ones is a NaN for f32/f16/bf16), so a
    // read-before-write shows up as NaN in the result instead of as whatever the
    // previous call happened to leave. Worth having because the natural bug here
    // is invisible otherwise: on the first key block `corr` is 0, so `acc =
    // corr*acc + t` silently discards finite garbage, and only a non-finite value
    // survives to be noticed.
    #[cfg(debug_assertions)]
    // SAFETY: `v` owns v.len() * size_of::<T>() initialized bytes; writing an
    // arbitrary bit pattern over them is fine for the Copy types used here.
    unsafe {
        let n = v.len() * size_of::<T>();
        std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), n).fill(0xFF);
    }
    v
}

/// Hand a buffer back for the next call on this thread.
pub(crate) fn give<T: 'static>(slot: &'static std::thread::LocalKey<Cell<Vec<T>>>, v: Vec<T>) {
    if v.len() <= SCRATCH_CAP {
        slot.with(|c| c.set(v));
    }
}

/// Tile sizes for [`flash_attention_f32`].
///
/// The score block is `block_m x block_n` f32 and both it and the `block_m x
/// dv` output block stay resident across a key block, so the product wants to
/// fit L2. Smaller `block_m` re-reads K and V once per query block; larger
/// `block_n` spends more on the score scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashParams {
    /// Queries per outer tile.
    pub block_m: usize,
    /// Keys per inner tile.
    pub block_n: usize,
}

impl FlashParams {
    /// Tiles for an `m x n` attention: one tile while the whole score matrix
    /// fits the 16 MB scratch budget, then key tiles of 1024 with as many
    /// queries as the budget allows (query tiling re-packs K and V, so it is the
    /// last thing to give).
    #[must_use]
    pub const fn auto(m: usize, n: usize) -> Self {
        const SCRATCH: usize = 4 << 20; // f32 elements
        if m == 0 || n == 0 || m <= SCRATCH / n {
            return Self {
                block_m: m,
                block_n: n,
            };
        }
        let block_n = if n < 1024 { n } else { 1024 };
        let cap = SCRATCH / block_n;
        Self {
            block_m: if m < cap { m } else { cap },
            block_n,
        }
    }
}

/// `O = softmax(scale * Q @ K^T) @ V` with tiles from [`FlashParams::auto`].
///
/// `q` is `m x d`, `k` is `n x d`, `v` is `n x dv`, `o` is `m x dv`, all
/// row-major. `scale` is the usual `1/sqrt(d)`. Equivalent to materializing the
/// `m x n` scores, row-softmaxing them, and multiplying by V -- but the scores
/// live only one `block_m x block_n` tile at a time, so memory traffic is
/// `O(m*d + n*d + n*dv)` instead of `O(m*n)`.
///
/// # Panics
/// Panics if any slice length is inconsistent with `m`, `n`, `d`, `dv`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    m: usize,
    n: usize,
    d: usize,
    dv: usize,
    scale: f32,
) {
    flash_attention_f32_with(q, k, v, o, m, n, d, dv, scale, FlashParams::auto(m, n));
}

/// [`flash_attention_f32`] with explicit tile sizes.
///
/// # Panics
/// Panics if any slice length is inconsistent with `m`, `n`, `d`, `dv`, or if a
/// tile size is zero.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_f32_with(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    m: usize,
    n: usize,
    d: usize,
    dv: usize,
    scale: f32,
    p: FlashParams,
) {
    assert_eq!(q.len(), checked_dim2(m, d), "q is m*d");
    assert_eq!(k.len(), checked_dim2(n, d), "k is n*d");
    assert_eq!(v.len(), checked_dim2(n, dv), "v is n*dv");
    assert_eq!(o.len(), checked_dim2(m, dv), "o is m*dv");
    if m == 0 || dv == 0 {
        return;
    }
    if n == 0 || d == 0 {
        o.fill(0.0);
        return;
    }
    assert!(
        p.block_m > 0 && p.block_n > 0,
        "flash tile sizes must be > 0"
    );

    let bm = p.block_m.min(m);
    let bn = p.block_n.min(n);
    let mut s = take(&F32_SCRATCH, checked_dims(bm, bn, 1));
    let mut row_max = vec![f32::NEG_INFINITY; bm];
    let mut row_sum = vec![0.0f32; bm];

    for i0 in (0..m).step_by(bm) {
        let mi = bm.min(m - i0);
        row_max[..mi].fill(f32::NEG_INFINITY);
        row_sum[..mi].fill(0.0);
        let qb = &q[i0 * d..(i0 + mi) * d];
        let ob = &mut o[i0 * dv..(i0 + mi) * dv];
        ob.fill(0.0);

        for j0 in (0..n).step_by(bn) {
            let bj = bn.min(n - j0);
            // S = scale * Qb @ Kj^T -- Kj^T is the (d x bj) transposed view of
            // the row-major (bj x d) key block, so row stride 1, col stride d.
            gemm_f32(
                mi,
                bj,
                d,
                &mut s,
                bn,
                1,
                qb,
                d,
                1,
                &k[j0 * d..],
                1,
                d,
                0.0,
                scale,
            );
            flash_block(&mut s, mi, bj, bn, ob, dv, &mut row_max, &mut row_sum);
            // O += P @ Vj.
            gemm_f32(
                mi,
                dv,
                bj,
                ob,
                dv,
                1,
                &s,
                bn,
                1,
                &v[j0 * dv..],
                dv,
                1,
                1.0,
                1.0,
            );
        }
        flash_finish(ob, mi, dv, &row_sum[..mi]);
    }
    give(&F32_SCRATCH, s);
}

/// One online-softmax step: fold the fresh score block into the running per-row
/// max/sum and rescale the output accumulator by the max correction.
#[allow(clippy::too_many_arguments)]
fn flash_block(
    s: &mut [f32],
    mi: usize,
    bj: usize,
    s_rs: usize,
    o: &mut [f32],
    dv: usize,
    row_max: &mut [f32],
    row_sum: &mut [f32],
) {
    // NEON, not SME -- no capability probe, just the platform gate.
    // SAFETY: `s` covers (mi-1)*s_rs + bj and `o` covers mi*dv (both checked by
    // the callers' slicing); row_max/row_sum are length >= mi. The pass reads
    // and writes only those ranges.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe {
        attn_flash_block_f32(
            s.as_mut_ptr(),
            mi,
            bj,
            s_rs as isize,
            o.as_mut_ptr(),
            dv,
            dv as isize,
            row_max.as_mut_ptr(),
            row_sum.as_mut_ptr(),
        );
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    for i in 0..mi {
        let sp = &mut s[i * s_rs..i * s_rs + bj];
        let prev = row_max[i];
        let mut mx = prev;
        for &x in sp.iter() {
            if x > mx {
                mx = x;
            }
        }
        // Nothing finite seen yet: leave the running state alone.
        if !mx.is_finite() {
            continue;
        }
        let corr = if prev.is_finite() {
            (prev - mx).exp()
        } else {
            0.0
        };
        let mut sum = 0.0f32;
        for x in sp.iter_mut() {
            let e = (*x - mx).exp();
            *x = e;
            sum += e;
        }
        row_sum[i] = row_sum[i].mul_add(corr, sum);
        row_max[i] = mx;
        // Exact 1.0 means the running max did not move, so the rescale is a
        // no-op; any other value (incl. 0.0 for the first block) must apply.
        if (corr - 1.0) != 0.0 {
            for t in &mut o[i * dv..(i + 1) * dv] {
                *t *= corr;
            }
        }
    }
}

/// Divide each output row by its accumulated softmax denominator.
fn flash_finish(o: &mut [f32], mi: usize, dv: usize, row_sum: &[f32]) {
    // NEON, not SME -- no capability probe, just the platform gate.
    // SAFETY: `o` is exactly mi*dv and `row_sum` is length mi.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe {
        attn_flash_finish_f32(o.as_mut_ptr(), mi, dv, dv as isize, row_sum.as_ptr());
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    for i in 0..mi {
        let l = row_sum[i];
        let inv = if l == 0.0 { 0.0 } else { 1.0 / l };
        for t in &mut o[i * dv..(i + 1) * dv] {
            *t *= inv;
        }
    }
}
