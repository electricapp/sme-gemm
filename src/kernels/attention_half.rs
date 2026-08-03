//! Half-precision flash attention (f16 and bf16).
//!
//! Same online-softmax schedule as the f32 path, but the scores come out of the
//! GEMM in half precision. All softmax statistics stay f32, `scale` is applied
//! in f32 rather than folded into the GEMM's half-precision `beta`, and the
//! output accumulates in f32 across key blocks -- so only the two GEMM operands
//! and the probabilities are ever half.

use std::cell::Cell;

use half::{bf16, f16};

use crate::element::Accuracy;
use crate::exec::{checked_dim2, checked_dims};
use crate::kernels::attention::{FlashParams, give, take};

/// The three per-block passes for one half dtype: the vectorized C versions on
/// Apple silicon, a portable scalar equivalent elsewhere. Both storage types are
/// `repr(transparent)` over `u16`, so one FFI shape covers them.
macro_rules! flash_half_passes {
    ($T:ty, $block:ident, $accum:ident, $finish:ident,
     $cblock:path, $caccum:path, $cfinish:path) => {
        #[allow(clippy::too_many_arguments)]
        fn $block(
            s: &mut [$T],
            mi: usize,
            bj: usize,
            s_rs: usize,
            scale: f32,
            row_max: &mut [f32],
            row_sum: &mut [f32],
            corr: &mut [f32],
        ) {
            // SAFETY: `s` covers (mi-1)*s_rs + bj; row_max/row_sum/corr are
            // length >= mi. The pass touches only those ranges.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            unsafe {
                $cblock(
                    s.as_mut_ptr().cast::<u16>(),
                    mi,
                    bj,
                    s_rs as isize,
                    scale,
                    row_max.as_mut_ptr(),
                    row_sum.as_mut_ptr(),
                    corr.as_mut_ptr(),
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            for i in 0..mi {
                let sp = &mut s[i * s_rs..i * s_rs + bj];
                let prev = row_max[i];
                let mut mx = prev;
                for x in sp.iter() {
                    let x = x.to_f32() * scale;
                    if x > mx {
                        mx = x;
                    }
                }
                if !mx.is_finite() {
                    corr[i] = 1.0;
                    continue;
                }
                let c = if prev.is_finite() {
                    (prev - mx).exp()
                } else {
                    0.0
                };
                let mut sum = 0.0f32;
                for x in sp.iter_mut() {
                    let e = (x.to_f32() * scale - mx).exp();
                    *x = <$T>::from_f32(e);
                    sum += e;
                }
                row_sum[i] = row_sum[i].mul_add(c, sum);
                row_max[i] = mx;
                corr[i] = c;
            }
        }

        fn $accum(acc: &mut [f32], t: &[$T], mi: usize, dv: usize, corr: &[f32]) {
            // SAFETY: `acc` and `t` both cover mi*dv; `corr` is length >= mi.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            unsafe {
                $caccum(
                    acc.as_mut_ptr(),
                    t.as_ptr().cast::<u16>(),
                    mi,
                    dv,
                    dv as isize,
                    corr.as_ptr(),
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            for i in 0..mi {
                for j in 0..dv {
                    acc[i * dv + j] = acc[i * dv + j].mul_add(corr[i], t[i * dv + j].to_f32());
                }
            }
        }

        fn $finish(out: &mut [$T], acc: &[f32], mi: usize, dv: usize, row_sum: &[f32]) {
            // SAFETY: `out` and `acc` both cover mi*dv; `row_sum` is >= mi.
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            unsafe {
                $cfinish(
                    out.as_mut_ptr().cast::<u16>(),
                    acc.as_ptr(),
                    mi,
                    dv,
                    dv as isize,
                    row_sum.as_ptr(),
                );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            for i in 0..mi {
                let l = row_sum[i];
                let inv = if l == 0.0 { 0.0 } else { 1.0 / l };
                for j in 0..dv {
                    out[i * dv + j] = <$T>::from_f32(acc[i * dv + j] * inv);
                }
            }
        }
    };
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::ffi::{
    attn_flash_accum_bf16, attn_flash_accum_f16, attn_flash_block_bf16, attn_flash_block_f16,
    attn_flash_finish_bf16, attn_flash_finish_f16,
};

flash_half_passes!(
    f16,
    block_f16,
    accum_f16,
    finish_f16,
    attn_flash_block_f16,
    attn_flash_accum_f16,
    attn_flash_finish_f16
);
flash_half_passes!(
    bf16,
    block_bf16,
    accum_bf16,
    finish_bf16,
    attn_flash_block_bf16,
    attn_flash_accum_bf16,
    attn_flash_finish_bf16
);

/// Body shared by the f16 and bf16 entry points.
///
/// `gemm` is the dtype's strided GEMM; `block`/`accum`/`finish` are the passes
/// generated above.
macro_rules! flash_half {
    (
        $(#[$meta:meta])*
        $name:ident, $with:ident, $T:ty, $gemm:path,
        $block:ident, $accum:ident, $finish:ident
    ) => {
        $(#[$meta])*
        ///
        /// # Panics
        /// Panics if any slice length is inconsistent with `m`, `n`, `d`, `dv`.
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            q: &[$T],
            k: &[$T],
            v: &[$T],
            o: &mut [$T],
            m: usize,
            n: usize,
            d: usize,
            dv: usize,
            scale: f32,
            mode: Accuracy,
        ) {
            $with(q, k, v, o, m, n, d, dv, scale, mode, FlashParams::auto(m, n));
        }

        #[doc = concat!("[`", stringify!($name), "`] with explicit tile sizes.")]
        ///
        /// # Panics
        /// Panics if any slice length is inconsistent with `m`, `n`, `d`, `dv`,
        /// or if a tile size is zero.
        #[allow(clippy::too_many_arguments)]
        pub fn $with(
            q: &[$T],
            k: &[$T],
            v: &[$T],
            o: &mut [$T],
            m: usize,
            n: usize,
            d: usize,
            dv: usize,
            scale: f32,
            mode: Accuracy,
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
                o.fill(<$T>::ZERO);
                return;
            }
            assert!(
                p.block_m > 0 && p.block_n > 0,
                "flash tile sizes must be > 0"
            );

            let bm = p.block_m.min(m);
            let bn = p.block_n.min(n);
            let (zero, one) = (<$T>::ZERO, <$T>::ONE);
            // Reused across calls; see the note on `take` in attention.rs.
            thread_local! {
                static S: Cell<Vec<$T>> = const { Cell::new(Vec::new()) };
                static P: Cell<Vec<$T>> = const { Cell::new(Vec::new()) };
                static A: Cell<Vec<f32>> = const { Cell::new(Vec::new()) };
            }
            let mut s = take(&S, checked_dims(bm, bn, 1));
            let mut prod = take(&P, checked_dims(bm, dv, 1));
            let mut acc = take(&A, checked_dims(bm, dv, 1));
            let mut row_max = vec![0.0f32; bm];
            let mut row_sum = vec![0.0f32; bm];
            let mut corr = vec![0.0f32; bm];

            for i0 in (0..m).step_by(bm) {
                let mi = bm.min(m - i0);
                row_max[..mi].fill(f32::NEG_INFINITY);
                row_sum[..mi].fill(0.0);
                acc[..mi * dv].fill(0.0);
                let qb = &q[i0 * d..(i0 + mi) * d];

                for j0 in (0..n).step_by(bn) {
                    let bj = bn.min(n - j0);
                    // S = Qb @ Kj^T -- Kj^T is the (d x bj) transposed view of
                    // the row-major (bj x d) key block: row stride 1, col d.
                    $gemm(
                        mi, bj, d, &mut s, bn, 1, qb, d, 1, &k[j0 * d..], 1, d, zero, one, mode,
                    );
                    $block(
                        &mut s,
                        mi,
                        bj,
                        bn,
                        scale,
                        &mut row_max,
                        &mut row_sum,
                        &mut corr,
                    );
                    $gemm(
                        mi, dv, bj, &mut prod, dv, 1, &s, bn, 1, &v[j0 * dv..], dv, 1, zero, one,
                        mode,
                    );
                    $accum(&mut acc, &prod, mi, dv, &corr);
                }
                $finish(&mut o[i0 * dv..], &acc, mi, dv, &row_sum);
            }
            give(&S, s);
            give(&P, prod);
            give(&A, acc);
        }
    };
}

flash_half! {
    /// `O = softmax(scale * Q @ K^T) @ V` in f16, a key block at a time.
    ///
    /// `q` is `m x d`, `k` is `n x d`, `v` is `n x dv`, `o` is `m x dv`, all
    /// row-major. `mode` selects the GEMM accumulator: [`Accuracy::Accurate`]
    /// widens to f32 (the right choice for attention logits),
    /// [`Accuracy::Fast`] uses the M5 non-widening f16 MOPA.
    ///
    /// Scores and probabilities are stored in f16, so expect f16-attention
    /// accuracy (~1e-3 relative), not the f32 path's ~1e-6.
    flash_attention_f16, flash_attention_f16_with, f16, crate::kernels::f16::gemm_f16,
    block_f16, accum_f16, finish_f16
}

flash_half! {
    /// `O = softmax(scale * Q @ K^T) @ V` in bf16, a key block at a time.
    ///
    /// See [`flash_attention_f16`]; bf16's narrower mantissa makes the stored
    /// scores and probabilities correspondingly coarser (~1e-2 relative).
    flash_attention_bf16, flash_attention_bf16_with, bf16, crate::kernels::bf16::gemm_bf16,
    block_bf16, accum_bf16, finish_bf16
}
