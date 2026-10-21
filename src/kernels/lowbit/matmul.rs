//! The low-bit GEMMs: `C = A @ dequant(B)` in f16 for 2-bit and 1-bit
//! weights, each with a fused-epilogue variant, and the reference they match
//! bit for bit.

use half::f16;

use super::{Q1Weights, Q2Weights};
use crate::epilogue::Epilogue;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::probe::caps;

/// Row-major `C = A @ dequant(B)` with B kept 2-bit resident (see
/// [`Q2Weights`]). `a` is `m x k` row-major f16, `c` is `m x n` row-major f16.
///
/// Each row of `a` is quantized to int8 per K-block inside the call (scale
/// `max|a|/127`, round to nearest even), and the products are exact integer
/// sums folded block by block into f32: activation rounding, about 0.4% of a
/// block's largest value, is the accuracy floor beyond the weights' own. One
/// row runs at the int8 dot-product rate, twice the Q4 GEMV's on M5, and the
/// weights are half Q4's bytes. Every row runs that GEMV, so for many rows
/// the call stays at it rather than reaching the f16 MOPA rate. Results do
/// not depend on `m` or on how the call splits over cores.
///
/// [`matmul_q2_ep`] fuses a bias/activation op-graph into the same store.
///
/// ```
/// use half::f16;
/// use sme_gemm::{Q2Params, Q2Weights, WeightLayout, matmul_q2};
/// let (n, k) = (64, 256);
/// let w: Vec<f32> = (0..n * k).map(|i| [-0.5, 0.0, 0.5][i % 3]).collect();
/// let q = Q2Weights::quantize(&w, WeightLayout::OutIn, n, k, Q2Params::default());
/// let (a, mut c) = (vec![f16::ONE; k], vec![f16::ZERO; n]);
/// matmul_q2(&a, &q, &mut c, 1);
/// ```
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the weights' `n`, `k`.
pub fn matmul_q2(a: &[f16], w: &Q2Weights, c: &mut [f16], m: usize) {
    lowbit_impl(a, Codes::Q2(w), c, m, &Epilogue::new());
}

/// [`matmul_q2`] with a fused epilogue: `C = ep(A @ dequant(B))`, applied at
/// the f16 store.
///
/// # Panics
/// Panics as [`matmul_q2`], plus if an epilogue operand length mismatches
/// `m`/`n`.
pub fn matmul_q2_ep(a: &[f16], w: &Q2Weights, c: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
    lowbit_impl(a, Codes::Q2(w), c, m, ep);
}

/// Row-major `C = A @ dequant(B)` with B kept 1-bit resident (see
/// [`Q1Weights`]).
///
/// This is [`matmul_q2`]'s int8 path with one LUTI4 expanding each group of
/// four signs, so the same dot-product rate from half the bytes.
///
/// ```
/// use half::f16;
/// use sme_gemm::{Q1Weights, WeightLayout, matmul_q1};
/// let (n, k) = (64, 256);
/// let w: Vec<f32> = (0..n * k).map(|i| [-0.5, 0.5][i % 2]).collect();
/// let q = Q1Weights::quantize(&w, WeightLayout::OutIn, n, k, 128);
/// let (a, mut c) = (vec![f16::ONE; k], vec![f16::ZERO; n]);
/// matmul_q1(&a, &q, &mut c, 1);
/// ```
///
/// # Panics
/// Panics if `a`/`c` lengths are inconsistent with `m` and the weights' `n`, `k`.
pub fn matmul_q1(a: &[f16], w: &Q1Weights, c: &mut [f16], m: usize) {
    lowbit_impl(a, Codes::Q1(w), c, m, &Epilogue::new());
}

/// [`matmul_q1`] with a fused epilogue, applied at the f16 store.
///
/// # Panics
/// Panics as [`matmul_q1`], plus if an epilogue operand length mismatches
/// `m`/`n`.
pub fn matmul_q1_ep(a: &[f16], w: &Q1Weights, c: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
    lowbit_impl(a, Codes::Q1(w), c, m, ep);
}

/// [`matmul_q2`] on the f16 MOPA path, for many rows.
///
/// `a` stays f16, the call unpacks the weights into f16 panels -- `scale *
/// codebook[code]`, rounded once -- and the dense f16 kernel multiplies them,
/// accumulating in f16 as [`Accum::F16`](crate::Accum::F16) does.
///
/// At prompt-sized row counts that is twice [`matmul_q2`]'s rate on M5, with
/// a different error: no activation rounding, but an f16 accumulator, whose
/// error grows with `k` (with random weights and activations about 0.65% of
/// the output's RMS at k = 2048 and 1.1% at 6144, against 0.4% for the int8
/// path). From 8 rows the
/// result is [`matmul_f16_packed`](crate::matmul_f16_packed) over the
/// dequantized weights bit for bit; fewer rows run the same MOPA, where the
/// dense GEMM would switch to its GEMV.
///
/// # Panics
/// As [`matmul_q2`].
pub fn matmul_q2_mopa(a: &[f16], w: &Q2Weights, c: &mut [f16], m: usize) {
    mopa_impl(a, Codes::Q2(w), c, m, &Epilogue::new());
}

/// [`matmul_q2_mopa`] with a fused epilogue, applied at the f16 store.
///
/// # Panics
/// As [`matmul_q2_ep`].
pub fn matmul_q2_mopa_ep(
    a: &[f16],
    w: &Q2Weights,
    c: &mut [f16],
    m: usize,
    ep: &Epilogue<'_, f16>,
) {
    mopa_impl(a, Codes::Q2(w), c, m, ep);
}

/// [`matmul_q1`] on the f16 MOPA path, as [`matmul_q2_mopa`]; a 1-bit
/// weight unpacks exactly (its scale, negated for a clear bit).
///
/// # Panics
/// As [`matmul_q1`].
pub fn matmul_q1_mopa(a: &[f16], w: &Q1Weights, c: &mut [f16], m: usize) {
    mopa_impl(a, Codes::Q1(w), c, m, &Epilogue::new());
}

/// [`matmul_q1_mopa`] with a fused epilogue, applied at the f16 store.
///
/// # Panics
/// As [`matmul_q1_ep`].
pub fn matmul_q1_mopa_ep(
    a: &[f16],
    w: &Q1Weights,
    c: &mut [f16],
    m: usize,
    ep: &Epilogue<'_, f16>,
) {
    mopa_impl(a, Codes::Q1(w), c, m, ep);
}

/// The weights either path reads, as the shared code sees them.
#[derive(Clone, Copy)]
enum Codes<'w> {
    Q2(&'w Q2Weights),
    Q1(&'w Q1Weights),
}

impl Codes<'_> {
    const fn dims(self) -> (usize, usize, usize) {
        match self {
            Self::Q2(w) => (w.n, w.k, w.params.block),
            Self::Q1(w) => (w.n, w.k, w.block),
        }
    }

    /// The int8 value of column `j`'s weight at depth `d`, before its scale.
    fn value(self, d: usize, j: usize) -> i32 {
        match self {
            Self::Q2(w) => i32::from(w.params.codebook[w.code(d, j)]),
            Self::Q1(w) => {
                if w.sign(d, j) {
                    1
                } else {
                    -1
                }
            }
        }
    }

    fn scale(self, blk: usize, j: usize) -> f32 {
        match self {
            Self::Q2(w) => w.scale(blk, j),
            Self::Q1(w) => w.scale(blk, j),
        }
    }

    /// The weights as the MOPA path unpacks them, row-major `k x n`.
    fn dequant(self) -> Vec<f16> {
        let (n, k, block) = self.dims();
        let mut b = vec![f16::ZERO; k * n];
        for (d, row) in b.chunks_exact_mut(n.max(1)).enumerate().take(k) {
            for (j, o) in row.iter_mut().enumerate() {
                let s = self.scale(d / block, j);
                *o = f16::from_f32(match self {
                    Self::Q2(w) => s * f32::from(w.params.codebook[w.code(d, j)]),
                    Self::Q1(w) => {
                        if w.sign(d, j) {
                            s
                        } else {
                            -s
                        }
                    }
                });
            }
        }
        b
    }

    /// Runs the int8 GEMV kernel, or with `mopa` the f16 MOPA path; non-zero
    /// when its scratch allocation failed and `c` is untouched.
    ///
    /// # Safety
    /// `a` / `c` hold `m*k` / `m*n` values; `ep` is null or a validated
    /// descriptor whose operands outlive the call.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    unsafe fn run(
        self,
        mopa: bool,
        m: usize,
        c: *mut u16,
        a: *const u16,
        ep: *const crate::ffi::EpDesc16,
    ) -> core::ffi::c_int {
        let (n, k, block) = self.dims();
        // SAFETY: the caller's contract covers a, c and ep; the codes and
        // scales describe the weights' (n, k) in the band-major layout the
        // kernels expect, and a codebook is 4 bytes.
        unsafe {
            match (self, mopa) {
                (Self::Q2(w), false) => {
                    let _busy = crate::warm::busy("gemm_sme_f16_q2", m, n, k);
                    crate::ffi::gemm_sme_f16_q2(
                        m,
                        n,
                        k,
                        c,
                        a,
                        w.codes.as_ptr(),
                        w.scales.as_ptr().cast::<u16>(),
                        w.params.codebook.as_ptr(),
                        block,
                        ep,
                    )
                }
                (Self::Q2(w), true) => {
                    let _busy = crate::warm::busy("gemm_sme_f16f16_q2", m, n, k);
                    crate::ffi::gemm_sme_f16f16_q2(
                        m,
                        n,
                        k,
                        c,
                        a,
                        w.codes.as_ptr(),
                        w.scales.as_ptr().cast::<u16>(),
                        w.params.codebook.as_ptr(),
                        block,
                        ep,
                    )
                }
                (Self::Q1(w), false) => {
                    let _busy = crate::warm::busy("gemm_sme_f16_q1", m, n, k);
                    crate::ffi::gemm_sme_f16_q1(
                        m,
                        n,
                        k,
                        c,
                        a,
                        w.signs.as_ptr(),
                        w.scales.as_ptr().cast::<u16>(),
                        block,
                        ep,
                    )
                }
                (Self::Q1(w), true) => {
                    let _busy = crate::warm::busy("gemm_sme_f16f16_q1", m, n, k);
                    crate::ffi::gemm_sme_f16f16_q1(
                        m,
                        n,
                        k,
                        c,
                        a,
                        w.signs.as_ptr(),
                        w.scales.as_ptr().cast::<u16>(),
                        block,
                        ep,
                    )
                }
            }
        }
    }

    /// [`Codes::run`] with `ep` resolved for the kernel; `true` when it ran.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn run_ep(
        self,
        mopa: bool,
        a: &[f16],
        c: &mut [f16],
        m: usize,
        ep: &Epilogue<'_, f16>,
    ) -> bool {
        let fnodes = crate::exec::resolve_nodes(&ep.nodes, ep.act, self.dims().0);
        let desc = crate::ffi::EpDesc16 {
            n_nodes: u32::try_from(fnodes.len()).expect("epilogue op-graph exceeds u32 nodes"),
            nodes: fnodes.as_ptr(),
        };
        let ep_ptr = if fnodes.is_empty() {
            core::ptr::null()
        } else {
            &raw const desc
        };
        // SAFETY: the callers check a / c against m*k / m*n and validate the
        // node operands, which outlive the call.
        let rc = unsafe {
            self.run(
                mopa,
                m,
                c.as_mut_ptr().cast::<u16>(),
                a.as_ptr().cast::<u16>(),
                ep_ptr,
            )
        };
        rc == 0
    }
}

/// The shapes both paths take; `false` when there is nothing to compute.
fn check(a: &[f16], w: Codes<'_>, c: &[f16], m: usize, ep: &Epilogue<'_, f16>) -> bool {
    let (n, k, _) = w.dims();
    assert_eq!(a.len(), crate::exec::checked_dim2(m, k), "a is m*k");
    assert_eq!(c.len(), crate::exec::checked_dim2(m, n), "c is m*n");
    crate::exec::validate_ep(ep, m, n);
    m != 0 && n != 0
}

fn lowbit_impl(a: &[f16], w: Codes<'_>, c: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
    if !check(a, w, c, m, ep) {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme2p1 && w.run_ep(false, a, c, m, ep) {
        return;
    }
    let n = w.dims().0;
    reference(a, w, c, m);
    crate::exec::apply_ep_scalar(c, m, n, n, 1, ep);
}

fn mopa_impl(a: &[f16], w: Codes<'_>, c: &mut [f16], m: usize, ep: &Epilogue<'_, f16>) {
    if !check(a, w, c, m, ep) {
        return;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if caps().sme_f16f16 && w.run_ep(true, a, c, m, ep) {
        return;
    }
    let (n, k, _) = w.dims();
    let p = crate::prepack_f16(&w.dequant(), n, k);
    crate::kernels::f16::f16_packed_ep_impl(a, &p, c, m, ep, f16::ONE, false);
}

/// One row's int8 activations and per-block scales, as `csrc/gemm_q2.cc`
/// quantizes them: `amax` NaN if any value is.
fn quantize_row(a: &[f16], block: usize, q: &mut [i8], s: &mut [f32]) {
    for ((ab, qb), sb) in a.chunks(block).zip(q.chunks_mut(block)).zip(s.iter_mut()) {
        let amax = ab.iter().fold(0f32, |m, x| {
            let x = x.to_f32().abs();
            if x.is_nan() || m.is_nan() {
                f32::NAN
            } else {
                m.max(x)
            }
        });
        let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
        *sb = amax / 127.0;
        for (o, x) in qb.iter_mut().zip(ab) {
            // Saturating, NaN to 0, as the kernel's convert and narrow.
            *o = (x.to_f32() * inv).round_ties_even() as i8;
        }
    }
}

/// The product without an epilogue, computed as the kernel does: exact
/// integer block sums, each folded as `total + f32(sum) * (scale * s)`, fused,
/// in K order.
fn reference(a: &[f16], w: Codes<'_>, c: &mut [f16], m: usize) {
    let (n, k, block) = w.dims();
    let nbk = k.div_ceil(block);
    let (mut q, mut s) = (vec![0i8; k], vec![0f32; nbk]);
    for (ar, cr) in a.chunks_exact(k.max(1)).zip(c.chunks_exact_mut(n)).take(m) {
        quantize_row(&ar[..k], block, &mut q, &mut s);
        for (j, out) in cr.iter_mut().enumerate() {
            let mut total = 0f32;
            for (blk, &sa) in s.iter().enumerate() {
                let (d0, d1) = (blk * block, ((blk + 1) * block).min(k));
                let sum: i32 = (d0..d1).map(|d| i32::from(q[d]) * w.value(d, j)).sum();
                // An i32 converts as SCVTF does: round to nearest even.
                #[allow(clippy::cast_precision_loss)]
                let f = sum as f32;
                total = f.mul_add(w.scale(blk, j) * sa, total);
            }
            *out = f16::from_f32(total);
        }
    }
    if k == 0 {
        c[..m * n].fill(f16::ZERO);
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::{Codes, reference};
    use crate::Epilogue;
    use crate::WeightLayout;
    use crate::kernels::lowbit::{Q1Weights, Q2Params, Q2Weights};

    fn data(seed: u64, len: usize) -> Vec<u8> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                (s >> 40) as u8
            })
            .collect()
    }

    const SHAPES: [(usize, usize, usize); 12] = [
        (1, 64, 256),
        (1, 65, 37),
        (1, 1, 1),
        (2, 200, 300),
        (3, 64, 16),
        (5, 130, 513),
        (6, 192, 256),
        (7, 100, 64),
        (12, 70, 200),
        (1, 2048, 1024),
        (4, 1536, 512),
        (9, 1024, 384),
    ];

    fn check(m: usize, n: usize, w: Codes<'_>, a: &[f16], what: &str) {
        let mut want = vec![f16::ZERO; m * n];
        reference(a, w, &mut want, m);
        let mut got = vec![f16::from_f32(9.0); m * n];
        match w {
            Codes::Q2(q) => crate::matmul_q2(a, q, &mut got, m),
            Codes::Q1(q) => crate::matmul_q1(a, q, &mut got, m),
        }
        for (i, (g, r)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                r.to_bits(),
                "{what}: c[{}][{}] = {g} vs {r}",
                i / n,
                i % n
            );
        }
    }

    /// The kernels give the reference's bits for every row count, ragged
    /// shapes, block sizes and codebooks, single-threaded and spread over the
    /// cores (n large enough to dispatch).
    #[test]
    fn kernels_match_reference_bits() {
        let books = [[-1i8, 0, 1, 0], [-2, -1, 1, 2], [-128, 127, 3, -7]];
        for (si, &(m, n, k)) in SHAPES.iter().enumerate() {
            for (bi, &block) in [16usize, 64, 128, 4096].iter().enumerate() {
                let nbk = k.div_ceil(block);
                let scales: Vec<f16> = data(si as u64 + 7, n * nbk)
                    .iter()
                    .map(|&b| f16::from_f32(f32::from(b) * 1e-3 - 0.1))
                    .collect();
                let a: Vec<f16> = data(si as u64 + 99, m * k)
                    .iter()
                    .map(|&b| f16::from_f32((f32::from(b) - 128.0) / 37.0))
                    .collect();
                let book = books[(si + bi) % books.len()];
                let p = Q2Params::new(block).with_codebook(book);
                let codes = data(si as u64 * 31 + bi as u64, (k * n).div_ceil(4));
                let q2 = Q2Weights::new(&codes, &scales, n, k, p);
                check(
                    m,
                    n,
                    Codes::Q2(&q2),
                    &a,
                    &format!("q2 {m}x{n}x{k} {block} {book:?}"),
                );
                let signs = data(si as u64 * 17 + bi as u64, (k * n).div_ceil(8));
                for layout in [WeightLayout::OutIn, WeightLayout::InOut] {
                    let q1 = Q1Weights::new(&signs, &scales, layout, n, k, block);
                    check(
                        m,
                        n,
                        Codes::Q1(&q1),
                        &a,
                        &format!("q1 {m}x{n}x{k} {block} {layout:?}"),
                    );
                }
            }
        }
    }

    /// The MOPA path against the eager dequant through the dense f16 kernel,
    /// bit for bit. The dense call gets at least 8 rows (zero rows appended),
    /// so it runs its MOPA too rather than its GEMV.
    fn check_mopa(m: usize, w: Codes<'_>, a: &[f16], ep: &Epilogue<'_, f16>, what: &str) {
        let (n, k, _) = w.dims();
        let mp = m.max(8);
        let mut ap = a.to_vec();
        ap.resize(mp * k, f16::ZERO);
        let mut want = vec![f16::ZERO; mp * n];
        let p = crate::prepack_f16(&w.dequant(), n, k);
        crate::kernels::f16::f16_packed_ep_impl(&ap, &p, &mut want, mp, ep, f16::ONE, false);
        let mut got = vec![f16::from_f32(9.0); m * n];
        match w {
            Codes::Q2(q) => crate::matmul_q2_mopa_ep(a, q, &mut got, m, ep),
            Codes::Q1(q) => crate::matmul_q1_mopa_ep(a, q, &mut got, m, ep),
        }
        for (i, (g, r)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                r.to_bits(),
                "{what}: c[{}][{}] = {g} vs {r}",
                i / n,
                i % n
            );
        }
    }

    /// Ragged shapes, K tails off the four-depth quad, odd tile counts, more
    /// N-blocks than ring slots, every block size and codebook, plain and with
    /// a fused bias and activation.
    #[test]
    fn mopa_matches_eager_dequant() {
        let books = [[-1i8, 0, 1, 0], [-2, -1, 1, 2], [-128, 127, 3, -7]];
        let shapes = [
            (8usize, 64usize, 256usize),
            (1, 65, 37),
            (3, 200, 301),
            (33, 130, 514),
            (40, 4200, 1030),
            (200, 1000, 2048),
        ];
        for (si, &(m, n, k)) in shapes.iter().enumerate() {
            let bias: Vec<f16> = (0..n)
                .map(|j| f16::from_f32(j as f32 * 0.01 - 1.0))
                .collect();
            for (bi, &block) in [16usize, 128, 4096].iter().enumerate() {
                let nbk = k.div_ceil(block);
                let scales: Vec<f16> = data(si as u64 + 3, n * nbk)
                    .iter()
                    .map(|&b| f16::from_f32(f32::from(b) * 1e-3 - 0.1))
                    .collect();
                let a: Vec<f16> = data(si as u64 + 77, m * k)
                    .iter()
                    .map(|&b| f16::from_f32((f32::from(b) - 128.0) / 61.0))
                    .collect();
                let book = books[(si + bi) % books.len()];
                let codes = data(si as u64 * 13 + bi as u64, (k * n).div_ceil(4));
                let p = Q2Params::new(block).with_codebook(book);
                let q2 = Q2Weights::new(&codes, &scales, n, k, p);
                let signs = data(si as u64 * 19 + bi as u64, (k * n).div_ceil(8));
                let q1 = Q1Weights::new(&signs, &scales, WeightLayout::OutIn, n, k, block);
                for (ei, ep) in [Epilogue::new(), Epilogue::new().add_col(&bias).silu()]
                    .iter()
                    .enumerate()
                {
                    let what = format!("{m}x{n}x{k} block {block} ep {ei}");
                    check_mopa(m, Codes::Q2(&q2), &a, ep, &format!("q2 {book:?} {what}"));
                    check_mopa(m, Codes::Q1(&q1), &a, ep, &format!("q1 {what}"));
                }
            }
        }
    }
}
