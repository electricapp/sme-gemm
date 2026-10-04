// SME f64 GEMM driver (Apple M5+, FEAT_SME_F64F64, FMOPA double-precision).
// SVL=512 gives 8 f64 lanes, so a 16x16 super-tile is four 8x8 ZA64 quadrants;
// one K-value per MOPA. Packing is plain [k, 8] per 8-wide band.
//
// Computes dst = alpha*dst + beta*(A @ B).

#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "epilogue.h"

// Pack one 8-wide band into [k, 8]: dst[d*8 + i] = src[i, d]. NEON fast paths
// for the full-8-lane case: lanes contiguous (B) -> copy per depth; depth
// contiguous (row-major A) -> 2x2 f64 transpose.
static void pack_band(double *dst, const double *src, long lane_stride, long depth_stride, size_t k,
                      size_t valid) {
    if (valid == 8 && lane_stride == 1) {
        for (size_t d = 0; d < k; d++)
            memcpy(dst + d * 8, src + (long)d * depth_stride, 8 * sizeof(double));
        return;
    }
    if (valid == 8 && depth_stride == 1) {
        // Depth-outer so the four lane groups of one depth fill its 64-byte line
        // consecutively; lane-group-outer walks the whole panel four times. Same
        // reasoning as the f32 twin.
        size_t kf = k & ~(size_t)1;
        for (size_t d = 0; d < kf; d += 2) {
            for (size_t rg = 0; rg < 8; rg += 2) {
                float64x2_t r0 = vld1q_f64(src + (long)(rg + 0) * lane_stride + d);
                float64x2_t r1 = vld1q_f64(src + (long)(rg + 1) * lane_stride + d);
                vst1q_f64(dst + (d + 0) * 8 + rg, vtrn1q_f64(r0, r1));
                vst1q_f64(dst + (d + 1) * 8 + rg, vtrn2q_f64(r0, r1));
            }
        }
        for (size_t d = kf; d < k; d++)
            for (size_t i = 0; i < 8; i++)
                dst[d * 8 + i] = src[(long)i * lane_stride + d];
        return;
    }
    for (size_t d = 0; d < k; d++)
        for (size_t i = 0; i < 8; i++)
            dst[d * 8 + i] =
                (i < valid) ? src[(long)i * lane_stride + (long)d * depth_stride] : 0.0;
}

// Zero the four ZA64 sub-tiles then initialize ZA[r][c] = bias[c] via rank-1
// FMOPAs (ones[r] * bias[c]). The 16x16 output is four 8x8 tiles: 0/1 = rows
// 0..7 x cols {0..7, 8..15}, 2/3 = rows 8..15 x same cols. The lo bias half
// (cols n0..n0+7) inits tiles 0,2; the hi half (n0+8..n0+15) inits tiles 1,3.
// Out of line so the fold does not perturb the store codegen.
__attribute__((noinline)) static void bias_init_za(svbool_t p64, const double *bias, size_t n0,
                                                   size_t n, size_t nc, size_t mr)
    __arm_streaming __arm_inout("za") {
    svzero_za();
    svfloat64_t v_ones = svdup_n_f64(1.0);
    size_t nlo = n0, nhi = n0 + 8;
    svbool_t plo = (nlo < n) ? svwhilelt_b64((uint64_t)nlo, (uint64_t)n) : svpfalse_b();
    svbool_t phi = (nhi < n) ? svwhilelt_b64((uint64_t)nhi, (uint64_t)n) : svpfalse_b();
    // Clamp the bias addresses for the all-false (pad) halves: no lane is read,
    // but forming bias + nlo/nhi past one-past-end would still be UB.
    svfloat64_t blo = svld1_f64(plo, (nlo < n) ? bias + nlo : bias);
    svfloat64_t bhi = svld1_f64(phi, (nhi < n) ? bias + nhi : bias);
    // svzero_za already cleared every tile; only seed the LIVE quadrants the store
    // will read. Narrow-N (nc<=8) leaves za1/za3 dead; narrow-M (mr<=8) za2/za3 --
    // skip their bias MOPAs (mirrors the F64_STEP_NARROW_* K-loop dispatch).
    int hi_n = nc > 8, hi_m = mr > 8;
    svmopa_za64_f64_m(0, p64, plo, v_ones, blo);
    if (hi_n) svmopa_za64_f64_m(1, p64, phi, v_ones, bhi);
    if (hi_m) svmopa_za64_f64_m(2, p64, plo, v_ones, blo);
    if (hi_n && hi_m) svmopa_za64_f64_m(3, p64, phi, v_ones, bhi);
}

// One K-step's MOPAs for the live ZA64 quadrants of the 16x16 super-tile
// (0=lo-M x lo-N, 1=lo-M x hi-N, 2=hi-M x lo-N, 3=hi-M x hi-N; 8-wide bands).
// NARROW-N (nc <= 8) leaves the hi-N band all zero-pad -> za1/za3 dead; NARROW-M
// (mr <= 8) leaves the hi-M band all-pad -> za2/za3 dead. Skipping the dead
// quadrants halves MOPA issue on small-m shapes (the store reads ZA by
// nc/mr predicates, so it is unaffected); the dispatch is hoisted out of the
// K-loop. f64 at tiny N is bandwidth-bound, so the win is modest, but the dead
// work is removed with no downside.
#define F64_STEP_FULL(al, ah, bl, bh)                                                              \
    svmopa_za64_f64_m(0, p64, p64, al, bl);                                                        \
    svmopa_za64_f64_m(1, p64, p64, al, bh);                                                        \
    svmopa_za64_f64_m(2, p64, p64, ah, bl);                                                        \
    svmopa_za64_f64_m(3, p64, p64, ah, bh)
#define F64_STEP_NARROW_N(al, ah, bl)                                                              \
    svmopa_za64_f64_m(0, p64, p64, al, bl);                                                        \
    svmopa_za64_f64_m(2, p64, p64, ah, bl)
#define F64_STEP_NARROW_M(al, bl, bh)                                                              \
    svmopa_za64_f64_m(0, p64, p64, al, bl);                                                        \
    svmopa_za64_f64_m(1, p64, p64, al, bh)

// Largest B (n*k*8 bytes) the direct-B path may read unpacked, and the flop
// count below which it stays on one cluster; see gemm_f32.c. The budget is well
// under f32's because an f64 N-tile is 8 lanes, not 32: unpacked B yields 64
// useful bytes per n*8 stride against f32's 128 per n*4, so an equal footprint
// costs twice the address span.
#define SME_F64_DIRECT_B_BYTES ((size_t)4 * 1024 * 1024)
#define SME_F64_SERIAL_FLOPS ((uint64_t)1 << 22)

// Workers for the dynamic M-chunk claiming; see gemm_f32.c.
#define SME_F64_MAX_WORKERS 8
static size_t f64_workers(size_t chunks) {
    if (chunks <= 3) return chunks;
    size_t w = chunks - 1;
    return w > SME_F64_MAX_WORKERS ? SME_F64_MAX_WORKERS : w;
}

// Resident-bytes budget for the cache blocking below, in 16-wide super-tiles
// (2 packed bands each). f64 panels are 2x the bytes of f32.
static size_t f64_budget(size_t k) {
    size_t tile_bytes = 2 * ep_cmul(k, 8) * sizeof(double);
    size_t b = (size_t)(24u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    return b < 2 ? 2 : b;
}

// N-block width, shared by the driver's outer jc loop and run_streaming's inner
// one so the two agree on block boundaries. Rounded down to an even split (see
// gemm_f32.c): the budget is a ceiling, not a target.
static size_t f64_nc_blk(size_t k, size_t n_tiles) {
    size_t nc = f64_budget(k) / 2;
    if (nc < 1) nc = 1;
    if (nc >= n_tiles) return n_tiles;
    size_t blocks = (n_tiles + nc - 1) / nc;
    return (n_tiles + blocks - 1) / blocks;
}

// m == 1: SME2 multi-vector FMLA into ZA vector groups instead of MOPA; see
// run_gemv in gemm_f16f16.c. 8-column bands, two per 16-wide tile; `abc` is
// [k][8] broadcast A. Past one row the 8x8 MOPA is cheaper.
__arm_locally_streaming __arm_new("za") static void run_gemv(double *dst, const double *abc,
                                                             const double *b_pack, size_t n,
                                                             size_t k, size_t nt_lo, size_t nt_hi,
                                                             double alpha, double beta,
                                                             int read_dst, const ep_desc_f64 *ep) {
    svbool_t p64 = svptrue_b64();
    svcount_t pn = svptrue_c64();
    const svfloat64_t va = svdup_n_f64(alpha);
    const svfloat64_t vb = svdup_n_f64(beta);
    int has_ep = ep && ep->n_nodes > 0;
    int pure = (beta == 1.0 && !read_dst && !has_ep);
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t per_band = ep_cmul(k, 8);
    size_t kfull = k & ~(size_t)3;
    svcount_t pt = svwhilelt_c64((uint64_t)0, (uint64_t)((k - kfull) * 8), 4);
    for (size_t nt = nt_lo; nt < nt_hi; nt += 2) {
        size_t T = nt_hi - nt < 2 ? 2 : 4; // bands
        const double *b = b_pack + 2 * nt * per_band;
        svzero_za();
        if (T == 4) {
            for (size_t d = 0; d < kfull; d += 4) {
                svfloat64x4_t A = svld1_f64_x4(pn, abc + d * 8);
                svmla_za64_f64_vg1x4(0, svld1_f64_x4(pn, b + d * 8), A);
                svmla_za64_f64_vg1x4(1, svld1_f64_x4(pn, b + per_band + d * 8), A);
                svmla_za64_f64_vg1x4(2, svld1_f64_x4(pn, b + 2 * per_band + d * 8), A);
                svmla_za64_f64_vg1x4(3, svld1_f64_x4(pn, b + 3 * per_band + d * 8), A);
            }
        } else {
            for (size_t t = 0; t < T; t++)
                for (size_t d = 0; d < kfull; d += 4)
                    svmla_za64_f64_vg1x4((uint32_t)t, svld1_f64_x4(pn, b + t * per_band + d * 8),
                                         svld1_f64_x4(pn, abc + d * 8));
        }
        if (kfull < k) {
            svfloat64x4_t A = svld1_f64_x4(pt, abc + kfull * 8);
            for (size_t t = 0; t < T; t++)
                svmla_za64_f64_vg1x4((uint32_t)t, svld1_f64_x4(pt, b + t * per_band + kfull * 8), A);
        }
        for (size_t t = 0; t < T; t += 2) {
            size_t n0 = (nt + t / 2) * 16;
            size_t nc = n - n0 < 16 ? n - n0 : 16;
            svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)(nc < 8 ? nc : 8));
            svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)(nc > 8 ? nc - 8 : 0));
            svfloat64x4_t ql = svread_za64_f64_vg1x4((uint32_t)t);
            svfloat64x4_t qh = svread_za64_f64_vg1x4((uint32_t)(t + 1));
            svfloat64_t lo = svadd_x(p64, svadd_x(p64, svget4(ql, 0), svget4(ql, 1)),
                                     svadd_x(p64, svget4(ql, 2), svget4(ql, 3)));
            svfloat64_t hi = svadd_x(p64, svadd_x(p64, svget4(qh, 0), svget4(qh, 1)),
                                     svadd_x(p64, svget4(qh, 2), svget4(qh, 3)));
            double *rp = dst + n0;
            double *rph = nc > 8 ? rp + 8 : rp;
            if (pure) {
                svst1_f64(plo, rp, lo);
                svst1_f64(phi, rph, hi);
            } else if (has_ep) {
#define EP_GEMV_RDL(s) lo
#define EP_GEMV_RDH(s) hi
                EP_STORE_TILE_ROWMAJOR_F64(p64, plo, phi, rp, 0, EP_GEMV_RDL, EP_GEMV_RDH, 0, 1, vb,
                                           va, read_dst, nodes, n_nodes, 0, n0);
#undef EP_GEMV_RDL
#undef EP_GEMV_RDH
            } else {
                lo = svmul_x(p64, lo, vb);
                hi = svmul_x(p64, hi, vb);
                if (read_dst) {
                    lo = svmla_x(p64, lo, svld1_f64(plo, rp), va);
                    hi = svmla_x(p64, hi, svld1_f64(phi, rph), va);
                }
                svst1_f64(plo, rp, lo);
                svst1_f64(phi, rph, hi);
            }
        }
    }
}

// [nt_lo, nt_hi) is the N-tile sub-range this invocation owns -- pass
// (0, n_tiles) for the whole problem, or a slice for the N-parallel flat-M path
// (see the driver). f64 N-tiles are independent (no dual-tile lockstep), so any
// chunk granularity is safe.
__arm_locally_streaming __arm_new("za") static void run_streaming(
    double *dst, long dst_cs, long dst_rs, const double *a_pack, const double *b_pack, size_t m,
    size_t n, size_t mt_lo, size_t mt_hi, size_t nt_lo, size_t nt_hi, size_t k, double alpha,
    double beta, int read_dst, const ep_desc_f64 *ep) {
    svbool_t p64 = svptrue_b64();
    const svfloat64_t z64 = svdup_n_f64(0.0);
    const svfloat64_t va = svdup_n_f64(alpha);
    const svfloat64_t vb = svdup_n_f64(beta);
    size_t per_tile = ep_cmul(k, 8);
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);
    // Fold a leading per-column bias (ADD_COL) into the ZA accumulator via rank-1
    // FMOPAs before the K-loop, then drop it; a bias-only graph then reverts to
    // the direct single-instruction store (free). The f64 accumulator domain
    // matches the bias domain, so this is exact. Requires beta==1 / !read_dst;
    // gated to bias-only (folding bias+activation perturbs store codegen).
    int fold_bias = ep && ep->n_nodes == 1 && beta == 1.0 && !read_dst &&
                    ep->nodes[0].op == EP_OP_ADD_COL;
    const double *fold_bias_ptr = fold_bias ? (const double *)ep->nodes[0].ptr : NULL;
    const ep_desc_f64 *ep_eff = fold_bias ? NULL : ep;
    int has_ep = ep_eff && ep_eff->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep_eff->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep_eff->n_nodes : 0;
    int has_tensor = has_ep && ep_has_tensor(nodes, n_nodes);
    int pure = (beta == 1.0 && !read_dst && !has_ep);

    // Cache blocking (BLIS jc->ic). nc_blk / mc_blk count 16-wide super-tiles
    // (= 2 packed bands each). The jc loop here only blocks within one M-chunk;
    // cross-chunk B reuse comes from the driver running this once per N-block
    // (see gemm_sme_f64_run_packed).
    size_t budget = f64_budget(k);
    size_t nt_span = nt_hi - nt_lo;
    size_t nc_blk = f64_nc_blk(k, nt_span);
    size_t mc_blk = budget / 2;
    if (mc_blk < 1) mc_blk = 1;
    if (mc_blk > mt_hi - mt_lo) mc_blk = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc_blk) {
        size_t jc_end = jc + nc_blk < nt_hi ? jc + nc_blk : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc_blk) {
            size_t ic_end = ic + mc_blk < mt_hi ? ic + mc_blk : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const double *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
                const double *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
                size_t m0 = mt * 16;
                size_t mr = m - m0 < 16 ? m - m0 : 16;
                size_t mr_lo = mr < 8 ? mr : 8;
                size_t mr_hi = mr > 8 ? mr - 8 : 0;
                for (size_t nt = jc; nt < jc_end; nt++) {
                    const double *b_lo = b_pack + (size_t)(2 * nt) * per_tile;
                    const double *b_hi = b_pack + (size_t)(2 * nt + 1) * per_tile;
                    size_t n0 = nt * 16;
                    size_t nc = n - n0 < 16 ? n - n0 : 16;

                    if (fold_bias) {
                        bias_init_za(p64, fold_bias_ptr, n0, n, nc, mr);
                    } else {
                        svzero_za();
                    }
                    size_t d = 0;
                    if (nc <= 8) { // narrow-N: hi-N band is pad, za1/za3 dead
                        for (; d + 2 <= k; d += 2) {
                            svfloat64_t al0 = svld1_f64(p64, a_lo + d * 8);
                            svfloat64_t ah0 = svld1_f64(p64, a_hi + d * 8);
                            svfloat64_t bl0 = svld1_f64(p64, b_lo + d * 8);
                            svfloat64_t al1 = svld1_f64(p64, a_lo + (d + 1) * 8);
                            svfloat64_t ah1 = svld1_f64(p64, a_hi + (d + 1) * 8);
                            svfloat64_t bl1 = svld1_f64(p64, b_lo + (d + 1) * 8);
                            F64_STEP_NARROW_N(al0, ah0, bl0);
                            F64_STEP_NARROW_N(al1, ah1, bl1);
                        }
                        for (; d < k; d++) {
                            svfloat64_t al = svld1_f64(p64, a_lo + d * 8);
                            svfloat64_t ah = svld1_f64(p64, a_hi + d * 8);
                            svfloat64_t bl = svld1_f64(p64, b_lo + d * 8);
                            F64_STEP_NARROW_N(al, ah, bl);
                        }
                    } else if (mr <= 8) { // narrow-M: hi-M band is pad, za2/za3 dead
                        for (; d + 2 <= k; d += 2) {
                            svfloat64_t al0 = svld1_f64(p64, a_lo + d * 8);
                            svfloat64_t bl0 = svld1_f64(p64, b_lo + d * 8);
                            svfloat64_t bh0 = svld1_f64(p64, b_hi + d * 8);
                            svfloat64_t al1 = svld1_f64(p64, a_lo + (d + 1) * 8);
                            svfloat64_t bl1 = svld1_f64(p64, b_lo + (d + 1) * 8);
                            svfloat64_t bh1 = svld1_f64(p64, b_hi + (d + 1) * 8);
                            F64_STEP_NARROW_M(al0, bl0, bh0);
                            F64_STEP_NARROW_M(al1, bl1, bh1);
                        }
                        for (; d < k; d++) {
                            svfloat64_t al = svld1_f64(p64, a_lo + d * 8);
                            svfloat64_t bl = svld1_f64(p64, b_lo + d * 8);
                            svfloat64_t bh = svld1_f64(p64, b_hi + d * 8);
                            F64_STEP_NARROW_M(al, bl, bh);
                        }
                    } else {
                        for (; d + 2 <= k; d += 2) {
                            svfloat64_t al0 = svld1_f64(p64, a_lo + d * 8);
                            svfloat64_t ah0 = svld1_f64(p64, a_hi + d * 8);
                            svfloat64_t bl0 = svld1_f64(p64, b_lo + d * 8);
                            svfloat64_t bh0 = svld1_f64(p64, b_hi + d * 8);
                            svfloat64_t al1 = svld1_f64(p64, a_lo + (d + 1) * 8);
                            svfloat64_t ah1 = svld1_f64(p64, a_hi + (d + 1) * 8);
                            svfloat64_t bl1 = svld1_f64(p64, b_lo + (d + 1) * 8);
                            svfloat64_t bh1 = svld1_f64(p64, b_hi + (d + 1) * 8);
                            F64_STEP_FULL(al0, ah0, bl0, bh0);
                            F64_STEP_FULL(al1, ah1, bl1, bh1);
                        }
                        for (; d < k; d++) {
                            svfloat64_t al = svld1_f64(p64, a_lo + d * 8);
                            svfloat64_t ah = svld1_f64(p64, a_hi + d * 8);
                            svfloat64_t bl = svld1_f64(p64, b_lo + d * 8);
                            svfloat64_t bh = svld1_f64(p64, b_hi + d * 8);
                            F64_STEP_FULL(al, ah, bl, bh);
                        }
                    }

                    if (col_major) {
                        svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)mr_lo);
                        svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)mr_hi);
                        for (size_t c = 0; c < (nc < 8 ? nc : 8); c++) {
                            double *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            // clamp so no one-past-end pointer is formed; the hi-M
                            // band is empty (phi all-false) when narrow-M (mr<=8).
                            double *colh = mr_hi ? col + 8 : col;
                            if (pure) {
                                svst1_ver_za64(0, (uint32_t)c, plo, col);
                                svst1_ver_za64(2, (uint32_t)c, phi, colh);
                            } else {
                                svfloat64_t accl = svread_ver_za64_f64_m(z64, p64, 0, (uint32_t)c);
                                svfloat64_t acch = svread_ver_za64_f64_m(z64, p64, 2, (uint32_t)c);
                                if (has_ep && !has_tensor) {
                                    ep_store_f64(p64, plo, col, accl, vb, va, read_dst, nodes,
                                                 n_nodes, 0, n0 + c, n0, m0);
                                    ep_store_f64(p64, phi, colh, acch, vb, va, read_dst, nodes,
                                                 n_nodes, 0, n0 + c, n0, m0 + 8);
                                } else if (has_ep) {
                                    double tmp[16];
                                    svst1_f64(plo, tmp, accl);
                                    svst1_f64(phi, tmp + 8, acch);
                                    for (size_t r = 0; r < mr; r++) {
                                        double v = tmp[r] * beta;
                                        v = ep_apply_nodes_scalar_f64(nodes, n_nodes, v, m0 + r,
                                                                      n0 + c);
                                        double *cell = col + (long)r * dst_rs;
                                        *cell = read_dst ? alpha * (*cell) + v : v;
                                    }
                                } else {
                                    svfloat64_t lo = svmul_x(p64, accl, vb);
                                    svfloat64_t hi = svmul_x(p64, acch, vb);
                                    if (read_dst) {
                                        lo = svmla_x(p64, lo, svld1_f64(plo, col), va);
                                        hi = svmla_x(p64, hi, svld1_f64(phi, colh), va);
                                    }
                                    svst1_f64(plo, col, lo);
                                    svst1_f64(phi, colh, hi);
                                }
                            }
                        }
                        for (size_t c = 8; c < nc; c++) {
                            uint32_t cc = (uint32_t)(c - 8);
                            double *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            // clamp so no one-past-end pointer is formed; the hi-M
                            // band is empty (phi all-false) when narrow-M (mr<=8).
                            double *colh = mr_hi ? col + 8 : col;
                            if (pure) {
                                svst1_ver_za64(1, cc, plo, col);
                                svst1_ver_za64(3, cc, phi, colh);
                            } else {
                                svfloat64_t accl = svread_ver_za64_f64_m(z64, p64, 1, cc);
                                svfloat64_t acch = svread_ver_za64_f64_m(z64, p64, 3, cc);
                                if (has_ep && !has_tensor) {
                                    ep_store_f64(p64, plo, col, accl, vb, va, read_dst, nodes,
                                                 n_nodes, 0, n0 + c, n0, m0);
                                    ep_store_f64(p64, phi, colh, acch, vb, va, read_dst, nodes,
                                                 n_nodes, 0, n0 + c, n0, m0 + 8);
                                } else if (has_ep) {
                                    double tmp[16];
                                    svst1_f64(plo, tmp, accl);
                                    svst1_f64(phi, tmp + 8, acch);
                                    for (size_t r = 0; r < mr; r++) {
                                        double v = tmp[r] * beta;
                                        v = ep_apply_nodes_scalar_f64(nodes, n_nodes, v, m0 + r,
                                                                      n0 + c);
                                        double *cell = col + (long)r * dst_rs;
                                        *cell = read_dst ? alpha * (*cell) + v : v;
                                    }
                                } else {
                                    svfloat64_t lo = svmul_x(p64, accl, vb);
                                    svfloat64_t hi = svmul_x(p64, acch, vb);
                                    if (read_dst) {
                                        lo = svmla_x(p64, lo, svld1_f64(plo, col), va);
                                        hi = svmla_x(p64, hi, svld1_f64(phi, colh), va);
                                    }
                                    svst1_f64(plo, col, lo);
                                    svst1_f64(phi, colh, hi);
                                }
                            }
                        }
                    } else if (row_major) {
                        svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)(nc < 8 ? nc : 8));
                        svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)(nc > 8 ? nc - 8 : 0));
                        double *rbase = dst + (long)m0 * dst_rs + (long)n0 * dst_cs;
                        if (has_ep) {
                            // NODE-MAJOR: read live ZA half-rows into Z registers in
                            // 4-row blocks and dispatch each node ONCE per block.
#define EP_RDL0(s) svread_hor_za64_f64_m(z64, p64, 0, (s))
#define EP_RDH0(s) svread_hor_za64_f64_m(z64, p64, 1, (s))
                            EP_STORE_TILE_ROWMAJOR_F64(p64, plo, phi, rbase, dst_rs, EP_RDL0, EP_RDH0,
                                                       0, mr_lo, vb, va, read_dst, nodes, n_nodes, m0,
                                                       n0);
#undef EP_RDL0
#undef EP_RDH0
#define EP_RDL2(s) svread_hor_za64_f64_m(z64, p64, 2, (s))
#define EP_RDH2(s) svread_hor_za64_f64_m(z64, p64, 3, (s))
                            EP_STORE_TILE_ROWMAJOR_F64(p64, plo, phi, rbase, dst_rs, EP_RDL2, EP_RDH2,
                                                       8, mr, vb, va, read_dst, nodes, n_nodes, m0,
                                                       n0);
#undef EP_RDL2
#undef EP_RDH2
                        } else
                        for (size_t r = 0; r < mr_lo; r++) {
                            double *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            // clamp so no one-past-end pointer is formed; the hi-N
                            // band is empty (phi all-false) when narrow-N (nc<=8).
                            double *rph = (nc > 8) ? rp + 8 : rp;
                            if (pure) {
                                svst1_hor_za64(0, (uint32_t)r, plo, rp);
                                svst1_hor_za64(1, (uint32_t)r, phi, rph);
                            } else {
                                svfloat64_t accl = svread_hor_za64_f64_m(z64, p64, 0, (uint32_t)r);
                                svfloat64_t acch = svread_hor_za64_f64_m(z64, p64, 1, (uint32_t)r);
                                svfloat64_t lo = svmul_x(p64, accl, vb);
                                svfloat64_t hi = svmul_x(p64, acch, vb);
                                if (read_dst) {
                                    lo = svmla_x(p64, lo, svld1_f64(plo, rp), va);
                                    hi = svmla_x(p64, hi, svld1_f64(phi, rph), va);
                                }
                                svst1_f64(plo, rp, lo);
                                svst1_f64(phi, rph, hi);
                            }
                        }
                        if (!has_ep)
                        for (size_t r = 8; r < mr; r++) {
                            uint32_t rr = (uint32_t)(r - 8);
                            double *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            // clamp so no one-past-end pointer is formed; the hi-N
                            // band is empty (phi all-false) when narrow-N (nc<=8).
                            double *rph = (nc > 8) ? rp + 8 : rp;
                            if (pure) {
                                svst1_hor_za64(2, rr, plo, rp);
                                svst1_hor_za64(3, rr, phi, rph);
                            } else {
                                svfloat64_t accl = svread_hor_za64_f64_m(z64, p64, 2, rr);
                                svfloat64_t acch = svread_hor_za64_f64_m(z64, p64, 3, rr);
                                svfloat64_t lo = svmul_x(p64, accl, vb);
                                svfloat64_t hi = svmul_x(p64, acch, vb);
                                if (read_dst) {
                                    lo = svmla_x(p64, lo, svld1_f64(plo, rp), va);
                                    hi = svmla_x(p64, hi, svld1_f64(phi, rph), va);
                                }
                                svst1_f64(plo, rp, lo);
                                svst1_f64(phi, rph, hi);
                            }
                        }
                    } else {
                        double scratch[16 * 16];
                        svbool_t pg = svptrue_b64();
                        for (uint32_t r = 0; r < 8; r++) {
                            svst1_f64(pg, scratch + (size_t)r * 16,
                                      svread_hor_za64_f64_m(z64, p64, 0, r));
                            svst1_f64(pg, scratch + (size_t)r * 16 + 8,
                                      svread_hor_za64_f64_m(z64, p64, 1, r));
                            svst1_f64(pg, scratch + (size_t)(8 + r) * 16,
                                      svread_hor_za64_f64_m(z64, p64, 2, r));
                            svst1_f64(pg, scratch + (size_t)(8 + r) * 16 + 8,
                                      svread_hor_za64_f64_m(z64, p64, 3, r));
                        }
                        for (size_t r = 0; r < mr; r++) {
                            for (size_t c = 0; c < nc; c++) {
                                double ab = scratch[r * 16 + c] * beta;
                                if (has_ep)
                                    ab = ep_apply_nodes_scalar_f64(nodes, n_nodes, ab, m0 + r,
                                                                   n0 + c);
                                double *cell =
                                    dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                                *cell = read_dst ? alpha * (*cell) + ab : ab;
                            }
                        }
                    }
                }
            }
        }
    }
}

static _Thread_local double *g_apack = NULL;
static _Thread_local size_t g_apack_cap = 0;
static double *apack_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1; // k==0: return a real (tiny) buffer, not a
                               // history-dependent NULL that looks like OOM.
    if (g_apack_cap < bytes) {
        free(g_apack);
        g_apack = (double *)malloc(bytes);
        g_apack_cap = g_apack ? bytes : 0;
    }
    return g_apack;
}

// malloc that never requests 0 bytes. With k==0 (per_tile==0) the pack sizes
// below are 0; a malloc(0) may return NULL, which the OOM checks would misread
// as a real allocation failure (spurious rc=-1 for a valid k==0 GEMM).
static void *xmalloc(size_t bytes) { return malloc(bytes ? bytes : 1); }

// The small/medium f64 path: B is packed per N-tile into a small scratch instead
#include "gemm_f64_small.h"

// Batched f64: `count` independent same-shape GEMMs in ONE streaming session, so
#include "gemm_f64_batched.h"

size_t gemm_sme_f64_packed_b_elems(size_t n, size_t k);
void gemm_sme_f64_packb(double *b_pack, const double *rhs, size_t n, size_t k, long rhs_rs,
                        long rhs_cs);
int gemm_sme_f64_run_packed(size_t m, size_t n, size_t k, double *dst, long dst_cs, long dst_rs,
                            int read_dst, const double *lhs, long lhs_cs, long lhs_rs,
                            const double *b_pack, double alpha, double beta,
                            const ep_desc_f64 *ep);

int gemm_sme_f64_run(size_t m, size_t n, size_t k, double *dst, long dst_cs, long dst_rs,
                     int read_dst, const double *lhs, long lhs_cs, long lhs_rs, const double *rhs,
                     long rhs_cs, long rhs_rs, double alpha, double beta, const ep_desc_f64 *ep) {
    if (m == 0 || n == 0) return 0;
    if (!read_dst) alpha = 0.0;

    size_t m_tiles = (m + 15) / 16;
    size_t per_tile = ep_cmul(k, 8);

    // Small + row-major B: direct B load (no B pack), light dispatch. See gemm_f32.c.
#define PACKA_SMALL(AP, MT0, MT1)                                                                  \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                             \
        size_t r0 = st * 8;                                                                        \
        size_t vr = (r0 < m) ? ((m - r0 < 8) ? (m - r0) : 8) : 0;                                  \
        pack_band((AP) + st * per_tile, vr ? lhs + (long)r0 * lhs_rs : lhs, lhs_rs, lhs_cs, k, vr); \
    }
    // Direct-B gate is B's footprint, and the serial gate a flop floor; see
    // gemm_f32.c for why.
    size_t b_budget = SME_F64_DIRECT_B_BYTES / sizeof(double);
    if (rhs_cs == 1 && ep_cmul(n, k) <= b_budget) {
        size_t sc_chunk = 2;
        size_t sc_n = (m_tiles + sc_chunk - 1) / sc_chunk;
        if (sc_n <= 1 || ep_flops(m, n, k) < SME_F64_SERIAL_FLOPS) {
            double *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
            if (a_pack) {
                PACKA_SMALL(a_pack, 0, m_tiles);
                run_small(dst, dst_cs, dst_rs, a_pack, rhs, rhs_rs, m, n, k, 0, m_tiles, alpha, beta,
                          read_dst, ep);
                return 0;
            }
        } else {
            double *a_pack = (double *)xmalloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
            if (a_pack) {
                // Chunks claimed off a shared cursor, packing kept with compute;
                // see gemm_sme_f32_run for why.
                _Atomic size_t cursor = 0;
                _Atomic size_t *cur = &cursor;
                dispatch_apply(f64_workers(sc_n),
                               dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                               ^(size_t w) {
                                 (void)w;
                                 for (;;) {
                                     size_t ci = atomic_fetch_add_explicit(cur, 1,
                                                                           memory_order_relaxed);
                                     size_t mt0 = ci * sc_chunk;
                                     if (mt0 >= m_tiles) break;
                                     size_t mt1 =
                                         mt0 + sc_chunk < m_tiles ? mt0 + sc_chunk : m_tiles;
                                     PACKA_SMALL(a_pack, mt0, mt1);
                                     run_small(dst, dst_cs, dst_rs, a_pack, rhs, rhs_rs, m, n, k,
                                               mt0, mt1, alpha, beta, read_dst, ep);
                                 }
                               });
                free(a_pack);
                return 0;
            }
        }
    }
#undef PACKA_SMALL

    double *b_pack = (double *)xmalloc(ep_cmul(gemm_sme_f64_packed_b_elems(n, k), sizeof(double)));
    if (!b_pack) return -1;
    gemm_sme_f64_packb(b_pack, rhs, n, k, rhs_rs, rhs_cs);

    int rc = gemm_sme_f64_run_packed(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs,
                                     b_pack, alpha, beta, ep);
    free(b_pack);
    return rc;
}

// Elements in a packed-B buffer for (n, k). Saturating (see gemm_i8i32.c).
size_t gemm_sme_f64_packed_b_elems(size_t n, size_t k) {
    return ep_cmul(2 * ((n + 15) / 16), ep_cmul(k, 8));
}

// Pack B once into a caller-allocated buffer of gemm_sme_f64_packed_b_elems
// doubles: [2*n_tiles][k, 8] bands, the layout run_streaming consumes.
void gemm_sme_f64_packb(double *b_pack, const double *rhs, size_t n, size_t k, long rhs_rs,
                        long rhs_cs) {
    size_t per_tile = ep_cmul(k, 8);
    size_t n_tiles = (n + 15) / 16;
    for (size_t st = 0; st < 2 * n_tiles; st++) {
        size_t c0 = st * 8;
        size_t vc = (c0 < n) ? ((n - c0 < 8) ? (n - c0) : 8) : 0;
        // vc==0 bands are zero-filled without reading src; clamp the base.
        pack_band(b_pack + st * per_tile, vc ? rhs + (long)c0 * rhs_cs : rhs, rhs_cs, rhs_rs, k, vc);
    }
}

// Packed-B f64 GEMM: caller supplies B pre-packed via gemm_sme_f64_packb. No
// run_small arm -- that reads B straight from a row-major rhs.
int gemm_sme_f64_run_packed(size_t m, size_t n, size_t k, double *dst, long dst_cs, long dst_rs,
                            int read_dst, const double *lhs, long lhs_cs, long lhs_rs,
                            const double *b_pack, double alpha, double beta,
                            const ep_desc_f64 *ep) {
    if (m == 0 || n == 0) return 0;
    if (!read_dst) alpha = 0.0;
    size_t m_tiles = (m + 15) / 16;
    size_t n_tiles = (n + 15) / 16;
    size_t per_tile = ep_cmul(k, 8);

#define PACKA_RANGE(AP, MT0, MT1)                                                                  \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                             \
        size_t r0 = st * 8;                                                                        \
        size_t vr = (r0 < m) ? ((m - r0 < 8) ? (m - r0) : 8) : 0;                                  \
        /* vr==0 bands are zero-filled without reading src; clamp the base so no   */               \
        /* out-of-bounds pointer is even formed (UB without a deref), as the B band above. */       \
        pack_band((AP) + st * per_tile, vr ? lhs + (long)r0 * lhs_rs : lhs, lhs_rs, lhs_cs, k, vr); \
    }

    size_t M_CHUNK = 2;
    size_t n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    int big = ep_flops(m, n, k) >= (1u << 21);
    if (m == 1 && dst_cs == 1) {
        double *abc = apack_scratch(ep_cmul(per_tile, sizeof(double)));
        if (!abc) return -1;
        for (size_t d = 0; d < k; d++) {
            float64x2_t v = vdupq_n_f64(lhs[(long)d * lhs_cs]);
            float64x2x4_t q = {{v, v, v, v}};
            vst1q_f64_x4(abc + d * 8, q);
        }
        size_t G_CHUNK = 4; // tiles per chunk
        size_t g_chunks = (n_tiles + G_CHUNK - 1) / G_CHUNK;
        if (!big || g_chunks < 3) {
            run_gemv(dst, abc, b_pack, n, k, 0, n_tiles, alpha, beta, read_dst, ep);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t nt0 = ci * G_CHUNK;
          size_t nt1 = nt0 + G_CHUNK < n_tiles ? nt0 + G_CHUNK : n_tiles;
          run_gemv(dst, abc, b_pack, n, k, nt0, nt1, alpha, beta, read_dst, ep);
        });
        return 0;
    }

    // Flat-M (few M-tiles) but large/wide: M is the only M-chunk axis, so the
    // M-parallel scheme below would run this on ONE cluster. Parallelize over N
    // instead -- pack all of A once (cheap; m_tiles is small) into a shared buffer
    // and hand N-tile chunks to both clusters. C columns are disjoint per chunk;
    // A and B are read-only and shared. f64 N-tiles carry no cross-tile state, so
    // any chunk granularity is safe. Needs >= 2 chunks to beat the serial path.
    // (Mirrors the f32 driver.)
    size_t N_CHUNK = 4;
    size_t nn_chunks = (n_tiles + N_CHUNK - 1) / N_CHUNK;
    if (n_chunks <= 1 && big && nn_chunks >= 2) {
        double *a_pack = (double *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
        if (a_pack) {
            PACKA_RANGE(a_pack, 0, m_tiles);
            dispatch_apply(nn_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
              size_t nt0 = ci * N_CHUNK;
              size_t nt1 = nt0 + N_CHUNK < n_tiles ? nt0 + N_CHUNK : n_tiles;
              run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, nt0, nt1, k, alpha,
                            beta, read_dst, ep);
            });
            free(a_pack);
            return 0;
        }
        // malloc failed: fall through to the serial path.
    }
    if (n_chunks <= 1 || !big) {
        double *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
        if (!a_pack) return -1;
        PACKA_RANGE(a_pack, 0, m_tiles);
        run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, 0, n_tiles, k, alpha,
                      beta, read_dst, ep);
        return 0;
    }
    double *a_pack = (double *)xmalloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
    if (!a_pack) return -1;

    // N-block loop OUTSIDE the M dispatch, so one B-block serves every M-chunk
    // while it is still L2-resident; see gemm_f32.c for the traffic argument.
    size_t nc_blk = f64_nc_blk(k, n_tiles);
    size_t n_blocks = (n_tiles + nc_blk - 1) / nc_blk;
    if (n_blocks > 1)
        dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t mt0 = ci * M_CHUNK;
          PACKA_RANGE(a_pack, mt0, mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles);
        });
    // Flat dispatch over (N-block, M-chunk), block-major; see gemm_i16i64.c.
    dispatch_apply(n_blocks * n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t it) {
      size_t ci = it % n_chunks, jc = it / n_chunks * nc_blk;
      size_t mt0 = ci * M_CHUNK;
      size_t mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
      size_t jc_end = jc + nc_blk < n_tiles ? jc + nc_blk : n_tiles;
      if (n_blocks == 1) PACKA_RANGE(a_pack, mt0, mt1);
      run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, mt0, mt1, jc, jc_end, k, alpha, beta,
                    read_dst, ep);
    });
#undef PACKA_RANGE
    free(a_pack);
    return 0;
}
