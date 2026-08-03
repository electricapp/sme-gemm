// SME f32 GEMM driver (Apple M4+, FMOPA single-precision). 32x32 super-tile =
// four 16x16 ZA32 quadrants; one K-value per MOPA (no widening, no zip). fp32
// in/out, so the epilogue is a direct ZA->memory store (pure C=A@B) or a
// read/scale/store. Packing is plain [k, 16] per 16-wide band.
//
// Computes dst = alpha*dst + beta*(A @ B).

#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "attention.h"
#include "epilogue.h"

// Pack one 16-wide band into [k, 16]: dst[d*16 + i] = src[i, d]. Two NEON fast
// paths for the full-16-lane case (a scalar transpose was the dominant
// small/medium-tile pack cost):
//   * lanes contiguous (B, lane_stride==1): each depth is a 16-float contiguous
//     run -> straight copy per depth.
//   * depth contiguous (row-major A, depth_stride==1): 4x4 f32 NEON transpose
//     turns 4 rows x 4 depths into 4 packed depth-vectors.
static void pack_band(float *dst, const float *src, long lane_stride, long depth_stride, size_t k,
                      size_t valid) {
    if (valid == 16 && lane_stride == 1) {
        for (size_t d = 0; d < k; d++)
            memcpy(dst + d * 16, src + (long)d * depth_stride, 16 * sizeof(float));
        return;
    }
    if (valid == 16 && depth_stride == 1) {
        // DEPTH-OUTER. Each 4x4 transpose stores 16 bytes at a 64-byte stride, so
        // with the lane group outer the loop makes four passes over the whole
        // k*16 panel, touching every destination line four times. Depth-outer
        // means the four lane groups of one depth fill that line back to back and
        // the panel is walked once.
        size_t kf = k & ~(size_t)3;
        for (size_t d = 0; d < kf; d += 4) {
            for (size_t rg = 0; rg < 16; rg += 4) {
                float32x4_t r0 = vld1q_f32(src + (long)(rg + 0) * lane_stride + d);
                float32x4_t r1 = vld1q_f32(src + (long)(rg + 1) * lane_stride + d);
                float32x4_t r2 = vld1q_f32(src + (long)(rg + 2) * lane_stride + d);
                float32x4_t r3 = vld1q_f32(src + (long)(rg + 3) * lane_stride + d);
                float32x4x2_t t01 = vtrnq_f32(r0, r1);
                float32x4x2_t t23 = vtrnq_f32(r2, r3);
                float32x4_t c0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
                float32x4_t c1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
                float32x4_t c2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
                float32x4_t c3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
                vst1q_f32(dst + (d + 0) * 16 + rg, c0);
                vst1q_f32(dst + (d + 1) * 16 + rg, c1);
                vst1q_f32(dst + (d + 2) * 16 + rg, c2);
                vst1q_f32(dst + (d + 3) * 16 + rg, c3);
            }
        }
        for (size_t d = kf; d < k; d++)
            for (size_t i = 0; i < 16; i++)
                dst[d * 16 + i] = src[(long)i * lane_stride + d];
        return;
    }
    for (size_t d = 0; d < k; d++)
        for (size_t i = 0; i < 16; i++)
            dst[d * 16 + i] =
                (i < valid) ? src[(long)i * lane_stride + (long)d * depth_stride] : 0.0f;
}

// Zero the four ZA32 sub-tiles then initialize ZA[r][c] = bias[c] via rank-1
// FMOPAs (ones[r] * bias[c]). The 32x32 output is held as four 16x16 tiles:
// 0/1 = rows 0..15 x cols {0..15, 16..31}, 2/3 = rows 16..31 x same cols. The
// lo bias half (cols n0..n0+15) inits tiles 0,2; the hi half (n0+16..n0+31)
// inits tiles 1,3. Out of line so the fold does not perturb the store codegen.
__attribute__((noinline)) static void bias_init_za(svbool_t p32, const float *bias, size_t n0,
                                                   size_t n, size_t nc, size_t mr)
    __arm_streaming __arm_inout("za") {
    svzero_za();
    svfloat32_t v_ones = svdup_n_f32(1.0f);
    size_t nlo = n0, nhi = n0 + 16;
    svbool_t plo = (nlo < n) ? svwhilelt_b32((uint64_t)nlo, (uint64_t)n) : svpfalse_b();
    svbool_t phi = (nhi < n) ? svwhilelt_b32((uint64_t)nhi, (uint64_t)n) : svpfalse_b();
    // Clamp the bias addresses for the all-false (pad) halves: no lane is read,
    // but forming bias + nlo/nhi past one-past-end would still be UB.
    svfloat32_t blo = svld1_f32(plo, (nlo < n) ? bias + nlo : bias);
    svfloat32_t bhi = svld1_f32(phi, (nhi < n) ? bias + nhi : bias);
    // svzero_za already cleared every tile; only seed the LIVE quadrants the store
    // will read. Narrow-N (nc<=16) leaves za1/za3 dead; narrow-M (mr<=16) za2/za3 --
    // skip their bias MOPAs (mirrors the F32_STEP_NARROW_* K-loop dispatch).
    int hi_n = nc > 16, hi_m = mr > 16;
    svmopa_za32_f32_m(0, p32, plo, v_ones, blo);
    if (hi_n) svmopa_za32_f32_m(1, p32, phi, v_ones, bhi);
    if (hi_m) svmopa_za32_f32_m(2, p32, plo, v_ones, blo);
    if (hi_n && hi_m) svmopa_za32_f32_m(3, p32, phi, v_ones, bhi);
}

// One K-step's MOPAs for the live ZA quadrants of the 32x32 super-tile
// (0=lo-M x lo-N, 1=lo-M x hi-N, 2=hi-M x lo-N, 3=hi-M x hi-N). A NARROW-N tile
// (nc <= 16) has the hi-N band all zero-pad, so za1/za3 are dead; a NARROW-M
// tile (mr <= 16) has the hi-M band all-pad, so za2/za3 are dead. Issuing only
// the live quadrants halves MOPA issue on those decode/GEMV shapes (the store,
// which reads ZA by nc/mr predicates, is unaffected). The dispatch is hoisted
// out of the K-loop, so the inner loop stays branch-free.
#define F32_STEP_FULL(al, ah, bl, bh)                                                              \
    svmopa_za32_f32_m(0, p32, p32, al, bl);                                                        \
    svmopa_za32_f32_m(1, p32, p32, al, bh);                                                        \
    svmopa_za32_f32_m(2, p32, p32, ah, bl);                                                        \
    svmopa_za32_f32_m(3, p32, p32, ah, bh)
#define F32_STEP_NARROW_N(al, ah, bl)                                                              \
    svmopa_za32_f32_m(0, p32, p32, al, bl);                                                        \
    svmopa_za32_f32_m(2, p32, p32, ah, bl)
#define F32_STEP_NARROW_M(al, bl, bh)                                                              \
    svmopa_za32_f32_m(0, p32, p32, al, bl);                                                        \
    svmopa_za32_f32_m(1, p32, p32, al, bh)

// [nt_lo, nt_hi) is the N-tile sub-range this invocation owns -- pass
// (0, n_tiles) for the whole problem, or a slice for the N-parallel flat-M path
// (see the driver). f32 N-tiles are independent (no dual-tile lockstep), so any
// chunk granularity is safe.
__arm_locally_streaming __arm_new("za") static void run_streaming(
    float *dst, long dst_cs, long dst_rs, const float *a_pack, const float *b_pack, size_t m,
    size_t n, size_t mt_lo, size_t mt_hi, size_t nt_lo, size_t nt_hi, size_t k, float alpha,
    float beta, int read_dst, const ep_desc_f32 *ep) {
    svbool_t p32 = svptrue_b32();
    const svfloat32_t z32 = svdup_n_f32(0.0f);
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    size_t per_tile = ep_cmul(k, 16);
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);
    // Fold a leading per-column bias (ADD_COL) into the ZA accumulator via rank-1
    // FMOPAs before the K-loop, then drop it; a bias-only graph then reverts to
    // the direct single-instruction store (free). The f32 accumulator domain
    // matches the bias domain, so this is exact. Requires beta==1 / !read_dst;
    // gated to bias-only (folding bias+activation perturbs store codegen).
    int fold_bias = ep && ep->n_nodes == 1 && beta == 1.0f && !read_dst &&
                    ep->nodes[0].op == EP_OP_ADD_COL;
    const float *fold_bias_ptr = fold_bias ? (const float *)ep->nodes[0].ptr : NULL;
    const ep_desc_f32 *ep_eff = fold_bias ? NULL : ep;
    int has_ep = ep_eff && ep_eff->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep_eff->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep_eff->n_nodes : 0;
    int has_tensor = has_ep && ep_has_tensor(nodes, n_nodes);
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    int has_reduce = ep_has_reduce(ep);

    // Cache blocking (BLIS jc->ic): keep the B-block resident in L2 so each B
    // panel streams from DRAM once per M-chunk instead of once per M-tile.
    // nc_blk/mc_blk count 32-wide SUPER-tiles (= 2 packed bands each), so the
    // budget CONSTANT here equals the total resident bytes (A-block + B-block):
    // resident = budget * super_bytes = the constant. mc_blk is additionally
    // capped at the M-chunk (M_CHUNK super-tiles), so in the parallel path only
    // nc_blk grows -- a wider B-block cuts A re-streaming. Cold 4096^3 f32 peaks
    // at a 24 MB working set (16 MB -> 76%, 24 -> 79%, 32 -> 71% as the B-block
    // spills the ~16 MB P-cluster L2). Whole problem -> single block (no
    // overhead). f32 panels are 2x the bytes of f16, so this path needs the
    // larger budget the f16 driver (8 MB, where the tile IS the super-tile)
    // doesn't.
    size_t tile_bytes = 2 * per_tile * sizeof(float);
    size_t budget = (size_t)(24u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    if (budget < 2) budget = 2;
    size_t nt_span = nt_hi - nt_lo;
    size_t nc_blk = budget / 2;
    if (nc_blk < 1) nc_blk = 1;
    if (nc_blk > nt_span || has_reduce) nc_blk = nt_span;
    size_t mc_blk = budget / 2;
    if (mc_blk < 1) mc_blk = 1;
    if (mc_blk > mt_hi - mt_lo) mc_blk = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc_blk) {
        size_t jc_end = jc + nc_blk < nt_hi ? jc + nc_blk : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc_blk) {
            size_t ic_end = ic + mc_blk < mt_hi ? ic + mc_blk : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const float *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
                const float *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
                size_t m0 = mt * 32;
                size_t mr = m - m0 < 32 ? m - m0 : 32;
                size_t mr_lo = mr < 16 ? mr : 16;
                size_t mr_hi = mr > 16 ? mr - 16 : 0;
                for (size_t nt = jc; nt < jc_end; nt++) {
                    const float *b_lo = b_pack + (size_t)(2 * nt) * per_tile;
                    const float *b_hi = b_pack + (size_t)(2 * nt + 1) * per_tile;
                    size_t n0 = nt * 32;
                    size_t nc = n - n0 < 32 ? n - n0 : 32;

                    if (fold_bias) {
                        bias_init_za(p32, fold_bias_ptr, n0, n, nc, mr);
                    } else {
                        svzero_za();
                    }
                    size_t d = 0;
                    if (nc <= 16) { // narrow-N: hi-N band is pad, za1/za3 dead
                        for (; d + 2 <= k; d += 2) {
                            svfloat32_t al0 = svld1_f32(p32, a_lo + d * 16);
                            svfloat32_t ah0 = svld1_f32(p32, a_hi + d * 16);
                            svfloat32_t bl0 = svld1_f32(p32, b_lo + d * 16);
                            svfloat32_t al1 = svld1_f32(p32, a_lo + (d + 1) * 16);
                            svfloat32_t ah1 = svld1_f32(p32, a_hi + (d + 1) * 16);
                            svfloat32_t bl1 = svld1_f32(p32, b_lo + (d + 1) * 16);
                            F32_STEP_NARROW_N(al0, ah0, bl0);
                            F32_STEP_NARROW_N(al1, ah1, bl1);
                        }
                        for (; d < k; d++) {
                            svfloat32_t al = svld1_f32(p32, a_lo + d * 16);
                            svfloat32_t ah = svld1_f32(p32, a_hi + d * 16);
                            svfloat32_t bl = svld1_f32(p32, b_lo + d * 16);
                            F32_STEP_NARROW_N(al, ah, bl);
                        }
                    } else if (mr <= 16) { // narrow-M: hi-M band is pad, za2/za3 dead
                        for (; d + 2 <= k; d += 2) {
                            svfloat32_t al0 = svld1_f32(p32, a_lo + d * 16);
                            svfloat32_t bl0 = svld1_f32(p32, b_lo + d * 16);
                            svfloat32_t bh0 = svld1_f32(p32, b_hi + d * 16);
                            svfloat32_t al1 = svld1_f32(p32, a_lo + (d + 1) * 16);
                            svfloat32_t bl1 = svld1_f32(p32, b_lo + (d + 1) * 16);
                            svfloat32_t bh1 = svld1_f32(p32, b_hi + (d + 1) * 16);
                            F32_STEP_NARROW_M(al0, bl0, bh0);
                            F32_STEP_NARROW_M(al1, bl1, bh1);
                        }
                        for (; d < k; d++) {
                            svfloat32_t al = svld1_f32(p32, a_lo + d * 16);
                            svfloat32_t bl = svld1_f32(p32, b_lo + d * 16);
                            svfloat32_t bh = svld1_f32(p32, b_hi + d * 16);
                            F32_STEP_NARROW_M(al, bl, bh);
                        }
                    } else {
                        for (; d + 2 <= k; d += 2) {
                            svfloat32_t al0 = svld1_f32(p32, a_lo + d * 16);
                            svfloat32_t ah0 = svld1_f32(p32, a_hi + d * 16);
                            svfloat32_t bl0 = svld1_f32(p32, b_lo + d * 16);
                            svfloat32_t bh0 = svld1_f32(p32, b_hi + d * 16);
                            svfloat32_t al1 = svld1_f32(p32, a_lo + (d + 1) * 16);
                            svfloat32_t ah1 = svld1_f32(p32, a_hi + (d + 1) * 16);
                            svfloat32_t bl1 = svld1_f32(p32, b_lo + (d + 1) * 16);
                            svfloat32_t bh1 = svld1_f32(p32, b_hi + (d + 1) * 16);
                            F32_STEP_FULL(al0, ah0, bl0, bh0);
                            F32_STEP_FULL(al1, ah1, bl1, bh1);
                        }
                        for (; d < k; d++) {
                            svfloat32_t al = svld1_f32(p32, a_lo + d * 16);
                            svfloat32_t ah = svld1_f32(p32, a_hi + d * 16);
                            svfloat32_t bl = svld1_f32(p32, b_lo + d * 16);
                            svfloat32_t bh = svld1_f32(p32, b_hi + d * 16);
                            F32_STEP_FULL(al, ah, bl, bh);
                        }
                    }

                    if (col_major) {
                        svbool_t plo = svwhilelt_b32((uint32_t)0, (uint32_t)mr_lo);
                        svbool_t phi = svwhilelt_b32((uint32_t)0, (uint32_t)mr_hi);
                        // tile pair (0,2) for N0..15, (1,3) for N16..31; constant tiles.
                        for (size_t c = 0; c < (nc < 16 ? nc : 16); c++) {
                            float *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            // clamp so no one-past-end pointer is formed; the hi-M
                            // band is empty (phi all-false) when narrow-M (mr<=16).
                            float *colh = mr_hi ? col + 16 : col;
                            if (pure) {
                                svst1_ver_za32(0, (uint32_t)c, plo, col);
                                svst1_ver_za32(2, (uint32_t)c, phi, colh);
                            } else {
                                svfloat32_t accl = svread_ver_za32_m(z32, p32, 0, (uint32_t)c);
                                svfloat32_t acch = svread_ver_za32_m(z32, p32, 2, (uint32_t)c);
                                if (has_ep && !has_tensor) {
                                    ep_store_f32(p32, plo, col, accl, vb, va, read_dst, nodes,
                                                 n_nodes, 0, n0 + c, n0, m0);
                                    ep_store_f32(p32, phi, colh, acch, vb, va, read_dst, nodes,
                                                 n_nodes, 0, n0 + c, n0, m0 + 16);
                                } else if (has_ep) {
                                    float tmp[32];
                                    svst1_f32(plo, tmp, accl);
                                    svst1_f32(phi, tmp + 16, acch);
                                    for (size_t r = 0; r < mr; r++) {
                                        float v = tmp[r] * beta;
                                        v = ep_apply_nodes_scalar_f32(nodes, n_nodes, v, m0 + r,
                                                                      n0 + c);
                                        float *cell = col + (long)r * dst_rs;
                                        *cell = read_dst ? alpha * (*cell) + v : v;
                                    }
                                } else {
                                    svfloat32_t lo = svmul_x(p32, accl, vb);
                                    svfloat32_t hi = svmul_x(p32, acch, vb);
                                    if (read_dst) {
                                        lo = svmla_x(p32, lo, svld1_f32(plo, col), va);
                                        hi = svmla_x(p32, hi, svld1_f32(phi, colh), va);
                                    }
                                    svst1_f32(plo, col, lo);
                                    svst1_f32(phi, colh, hi);
                                }
                            }
                        }
                        for (size_t c = 16; c < nc; c++) {
                            uint32_t cc = (uint32_t)(c - 16);
                            float *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            // clamp so no one-past-end pointer is formed; the hi-M
                            // band is empty (phi all-false) when narrow-M (mr<=16).
                            float *colh = mr_hi ? col + 16 : col;
                            if (pure) {
                                svst1_ver_za32(1, cc, plo, col);
                                svst1_ver_za32(3, cc, phi, colh);
                            } else {
                                svfloat32_t accl = svread_ver_za32_m(z32, p32, 1, cc);
                                svfloat32_t acch = svread_ver_za32_m(z32, p32, 3, cc);
                                if (has_ep && !has_tensor) {
                                    ep_store_f32(p32, plo, col, accl, vb, va, read_dst, nodes,
                                                 n_nodes, 0, n0 + c, n0, m0);
                                    ep_store_f32(p32, phi, colh, acch, vb, va, read_dst, nodes,
                                                 n_nodes, 0, n0 + c, n0, m0 + 16);
                                } else if (has_ep) {
                                    float tmp[32];
                                    svst1_f32(plo, tmp, accl);
                                    svst1_f32(phi, tmp + 16, acch);
                                    for (size_t r = 0; r < mr; r++) {
                                        float v = tmp[r] * beta;
                                        v = ep_apply_nodes_scalar_f32(nodes, n_nodes, v, m0 + r,
                                                                      n0 + c);
                                        float *cell = col + (long)r * dst_rs;
                                        *cell = read_dst ? alpha * (*cell) + v : v;
                                    }
                                } else {
                                    svfloat32_t lo = svmul_x(p32, accl, vb);
                                    svfloat32_t hi = svmul_x(p32, acch, vb);
                                    if (read_dst) {
                                        lo = svmla_x(p32, lo, svld1_f32(plo, col), va);
                                        hi = svmla_x(p32, hi, svld1_f32(phi, colh), va);
                                    }
                                    svst1_f32(plo, col, lo);
                                    svst1_f32(phi, colh, hi);
                                }
                            }
                        }
                    } else if (row_major) {
                        svbool_t plo = svwhilelt_b32((uint32_t)0, (uint32_t)(nc < 16 ? nc : 16));
                        svbool_t phi =
                            svwhilelt_b32((uint32_t)0, (uint32_t)(nc > 16 ? nc - 16 : 0));
                        float *rbase = dst + (long)m0 * dst_rs + (long)n0 * dst_cs;
                        if (has_ep) {
                            // NODE-MAJOR: read the live ZA half-rows directly into Z
                            // registers in 4-row blocks (lo from tile 0/2, hi from
                            // tile 1/3) and dispatch each op-graph node ONCE per block
                            // across all halves; invariant operands loaded once/node.
#define EP_RDL0(s) svread_hor_za32_m(z32, p32, 0, (s))
#define EP_RDH0(s) svread_hor_za32_m(z32, p32, 1, (s))
                            EP_STORE_TILE_ROWMAJOR_F32(p32, plo, phi, rbase, dst_rs, EP_RDL0, EP_RDH0,
                                                       0, mr_lo, vb, va, read_dst, nodes, n_nodes, m0,
                                                       n0);
#undef EP_RDL0
#undef EP_RDH0
#define EP_RDL2(s) svread_hor_za32_m(z32, p32, 2, (s))
#define EP_RDH2(s) svread_hor_za32_m(z32, p32, 3, (s))
                            EP_STORE_TILE_ROWMAJOR_F32(p32, plo, phi, rbase, dst_rs, EP_RDL2, EP_RDH2,
                                                       16, mr, vb, va, read_dst, nodes, n_nodes, m0,
                                                       n0);
#undef EP_RDL2
#undef EP_RDH2
                        } else
                        for (size_t r = 0; r < mr_lo; r++) {
                            float *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            // clamp so no one-past-end pointer is formed; the hi-N
                            // band is empty (phi all-false) when narrow-N (nc<=16).
                            float *rph = (nc > 16) ? rp + 16 : rp;
                            if (pure) {
                                svst1_hor_za32(0, (uint32_t)r, plo, rp);
                                svst1_hor_za32(1, (uint32_t)r, phi, rph);
                            } else {
                                svfloat32_t accl = svread_hor_za32_m(z32, p32, 0, (uint32_t)r);
                                svfloat32_t acch = svread_hor_za32_m(z32, p32, 1, (uint32_t)r);
                                svfloat32_t lo = svmul_x(p32, accl, vb);
                                svfloat32_t hi = svmul_x(p32, acch, vb);
                                if (read_dst) {
                                    lo = svmla_x(p32, lo, svld1_f32(plo, rp), va);
                                    hi = svmla_x(p32, hi, svld1_f32(phi, rph), va);
                                }
                                svst1_f32(plo, rp, lo);
                                svst1_f32(phi, rph, hi);
                            }
                        }
                        if (!has_ep)
                        for (size_t r = 16; r < mr; r++) {
                            uint32_t rr = (uint32_t)(r - 16);
                            float *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            // clamp so no one-past-end pointer is formed; the hi-N
                            // band is empty (phi all-false) when narrow-N (nc<=16).
                            float *rph = (nc > 16) ? rp + 16 : rp;
                            if (pure) {
                                svst1_hor_za32(2, rr, plo, rp);
                                svst1_hor_za32(3, rr, phi, rph);
                            } else {
                                svfloat32_t accl = svread_hor_za32_m(z32, p32, 2, rr);
                                svfloat32_t acch = svread_hor_za32_m(z32, p32, 3, rr);
                                svfloat32_t lo = svmul_x(p32, accl, vb);
                                svfloat32_t hi = svmul_x(p32, acch, vb);
                                if (read_dst) {
                                    lo = svmla_x(p32, lo, svld1_f32(plo, rp), va);
                                    hi = svmla_x(p32, hi, svld1_f32(phi, rph), va);
                                }
                                svst1_f32(plo, rp, lo);
                                svst1_f32(phi, rph, hi);
                            }
                        }
                    } else {
                        float scratch[32 * 32];
                        svbool_t pg = svptrue_b32();
                        for (uint32_t r = 0; r < 16; r++) {
                            svst1_f32(pg, scratch + (size_t)r * 32,
                                      svread_hor_za32_m(z32, p32, 0, r));
                            svst1_f32(pg, scratch + (size_t)r * 32 + 16,
                                      svread_hor_za32_m(z32, p32, 1, r));
                            svst1_f32(pg, scratch + (size_t)(16 + r) * 32,
                                      svread_hor_za32_m(z32, p32, 2, r));
                            svst1_f32(pg, scratch + (size_t)(16 + r) * 32 + 16,
                                      svread_hor_za32_m(z32, p32, 3, r));
                        }
                        for (size_t r = 0; r < mr; r++) {
                            for (size_t c = 0; c < nc; c++) {
                                float ab = scratch[r * 32 + c] * beta;
                                if (has_ep)
                                    ab = ep_apply_nodes_scalar_f32(nodes, n_nodes, ab, m0 + r,
                                                                   n0 + c);
                                float *cell =
                                    dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                                *cell = read_dst ? alpha * (*cell) + ab : ab;
                            }
                        }
                    }
                } // nt
                if (has_reduce) {
                    // ONE horizontal reduce per row, over the whole finished row,
                    // right after this M-tile's last N-tile wrote it -- so the row
                    // is still L1-resident. Doing it per (row, N-tile) instead was
                    // measured 2.2x slower than the GEMM itself: svaddv/svmaxv are
                    // serializing cross-lane ops, and n_tiles of them per row
                    // dwarfs the sweep they save. has_reduce forces a single
                    // N-block, so [nt_lo, nt_hi) here is the complete row.
                    for (size_t r = 0; r < mr; r++) {
                        const float *rp = dst + (long)(m0 + r) * dst_rs;
                        // Accumulate into VECTORS and reduce across lanes exactly
                        // once per row. Calling svaddv/svmaxv per 16-lane chunk
                        // instead builds an n/16-long serial dependency chain of
                        // cross-lane ops -- measured 3.5x the whole GEMM.
                        svfloat32_t vs = svdup_n_f32(0.0f);
                        svfloat32_t vm = svdup_n_f32(-INFINITY);
                        size_t j = 0;
                        for (; j + 16 <= n; j += 16) {
                            svfloat32_t v = svld1_f32(p32, rp + j);
                            vs = svadd_f32_x(p32, vs, v);
                            vm = svmaxnm_f32_x(p32, vm, v);
                        }
                        if (j < n) {
                            svbool_t pt = svwhilelt_b32((uint32_t)j, (uint32_t)n);
                            svfloat32_t v = svld1_f32(pt, rp + j);
                            // Identity in the inactive lanes so the tail cannot
                            // perturb either accumulator.
                            vs = svadd_f32_m(pt, vs, v);
                            vm = svmaxnm_f32_m(pt, vm, v);
                        }
                        if (ep->row_sum) ep->row_sum[m0 + r] = svaddv_f32(p32, vs);
                        if (ep->row_max) ep->row_max[m0 + r] = svmaxv_f32(p32, vm);
                    }
                }
            }
        }
    }
}

// Per-thread reusable A-pack scratch (no per-call malloc/free on the serial
// small/medium path). Grows monotonically, lives for the thread's lifetime.
static _Thread_local float *g_apack = NULL;
static _Thread_local size_t g_apack_cap = 0;
static float *apack_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1; // k==0: return a real (tiny) buffer, not a
                               // history-dependent NULL on fresh threads
    if (g_apack_cap < bytes) {
        free(g_apack);
        g_apack = (float *)malloc(bytes);
        g_apack_cap = g_apack ? bytes : 0;
    }
    return g_apack;
}

// The small/medium f32 path: B is packed per N-tile into a small scratch instead
#include "gemm_f32_small.h"

// Batched f32: `count` independent same-shape GEMMs in ONE streaming session, so
#include "gemm_f32_batched.h"

size_t gemm_sme_f32_packed_b_elems(size_t n, size_t k);
void gemm_sme_f32_packb(float *b_pack, const float *rhs, size_t n, size_t k, long rhs_rs,
                        long rhs_cs);
int gemm_sme_f32_run_packed(size_t m, size_t n, size_t k, float *dst, long dst_cs, long dst_rs,
                            int read_dst, const float *lhs, long lhs_cs, long lhs_rs,
                            const float *b_pack, float alpha, float beta, const ep_desc_f32 *ep);

int gemm_sme_f32_run(size_t m, size_t n, size_t k, float *dst, long dst_cs, long dst_rs,
                     int read_dst, const float *lhs, long lhs_cs, long lhs_rs, const float *rhs,
                     long rhs_cs, long rhs_rs, float alpha, float beta, const ep_desc_f32 *ep) {
    if (m == 0 || n == 0) return 0;
    if (!read_dst) alpha = 0.0f;

    size_t m_tiles = (m + 31) / 32;
    size_t per_tile = ep_cmul(k, 16);

    // Small + row-major B: load B direct (no B pack -- B[d,n0..] is contiguous).
    // Serial when a single chunk; otherwise a light dispatch over both clusters
    // (no B pack and a shared A-pack, so much cheaper than the main path's
    // per-call B packing). The full parallel path's overhead dominates here.
#define PACKA_SMALL(AP, MT0, MT1)                                                                   \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                             \
        size_t r0 = st * 16;                                                                        \
        size_t vr = (r0 < m) ? ((m - r0 < 16) ? (m - r0) : 16) : 0;                                 \
        pack_band((AP) + st * per_tile, vr ? lhs + (long)r0 * lhs_rs : lhs, lhs_rs, lhs_cs, k, vr);  \
    }
    // run_small has no reduction block; reductions take run_streaming.
    if (rhs_cs == 1 && !ep_has_reduce(ep) && ep_flops(m, n, k) < (1u << 25)) {
        size_t sc_chunk = 2;
        size_t sc_n = (m_tiles + sc_chunk - 1) / sc_chunk;
        if (sc_n <= 1) {
            float *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
            if (a_pack) {
                PACKA_SMALL(a_pack, 0, m_tiles);
                run_small(dst, dst_cs, dst_rs, a_pack, rhs, rhs_rs, m, n, k, 0, m_tiles, alpha, beta,
                          read_dst, ep);
                return 0;
            }
        } else {
            float *a_pack = (float *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
            if (a_pack) {
                dispatch_apply(sc_n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                               ^(size_t ci) {
                                 size_t mt0 = ci * sc_chunk;
                                 size_t mt1 = mt0 + sc_chunk < m_tiles ? mt0 + sc_chunk : m_tiles;
                                 PACKA_SMALL(a_pack, mt0, mt1);
                                 run_small(dst, dst_cs, dst_rs, a_pack, rhs, rhs_rs, m, n, k, mt0,
                                           mt1, alpha, beta, read_dst, ep);
                               });
                free(a_pack);
                return 0;
            }
        }
    }
#undef PACKA_SMALL

    // B packed once, shared across chunks (the reused operand).
    float *b_pack = (float *)malloc(ep_cmul(gemm_sme_f32_packed_b_elems(n, k), sizeof(float)));
    if (!b_pack) return -1;
    gemm_sme_f32_packb(b_pack, rhs, n, k, rhs_rs, rhs_cs);

    int rc = gemm_sme_f32_run_packed(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs,
                                     b_pack, alpha, beta, ep);
    free(b_pack);
    return rc;
}

// Elements in a packed-B buffer for (n, k). Saturating (see gemm_i8i32.c).
size_t gemm_sme_f32_packed_b_elems(size_t n, size_t k) {
    return ep_cmul(2 * ((n + 31) / 32), ep_cmul(k, 16));
}

// Pack B once into a caller-allocated buffer of gemm_sme_f32_packed_b_elems
// floats: [2*n_tiles][k, 16] bands, the layout run_streaming consumes.
void gemm_sme_f32_packb(float *b_pack, const float *rhs, size_t n, size_t k, long rhs_rs,
                        long rhs_cs) {
    size_t per_tile = ep_cmul(k, 16);
    size_t n_tiles = (n + 31) / 32;
    for (size_t st = 0; st < 2 * n_tiles; st++) {
        size_t c0 = st * 16;
        size_t vc = (c0 < n) ? ((n - c0 < 16) ? (n - c0) : 16) : 0;
        // vc==0 bands are zero-filled without reading src; clamp the base so no
        // out-of-bounds pointer is even formed.
        pack_band(b_pack + st * per_tile, vc ? rhs + (long)c0 * rhs_cs : rhs, rhs_cs, rhs_rs, k, vc);
    }
}

// Packed-B f32 GEMM: caller supplies B pre-packed via gemm_sme_f32_packb, so only
// A is packed per call. No run_small arm -- that one reads B straight from a
// row-major rhs, which a prepacked panel is not.
int gemm_sme_f32_run_packed(size_t m, size_t n, size_t k, float *dst, long dst_cs, long dst_rs,
                            int read_dst, const float *lhs, long lhs_cs, long lhs_rs,
                            const float *b_pack, float alpha, float beta, const ep_desc_f32 *ep) {
    if (m == 0 || n == 0) return 0;
    if (!read_dst) alpha = 0.0f;
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    size_t per_tile = ep_cmul(k, 16);

    // A band: lane = M-row (stride lhs_rs), depth = K (stride lhs_cs).
#define PACKA_RANGE(AP, MT0, MT1)                                                                  \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                            \
        size_t r0 = st * 16;                                                                       \
        size_t vr = (r0 < m) ? ((m - r0 < 16) ? (m - r0) : 16) : 0;                                \
        /* vr==0 bands are zero-filled without reading src; clamp the base so no   */              \
        /* out-of-bounds pointer is even formed (UB without a deref). */                           \
        pack_band((AP) + st * per_tile, vr ? lhs + (long)r0 * lhs_rs : lhs, lhs_rs, lhs_cs, k, vr);\
    }

    size_t M_CHUNK = 2;
    size_t n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    int big = ep_flops(m, n, k) >= (1u << 21);

    // Flat-M (few M-tiles) but large/wide: M is the only M-chunk axis, so the
    // M-parallel scheme below would run this on ONE cluster. Parallelize over N
    // instead -- pack all of A once (cheap; m_tiles is small) into a shared buffer
    // and hand N-tile chunks to both clusters. C columns are disjoint per chunk;
    // A and B are read-only and shared. f32 N-tiles carry no cross-tile state, so
    // any chunk granularity is safe. Needs >= 2 chunks to beat the serial path.
    size_t N_CHUNK = 4;
    size_t nn_chunks = (n_tiles + N_CHUNK - 1) / N_CHUNK;
    // A row spans all N-tiles, so N-parallel would split its reduction across
    // threads. M-parallel is safe (chunks own disjoint rows).
    if (n_chunks <= 1 && big && nn_chunks >= 2 && !ep_has_reduce(ep)) {
        float *a_pack = (float *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
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
        float *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
        if (!a_pack) return -1;
        PACKA_RANGE(a_pack, 0, m_tiles);
        run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, 0, n_tiles, k, alpha,
                      beta, read_dst, ep);
        return 0;
    }
    float *a_pack = (float *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
    if (!a_pack) return -1;
    dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
      size_t mt0 = ci * M_CHUNK;
      size_t mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
      PACKA_RANGE(a_pack, mt0, mt1);
      run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, mt0, mt1, 0, n_tiles, k, alpha, beta,
                    read_dst, ep);
    });
#undef PACKA_RANGE
    free(a_pack);
    return 0;
}

// C = softmax_rows(A @ B), B pre-packed. One GEMM (whose store also produces the
// per-row maxima) followed by the parallel NEON softmax pass in attention.c.
int gemm_sme_f32_softmax(size_t m, size_t n, size_t k, float *dst, const float *lhs, long lhs_cs,
                         long lhs_rs, const float *b_pack, float *row_max) {
    if (m == 0 || n == 0) return 0;
    ep_desc_f32 ep = {0, NULL, NULL, row_max};
    int rc = gemm_sme_f32_run_packed(m, n, k, dst, 1, (long)n, 0, lhs, lhs_cs, lhs_rs, b_pack, 0.0f,
                                     1.0f, &ep);
    if (rc != 0) return rc;
    attn_softmax_rows_f32(dst, m, n, (long)n, row_max);
    return 0;
}
