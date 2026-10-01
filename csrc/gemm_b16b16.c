// SME b16b16 non-widening bf16 GEMM driver (Apple M5+, FEAT_SME_B16B16). Mirror
// of the f16f16 driver: each MOPA is a 32x32 bf16 outer product, and two
// independent ZA16 tiles run in lockstep (za0 for nt, za1 for nt+1) so
// consecutive MOPAs never hit the same tile.
//
// Accumulation is bf16; the alpha/beta combine uses FEAT_SME_B16B16 /
// FEAT_SVE_B16B16 arithmetic (svmul_bf16/svmla_bf16). Single pass over K.
// Computes dst = alpha*dst + beta*(A @ B).

#include "epilogue.h"
#include "transpose16.h"
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef __bf16 bf16;

static inline bf16 bf16_from_bits(uint16_t b) {
    bf16 h;
    __builtin_memcpy(&h, &b, sizeof h);
    return h;
}

// Pack one 32-row M-tile of A into [K, 32]: dst[d*32 + i] = A[i, d].
static void pack_a_tile(bf16 *dst, const bf16 *a, long rs, long cs, size_t k, size_t vr) {
    if (rs == 1 && vr == 32) {
        for (size_t d = 0; d < k; d++)
            memcpy(dst + d * 32, a + (long)d * cs, 32 * sizeof(bf16));
        return;
    }
    for (size_t d = 0; d < k; d++)
        for (size_t i = 0; i < 32; i++)
            dst[d * 32 + i] = (i < vr) ? a[(long)i * rs + (long)d * cs] : (bf16)0.0f;
}

// Transpose 32 depth-contiguous lanes into a [K, 32] tile; see the f16f16 twin.
// Serves both row-major A (rows as lanes) and column-major B (columns as lanes).
static void pack_lanes_depth_major(bf16 *dt, const bf16 *src, long lane_stride, size_t k,
                                   size_t valid) {
    size_t kfull = k & ~(size_t)7;
    size_t lfull = valid & ~(size_t)7;
    size_t kfull2 = kfull & ~(size_t)15;
    for (size_t lg = 0; lg < lfull; lg += 8) {
        size_t kg = 0;
        for (; kg < kfull2; kg += 16) {
            uint16x8_t r[8], s[8];
            for (int i = 0; i < 8; i++) {
                const uint16_t *lane = (const uint16_t *)(src + (long)(lg + i) * lane_stride + kg);
                r[i] = vld1q_u16(lane);
                s[i] = vld1q_u16(lane + 8);
            }
            transpose_8x8_u16(r);
            transpose_8x8_u16(s);
            for (int i = 0; i < 8; i++) {
                vst1q_u16((uint16_t *)(dt + (kg + i) * 32 + lg), r[i]);
                vst1q_u16((uint16_t *)(dt + (kg + 8 + i) * 32 + lg), s[i]);
            }
        }
        for (; kg < kfull; kg += 8) {
            uint16x8_t r[8];
            for (int i = 0; i < 8; i++)
                r[i] = vld1q_u16((const uint16_t *)(src + (long)(lg + i) * lane_stride + kg));
            transpose_8x8_u16(r);
            for (int i = 0; i < 8; i++)
                vst1q_u16((uint16_t *)(dt + (kg + i) * 32 + lg), r[i]);
        }
    }
    // Lane tail, depth-outer -- lane-outer re-walks the whole panel per lane.
    for (size_t d = 0; d < k && lfull < 32; d++) {
        bf16 *dp = dt + d * 32;
        for (size_t i = lfull; i < valid; i++)
            dp[i] = src[(long)i * lane_stride + d];
        for (size_t i = valid; i < 32; i++)
            dp[i] = (bf16)0.0f;
    }
    for (size_t i = 0; i < lfull; i++)
        for (size_t d = kfull; d < k; d++)
            dt[d * 32 + i] = src[(long)i * lane_stride + d];
}

// Pack one 32-col N-tile of B into [K, 32]: dst[d*32 + j] = B[d, j].
static void pack_b_tile(bf16 *dst, const bf16 *b, long rs, long cs, size_t k, size_t vc) {
    if (cs == 1 && vc == 32) {
        for (size_t d = 0; d < k; d++)
            memcpy(dst + d * 32, b + (long)d * rs, 32 * sizeof(bf16));
        return;
    }
    if (rs == 1) {
        // Col-major B (transposed-B / Q@K^T): columns are the depth-contiguous
        // lanes, so this is the A-pack's transpose with columns for rows.
        pack_lanes_depth_major(dst, b, cs, k, vc);
        return;
    }
    for (size_t d = 0; d < k; d++)
        for (size_t j = 0; j < 32; j++)
            dst[d * 32 + j] = (j < vc) ? b[(long)d * rs + (long)j * cs] : (bf16)0.0f;
}

// Narrow/combine/store one extracted ZA16 column or row (already 32 contiguous
// bf16 for the stored direction). `acc` is the raw bf16 outer-product slice.
static inline void store_slice(svbool_t p16, svbool_t pst, bf16 *ptr, svbfloat16_t acc,
                               svbfloat16_t vb, int read_dst, svbfloat16_t va) __arm_streaming {
    acc = svmul_bf16_x(p16, acc, vb);
    if (read_dst) {
        acc = svmla_bf16_x(p16, acc, svld1_bf16(pst, ptr), va);
    }
    svst1_bf16(pst, ptr, acc);
}

// Zero ZA tiles 0/1 then initialize ZA[r][c] = bias[c] via a rank-1 FMOPA
// (ones[r] * bias[c]). Kept out of line so the bias-fold path does not perturb
// the layout-sensitive inlined store code of run_streaming. nt is the first of
// the two N-tiles; partial column tiles predicate the bias load.
__attribute__((noinline)) static void bias_init_za(svbool_t p16, const bf16 *bias, size_t nt,
                                                   size_t n) __arm_streaming __arm_inout("za") {
    svzero_za();
    svbfloat16_t v_ones = svdup_n_bf16((bf16)1.0f);
    size_t bn0 = (nt + 0) * 32;
    size_t bn1 = (nt + 1) * 32;
    svbool_t pb0 = (bn0 < n) ? svwhilelt_b16((uint64_t)bn0, (uint64_t)n) : svpfalse_b();
    svbool_t pb1 = (bn1 < n) ? svwhilelt_b16((uint64_t)bn1, (uint64_t)n) : svpfalse_b();
    // Clamp the pad-tile address: pb1 may be all-false (no lane is read), but
    // forming bias + bn1 past one-past-end would still be UB.
    svmopa_za16_bf16_m(0, p16, pb0, v_ones, svld1_bf16(pb0, bias + bn0));
    svmopa_za16_bf16_m(1, p16, pb1, v_ones, svld1_bf16(pb1, (bn1 < n) ? bias + bn1 : bias));
}

// Narrow-N (n_tiles == 1, n <= 32) row-major fast path -- bf16 mirror of
// gemm_f16f16.c run_narrow_rowmajor. The dual-tile lockstep wastes za1 on the pad
// N-tile when there is only one N-tile; pair two M-tiles against the single
// B-tile instead (za0 = A[mt] @ B, za1 = A[mt+1] @ B) so both ZA tiles stay
// productive. Row-major only; bias applied as a node (not folded) so the store
// reuses the existing pure / node-major primitives. run_streaming is untouched.
__arm_locally_streaming __arm_new("za") static void run_narrow_rowmajor(
    bf16 *dst, long dst_cs, long dst_rs, const bf16 *a_pack, const bf16 *b_pack, size_t m, size_t n,
    size_t k, size_t mt_lo, size_t mt_hi, size_t nt, float alpha, float beta, int read_dst,
    const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    svcount_t pn = svptrue_c16();
    const svbfloat16_t z16 = svdup_n_bf16((bf16)0.0f);
    const svbfloat16_t va = svdup_n_bf16((bf16)alpha);
    const svbfloat16_t vb = svdup_n_bf16((bf16)beta);
    size_t per_tile = ep_cmul(k, 32);
    int has_ep = ep && ep->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    // `nt` is the single N-tile this pass covers: tile 0 for n<=32, or the odd
    // trailing tile when the dual-N loop has run out of pairs.
    size_t n0 = nt * 32;
    size_t ncols = (n - n0 < 32) ? (n - n0) : 32;
    svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
    const bf16 *b0 = b_pack + nt * per_tile;

    for (size_t mt = mt_lo; mt < mt_hi; mt += 2) {
        const bf16 *a0t = a_pack + mt * per_tile;
        int have1 = (mt + 1 < mt_hi);
        const bf16 *a1t = have1 ? a_pack + (mt + 1) * per_tile : a0t;
        size_t m0_0 = mt * 32;
        size_t mrows0 = (m - m0_0 < 32) ? (m - m0_0) : 32;
        size_t m0_1 = (mt + 1) * 32;
        size_t mrows1 = have1 ? ((m - m0_1 < 32) ? (m - m0_1) : 32) : 0;

        svzero_za();
        size_t d = 0;
        for (; d + 4 <= k; d += 4) {
            svbfloat16x4_t A0 = svld1_bf16_x4(pn, a0t + d * 32);
            svbfloat16x4_t A1 = svld1_bf16_x4(pn, a1t + d * 32);
            svbfloat16x4_t U = svld1_bf16_x4(pn, b0 + d * 32);
            svbfloat16_t u0 = svget4_bf16(U, 0), u1 = svget4_bf16(U, 1), u2 = svget4_bf16(U, 2),
                         u3 = svget4_bf16(U, 3);
            svmopa_za16_bf16_m(0, p16, p16, svget4_bf16(A0, 0), u0);
            svmopa_za16_bf16_m(1, p16, p16, svget4_bf16(A1, 0), u0);
            svmopa_za16_bf16_m(0, p16, p16, svget4_bf16(A0, 1), u1);
            svmopa_za16_bf16_m(1, p16, p16, svget4_bf16(A1, 1), u1);
            svmopa_za16_bf16_m(0, p16, p16, svget4_bf16(A0, 2), u2);
            svmopa_za16_bf16_m(1, p16, p16, svget4_bf16(A1, 2), u2);
            svmopa_za16_bf16_m(0, p16, p16, svget4_bf16(A0, 3), u3);
            svmopa_za16_bf16_m(1, p16, p16, svget4_bf16(A1, 3), u3);
        }
        for (; d < k; d++) {
            svbfloat16_t u = svld1_bf16(p16, b0 + d * 32);
            svmopa_za16_bf16_m(0, p16, p16, svld1_bf16(p16, a0t + d * 32), u);
            svmopa_za16_bf16_m(1, p16, p16, svld1_bf16(p16, a1t + d * 32), u);
        }

        if (pure) {
            for (size_t r = 0; r < mrows0; r++)
                svst1_hor_za16(0, (uint32_t)r, pst,
                               dst + (long)(m0_0 + r) * dst_rs + (long)n0 * dst_cs);
            if (have1)
                for (size_t r = 0; r < mrows1; r++)
                    svst1_hor_za16(1, (uint32_t)r, pst,
                                   dst + (long)(m0_1 + r) * dst_rs + (long)n0 * dst_cs);
        } else if (has_ep) {
            bf16 *base0 = dst + (long)m0_0 * dst_rs + (long)n0 * dst_cs;
#define EP_RD0(s) svread_hor_za16_bf16_m(z16, p16, 0, (s))
            EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, base0, dst_rs, EP_RD0, mrows0, vb, va, read_dst,
                                        nodes, n_nodes, m0_0, n0);
#undef EP_RD0
            if (have1) {
                bf16 *base1 = dst + (long)m0_1 * dst_rs + (long)n0 * dst_cs;
#define EP_RD1(s) svread_hor_za16_bf16_m(z16, p16, 1, (s))
                EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, base1, dst_rs, EP_RD1, mrows1, vb, va, read_dst,
                                            nodes, n_nodes, m0_1, n0);
#undef EP_RD1
            }
        } else {
            for (size_t r = 0; r < mrows0; r++)
                store_slice(p16, pst, dst + (long)(m0_0 + r) * dst_rs + (long)n0 * dst_cs,
                            svread_hor_za16_bf16_m(z16, p16, 0, (uint32_t)r), vb, read_dst, va);
            if (have1)
                for (size_t r = 0; r < mrows1; r++)
                    store_slice(p16, pst, dst + (long)(m0_1 + r) * dst_rs + (long)n0 * dst_cs,
                                svread_hor_za16_bf16_m(z16, p16, 1, (uint32_t)r), vb, read_dst, va);
        }
    }
}

// m == 1 GEMV; see gemm_f16f16.c.
#define GEMV_T 4
#define GEMV_MAXR 4 // rows; past this the MOPA path does fewer ZA ops
__arm_locally_streaming __arm_new("za") static void run_gemv(bf16 *dst, long dst_rs, const bf16 *abc,
                                                             size_t rows, const bf16 *b_pack, size_t n,
                                                             size_t k, size_t nt_lo, size_t nt_hi,
                                                             float alpha, float beta, int read_dst,
                                                             const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    svcount_t pn = svptrue_c16();
    const svbfloat16_t va = svdup_n_bf16((bf16)alpha);
    const svbfloat16_t vb = svdup_n_bf16((bf16)beta);
    int has_ep = ep && ep->n_nodes > 0;
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t per_tile = ep_cmul(k, 32); // also the stride between abc rows
    size_t kfull = k & ~(size_t)3;
    // K tail: predicated loads zero the dead depths.
    svcount_t pt = svwhilelt_c16((uint64_t)0, (uint64_t)((k - kfull) * 32), 4);
    for (size_t nt = nt_lo; nt < nt_hi; nt += GEMV_T) {
        size_t T = nt_hi - nt < GEMV_T ? nt_hi - nt : GEMV_T;
        const bf16 *b = b_pack + nt * per_tile;
        svzero_za();
        if (T == GEMV_T) {
            for (size_t d = 0; d < kfull; d += 4) {
                svbfloat16x4_t B0 = svld1_bf16_x4(pn, b + d * 32);
                svbfloat16x4_t B1 = svld1_bf16_x4(pn, b + per_tile + d * 32);
                svbfloat16x4_t B2 = svld1_bf16_x4(pn, b + 2 * per_tile + d * 32);
                svbfloat16x4_t B3 = svld1_bf16_x4(pn, b + 3 * per_tile + d * 32);
                for (size_t r = 0; r < rows; r++) {
                    svbfloat16x4_t A = svld1_bf16_x4(pn, abc + r * per_tile + d * 32);
                    uint32_t w = (uint32_t)(4 * r);
                    svmla_za16_bf16_vg1x4(w, B0, A);
                    svmla_za16_bf16_vg1x4(w + 1, B1, A);
                    svmla_za16_bf16_vg1x4(w + 2, B2, A);
                    svmla_za16_bf16_vg1x4(w + 3, B3, A);
                }
            }
        } else {
            for (size_t t = 0; t < T; t++)
                for (size_t d = 0; d < kfull; d += 4) {
                    svbfloat16x4_t B = svld1_bf16_x4(pn, b + t * per_tile + d * 32);
                    for (size_t r = 0; r < rows; r++)
                        svmla_za16_bf16_vg1x4((uint32_t)(4 * r + t), B,
                                              svld1_bf16_x4(pn, abc + r * per_tile + d * 32));
                }
        }
        if (kfull < k)
            for (size_t t = 0; t < T; t++) {
                svbfloat16x4_t B = svld1_bf16_x4(pt, b + t * per_tile + kfull * 32);
                for (size_t r = 0; r < rows; r++)
                    svmla_za16_bf16_vg1x4((uint32_t)(4 * r + t), B,
                                          svld1_bf16_x4(pt, abc + r * per_tile + kfull * 32));
            }
        for (size_t r = 0; r < rows; r++)
            for (size_t t = 0; t < T; t++) {
                size_t n0 = (nt + t) * 32;
                size_t ncols = n - n0 < 32 ? n - n0 : 32;
                svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                svbfloat16x4_t q = svread_za16_bf16_vg1x4((uint32_t)(4 * r + t));
                svbfloat16_t acc = svadd_bf16_x(p16, svadd_bf16_x(p16, svget4(q, 0), svget4(q, 1)),
                                         svadd_bf16_x(p16, svget4(q, 2), svget4(q, 3)));
                bf16 *out = dst + (long)r * dst_rs + n0;
                if (pure) {
                    svst1_bf16(pst, out, acc);
                } else if (has_ep) {
#define EP_GEMV_RD(s) acc
                    EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, out, dst_rs, EP_GEMV_RD, 1, vb, va, read_dst,
                                               nodes, n_nodes, r, n0);
#undef EP_GEMV_RD
                } else {
                    store_slice(p16, pst, out, acc, vb, read_dst, va);
                }
            }
    }
}

// abc[d*32 + j] = A[0, d].
static void bcast_row(bf16 *abc, const bf16 *a, long lhs_cs, size_t k) {
    const uint16_t *src = (const uint16_t *)a;
    uint16_t *out = (uint16_t *)abc;
    for (size_t d = 0; d < k; d++) {
        uint16x8_t v = vdupq_n_u16(src[(long)d * lhs_cs]);
        uint16x8x4_t q = {{v, v, v, v}};
        vst1q_u16_x4(out + d * 32, q);
    }
}

// See f16_budget / f16_nc_blk.
static size_t b16_budget(size_t k) {
    size_t tile_bytes = ep_cmul(k, 32) * sizeof(bf16);
    size_t b = (8u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    return b < 4 ? 4 : b;
}

static size_t b16_nc_blk(size_t k, size_t span) {
    size_t nc = (b16_budget(k) / 2) & ~(size_t)1;
    if (nc < 2) nc = 2;
    if (nc >= span) return span;
    size_t blocks = (span + nc - 1) / nc;
    size_t w = (span + blocks - 1) / blocks;
    return (w + 1) & ~(size_t)1;
}

// [nt_lo, nt_hi): the even-aligned N-tile (pad) sub-range this invocation owns
// (whole problem = (0, n_tiles_pad); a slice for the N-parallel flat-M path).
// See gemm_f16f16.c run_streaming and gemm_sme_b16b16_run_packed.
__arm_locally_streaming __arm_new("za") static void run_streaming(
    bf16 *dst, long dst_cs, long dst_rs, const bf16 *a_pack, const bf16 *b_pack, size_t m, size_t n,
    size_t k, size_t mt_lo, size_t mt_hi, size_t n_tiles, size_t nt_lo, size_t nt_hi, float alpha,
    float beta, int read_dst, const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    // SME2 multi-vector LD1H: one load pulls 4 contiguous depth-vectors of a
    // packed [k,32] panel (z-group), 3 loads vs 12. Throughput-neutral (MOPA-
    // bound) but fewer uops and less load-port pressure under multi-cluster.
    svcount_t pn = svptrue_c16();
    const svbfloat16_t z16 = svdup_n_bf16((bf16)0.0f);
    const svbfloat16_t va = svdup_n_bf16((bf16)alpha);
    const svbfloat16_t vb = svdup_n_bf16((bf16)beta);
    size_t per_tile = ep_cmul(k, 32);
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);
    // Pure overwrite (C = A@B): the epilogue is just "ZA -> C", which the
    // SME `st1h {zaNv}` family does in ONE instruction per slice.
    // A fused epilogue (bias/activation) needs the accumulator in registers.
    // Fold a leading per-column bias (ADD_COL) into the ZA accumulator via a
    // rank-1 FMOPA before the K-loop MOPAs, then drop it from the store graph; a
    // bias-only graph then reverts to the direct single-instruction store (free).
    // Requires beta==1 / !read_dst. Bias is added in the bf16 ZA element domain
    // (bias-first, then accumulate) -- a rounding-order change within the bf16
    // sqrt(k) test tolerance. Gated to bias-only (folding bias+activation perturbs
    // the layout-sensitive store codegen).
    int fold_bias = ep && ep->n_nodes == 1 && beta == 1.0f && !read_dst &&
                    ep->nodes[0].op == EP_OP_ADD_COL;
    const bf16 *fold_bias_ptr = fold_bias ? (const bf16 *)ep->nodes[0].ptr : NULL;
    const ep_desc16 *ep_eff = fold_bias ? NULL : ep;
    int has_ep = ep_eff && ep_eff->n_nodes > 0;
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    const EpNode *nodes = has_ep ? ep_eff->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep_eff->n_nodes : 0;
    int has_tensor = has_ep && ep_has_tensor(nodes, n_nodes);

    size_t budget_tiles = b16_budget(k);
    size_t nt_span = nt_hi - nt_lo;
    size_t nc = b16_nc_blk(k, nt_span);
    size_t mc = budget_tiles / 2;
    if (mc < 1) mc = 1;
    if (mc > mt_hi - mt_lo) mc = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc) {
        size_t jc_end = jc + nc < nt_hi ? jc + nc : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc) {
            size_t ic_end = ic + mc < mt_hi ? ic + mc : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const bf16 *at = a_pack + mt * per_tile;
                size_t m0 = mt * 32;
                size_t mrows = (m - m0 < 32) ? (m - m0) : 32;

                for (size_t nt = jc; nt < jc_end; nt += 2) {
                    const bf16 *b0 = b_pack + (nt + 0) * per_tile;
                    const bf16 *b1 = b_pack + (nt + 1) * per_tile;

                    if (fold_bias) {
                        bias_init_za(p16, fold_bias_ptr, nt, n);
                    } else {
                        svzero_za();
                    }
                    size_t d = 0;
                    for (; d + 4 <= k; d += 4) {
                        svbfloat16x4_t A = svld1_bf16_x4(pn, at + d * 32);
                        svbfloat16x4_t U = svld1_bf16_x4(pn, b0 + d * 32);
                        svbfloat16x4_t V = svld1_bf16_x4(pn, b1 + d * 32);
                        svbfloat16_t a0 = svget4_bf16(A, 0), a1 = svget4_bf16(A, 1),
                                     a2 = svget4_bf16(A, 2), a3 = svget4_bf16(A, 3);
                        svmopa_za16_bf16_m(0, p16, p16, a0, svget4_bf16(U, 0));
                        svmopa_za16_bf16_m(1, p16, p16, a0, svget4_bf16(V, 0));
                        svmopa_za16_bf16_m(0, p16, p16, a1, svget4_bf16(U, 1));
                        svmopa_za16_bf16_m(1, p16, p16, a1, svget4_bf16(V, 1));
                        svmopa_za16_bf16_m(0, p16, p16, a2, svget4_bf16(U, 2));
                        svmopa_za16_bf16_m(1, p16, p16, a2, svget4_bf16(V, 2));
                        svmopa_za16_bf16_m(0, p16, p16, a3, svget4_bf16(U, 3));
                        svmopa_za16_bf16_m(1, p16, p16, a3, svget4_bf16(V, 3));
                    }
                    for (; d < k; d++) {
                        svbfloat16_t av = svld1_bf16(p16, at + d * 32);
                        svmopa_za16_bf16_m(0, p16, p16, av, svld1_bf16(p16, b0 + d * 32));
                        svmopa_za16_bf16_m(1, p16, p16, av, svld1_bf16(p16, b1 + d * 32));
                    }

                    for (size_t t = 0; t < 2; t++) {
                        size_t ntt = nt + t;
                        if (ntt >= n_tiles) break;
                        size_t n0 = ntt * 32;
                        size_t ncols = (n - n0 < 32) ? (n - n0) : 32;
                        if (col_major) {
                            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)mrows);
                            if (pure) {
                                if (t == 0)
                                    for (size_t c = 0; c < ncols; c++)
                                        svst1_ver_za16(0, (uint32_t)c, pst,
                                                       dst + (long)(n0 + c) * dst_cs +
                                                           (long)m0 * dst_rs);
                                else
                                    for (size_t c = 0; c < ncols; c++)
                                        svst1_ver_za16(1, (uint32_t)c, pst,
                                                       dst + (long)(n0 + c) * dst_cs +
                                                           (long)m0 * dst_rs);
                            } else {
                                // Col slice (M-oriented, vector spans M): ROW is the
                                // contiguous M-vector load; COL/SCALAR splat; TENSOR
                                // (col-strided) routes to the scalar path below.
                                for (size_t c = 0; c < ncols; c++) {
                                    bf16 *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                                    svbfloat16_t acc =
                                        (t == 0) ? svread_ver_za16_bf16_m(z16, p16, 0, (uint32_t)c)
                                                 : svread_ver_za16_bf16_m(z16, p16, 1, (uint32_t)c);
                                    if (has_ep && !has_tensor) {
                                        ep_store_bf16(p16, pst, col, acc, vb, va, read_dst, nodes,
                                                      n_nodes, /*span_n=*/0, /*i=*/n0 + c, n0, m0);
                                    } else if (has_ep) {
                                        // TENSOR present: strided gather over rows ->
                                        // materialize the M-column, apply scalar-wise.
                                        bf16 tmp[32];
                                        svst1_bf16(pst, tmp, acc);
                                        for (size_t r = 0; r < mrows; r++) {
                                            float v = (float)tmp[r] * beta;
                                            v = ep_apply_nodes_scalar_bf16(nodes, n_nodes, v, m0 + r,
                                                                           n0 + c);
                                            bf16 *cell = col + (long)r * dst_rs;
                                            *cell = (bf16)(read_dst ? alpha * (float)(*cell) + v : v);
                                        }
                                    } else {
                                        store_slice(p16, pst, col, acc, vb, read_dst, va);
                                    }
                                }
                            }
                        } else if (row_major) {
                            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                            if (pure) {
                                if (t == 0)
                                    for (size_t r = 0; r < mrows; r++)
                                        svst1_hor_za16(0, (uint32_t)r, pst,
                                                       dst + (long)(m0 + r) * dst_rs +
                                                           (long)n0 * dst_cs);
                                else
                                    for (size_t r = 0; r < mrows; r++)
                                        svst1_hor_za16(1, (uint32_t)r, pst,
                                                       dst + (long)(m0 + r) * dst_rs +
                                                           (long)n0 * dst_cs);
                            } else if (has_ep) {
                                // Row slice (N-oriented, vector spans N), NODE-MAJOR:
                                // read the live ZA tile directly into Z registers in
                                // 4-row blocks and dispatch each op-graph node ONCE
                                // per block (invariant operands loaded once per node);
                                // no memory bounce.
                                bf16 *base = dst + (long)m0 * dst_rs + (long)n0 * dst_cs;
                                if (t == 0) {
#define EP_RD0(s) svread_hor_za16_bf16_m(z16, p16, 0, (s))
                                    EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, base, dst_rs, EP_RD0,
                                                                mrows, vb, va, read_dst, nodes,
                                                                n_nodes, m0, n0);
#undef EP_RD0
                                } else {
#define EP_RD1(s) svread_hor_za16_bf16_m(z16, p16, 1, (s))
                                    EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, base, dst_rs, EP_RD1,
                                                                mrows, vb, va, read_dst, nodes,
                                                                n_nodes, m0, n0);
#undef EP_RD1
                                }
                            } else {
                                for (size_t r = 0; r < mrows; r++) {
                                    bf16 *rowp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                                    svbfloat16_t acc =
                                        (t == 0) ? svread_hor_za16_bf16_m(z16, p16, 0, (uint32_t)r)
                                                 : svread_hor_za16_bf16_m(z16, p16, 1, (uint32_t)r);
                                    store_slice(p16, pst, rowp, acc, vb, read_dst, va);
                                }
                            }
                        } else {
                            bf16 scratch[32 * 32];
                            svbool_t pg = svwhilelt_b16((uint64_t)0, (uint64_t)32);
                            for (uint32_t r = 0; r < 32; r++) {
                                svbfloat16_t row = (t == 0)
                                                       ? svread_hor_za16_bf16_m(z16, p16, 0, r)
                                                       : svread_hor_za16_bf16_m(z16, p16, 1, r);
                                svst1_bf16(pg, scratch + (size_t)r * 32, row);
                            }
                            for (size_t r = 0; r < mrows; r++)
                                for (size_t c = 0; c < ncols; c++) {
                                    float ab = (float)scratch[r * 32 + c] * beta;
                                    if (has_ep)
                                        ab = ep_apply_nodes_scalar_bf16(nodes, n_nodes, ab, m0 + r,
                                                                        n0 + c);
                                    bf16 *cell =
                                        dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                                    *cell = (bf16)(read_dst ? alpha * (float)(*cell) + ab : ab);
                                }
                        }
                    }
                }
            } // mt
        } // ic
    } // jc
}

// Pack A into [m_tiles, k, 32]. Fast paths for col-major (lhs_rs==1, memcpy)
// and row-major (lhs_cs==1, cache-blocked transpose); strided gather otherwise.
static void packa(bf16 *a_pack, const bf16 *a, size_t m, size_t k, long lhs_rs, long lhs_cs,
                  size_t mt_lo, size_t mt_hi, size_t per_tile) {
    if (lhs_rs == 1) {
        const size_t TB = 8;
        for (size_t tb = mt_lo; tb < mt_hi; tb += TB) {
            size_t tb_end = tb + TB < mt_hi ? tb + TB : mt_hi;
            for (size_t d = 0; d < k; d++) {
                const bf16 *col = a + (long)d * lhs_cs;
                for (size_t mt = tb; mt < tb_end; mt++) {
                    size_t r0 = mt * 32;
                    size_t vr = (m - r0 < 32) ? (m - r0) : 32;
                    bf16 *dp = a_pack + mt * per_tile + d * 32;
                    if (vr == 32) {
                        memcpy(dp, col + r0, 32 * sizeof(bf16));
                    } else {
                        memcpy(dp, col + r0, vr * sizeof(bf16));
                        for (size_t i = vr; i < 32; i++)
                            dp[i] = (bf16)0.0f;
                    }
                }
            }
        }
    } else if (lhs_cs == 1) {
        // Row-major A: rows are the depth-contiguous lanes.
        for (size_t mt = mt_lo; mt < mt_hi; mt++) {
            size_t r0 = mt * 32;
            size_t mrows = (m - r0 < 32) ? (m - r0) : 32;
            pack_lanes_depth_major(a_pack + mt * per_tile, a + (long)r0 * lhs_rs, lhs_rs, k, mrows);
        }
    } else {
        for (size_t mt = mt_lo; mt < mt_hi; mt++) {
            size_t r0 = mt * 32;
            size_t vr = (m - r0 < 32) ? (m - r0) : 32;
            pack_a_tile(a_pack + mt * per_tile, a + (long)r0 * lhs_rs, lhs_rs, lhs_cs, k, vr);
        }
    }
}

// Pack B into [n_tiles_pad, k, 32]; caller zeroes pad tiles.
static void packb(bf16 *b_pack, const bf16 *b, size_t n, size_t k, long rhs_rs, long rhs_cs,
                  size_t n_tiles, size_t per_tile) {
    if (rhs_cs == 1) {
        const size_t TB = 8;
        for (size_t tb = 0; tb < n_tiles; tb += TB) {
            size_t tb_end = tb + TB < n_tiles ? tb + TB : n_tiles;
            for (size_t d = 0; d < k; d++) {
                const bf16 *row = b + (long)d * rhs_rs;
                for (size_t nt = tb; nt < tb_end; nt++) {
                    size_t c0 = nt * 32;
                    size_t vc = (n - c0 < 32) ? (n - c0) : 32;
                    memcpy(b_pack + nt * per_tile + d * 32, row + c0, vc * sizeof(bf16));
                }
            }
        }
    } else {
        for (size_t nt = 0; nt < n_tiles; nt++) {
            size_t c0 = nt * 32;
            size_t vc = (n - c0 < 32) ? (n - c0) : 32;
            pack_b_tile(b_pack + nt * per_tile, b + (long)c0 * rhs_cs, rhs_rs, rhs_cs, k, vc);
        }
    }
}

// Per-thread reusable A-pack scratch (no per-call malloc/free on the serial
// small/medium path). Grows monotonically, lives for the thread's lifetime.
static _Thread_local bf16 *g_apack = NULL;
static _Thread_local size_t g_apack_cap = 0;
static bf16 *apack_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1; // k==0: return a real (tiny) buffer, not a
                               // history-dependent NULL on fresh threads
    if (g_apack_cap < bytes) {
        free(g_apack);
        g_apack = (bf16 *)malloc(bytes);
        g_apack_cap = g_apack ? bytes : 0;
    }
    return g_apack;
}

// Packed-B GEMM: caller supplies B pre-packed via gemm_sme_b16b16_packb. Only
// A is packed here -- the inference pattern (weights packed once, reused).
int gemm_sme_b16b16_run_packed(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs,
                               long dst_rs, int read_dst, const uint16_t *lhs, long lhs_cs,
                               long lhs_rs, const uint16_t *b_pack, uint16_t alpha_bits,
                               uint16_t beta_bits, const ep_desc16 *ep) {
    if (m == 0 || n == 0) return 0;
    float alpha = read_dst ? (float)bf16_from_bits(alpha_bits) : 0.0f;
    float beta = (float)bf16_from_bits(beta_bits);
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    size_t n_tiles_pad = (n_tiles + 1) & ~(size_t)1;
    size_t per_tile = ep_cmul(k, 32);
    const bf16 *a = (const bf16 *)lhs;
    const bf16 *bp = (const bf16 *)b_pack;
    bf16 *d = (bf16 *)dst;

    // Finer chunks so dispatch_apply work-steals to balance the slower E-cluster
    // SME unit against the P-cluster (a coarse 2-way split stalls on the slow half).
    size_t M_CHUNK = 2;
    size_t n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    int big = ep_flops(m, n, k) >= (1u << 21);
    if (m <= GEMV_MAXR && dst_cs == 1) {
        bf16 *abc = apack_scratch(ep_cmul(ep_cmul(m, per_tile), sizeof(bf16)));
        if (!abc) return -1;
        for (size_t r = 0; r < m; r++) bcast_row(abc + r * per_tile, a + (long)r * lhs_rs, lhs_cs, k);
        size_t G_CHUNK = 4; // tiles per chunk
        size_t g_chunks = (n_tiles + G_CHUNK - 1) / G_CHUNK;
        if (!big || g_chunks < 3) {
            run_gemv(d, dst_rs, abc, m, bp, n, k, 0, n_tiles, alpha, beta, read_dst, ep);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t nt0 = ci * G_CHUNK;
          size_t nt1 = nt0 + G_CHUNK < n_tiles ? nt0 + G_CHUNK : n_tiles;
          run_gemv(d, dst_rs, abc, m, bp, n, k, nt0, nt1, alpha, beta, read_dst, ep);
        });
        return 0;
    }
    // n_tiles==1 (n<=32) row-major: pair two M-tiles against the single B-tile
    // (run_narrow_rowmajor) instead of wasting za1 on the pad N-tile. See f16f16.
    int narrow = (n_tiles == 1 && dst_cs == 1);
    // Odd N-tile tail: hand the unpaired last tile to run_narrow_rowmajor so
    // both ZA tiles stay productive instead of za1 taking the zero pad tile.
    // Skipped when the bias folds -- run_streaming folds it into ZA and
    // run_narrow_rowmajor applies it at the store, which would round the two
    // halves of one output differently. See the f16f16 twin.
    int fold_bias_here =
        ep && ep->n_nodes == 1 && beta == 1.0f && !read_dst && ep->nodes[0].op == EP_OP_ADD_COL;
    int odd_tail = (n_tiles > 1) && (n_tiles & 1) && dst_cs == 1 && !fold_bias_here;
    // Flat-M: parallelize over N (both clusters) instead of running serially on
    // one. Pack all of A once (m_tiles is small) into a shared buffer, hand
    // even-aligned N-tile-pair chunks to the workers. See gemm_f16f16.c.
    size_t N_CHUNK = 4;
    size_t nn_chunks = (n_tiles_pad + N_CHUNK - 1) / N_CHUNK;
    // Three chunks, not two -- a 2-way split loses to serial here. See the long
    // note in gemm_f16f16.c; measured 1.03-1.13x on bf16 as well.
    if (n_chunks <= 1 && big && nn_chunks >= 3) {
        bf16 *a_pack = (bf16 *)malloc(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(bf16)));
        if (a_pack) {
            packa(a_pack, a, m, k, lhs_rs, lhs_cs, 0, m_tiles, per_tile);
            dispatch_apply(nn_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
              size_t nt0 = ci * N_CHUNK;
              size_t nt1 = nt0 + N_CHUNK < n_tiles_pad ? nt0 + N_CHUNK : n_tiles_pad;
              // The chunk holding the unpaired last tile stops one short and runs
              // that tile M-paired instead; see the f16f16 twin.
              if (odd_tail && nt0 <= n_tiles - 1 && n_tiles - 1 < nt1) {
                  run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, nt0,
                                n_tiles - 1, alpha, beta, read_dst, ep);
                  run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles,
                                      n_tiles - 1, alpha, beta, read_dst, ep);
              } else {
                  run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, nt0,
                                nt1, alpha, beta, read_dst, ep);
              }
            });
            free(a_pack);
            return 0;
        }
    }
    if (n_chunks <= 1 || !big) {
        // Serial: per-thread reusable scratch, no per-call malloc/free.
        bf16 *a_pack = apack_scratch(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(bf16)));
        if (!a_pack) return -1;
        packa(a_pack, a, m, k, lhs_rs, lhs_cs, 0, m_tiles, per_tile);
        if (narrow) {
            run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, 0, alpha,
                                beta, read_dst, ep);
        } else if (odd_tail) {
            run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, 0,
                          n_tiles - 1, alpha, beta, read_dst, ep);
            run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles - 1,
                                alpha, beta, read_dst, ep);
        } else {
            run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, 0,
                          n_tiles_pad, alpha, beta, read_dst, ep);
        }
        return 0;
    }
    bf16 *a_pack = (bf16 *)malloc(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(bf16)));
    if (!a_pack) return -1;
    if (narrow) {
        dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t mt0 = ci * M_CHUNK;
          size_t mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
          packa(a_pack, a, m, k, lhs_rs, lhs_cs, mt0, mt1, per_tile);
          run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, mt0, mt1, 0, alpha, beta,
                              read_dst, ep);
        });
        free(a_pack);
        return 0;
    }
    // N-blocks outside the M dispatch so each B-block stays L2-resident; the
    // odd tail tile rides with the last block.
    size_t nt_end = odd_tail ? n_tiles - 1 : n_tiles_pad;
    size_t nc_blk = b16_nc_blk(k, nt_end);
    size_t n_blocks = (nt_end + nc_blk - 1) / nc_blk;
    if (n_blocks > 1)
        dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t mt0 = ci * M_CHUNK;
          packa(a_pack, a, m, k, lhs_rs, lhs_cs, mt0, mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles,
                per_tile);
        });
    // Flat dispatch over (N-block, M-chunk), block-major; see gemm_i16i64.c.
    dispatch_apply(n_blocks * n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t it) {
      size_t ci = it % n_chunks, jc = it / n_chunks * nc_blk;
      size_t mt0 = ci * M_CHUNK;
      size_t mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
      size_t jc_end = jc + nc_blk < nt_end ? jc + nc_blk : nt_end;
      if (n_blocks == 1) packa(a_pack, a, m, k, lhs_rs, lhs_cs, mt0, mt1, per_tile);
      run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, mt0, mt1, n_tiles, jc, jc_end, alpha, beta,
                    read_dst, ep);
      if (odd_tail && jc_end == nt_end)
          run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, mt0, mt1, n_tiles - 1, alpha,
                              beta, read_dst, ep);
    });
    free(a_pack);
    return 0;
}

// Pack B once into a caller-allocated, zeroed buffer of
// gemm_sme_b16b16_packed_b_elems(n,k) bf16. Reuse across many GEMMs (weights).
void gemm_sme_b16b16_packb(uint16_t *b_pack, const uint16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs) {
    packb((bf16 *)b_pack, (const bf16 *)rhs, n, k, rhs_rs, rhs_cs, (n + 31) / 32, ep_cmul(k, 32));
}

// Saturating: SIZE_MAX on overflow (see gemm_sme_i8i32_packed_b_elems).
size_t gemm_sme_b16b16_packed_b_elems(size_t n, size_t k) {
    size_t n_tiles_pad = (((n + 31) / 32) + 1) & ~(size_t)1;
    return ep_cmul(ep_cmul(n_tiles_pad, k), 32);
}

int gemm_sme_b16b16_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                        int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                        const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                        uint16_t beta_bits) {
    if (m == 0 || n == 0) return 0;
    size_t per_tile = ep_cmul(k, 32);
    size_t n_tiles_pad = (((n + 31) / 32) + 1) & ~(size_t)1;
    bf16 *b_pack = (bf16 *)calloc(ep_cmul(n_tiles_pad, per_tile), sizeof(bf16));
    if (!b_pack) return -1;
    packb(b_pack, (const bf16 *)rhs, n, k, rhs_rs, rhs_cs, (n + 31) / 32, per_tile);
    int rc = gemm_sme_b16b16_run_packed(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs,
                                        (const uint16_t *)b_pack, alpha_bits, beta_bits, NULL);
    free(b_pack);
    return rc;
}

// One row-major GEMM (C = A@B overwrite), already in streaming mode. B is read
// straight from the row-major input (a 32-column tile at depth d is contiguous)
// and each A tile is transposed through ZA tile 0 into `a_scr` ([k][32]), so
// nothing is packed outside the streaming session. `ep` applies the op-graph
// at the per-row store, as run_streaming's row-major path does.
static void batch_one(bf16 *dst, const bf16 *a, const bf16 *b, bf16 *a_scr, size_t m, size_t n,
                      size_t k, size_t n_tiles, const EpNode *nodes,
                      uint32_t n_nodes) __arm_streaming __arm_inout("za") {
    svbool_t p16 = svptrue_b16();
    const svbfloat16_t z16 = svdup_n_bf16((bf16)0.0f);
    const svbfloat16_t vb = svdup_n_bf16((bf16)1.0f);
    const svbfloat16_t va = svdup_n_bf16((bf16)0.0f);
    int has_ep = nodes && n_nodes > 0;
    size_t m_tiles = (m + 31) / 32;
    for (size_t mt = 0; mt < m_tiles; mt++) {
        size_t m0 = mt * 32;
        size_t mrows = (m - m0 < 32) ? (m - m0) : 32;
        for (size_t d0 = 0; d0 < k; d0 += 32) {
            size_t dc = k - d0 < 32 ? k - d0 : 32;
            svbool_t pk = svwhilelt_b16((uint64_t)0, (uint64_t)dc);
            svzero_za();
            for (size_t r = 0; r < mrows; r++)
                svld1_hor_za16(0, (uint32_t)r, pk, a + (m0 + r) * k + d0);
            for (size_t c = 0; c < dc; c++)
                svst1_ver_za16(0, (uint32_t)c, p16, a_scr + (d0 + c) * 32);
        }
        for (size_t nt = 0; nt < n_tiles; nt += 2) {
            size_t n0 = nt * 32;
            size_t n1 = n0 + 32;
            svbool_t pb0 = svwhilelt_b16((uint64_t)n0, (uint64_t)n);
            svbool_t pb1 = n1 < n ? svwhilelt_b16((uint64_t)n1, (uint64_t)n) : svpfalse_b();
            const bf16 *b1 = n1 < n ? b + n1 : b;
            svzero_za();
            for (size_t d = 0; d < k; d++) {
                svbfloat16_t av = svld1_bf16(p16, a_scr + d * 32);
                svmopa_za16_bf16_m(0, p16, p16, av, svld1_bf16(pb0, b + d * n + n0));
                svmopa_za16_bf16_m(1, p16, p16, av, svld1_bf16(pb1, b1 + d * n));
            }
            for (size_t t = 0; t < 2; t++) {
                size_t ntt = nt + t;
                if (ntt >= n_tiles) break;
                size_t nc0 = ntt * 32;
                size_t ncols = (n - nc0 < 32) ? (n - nc0) : 32;
                svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                if (has_ep) {
                    bf16 *base = dst + m0 * n + nc0;
                    if (t == 0) {
#define EP_RD0(s) svread_hor_za16_bf16_m(z16, p16, 0, (s))
                        EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, base, (long)n, EP_RD0, mrows, vb, va,
                                                   /*read_dst=*/0, nodes, n_nodes, m0, nc0);
#undef EP_RD0
                    } else {
#define EP_RD1(s) svread_hor_za16_bf16_m(z16, p16, 1, (s))
                        EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, base, (long)n, EP_RD1, mrows, vb, va,
                                                   /*read_dst=*/0, nodes, n_nodes, m0, nc0);
#undef EP_RD1
                    }
                } else if (t == 0) {
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(0, (uint32_t)r, pst, dst + (m0 + r) * n + nc0);
                } else {
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(1, (uint32_t)r, pst, dst + (m0 + r) * n + nc0);
                }
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void run_batched(
    size_t count, bf16 *dst, const bf16 *lhs, const bf16 *rhs, bf16 *a_scr, size_t m, size_t n,
    size_t k, size_t n_tiles, const EpNode *nodes, uint32_t n_nodes) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(dst + bi * m * n, lhs + bi * m * k, rhs + bi * k * n, a_scr, m, n, k, n_tiles,
                  nodes, n_nodes);
}

// Batched bf16 GEMM with an optional fused op-graph epilogue applied to EACH
// item (same nodes/operands for all items). `ep` NULL / n_nodes==0 is the raw
// path. The epilogue runs the whole op-graph in f32 (the bf16 accumulator and
// operands are upcast, computed, and rounded back to bf16), so it supports the
// full op set -- divide/sqrt/transcendentals included. See
// gemm_sme_f16f16_batched_ep.
int gemm_sme_b16b16_batched_ep(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                               const uint16_t *lhs, const uint16_t *rhs, const ep_desc16 *ep) {
    if (count == 0 || m == 0 || n == 0) return 0;
    int has_ep = ep && ep->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t n_tiles = (n + 31) / 32;
    // A scratch for one M-tile, rounded up to whole 32-deep transpose blocks.
    bf16 *a_scr = apack_scratch(ep_cmul(ep_cmul((k + 31) & ~(size_t)31, 32), sizeof(bf16)));
    if (!a_scr) return -1;
    run_batched(count, (bf16 *)dst, (const bf16 *)lhs, (const bf16 *)rhs, a_scr, m, n, k, n_tiles,
                nodes, n_nodes);
    return 0;
}

// Raw batched bf16 (no epilogue) -- unchanged ABI, thin wrapper.
int gemm_sme_b16b16_batched(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                            const uint16_t *lhs, const uint16_t *rhs) {
    return gemm_sme_b16b16_batched_ep(count, m, n, k, dst, lhs, rhs, NULL);
}

// The 4-bit-resident path, kept in its own file (it needs `packa` and the bf16
// typedef, so it is included rather than compiled separately).
#include "gemm_b16b16_q4.h"
