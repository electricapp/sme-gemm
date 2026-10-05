// SME f16f16 non-widening GEMM driver (Apple M5+, FEAT_SME_F16F16): each MOPA
// is a 32x32 fp16 outer product, and two independent ZA16 tiles run in lockstep
// (za0 for N-tile nt, za1 for nt+1) so consecutive MOPAs never hit the same
// tile. ZA16 consumes one K-value per MOPA, so the packed panels are plain
// [K, 32] tile-major (32 contiguous lanes per depth).
//
// Computes dst = alpha*dst + beta*(A @ B); fp16 accumulate, fp16 alpha/beta
// combine. Single pass over K, single-threaded per cluster.

#include "epilogue.h"
#include "neon_act.h"
#include "panel_ring.h"
#include "transpose16.h"
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef __fp16 f16;

static inline f16 f16_from_bits(uint16_t b) {
    f16 h;
    __builtin_memcpy(&h, &b, sizeof h);
    return h;
}

// Pack one 32-row M-tile of A into [K, 32]: dst[d*32 + i] = A[i, d].
static void pack_a_tile(f16 *dst, const f16 *a, long rs, long cs, size_t k, size_t vr) {
    if (rs == 1 && vr == 32) {
        // Fast path: rows contiguous, full tile -> 64-byte copy per depth.
        for (size_t d = 0; d < k; d++)
            memcpy(dst + d * 32, a + (long)d * cs, 32 * sizeof(f16));
        return;
    }
    for (size_t d = 0; d < k; d++)
        for (size_t i = 0; i < 32; i++)
            dst[d * 32 + i] = (i < vr) ? a[(long)i * rs + (long)d * cs] : (f16)0.0f;
}

// Transpose 32 depth-contiguous lanes into a [K, 32] tile: dst[d*32 + i] is lane
// i at depth d, where lane i starts at src + i*lane_stride and runs contiguously
// in depth. Lanes >= `valid` are zero-filled.
//
// This is the same shape for both operands -- row-major A (each ROW contiguous in
// depth) and column-major B (each COLUMN contiguous in depth) -- so both packs
// call it. NEON 8x8 transpose over the aligned bulk; the lane and depth edges go
// scalar, depth-outer so their stores stay within a cache line.
static void pack_lanes_depth_major(f16 *dt, const f16 *src, long lane_stride, size_t k,
                                   size_t valid) {
    size_t kfull = k & ~(size_t)7;
    size_t lfull = valid & ~(size_t)7;
    // Interleave two independent 8x8 transposes per step: the vtrn ladder is a
    // long dependency chain over strided loads, so a single block runs
    // latency-bound (~14 GB/s). Two blocks in flight overlap their chains and the
    // strided-load latency -> the pack stops being the small/medium-tile
    // bottleneck.
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
    // Lane tail (i in [lfull,32), all depths): real lanes up to `valid`, zero
    // past them. DEPTH-OUTER on purpose. The lane-outer form strides 32 elements
    // (one cache line) per store, so it re-walks the whole k*32 panel once per
    // lane -- at valid==1, lfull is 0 and that is 32 passes over the panel to
    // write a single lane, with 31 of them pure zero fill.
    for (size_t d = 0; d < k && lfull < 32; d++) {
        f16 *dp = dt + d * 32;
        for (size_t i = lfull; i < valid; i++)
            dp[i] = src[(long)i * lane_stride + d];
        for (size_t i = valid; i < 32; i++)
            dp[i] = (f16)0.0f;
    }
    // Depth tail (whole lanes, depths [kfull,k)).
    for (size_t i = 0; i < lfull; i++)
        for (size_t d = kfull; d < k; d++)
            dt[d * 32 + i] = src[(long)i * lane_stride + d];
}

// Pack one 32-col N-tile of B into [K, 32]: dst[d*32 + j] = B[d, j].
static void pack_b_tile(f16 *dst, const f16 *b, long rs, long cs, size_t k, size_t vc) {
    if (cs == 1 && vc == 32) {
        // Row-major B (e.g. [N,K] weights): contiguous 32 N-values per depth.
        for (size_t d = 0; d < k; d++)
            memcpy(dst + d * 32, b + (long)d * rs, 32 * sizeof(f16));
        return;
    }
    if (rs == 1) {
        // Col-major B (the transposed-B / Q@K^T case): each N-column is
        // contiguous in depth, which is the A-pack's layout with columns for
        // rows -- so it takes the same NEON transpose.
        pack_lanes_depth_major(dst, b, cs, k, vc);
        return;
    }
    for (size_t d = 0; d < k; d++)
        for (size_t j = 0; j < 32; j++)
            dst[d * 32 + j] = (j < vc) ? b[(long)d * rs + (long)j * cs] : (f16)0.0f;
}

// Narrow/combine/store one extracted ZA16 column or row (already 32 contiguous
// f16 for the stored direction). `acc` is the raw fp16 outer-product slice.
static inline void store_slice(svbool_t p16, svbool_t pst, f16 *ptr, svfloat16_t acc,
                               svfloat16_t vb, int read_dst, svfloat16_t va) __arm_streaming {
    acc = svmul_x(p16, acc, vb);
    if (read_dst) {
        acc = svmla_x(p16, acc, svld1_f16(pst, ptr), va);
    }
    svst1_f16(pst, ptr, acc);
}

// Zero ZA tiles 0/1 then initialize ZA[r][c] = bias[c] via a rank-1 FMOPA
// (ones[r] * bias[c]). Kept out of line so the bias-fold path does not perturb
// the (layout-sensitive) inlined store code of run_streaming. nt is the first of
// the two N-tiles; partial column tiles predicate the bias load.
__attribute__((noinline)) static void bias_init_za(svbool_t p16, const f16 *bias, size_t nt,
                                                   size_t n) __arm_streaming __arm_inout("za") {
    svzero_za();
    svfloat16_t v_ones = svdup_n_f16((f16)1.0f);
    size_t bn0 = (nt + 0) * 32;
    size_t bn1 = (nt + 1) * 32;
    svbool_t pb0 = (bn0 < n) ? svwhilelt_b16((uint64_t)bn0, (uint64_t)n) : svpfalse_b();
    svbool_t pb1 = (bn1 < n) ? svwhilelt_b16((uint64_t)bn1, (uint64_t)n) : svpfalse_b();
    // Clamp the pad-tile address: pb1 may be all-false (no lane is read), but
    // forming bias + bn1 past one-past-end would still be UB.
    svmopa_za16_m(0, p16, pb0, v_ones, svld1_f16(pb0, bias + bn0));
    svmopa_za16_m(1, p16, pb1, v_ones, svld1_f16(pb1, (bn1 < n) ? bias + bn1 : bias));
}

// Narrow-N (n_tiles == 1, n <= 32) row-major fast path. The dual-tile lockstep in
// run_streaming pairs N-tiles (za0 = nt, za1 = nt+1); when there is only ONE
// N-tile, za1 accumulates the zeroed pad tile -- ~half the MOPAs are wasted (n=32
// measured ~2x slow per flop vs n=64). Here both ZA tiles stay productive by
// pairing two M-tiles against the single B-tile: za0 = A[mt] @ B, za1 = A[mt+1] @ B.
// Row-major only (the small-m output layout); col-major / strided n_tiles==1
// stays on the general path. Bias is NOT folded (applied as a normal node at the
// store), so the store reuses run_streaming's pure / node-major row-major
// primitives verbatim -- no new store codegen, and run_streaming is untouched.
// `b_pack` points at tile `nt` itself.
__arm_locally_streaming __arm_new("za") static void run_narrow_rowmajor(
    f16 *dst, long dst_cs, long dst_rs, const f16 *a_pack, const f16 *b_pack, size_t m, size_t n,
    size_t k, size_t mt_lo, size_t mt_hi, size_t nt, float alpha, float beta, int read_dst,
    const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    svcount_t pn = svptrue_c16();
    const svfloat16_t z16 = svdup_n_f16((f16)0.0f);
    const svfloat16_t va = svdup_n_f16((f16)alpha);
    const svfloat16_t vb = svdup_n_f16((f16)beta);
    size_t per_tile = ep_cmul(k, 32);
    int has_ep = ep && ep->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    // `nt` is the single N-tile this pass covers: tile 0 for the n<=32 case, or
    // the odd trailing tile when the dual-N loop has run out of pairs.
    size_t n0 = nt * 32;
    size_t ncols = (n - n0 < 32) ? (n - n0) : 32;
    svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
    const f16 *b0 = b_pack;

    for (size_t mt = mt_lo; mt < mt_hi; mt += 2) {
        const f16 *a0t = a_pack + mt * per_tile;
        int have1 = (mt + 1 < mt_hi);
        const f16 *a1t = have1 ? a_pack + (mt + 1) * per_tile : a0t;
        size_t m0_0 = mt * 32;
        size_t mrows0 = (m - m0_0 < 32) ? (m - m0_0) : 32;
        size_t m0_1 = (mt + 1) * 32;
        size_t mrows1 = have1 ? ((m - m0_1 < 32) ? (m - m0_1) : 32) : 0;

        // za0 = A[mt] @ B, za1 = A[mt+1] @ B. Both share the single B-vector per
        // depth, so the two MOPAs issue back-to-back with no RAW stall (distinct
        // tiles), exactly like the dual-N scheme -- but both outputs are real.
        svzero_za();
        size_t d = 0;
        for (; d + 4 <= k; d += 4) {
            svfloat16x4_t A0 = svld1_f16_x4(pn, a0t + d * 32);
            svfloat16x4_t A1 = svld1_f16_x4(pn, a1t + d * 32);
            svfloat16x4_t U = svld1_f16_x4(pn, b0 + d * 32);
            svfloat16_t u0 = svget4_f16(U, 0), u1 = svget4_f16(U, 1), u2 = svget4_f16(U, 2),
                        u3 = svget4_f16(U, 3);
            svmopa_za16_m(0, p16, p16, svget4_f16(A0, 0), u0);
            svmopa_za16_m(1, p16, p16, svget4_f16(A1, 0), u0);
            svmopa_za16_m(0, p16, p16, svget4_f16(A0, 1), u1);
            svmopa_za16_m(1, p16, p16, svget4_f16(A1, 1), u1);
            svmopa_za16_m(0, p16, p16, svget4_f16(A0, 2), u2);
            svmopa_za16_m(1, p16, p16, svget4_f16(A1, 2), u2);
            svmopa_za16_m(0, p16, p16, svget4_f16(A0, 3), u3);
            svmopa_za16_m(1, p16, p16, svget4_f16(A1, 3), u3);
        }
        for (; d < k; d++) {
            svfloat16_t u = svld1_f16(p16, b0 + d * 32);
            svmopa_za16_m(0, p16, p16, svld1_f16(p16, a0t + d * 32), u);
            svmopa_za16_m(1, p16, p16, svld1_f16(p16, a1t + d * 32), u);
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
            f16 *base0 = dst + (long)m0_0 * dst_rs + (long)n0 * dst_cs;
#define EP_RD0(s) svread_hor_za16_m(z16, p16, 0, (s))
            EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base0, dst_rs, EP_RD0, mrows0, vb, va, read_dst,
                                       nodes, n_nodes, m0_0, n0);
#undef EP_RD0
            if (have1) {
                f16 *base1 = dst + (long)m0_1 * dst_rs + (long)n0 * dst_cs;
#define EP_RD1(s) svread_hor_za16_m(z16, p16, 1, (s))
                EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base1, dst_rs, EP_RD1, mrows1, vb, va, read_dst,
                                           nodes, n_nodes, m0_1, n0);
#undef EP_RD1
            }
        } else {
            // beta != 1 or read_dst (no ep): scale / accumulate per row.
            for (size_t r = 0; r < mrows0; r++)
                store_slice(p16, pst, dst + (long)(m0_0 + r) * dst_rs + (long)n0 * dst_cs,
                            svread_hor_za16_m(z16, p16, 0, (uint32_t)r), vb, read_dst, va);
            if (have1)
                for (size_t r = 0; r < mrows1; r++)
                    store_slice(p16, pst, dst + (long)(m0_1 + r) * dst_rs + (long)n0 * dst_cs,
                                svread_hor_za16_m(z16, p16, 1, (uint32_t)r), vb, read_dst, va);
        }
    }
}

// m == 1: SME2 multi-vector FMLA into ZA vector groups (4 depths per op) instead
// of MOPA, which wastes 31/32 of its work on one row. `abc` is A broadcast [k][32];
// b_pack points at tile nt_lo.
#define GEMV_T 4
#define GEMV_MAXR 4 // rows; past this the MOPA path does fewer ZA ops
__arm_locally_streaming __arm_new("za") static void run_gemv(f16 *dst, long dst_rs, const f16 *abc,
                                                             size_t rows, const f16 *b_pack, size_t n,
                                                             size_t k, size_t nt_lo, size_t nt_hi,
                                                             float alpha, float beta, int read_dst,
                                                             const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    svcount_t pn = svptrue_c16();
    const svfloat16_t va = svdup_n_f16((f16)alpha);
    const svfloat16_t vb = svdup_n_f16((f16)beta);
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
        const f16 *b = b_pack + (nt - nt_lo) * per_tile;
        svzero_za();
        if (T == GEMV_T) {
            for (size_t d = 0; d < kfull; d += 4) {
                svfloat16x4_t B0 = svld1_f16_x4(pn, b + d * 32);
                svfloat16x4_t B1 = svld1_f16_x4(pn, b + per_tile + d * 32);
                svfloat16x4_t B2 = svld1_f16_x4(pn, b + 2 * per_tile + d * 32);
                svfloat16x4_t B3 = svld1_f16_x4(pn, b + 3 * per_tile + d * 32);
                for (size_t r = 0; r < rows; r++) {
                    svfloat16x4_t A = svld1_f16_x4(pn, abc + r * per_tile + d * 32);
                    uint32_t w = (uint32_t)(4 * r);
                    svmla_za16_f16_vg1x4(w, B0, A);
                    svmla_za16_f16_vg1x4(w + 1, B1, A);
                    svmla_za16_f16_vg1x4(w + 2, B2, A);
                    svmla_za16_f16_vg1x4(w + 3, B3, A);
                }
            }
        } else {
            for (size_t t = 0; t < T; t++)
                for (size_t d = 0; d < kfull; d += 4) {
                    svfloat16x4_t B = svld1_f16_x4(pn, b + t * per_tile + d * 32);
                    for (size_t r = 0; r < rows; r++)
                        svmla_za16_f16_vg1x4((uint32_t)(4 * r + t), B,
                                              svld1_f16_x4(pn, abc + r * per_tile + d * 32));
                }
        }
        if (kfull < k)
            for (size_t t = 0; t < T; t++) {
                svfloat16x4_t B = svld1_f16_x4(pt, b + t * per_tile + kfull * 32);
                for (size_t r = 0; r < rows; r++)
                    svmla_za16_f16_vg1x4((uint32_t)(4 * r + t), B,
                                          svld1_f16_x4(pt, abc + r * per_tile + kfull * 32));
            }
        for (size_t r = 0; r < rows; r++)
            for (size_t t = 0; t < T; t++) {
                size_t n0 = (nt + t) * 32;
                size_t ncols = n - n0 < 32 ? n - n0 : 32;
                svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                svfloat16x4_t q = svread_za16_f16_vg1x4((uint32_t)(4 * r + t));
                svfloat16_t acc = svadd_f16_x(p16, svadd_f16_x(p16, svget4(q, 0), svget4(q, 1)),
                                         svadd_f16_x(p16, svget4(q, 2), svget4(q, 3)));
                f16 *out = dst + (long)r * dst_rs + n0;
                if (pure) {
                    svst1_f16(pst, out, acc);
                } else if (has_ep) {
#define EP_GEMV_RD(s) acc
                    EP_STORE_TILE_ROWMAJOR_F16(p16, pst, out, dst_rs, EP_GEMV_RD, 1, vb, va, read_dst,
                                               nodes, n_nodes, r, n0);
#undef EP_GEMV_RD
                } else {
                    store_slice(p16, pst, out, acc, vb, read_dst, va);
                }
            }
    }
}

// Row-major-B GEMV read in place (`b_rm` is column nt_lo*32, row stride b_rs).
// One load is four tiles at one depth, so it takes the tuple-by-vector FMLA,
// which issues ~25% faster than run_gemv's tuple-by-tuple form; group (r, g, p)
// holds tile group g's depth phase p. Loads go depth-major across the pass's
// tile groups: down one group at a power-of-two stride every load hits the same
// L2 bank (1.8x slower, measured). Each tile still sums its four depth phases in
// run_gemv's order, so the two agree bit for bit.
//
// One pass's accumulate, inlined per row count so the row loops unroll and A
// stays in registers (a runtime count spills). BG = 4 / rows tile groups.
__attribute__((always_inline)) static inline void
gemv_rm_acc(size_t rows, const f16 *b, long b_rs, const f16 *abc, size_t per_tile, size_t k,
            size_t G, svcount_t pc0, svcount_t pc1, svcount_t pc2,
            svcount_t pc3) __arm_streaming __arm_inout("za") {
    svbool_t p16 = svptrue_b16();
    svcount_t pn = svptrue_c16();
    size_t BG = 4 / rows, kfull = k & ~(size_t)3;
#define GEMV_FMLA(r, p, a_)                                                                        \
    if (rows > (r)) {                                                                              \
        uint32_t w = (uint32_t)(4 * (r) * BG + (p));                                               \
        svmla_single_za16_f16_vg1x4(w, B0, a_);                                                    \
        if (G > 1) svmla_single_za16_f16_vg1x4(w + 4, B1, a_);                                     \
        if (G > 2) svmla_single_za16_f16_vg1x4(w + 8, B2, a_);                                     \
        if (G > 3) svmla_single_za16_f16_vg1x4(w + 12, B3, a_);                                    \
    }
#define GEMV_ROW(p, row, AV)                                                                       \
    {                                                                                              \
        svfloat16x4_t B0 = svld1_f16_x4(pc0, (row)), B1 = B0, B2 = B0, B3 = B0;                    \
        if (G > 1) B1 = svld1_f16_x4(pc1, (row) + 128);                                            \
        if (G > 2) B2 = svld1_f16_x4(pc2, (row) + 256);                                            \
        if (G > 3) B3 = svld1_f16_x4(pc3, (row) + 384);                                            \
        GEMV_FMLA(0, p, AV(0))                                                                     \
        GEMV_FMLA(1, p, AV(1))                                                                     \
        GEMV_FMLA(2, p, AV(2))                                                                     \
        GEMV_FMLA(3, p, AV(3))                                                                     \
    }
    size_t d = 0;
    for (; d < kfull; d += 4) {
        const f16 *b0 = b + (long)d * b_rs;
        svfloat16x4_t A0 = svld1_f16_x4(pn, abc + d * 32), A1 = A0, A2 = A0, A3 = A0;
        if (rows > 1) A1 = svld1_f16_x4(pn, abc + per_tile + d * 32);
        if (rows > 2) A2 = svld1_f16_x4(pn, abc + 2 * per_tile + d * 32);
        if (rows > 3) A3 = svld1_f16_x4(pn, abc + 3 * per_tile + d * 32);
#define AV0(r) svget4(A##r, 0)
#define AV1(r) svget4(A##r, 1)
#define AV2(r) svget4(A##r, 2)
#define AV3(r) svget4(A##r, 3)
        GEMV_ROW(0, b0, AV0)
        GEMV_ROW(1, b0 + b_rs, AV1)
        GEMV_ROW(2, b0 + 2 * b_rs, AV2)
        GEMV_ROW(3, b0 + 3 * b_rs, AV3)
#undef AV0
#undef AV1
#undef AV2
#undef AV3
    }
#define AVT(r) svld1_f16(p16, abc + (r) * per_tile + d * 32)
    for (; d < k; d++)
        GEMV_ROW(d - kfull, b + (long)d * b_rs, AVT)
#undef AVT
#undef GEMV_ROW
#undef GEMV_FMLA
}

__arm_locally_streaming __arm_new("za") static void run_gemv_rm(
    f16 *dst, long dst_rs, const f16 *abc, size_t rows, const f16 *b_rm, long b_rs, size_t n,
    size_t k, size_t nt_lo, size_t nt_hi, float alpha, float beta, int read_dst) {
    svbool_t p16 = svptrue_b16();
    const svfloat16_t va = svdup_n_f16((f16)alpha);
    const svfloat16_t vb = svdup_n_f16((f16)beta);
    int pure = (beta == 1.0f && !read_dst);
    size_t per_tile = ep_cmul(k, 32); // the stride between abc rows
    size_t BG = 4 / rows;             // tile groups (128 columns) per pass
    for (size_t nt = nt_lo; nt < nt_hi; nt += 4 * BG) {
        size_t G = (nt_hi - nt + 3) / 4 < BG ? (nt_hi - nt + 3) / 4 : BG;
        const f16 *b = b_rm + (nt - nt_lo) * 32;
        svcount_t pc0 = svwhilelt_c16((uint64_t)nt * 32, (uint64_t)n, 4);
        svcount_t pc1 = svwhilelt_c16((uint64_t)nt * 32 + 128, (uint64_t)n, 4);
        svcount_t pc2 = svwhilelt_c16((uint64_t)nt * 32 + 256, (uint64_t)n, 4);
        svcount_t pc3 = svwhilelt_c16((uint64_t)nt * 32 + 384, (uint64_t)n, 4);
        svzero_za();
        if (rows == 1)
            gemv_rm_acc(1, b, b_rs, abc, per_tile, k, G, pc0, pc1, pc2, pc3);
        else if (rows == 2)
            gemv_rm_acc(2, b, b_rs, abc, per_tile, k, G, pc0, pc1, pc2, pc3);
        else if (rows == 3)
            gemv_rm_acc(3, b, b_rs, abc, per_tile, k, G, pc0, pc1, pc2, pc3);
        else
            gemv_rm_acc(4, b, b_rs, abc, per_tile, k, G, pc0, pc1, pc2, pc3);
        for (size_t r = 0; r < rows; r++)
            for (size_t g = 0; g < G; g++) {
                uint32_t w = (uint32_t)(4 * (r * BG + g));
                svfloat16x4_t Q0 = svread_za16_f16_vg1x4(w), Q1 = svread_za16_f16_vg1x4(w + 1);
                svfloat16x4_t Q2 = svread_za16_f16_vg1x4(w + 2), Q3 = svread_za16_f16_vg1x4(w + 3);
                size_t ntg = nt + 4 * g, T = nt_hi - ntg < 4 ? nt_hi - ntg : 4;
            // Tile i = its four phase partials, summed in run_gemv's order.
#define GEMV_ST(i)                                                                                 \
    if (i < T) {                                                                                   \
        size_t n0 = (ntg + i) * 32;                                                                \
        svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)(n - n0 < 32 ? n - n0 : 32));          \
        svfloat16_t acc = svadd_f16_x(p16, svadd_f16_x(p16, svget4(Q0, i), svget4(Q1, i)),         \
                                      svadd_f16_x(p16, svget4(Q2, i), svget4(Q3, i)));             \
        f16 *out = dst + (long)r * dst_rs + n0;                                                    \
        if (pure)                                                                                  \
            svst1_f16(pst, out, acc);                                                              \
        else                                                                                       \
            store_slice(p16, pst, out, acc, vb, read_dst, va);                                     \
    }
                GEMV_ST(0)
                GEMV_ST(1)
                GEMV_ST(2)
                GEMV_ST(3)
#undef GEMV_ST
            }
    }
}

// abc[d*32 + j] = A[0, d].
static void bcast_row(f16 *abc, const f16 *a, long lhs_cs, size_t k) {
    for (size_t d = 0; d < k; d++) {
        float16x8_t v = vdupq_n_f16(a[(long)d * lhs_cs]);
        float16x8x4_t q = {{v, v, v, v}};
        vst1q_f16_x4(abc + d * 32, q);
    }
}

// =============================================================================
// run_streaming: the SME hot loop. Computes the tile-blocked GEMM + epilogue for
// the M-tile range [mt_lo, mt_hi) of this (cluster's) chunk.
//
// INVARIANTS / WHY IT IS SHAPED THIS WAY -- read before touching this function.
//
// STREAMING-MODE REGION. The whole body runs in SME streaming mode: the
// `__arm_locally_streaming` attribute makes the compiler emit SMSTART on entry
// and SMSTOP on return, and `__arm_new("za")` gives it a fresh ZA state. So the
// SMSTART/SMSTOP boundaries are the function boundaries -- everything between is
// streaming-SVE (SVL=512: 32 f16 lanes per vector) with ZA live. Consequences:
// the SVE register width is the streaming width, ZA must be zeroed/initialized
// before use (svzero_za / bias_init_za), and libm / non-streaming code is NOT
// callable here (hence the in-register transcendental approximations in
// epilogue.h). The caller pays this streaming entry once per chunk; run_batched
// elsewhere reuses one region across many items for the same reason.
//
// ZA TILE LAYOUT. The kernel keeps TWO independent 32x32 ZA16 (fp16) tiles in
// flight per N-pair: za0 accumulates N-tile `nt`, za1 accumulates `nt+1`. They
// run in lockstep so the two MOPAs per K-step issue back-to-back with no
// read-after-write stall on a single tile (the K-loop is unrolled by 4 so the
// za0->za0 RAW distance is fully hidden by the interleaved za1 work). Each MOPA
// is a rank-1 fp16 outer product consuming one K-value; the packed panels are
// [K,32] tile-major, so one depth step is 32 contiguous lanes.
//
// HARD-CODED ZA TILE INDICES. The tile number in every svmopa_za16_m(0/1, ...),
// svread_{hor,ver}_za16_m(.., 0/1, ..) and svst1_{hor,ver}_za16(0/1, ..) is an
// INSTRUCTION IMMEDIATE -- it is encoded in the opcode and CANNOT be a runtime
// variable. That is why tiles 0 and 1 are spelled out literally in two parallel
// code arms (the `if (t == 0) ... else ...` and the EP_RD0/EP_RD1 macro pair)
// instead of being indexed by a loop variable. Parameterizing "which ZA tile"
// costs the dual-tile pipelining that the immediate enables.
//
// STRUCTURE (one large function on purpose):
//   1. Setup: predicates, alpha/beta splats, the bias-fold decision (a leading
//      ADD_COL bias, with beta==1 && !read_dst, is folded into ZA via a rank-1
//      FMOPA so the store reverts to the pure fast path), and `pure` (overwrite,
//      no epilogue) detection.
//   2. Cache blocking: jc->ic->mt->nt loops sized so the A- and B-blocks fit the
//      shared L2 (when the whole problem fits, the outer loops run once).
//   3. K-loop: per (mt, nt-pair), zero/bias-init ZA, then the unrolled-by-4
//      multi-vector LD1H + 8-MOPA inner loop over K, plus a scalar K-tail.
//   4. Store / epilogue hook, per N-tile, branching on output layout:
//        - `pure`: direct single-instruction ZA->C store (st1h vertical for
//          col-major, horizontal for row-major) -- no read/scale/store.
//        - col-major with epilogue: read each ZA vertical slice, run the op-graph
//          (TENSOR ops route to the scalar path, being a strided gather).
//        - row-major with epilogue: the NODE-MAJOR register-resident store
//          (EP_STORE_TILE_ROWMAJOR_F16) -- read a 4-row block, dispatch each node
//          once per block. See epilogue.h.
//        - fully strided: materialize a 32x32 scratch tile and store scalar-wise.
//
// REGISTER-FRAGILE / DO NOT SPLIT. This function is intentionally monolithic. The
// store arms inline so the live ZA slice never bounces through memory, and the
// codegen here is sensitive to register pressure -- the bias-fold is restricted to
// the bias-ONLY case precisely because folding through the relu store perturbs
// this layout and costs more than it saves. Splitting run_streaming into smaller
// functions costs ~8% of shared GEMM throughput: the streaming/ZA state and the
// inlined store stop composing. The monolith is load-bearing; benchmark ANY
// restructuring against examples/bench.rs and examples/epilogue_bench.rs (and run
// the correctness suite) before committing.
// =============================================================================
// L2 blocking budget in 32-wide tiles.
static size_t f16_budget(size_t k) {
    size_t tile_bytes = ep_cmul(k, 32) * sizeof(f16);
    size_t b = (8u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    return b < 4 ? 4 : b;
}

// N-block width shared by the driver and run_streaming; even, split evenly.
static size_t f16_nc_blk(size_t k, size_t span) {
    size_t nc = (f16_budget(k) / 2) & ~(size_t)1;
    if (nc < 2) nc = 2;
    if (nc >= span) return span;
    size_t blocks = (span + nc - 1) / nc;
    size_t w = (span + blocks - 1) / blocks;
    return (w + 1) & ~(size_t)1;
}

// [nt_lo, nt_hi) is the N-tile (pad) sub-range this invocation owns -- pass
// (0, n_tiles_pad) for the whole problem, or an even-aligned slice for the
// N-parallel flat-M path (see gemm_sme_f16f16_run_packed). Must be pair-aligned
// (even) so the dual-tile lockstep never splits a pair across workers. `b_pack`
// points at tile nt_lo, so a caller can hand in one N-block's panel on its own.
__arm_locally_streaming __arm_new("za") static void run_streaming(
    f16 *dst, long dst_cs, long dst_rs, const f16 *a_pack, const f16 *b_pack, size_t m, size_t n,
    size_t k, size_t mt_lo, size_t mt_hi, size_t n_tiles, size_t nt_lo, size_t nt_hi, float alpha,
    float beta, int read_dst, const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    // SME2 multi-vector LD1H: one load pulls 4 contiguous depth-vectors of a
    // packed [k,32] panel into a z-group, cutting the inner loop from 12 single
    // loads to 3. Throughput-neutral (the MOPA unit is the bottleneck), but
    // fewer uops and less load-port pressure under multi-cluster contention.
    svcount_t pn = svptrue_c16();
    const svfloat16_t z16 = svdup_n_f16((f16)0.0f);
    const svfloat16_t va = svdup_n_f16((f16)alpha);
    const svfloat16_t vb = svdup_n_f16((f16)beta);
    size_t per_tile = ep_cmul(k, 32);
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);
    // Pure overwrite (C = A@B): the epilogue is just "ZA -> C", which the
    // SME `st1h {zaNv}` family does in ONE instruction per slice -- no
    // read/scale/store. This removes ~2/3 of the epilogue ops, the dominant
    // cost at small K (tall-skinny shapes).
    // A fused epilogue (bias/activation) needs the accumulator in registers,
    // so it cannot use the single-instruction direct-store path.
    // Fold a leading per-column bias (ADD_COL) into the ZA accumulator before the
    // K-loop MOPAs via a rank-1 FMOPA (ones[r] * bias[c] = bias[c]), then drop it
    // from the store-time op-graph. Requires beta==1 (the store applies beta*acc,
    // which would also scale a pre-loaded bias) and !read_dst. The fold adds bias
    // in the f16 ZA element domain (bias-first, then accumulate) instead of after
    // accumulation -- a rounding-order change that stays within the sqrt(k) test
    // tolerance for the f16f16-Fast path. Gated to a bias-ONLY graph: the bias
    // then vanishes (the store reverts to the direct single-instruction path) for
    // free. (bias+activation keeps the cheaper in-store add: folding it perturbs
    // the layout-sensitive store codegen and costs more on the relu store than
    // the fold saves, so the fold is restricted to the bias-only case.)
    int fold_bias = ep && ep->n_nodes == 1 && beta == 1.0f && !read_dst &&
                    ep->nodes[0].op == EP_OP_ADD_COL;
    const f16 *fold_bias_ptr = fold_bias ? (const f16 *)ep->nodes[0].ptr : NULL;
    const ep_desc16 *ep_eff = fold_bias ? NULL : ep;
    int has_ep = ep_eff && ep_eff->n_nodes > 0;
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    const EpNode *nodes = has_ep ? ep_eff->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep_eff->n_nodes : 0;
    int has_tensor = has_ep && ep_has_tensor(nodes, n_nodes);

    // Cache blocking (BLIS jc->ic->mt->nt). Each packed tile-panel is
    // k*32*2 bytes. Pick N- and M-block tile counts so the B-block and the
    // A-block both fit L2: the B-block stays resident across the inner M-tiles,
    // the A-block across the inner N-tiles, so each panel streams from DRAM once
    // instead of once per opposing tile. When the whole problem already fits,
    // the blocks span everything and the outer loops run once (no overhead vs
    // the flat path). This is what keeps 4096-and-up compute-bound instead of
    // collapsing to DRAM bandwidth.
    //
    // The 8 MB is sized to the SMALLER cluster, not the P-cluster: chunks go to
    // both, and on M5 the E-cluster L2 is 6 MB against the P-cluster's 16
    // (hw.perflevel1/0.l2cachesize), so a P-sized block thrashes the E workers
    // that share the loop. Sweeping 8 / 16 / 24 MB over 2048^3 through
    // 1024x8192x8192, f16 and bf16, moves nothing outside +-3% run-to-run noise
    // and not monotonically -- these shapes are MOPA-bound, so the budget is not
    // a live knob here. (Q4 is the exception: there it also sets how many full
    // dequant passes run, which is why 16 MB wins in gemm_f16f16_q4.h.)
    size_t budget_tiles = f16_budget(k);
    size_t nt_span = nt_hi - nt_lo;
    size_t nc = f16_nc_blk(k, nt_span);
    size_t mc = budget_tiles / 2;
    if (mc < 1) mc = 1;
    if (mc > mt_hi - mt_lo) mc = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc) {
        size_t jc_end = jc + nc < nt_hi ? jc + nc : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc) {
            size_t ic_end = ic + mc < mt_hi ? ic + mc : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const f16 *at = a_pack + mt * per_tile;
                size_t m0 = mt * 32;
                size_t mrows = (m - m0 < 32) ? (m - m0) : 32;

                for (size_t nt = jc; nt < jc_end; nt += 2) {
                    const f16 *b0 = b_pack + (nt - nt_lo) * per_tile;
                    const f16 *b1 = b_pack + (nt + 1 - nt_lo) * per_tile;

                    // Dual-tile accumulate: za0 and za1 are independent, so the two
                    // MOPAs per K-step issue back-to-back with no RAW stall. Unrolled
                    // by 4 so the A-loads run ahead of the MOPAs and the za0/za0 RAW
                    // distance is fully covered by the interleaved za1 work.
                    if (fold_bias) {
                        bias_init_za(p16, fold_bias_ptr, nt, n);
                    } else {
                        svzero_za();
                    }
                    size_t d = 0;
                    for (; d + 4 <= k; d += 4) {
                        // One multi-vector LD1H per panel loads 4 contiguous depth-
                        // vectors (z-group) -- 3 load instructions instead of 12, so
                        // the load port does not gate the 8 MOPAs.
                        svfloat16x4_t A = svld1_f16_x4(pn, at + d * 32);
                        svfloat16x4_t U = svld1_f16_x4(pn, b0 + d * 32);
                        svfloat16x4_t V = svld1_f16_x4(pn, b1 + d * 32);
                        svfloat16_t a0 = svget4_f16(A, 0), a1 = svget4_f16(A, 1),
                                    a2 = svget4_f16(A, 2), a3 = svget4_f16(A, 3);
                        svmopa_za16_m(0, p16, p16, a0, svget4_f16(U, 0));
                        svmopa_za16_m(1, p16, p16, a0, svget4_f16(V, 0));
                        svmopa_za16_m(0, p16, p16, a1, svget4_f16(U, 1));
                        svmopa_za16_m(1, p16, p16, a1, svget4_f16(V, 1));
                        svmopa_za16_m(0, p16, p16, a2, svget4_f16(U, 2));
                        svmopa_za16_m(1, p16, p16, a2, svget4_f16(V, 2));
                        svmopa_za16_m(0, p16, p16, a3, svget4_f16(U, 3));
                        svmopa_za16_m(1, p16, p16, a3, svget4_f16(V, 3));
                    }
                    for (; d < k; d++) {
                        svfloat16_t av = svld1_f16(p16, at + d * 32);
                        svmopa_za16_m(0, p16, p16, av, svld1_f16(p16, b0 + d * 32));
                        svmopa_za16_m(1, p16, p16, av, svld1_f16(p16, b1 + d * 32));
                    }

                    // Epilogue for each of the two N-tiles (skip the odd padding tile).
                    for (size_t t = 0; t < 2; t++) {
                        size_t ntt = nt + t;
                        if (ntt >= n_tiles) break;
                        size_t n0 = ntt * 32;
                        size_t ncols = (n - n0 < 32) ? (n - n0) : 32;
                        if (col_major) {
                            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)mrows);
                            if (pure) {
                                // Direct ZA vertical-slice -> contiguous M-column.
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
                                // contiguous M-vector load (ptr+m0); COL/SCALAR splat;
                                // TENSOR (col-strided) routes to the scalar path below.
                                for (size_t c = 0; c < ncols; c++) {
                                    f16 *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                                    svfloat16_t acc =
                                        (t == 0) ? svread_ver_za16_m(z16, p16, 0, (uint32_t)c)
                                                 : svread_ver_za16_m(z16, p16, 1, (uint32_t)c);
                                    if (has_ep && !has_tensor) {
                                        ep_store_f16(p16, pst, col, acc, vb, va, read_dst, nodes,
                                                     n_nodes, /*span_n=*/0, /*i=*/n0 + c, n0, m0);
                                    } else if (has_ep) {
                                        // TENSOR present: col-major tensor is a strided
                                        // gather, so materialize the M-column and apply
                                        // the whole op-graph scalar-wise.
                                        f16 tmp[32];
                                        svst1_f16(pst, tmp, acc);
                                        for (size_t r = 0; r < mrows; r++) {
                                            float v = (float)tmp[r] * beta;
                                            v = ep_apply_nodes_scalar_f16(nodes, n_nodes, v, m0 + r,
                                                                          n0 + c);
                                            f16 *cell = col + (long)r * dst_rs;
                                            *cell = (f16)(read_dst ? alpha * (float)(*cell) + v : v);
                                        }
                                    } else {
                                        store_slice(p16, pst, col, acc, vb, read_dst, va);
                                    }
                                }
                            }
                        } else if (row_major) {
                            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                            if (pure) {
                                // Direct ZA horizontal-slice -> contiguous N-row.
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
                                // per block across all its rows (invariant COL vector
                                // / scalars / act constants loaded once per node, not
                                // once per row); no memory bounce.
                                f16 *base = dst + (long)m0 * dst_rs + (long)n0 * dst_cs;
                                if (t == 0) {
#define EP_RD0(s) svread_hor_za16_m(z16, p16, 0, (s))
                                    EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, dst_rs, EP_RD0, mrows,
                                                               vb, va, read_dst, nodes, n_nodes, m0,
                                                               n0);
#undef EP_RD0
                                } else {
#define EP_RD1(s) svread_hor_za16_m(z16, p16, 1, (s))
                                    EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, dst_rs, EP_RD1, mrows,
                                                               vb, va, read_dst, nodes, n_nodes, m0,
                                                               n0);
#undef EP_RD1
                                }
                            } else {
                                for (size_t r = 0; r < mrows; r++) {
                                    f16 *rowp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                                    svfloat16_t acc =
                                        (t == 0) ? svread_hor_za16_m(z16, p16, 0, (uint32_t)r)
                                                 : svread_hor_za16_m(z16, p16, 1, (uint32_t)r);
                                    store_slice(p16, pst, rowp, acc, vb, read_dst, va);
                                }
                            }
                        } else {
                            f16 scratch[32 * 32];
                            svbool_t pg = svwhilelt_b16((uint64_t)0, (uint64_t)32);
                            for (uint32_t r = 0; r < 32; r++) {
                                svfloat16_t row = (t == 0) ? svread_hor_za16_m(z16, p16, 0, r)
                                                           : svread_hor_za16_m(z16, p16, 1, r);
                                svst1_f16(pg, scratch + (size_t)r * 32, row);
                            }
                            for (size_t r = 0; r < mrows; r++)
                                for (size_t c = 0; c < ncols; c++) {
                                    float ab = (float)scratch[r * 32 + c] * beta;
                                    if (has_ep)
                                        ab = ep_apply_nodes_scalar_f16(nodes, n_nodes, ab, m0 + r,
                                                                       n0 + c);
                                    f16 *cell =
                                        dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                                    *cell = (f16)(read_dst ? alpha * (float)(*cell) + ab : ab);
                                }
                        }
                    }
                }
            } // mt
        } // ic
    } // jc
}

// Row-major B read in place (`b` is B's column 0, row stride b_rs) for the tile
// pair (nt, nt+1): a B row holds both tiles contiguously -- one 128-byte line --
// so one x2 load per depth feeds the two MOPAs. With `cap` set the pass also
// writes the pair out as packed [2][k][32] tiles, so later M-tiles can run the
// packed kernel instead of re-walking B's rows. Row-major C only.
__arm_locally_streaming __arm_new("za") static void run_direct(
    f16 *dst, long dst_rs, const f16 *a_pack, const f16 *b, long b_rs, size_t m, size_t n, size_t k,
    size_t mt_lo, size_t mt_hi, size_t nt, f16 *cap, float alpha, float beta, int read_dst) {
    svbool_t p16 = svptrue_b16();
    svcount_t pn = svptrue_c16();
    const svfloat16_t z16 = svdup_n_f16((f16)0.0f);
    const svfloat16_t va = svdup_n_f16((f16)alpha);
    const svfloat16_t vb = svdup_n_f16((f16)beta);
    int pure = (beta == 1.0f && !read_dst);
    size_t per_tile = ep_cmul(k, 32), kfull = k & ~(size_t)3, n_tiles = (n + 31) / 32;
    const f16 *bt = b + nt * 32;
    svcount_t pc = svwhilelt_c16((uint64_t)nt * 32, (uint64_t)n, 2);
#define DIRECT_STEP(a, row, d)                                                                     \
    {                                                                                              \
        svfloat16x2_t B_ = svld1_f16_x2(pc, (row));                                                \
        svmopa_za16_m(0, p16, p16, (a), svget2(B_, 0));                                            \
        svmopa_za16_m(1, p16, p16, (a), svget2(B_, 1));                                            \
        if (cap) {                                                                                 \
            svst1_f16(p16, cap + (d) * 32, svget2(B_, 0));                                         \
            svst1_f16(p16, cap + per_tile + (d) * 32, svget2(B_, 1));                              \
        }                                                                                          \
    }
    for (size_t mt = mt_lo; mt < mt_hi; mt++, cap = NULL) {
        const f16 *at = a_pack + mt * per_tile;
        svzero_za();
        size_t d = 0;
        for (; d < kfull; d += 4) {
            svfloat16x4_t A = svld1_f16_x4(pn, at + d * 32);
            const f16 *r = bt + (long)d * b_rs;
            DIRECT_STEP(svget4(A, 0), r, d)
            DIRECT_STEP(svget4(A, 1), r + b_rs, d + 1)
            DIRECT_STEP(svget4(A, 2), r + 2 * b_rs, d + 2)
            DIRECT_STEP(svget4(A, 3), r + 3 * b_rs, d + 3)
        }
        for (; d < k; d++)
            DIRECT_STEP(svld1_f16(p16, at + d * 32), bt + (long)d * b_rs, d)
#undef DIRECT_STEP
        size_t m0 = mt * 32, mrows = m - m0 < 32 ? m - m0 : 32;
        for (size_t t = 0; t < 2 && nt + t < n_tiles; t++) {
            size_t n0 = (nt + t) * 32;
            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)(n - n0 < 32 ? n - n0 : 32));
            f16 *row = dst + (long)m0 * dst_rs + (long)n0;
            for (size_t r = 0; r < mrows; r++, row += dst_rs) {
                if (pure && t == 0)
                    svst1_hor_za16(0, (uint32_t)r, pst, row);
                else if (pure)
                    svst1_hor_za16(1, (uint32_t)r, pst, row);
                else
                    store_slice(p16, pst, row,
                                t ? svread_hor_za16_m(z16, p16, 1, (uint32_t)r)
                                  : svread_hor_za16_m(z16, p16, 0, (uint32_t)r),
                                vb, read_dst, va);
            }
        }
    }
}

// Pack A into [m_tiles, k, 32]. Fast paths for col-major (lhs_rs==1, memcpy)
// and row-major (lhs_cs==1, cache-blocked transpose); strided gather otherwise.
static void packa(f16 *a_pack, const f16 *a, size_t m, size_t k, long lhs_rs, long lhs_cs,
                  size_t mt_lo, size_t mt_hi, size_t per_tile) {
    if (lhs_rs == 1) {
        const size_t TB = 8; // tiles per block: bounds the write streams
        for (size_t tb = mt_lo; tb < mt_hi; tb += TB) {
            size_t tb_end = tb + TB < mt_hi ? tb + TB : mt_hi;
            for (size_t d = 0; d < k; d++) {
                const f16 *col = a + (long)d * lhs_cs;
                for (size_t mt = tb; mt < tb_end; mt++) {
                    size_t r0 = mt * 32;
                    size_t vr = (m - r0 < 32) ? (m - r0) : 32;
                    f16 *dp = a_pack + mt * per_tile + d * 32;
                    if (vr == 32) {
                        memcpy(dp, col + r0, 32 * sizeof(f16));
                    } else {
                        memcpy(dp, col + r0, vr * sizeof(f16));
                        for (size_t i = vr; i < 32; i++)
                            dp[i] = (f16)0.0f;
                    }
                }
            }
        }
    } else if (lhs_cs == 1) {
        // Row-major A: each row is contiguous in depth, which is exactly the
        // layout pack_lanes_depth_major transposes (rows as lanes).
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

// Pack B into [n_tiles_pad, k, 32]; caller zeroes pad tiles. Fast path for
// row-major weights (rhs_cs==1); strided gather otherwise.
static void packb(f16 *b_pack, const f16 *b, size_t n, size_t k, long rhs_rs, long rhs_cs,
                  size_t n_tiles, size_t per_tile) {
    if (rhs_cs == 1) {
        const size_t TB = 8;
        for (size_t tb = 0; tb < n_tiles; tb += TB) {
            size_t tb_end = tb + TB < n_tiles ? tb + TB : n_tiles;
            for (size_t d = 0; d < k; d++) {
                const f16 *row = b + (long)d * rhs_rs;
                for (size_t nt = tb; nt < tb_end; nt++) {
                    size_t c0 = nt * 32;
                    size_t vc = (n - c0 < 32) ? (n - c0) : 32;
                    memcpy(b_pack + nt * per_tile + d * 32, row + c0, vc * sizeof(f16));
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
static _Thread_local f16 *g_apack = NULL;
static _Thread_local size_t g_apack_cap = 0;
static f16 *apack_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1; // k==0: return a real (tiny) buffer, not a
                               // history-dependent NULL on fresh threads
    if (g_apack_cap < bytes) {
        free(g_apack);
        g_apack = (f16 *)malloc(bytes);
        g_apack_cap = g_apack ? bytes : 0;
    }
    return g_apack;
}

// Packed-B GEMM: caller supplies B pre-packed via gemm_sme_f16f16_packb. Only
// A is packed here -- the inference pattern (weights packed once, reused).
static int run_packed_core(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                           int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                           const uint16_t *b_pack, uint16_t alpha_bits, uint16_t beta_bits,
                           const ep_desc16 *ep) {
    if (m == 0 || n == 0) return 0;
    float alpha = read_dst ? (float)f16_from_bits(alpha_bits) : 0.0f;
    float beta = (float)f16_from_bits(beta_bits);
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    size_t n_tiles_pad = (n_tiles + 1) & ~(size_t)1;
    size_t per_tile = ep_cmul(k, 32);
    const f16 *a = (const f16 *)lhs;
    const f16 *bp = (const f16 *)b_pack;
    f16 *d = (f16 *)dst;

    int big = ep_flops(m, n, k) >= (1u << 21);
    if (m <= GEMV_MAXR && dst_cs == 1) {
        f16 *abc = apack_scratch(ep_cmul(ep_cmul(m, per_tile), sizeof(f16)));
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
          run_gemv(d, dst_rs, abc, m, bp + nt0 * per_tile, n, k, nt0, nt1, alpha, beta, read_dst,
                   ep);
        });
        return 0;
    }

    // n_tiles==1 (n<=32) row-major: the dual-tile scheme wastes za1 on the pad
    // N-tile (~2x slow per flop). run_narrow_rowmajor pairs two M-tiles against the
    // single B-tile instead. Same M-chunk parallelism; just a different inner loop.
    int narrow = (n_tiles == 1 && dst_cs == 1);
    // ODD N-TILE TAIL. run_streaming pairs N-tiles into za0/za1, so an odd tile
    // count leaves the final pair's za1 accumulating the zero pad tile -- it
    // wastes 1/(2*ceil(n_tiles/2)) of all MOPA issue, measured ~10% at
    // n_tiles=9 (3.86 vs 4.30/4.32 TF/s at the even neighbours). Handing that
    // last tile to run_narrow_rowmajor instead pairs two M-tiles against it, so
    // both ZA tiles stay productive. A-traffic is unchanged: the tile was
    // already being read once either way.
    //
    // Not when the bias folds. run_streaming folds a leading ADD_COL into ZA via
    // a rank-1 MOPA while run_narrow_rowmajor applies it at the store, so
    // splitting a folded call would round the two halves of one output
    // differently. Same predicate as `fold_bias` inside run_streaming.
    int fold_bias_here =
        ep && ep->n_nodes == 1 && beta == 1.0f && !read_dst && ep->nodes[0].op == EP_OP_ADD_COL;
    int odd_tail = (n_tiles > 1) && (n_tiles & 1) && dst_cs == 1 && !fold_bias_here;
    // Two tiles per chunk. The clusters are asymmetric, so fine chunks let
    // dispatch_apply work-steal instead of stalling on the slow half: 2 beats 4
    // by ~5% geomean and 9-13% at 256^3/512^3/4096^3. 1 beats 2 on 1024^3/2048^3
    // but loses 8% on 16384x512x512. Must stay EVEN -- run_narrow_rowmajor pairs
    // M-tiles, and an odd chunk would idle za1.
    size_t M_CHUNK = 2;
    size_t n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    // Flat-M (few M-tiles) but large and wide: M is the only parallel axis, so the
    // M-chunk scheme would run this on ONE cluster. Parallelize over N instead --
    // pack all of A once (cheap, m_tiles is small) into a shared buffer, then hand
    // even-aligned N-tile-pair chunks to both clusters. C columns per chunk are
    // disjoint; A and B are read-only and shared.
    // Tiles per chunk; must stay even so a dual-tile pair is never split. Every
    // chunk re-streams the whole A-panel for its N-range, so halving this to 2
    // buys parallelism more slowly than it adds traffic: 0.87-1.03x across
    // m<=64, n=64..512, k=4096..8192, worse on 5 of 7 shapes.
    size_t N_CHUNK = 4;
    size_t nn_chunks = (n_tiles_pad + N_CHUNK - 1) / N_CHUNK;
    // THREE chunks, not two. A 2-way split is slower than not splitting at all:
    // 1.03-1.15x serial across n=160..256, k=512..16384, m=1..64, both dtypes,
    // fading to parity by k=32768 where B-streaming dominates. The clusters are
    // asymmetric, so a 2-way split's makespan is the E-cluster's half, which
    // costs more than running the whole thing on the caller's cluster; from 3
    // chunks up dispatch_apply can work-steal and the split pays again.
    //
    // Not the same as the M-chunk path below, where a 2-way split *wins*
    // (forcing it serial measures 0.78-0.97x): those chunks pack their own A
    // slice, so a second cluster overlaps NEON packing with MOPA issue. Here A
    // is packed once up front and the chunks are pure MOPA, with nothing to
    // overlap. Nor is it the allocator -- swapping this malloc for the
    // thread-local scratch measures 1.00-1.01x.
    if (n_chunks <= 1 && big && nn_chunks >= 3) {
        f16 *a_pack = (f16 *)malloc(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(f16)));
        if (a_pack) {
            packa(a_pack, a, m, k, lhs_rs, lhs_cs, 0, m_tiles, per_tile);
            dispatch_apply(nn_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
              size_t nt0 = ci * N_CHUNK;
              size_t nt1 = nt0 + N_CHUNK < n_tiles_pad ? nt0 + N_CHUNK : n_tiles_pad;
              // The chunk holding the unpaired last tile stops one short and runs
              // that tile M-paired instead, as the serial path does. nt0 is a
              // multiple of N_CHUNK and n_tiles-1 is even here, so the truncated
              // range stays pair-aligned (and may be empty, which is fine).
              if (odd_tail && nt0 <= n_tiles - 1 && n_tiles - 1 < nt1) {
                  run_streaming(d, dst_cs, dst_rs, a_pack, bp + nt0 * per_tile, m, n, k, 0, m_tiles,
                                n_tiles, nt0, n_tiles - 1, alpha, beta, read_dst, ep);
                  run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp + (n_tiles - 1) * per_tile, m,
                                      n, k, 0, m_tiles, n_tiles - 1, alpha, beta, read_dst, ep);
              } else {
                  run_streaming(d, dst_cs, dst_rs, a_pack, bp + nt0 * per_tile, m, n, k, 0, m_tiles,
                                n_tiles, nt0, nt1, alpha, beta, read_dst, ep);
              }
            });
            free(a_pack);
            return 0;
        }
        // malloc failed: fall through to the serial path.
    }
    if (n_chunks <= 1 || !big) {
        // Serial: per-thread reusable scratch, no per-call malloc/free.
        f16 *a_pack = apack_scratch(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(f16)));
        if (!a_pack) return -1;
        packa(a_pack, a, m, k, lhs_rs, lhs_cs, 0, m_tiles, per_tile);
        if (narrow) {
            run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, 0, alpha, beta,
                                read_dst, ep);
        } else if (odd_tail) {
            run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, 0,
                          n_tiles - 1, alpha, beta, read_dst, ep);
            run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp + (n_tiles - 1) * per_tile, m, n, k,
                                0, m_tiles, n_tiles - 1, alpha, beta, read_dst, ep);
        } else {
            run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, 0,
                          n_tiles_pad, alpha, beta, read_dst, ep);
        }
        return 0;
    }
    f16 *a_pack = (f16 *)malloc(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(f16)));
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
    size_t nc_blk = f16_nc_blk(k, nt_end);
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
      run_streaming(d, dst_cs, dst_rs, a_pack, bp + jc * per_tile, m, n, k, mt0, mt1, n_tiles, jc,
                    jc_end, alpha, beta, read_dst, ep);
      if (odd_tail && jc_end == nt_end)
          run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp + (n_tiles - 1) * per_tile, m, n, k,
                              mt0, mt1, n_tiles - 1, alpha, beta, read_dst, ep);
    });
    free(a_pack);
    return 0;
}

// A trailing gelu/silu/sigmoid/tanh runs as a NEON pass over the output after
// the kernel (neon_act.h): in a streaming epilogue it costs ~2.5 ns an output.
int gemm_sme_f16f16_run_packed(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs,
                               long dst_rs, int read_dst, const uint16_t *lhs, long lhs_cs,
                               long lhs_rs, const uint16_t *b_pack, uint16_t alpha_bits,
                               uint16_t beta_bits, const ep_desc16 *ep) {
    ep_desc16 rest;
    uint32_t act = (dst_cs != 1 && dst_rs != 1) ? EP_ACT_NONE : na_split(ep, &rest);
    const ep_desc16 *kep = na_kernel_ep(ep, act, &rest);
    int rc = run_packed_core(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, b_pack,
                             alpha_bits, beta_bits, kep);
    if (rc == 0 && act) na_post_f16(dst, m, n, dst_rs, dst_cs, act);
    return rc;
}

// Pack B once into a caller-allocated, zeroed buffer of
// gemm_sme_f16f16_packed_b_elems(n,k) f16. Reuse across many GEMMs (weights).
void gemm_sme_f16f16_packb(uint16_t *b_pack, const uint16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs) {
    packb((f16 *)b_pack, (const f16 *)rhs, n, k, rhs_rs, rhs_cs, (n + 31) / 32, ep_cmul(k, 32));
}

// Saturating: SIZE_MAX on overflow (see gemm_sme_i8i32_packed_b_elems).
size_t gemm_sme_f16f16_packed_b_elems(size_t n, size_t k) {
    size_t n_tiles_pad = (((n + 31) / 32) + 1) & ~(size_t)1;
    return ep_cmul(ep_cmul(n_tiles_pad, k), 32);
}

// Pack N-tiles [t0, t1) of B into consecutive [k, 32] tiles at dst, zeroing
// columns past n.
static void pack_b_range(f16 *dst, const f16 *b, long rs, long cs, size_t n, size_t k, size_t t0,
                         size_t t1, size_t per_tile) {
    if (cs != 1) {
        for (size_t t = t0; t < t1; t++) {
            size_t vc = n - t * 32 < 32 ? n - t * 32 : 32;
            pack_b_tile(dst + (t - t0) * per_tile, b + (long)(t * 32) * cs, rs, cs, k, vc);
        }
        return;
    }
    size_t tf = t1 < n / 32 ? t1 : n / 32; // full tiles end
    for (size_t d = 0; d < k; d++) {
        const f16 *row = b + (long)d * rs;
        f16 *dp = dst + d * 32;

        for (size_t t = t0; t < tf; t++)
            memcpy(dp + (t - t0) * per_tile, row + t * 32, 32 * sizeof(f16));
        if (tf < t1) {
            size_t vc = n - tf * 32;
            memcpy(dp + (tf - t0) * per_tile, row + tf * 32, vc * sizeof(f16));
            memset(dp + (tf - t0) * per_tile + vc, 0, (32 - vc) * sizeof(f16));
        }
    }
}

// MOPA driver for a B built per call (plain f16 unpacked, Q4): each N-block of B
// is built once into a ring slot (panel_ring.h) and shared by every M-chunk.
// `fill(out, t0, t1)` writes N-tiles [t0, t1), t1 <= n_tiles, as [k, 32] tiles;
// a unit covers ~unit_vals values.
static int run_panels(f16 *dst, long dst_cs, long dst_rs, const f16 *a, long lhs_rs, long lhs_cs,
                      size_t m, size_t n, size_t k, float alpha, float beta, int read_dst,
                      const ep_desc16 *ep, size_t unit_vals,
                      void (^fill)(f16 *out, size_t t0, size_t t1)) {
    size_t m_tiles = (m + 31) / 32, n_tiles = (n + 31) / 32;
    size_t n_tiles_pad = (n_tiles + 1) & ~(size_t)1;
    size_t per_tile = ep_cmul(k, 32);
    // As gemm_sme_f16f16_run_packed: the odd last tile goes M-paired through
    // run_narrow_rowmajor, unless a lone bias folds into ZA.
    int fold_bias =
        ep && ep->n_nodes == 1 && beta == 1.0f && !read_dst && ep->nodes[0].op == EP_OP_ADD_COL;
    int odd_tail = (n_tiles & 1) && dst_cs == 1 && !fold_bias;
    size_t nt_end = odd_tail ? n_tiles - 1 : n_tiles_pad;
    size_t nc_blk = nt_end ? f16_nc_blk(k, nt_end) : 2;
    size_t n_blocks = nt_end ? (nt_end + nc_blk - 1) / nc_blk : 1;
    size_t M_CHUNK = 2, n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    // GEMM items per block: M-chunks x N-subranges, at least RING_ITEMS of them.
    size_t n_sub = (RING_ITEMS + n_chunks - 1) / n_chunks, max_sub = (nc_blk + 1) / 2;
    if (n_sub > max_sub) n_sub = max_sub;
    size_t ns = ((nc_blk + n_sub - 1) / n_sub + 1) & ~(size_t)1; // tiles per subrange
    n_sub = (nc_blk + ns - 1) / ns;
    size_t slot_tiles = nc_blk + 1; // the last block also carries the odd or pad tile
    size_t n_slots = n_blocks < RING_SLOTS ? n_blocks : RING_SLOTS;
    size_t dt = per_tile ? unit_vals / per_tile : 1;
    if (dt < 1) dt = 1;
    if (dt > slot_tiles) dt = slot_tiles;
    size_t n_d = (slot_tiles + dt - 1) / dt;

    size_t slot_elems = ep_cmul(slot_tiles, per_tile);
    _Atomic size_t *ctr;
    f16 *pool = (f16 *)ring_pool(2 * n_blocks + 2 + n_chunks,
                                 ep_cmul(ep_cmul(n_slots, slot_elems), sizeof(f16)), &ctr);
    f16 *a_pack = apack_scratch(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(f16)));
    if (!pool || !a_pack) return -1;
    _Atomic size_t *a_done = ctr + 2 * n_blocks + 2;
    if (n_chunks <= 1) packa(a_pack, a, m, k, lhs_rs, lhs_cs, 0, m_tiles, per_tile);

    ring_run(
        n_blocks, n_d, n_chunks * n_sub, ctr,
        ^(size_t sd, size_t j) {
          size_t lo = sd * nc_blk;
          f16 *slot = pool + (sd % RING_SLOTS) * slot_elems;
          size_t hi = sd + 1 < n_blocks ? lo + nc_blk : (odd_tail ? n_tiles : n_tiles_pad);
          size_t t0 = lo + j * dt, t1 = t0 + dt < hi ? t0 + dt : hi;
          size_t tq = t1 < n_tiles ? t1 : n_tiles;
          if (t0 < tq) fill(slot + (t0 - lo) * per_tile, t0, tq);
          for (size_t t = t0 > tq ? t0 : tq; t < t1; t++) // zero pad tile
              memset(slot + (t - lo) * per_tile, 0, per_tile * sizeof(f16));
        },
        ^(size_t sg, size_t j) {
          size_t lo = sg * nc_blk;
          f16 *slot = pool + (sg % RING_SLOTS) * slot_elems;
          size_t jc_end = lo + nc_blk < nt_end ? lo + nc_blk : nt_end;
          size_t ci = j / n_sub, sj = j % n_sub;
          size_t mt0 = ci * M_CHUNK, mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
          size_t n0 = lo + sj * ns < jc_end ? lo + sj * ns : jc_end;
          size_t n1 = n0 + ns < jc_end ? n0 + ns : jc_end;
          // Block 0's first item per M-chunk packs that chunk of A; the rest wait
          // only for the pack, not for the item's MOPAs.
          if (n_chunks > 1 && sg == 0 && sj == 0) {
              packa(a_pack, a, m, k, lhs_rs, lhs_cs, mt0, mt1, per_tile);
              atomic_store_explicit(&a_done[ci], 1, memory_order_release);
          } else if (n_chunks > 1) {
              while (!atomic_load_explicit(&a_done[ci], memory_order_acquire))
                  __builtin_arm_yield();
          }
          if (n0 < n1)
              run_streaming(dst, dst_cs, dst_rs, a_pack, slot + (n0 - lo) * per_tile, m, n, k, mt0,
                            mt1, n_tiles, n0, n1, alpha, beta, read_dst, ep);
          // The odd tail rides with each M-chunk's last subrange of the last block.
          if (odd_tail && jc_end == nt_end && sj == (jc_end > lo ? (jc_end - lo - 1) / ns : 0))
              run_narrow_rowmajor(dst, dst_cs, dst_rs, a_pack, slot + (n_tiles - 1 - lo) * per_tile,
                                  m, n, k, mt0, mt1, n_tiles - 1, alpha, beta, read_dst, ep);
        });
    return 0;
}

// Small m: each B tile meets few M-chunks, so a shared ring only adds a
// cross-core (often cross-cluster) round trip per tile. Instead each worker packs
// its own N-chunk into a private panel and multiplies it while it is in cache.
// The first units pack A by M-chunk, so that runs in parallel too.
static int run_cols(f16 *dst, long dst_cs, long dst_rs, const f16 *a, long lhs_rs, long lhs_cs,
                    size_t m, size_t n, size_t k, float alpha, float beta, int read_dst,
                    const f16 *b, long rhs_rs, long rhs_cs) {
    size_t m_tiles = (m + 31) / 32, n_tiles = (n + 31) / 32;
    size_t per_tile = ep_cmul(k, 32);
    int odd_tail = (n_tiles & 1) && dst_cs == 1;
    size_t nt_end = odd_tail ? n_tiles - 1 : (n_tiles + 1) & ~(size_t)1;
    size_t NC = 2; // tiles per item (even); 4 halves A traffic but loses 0.5-2x on balance
    size_t M_CHUNK = 2, n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    // Row-major B and C: multiply straight from B's rows (run_direct). Past two
    // M-tiles at a power-of-two row stride the first pass's loads all hit one L2
    // bank, and the NEON pack below wins instead (128x8192x4096: 4.78 against
    // 4.21 TF/s; at 64 rows, or a 24 KB stride, direct wins by 1.1-1.6x).
    size_t row_bytes = (size_t)rhs_rs * sizeof(f16);
    int direct = rhs_cs == 1 && dst_cs == 1 && rhs_rs > 0 &&
                 (m_tiles <= 2 || (row_bytes & (row_bytes - 1)) != 0);
    size_t n_items = direct ? (n_tiles + 1) / 2 : nt_end ? (nt_end + NC - 1) / NC : 1;
    size_t n_units = n_chunks + n_items;
    size_t W = n_units < 12 ? n_units : 12;
    size_t panel = ep_cmul(NC + 1, per_tile);
    _Atomic size_t *ctr;
    f16 *pool = (f16 *)ring_pool(2 + n_chunks, ep_cmul(ep_cmul(W, panel), sizeof(f16)), &ctr);
    f16 *a_pack = apack_scratch(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(f16)));
    if (!pool || !a_pack) return -1;
    _Atomic size_t *cur = ctr, *a_cnt = ctr + 1, *a_done = ctr + 2;
    dispatch_apply(W, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t w) {
      f16 *bt = pool + w * panel;
      for (size_t i; (i = atomic_fetch_add_explicit(cur, 1, memory_order_relaxed)) < n_units;) {
          if (i < n_chunks) {
              size_t mt1 = (i + 1) * M_CHUNK < m_tiles ? (i + 1) * M_CHUNK : m_tiles;
              packa(a_pack, a, m, k, lhs_rs, lhs_cs, i * M_CHUNK, mt1, per_tile);
              atomic_store_explicit(&a_done[i], 1, memory_order_release);
              atomic_fetch_add_explicit(a_cnt, 1, memory_order_release);
              continue;
          }
          size_t it = i - n_chunks, mt_lo = 0;
          size_t t0 = it * NC, t1 = t0 + NC < nt_end ? t0 + NC : nt_end;
          int tail = 0;
          f16 *bo = bt + NC * per_tile;
          if (direct) {
              // M-tile 0 multiplies straight from B's rows, capturing the pair as
              // packed tiles when more M-tiles follow; those run the packed kernel.
              t0 = 2 * it, t1 = t0 + 2, mt_lo = 1;
              while (!atomic_load_explicit(&a_done[0], memory_order_acquire))
                  __builtin_arm_yield();
              run_direct(dst, dst_rs, a_pack, b, rhs_rs, m, n, k, 0, 1, t0, m_tiles > 1 ? bt : NULL,
                         alpha, beta, read_dst);
          } else {
              size_t tq = t1 < n_tiles ? t1 : n_tiles;
              if (t0 < tq) pack_b_range(bt, b, rhs_rs, rhs_cs, n, k, t0, tq, per_tile);
              for (size_t t = t0 > tq ? t0 : tq; t < t1; t++) // zero pad tile
                  memset(bt + (t - t0) * per_tile, 0, per_tile * sizeof(f16));
              tail = odd_tail && it == n_items - 1;
              if (tail) pack_b_range(bo, b, rhs_rs, rhs_cs, n, k, n_tiles - 1, n_tiles, per_tile);
          }
          // All of A packed: one pass. Otherwise M-chunk by M-chunk as each lands.
          int all = atomic_load_explicit(a_cnt, memory_order_acquire) == n_chunks;
          for (size_t ci = 0; ci < (all ? 1 : n_chunks); ci++) {
              size_t mt0 = all ? 0 : ci * M_CHUNK;
              size_t mt1 = all ? m_tiles : (mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles);
              if (mt0 < mt_lo) mt0 = mt_lo;
              if (mt0 >= mt1) continue;
              if (!all)
                  while (!atomic_load_explicit(&a_done[ci], memory_order_acquire))
                      __builtin_arm_yield();
              if (t0 < t1)
                  run_streaming(dst, dst_cs, dst_rs, a_pack, bt, m, n, k, mt0, mt1, n_tiles, t0, t1,
                                alpha, beta, read_dst, NULL);
              if (tail)
                  run_narrow_rowmajor(dst, dst_cs, dst_rs, a_pack, bo, m, n, k, mt0, mt1,
                                      n_tiles - 1, alpha, beta, read_dst, NULL);
          }
      }
    });
    return 0;
}

// m <= GEMV_MAXR with B not row-major (transposed, strided): each worker packs
// its own N-chunk and runs the packed GEMV on it while it is in cache.
static int run_gemv_cols(f16 *dst, long dst_rs, const f16 *a, long lhs_rs, long lhs_cs, size_t m,
                         size_t n, size_t k, float alpha, float beta, int read_dst, const f16 *b,
                         long rhs_rs, long rhs_cs) {
    size_t n_tiles = (n + 31) / 32, per_tile = ep_cmul(k, 32);
    size_t G = 4, n_items = (n_tiles + G - 1) / G;
    size_t W = n_items < 12 ? n_items : 12;
    _Atomic size_t *cur;
    f16 *pool = (f16 *)ring_pool(1, ep_cmul(ep_cmul(W * G, per_tile), sizeof(f16)), &cur);
    f16 *abc = apack_scratch(ep_cmul(ep_cmul(m, per_tile), sizeof(f16)));
    if (!pool || !abc) return -1;
    for (size_t r = 0; r < m; r++)
        bcast_row(abc + r * per_tile, a + (long)r * lhs_rs, lhs_cs, k);
    dispatch_apply(W, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t w) {
      f16 *bt = pool + w * G * per_tile;
      for (size_t i; (i = atomic_fetch_add_explicit(cur, 1, memory_order_relaxed)) < n_items;) {
          size_t t0 = i * G, t1 = t0 + G < n_tiles ? t0 + G : n_tiles;
          pack_b_range(bt, b, rhs_rs, rhs_cs, n, k, t0, t1, per_tile);
          run_gemv(dst, dst_rs, abc, m, bt, n, k, t0, t1, alpha, beta, read_dst, NULL);
      }
    });
    return 0;
}

int gemm_sme_f16f16_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                        int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                        const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                        uint16_t beta_bits) {
    if (m == 0 || n == 0) return 0;
    size_t per_tile = ep_cmul(k, 32);
    int big = ep_flops(m, n, k) >= (1u << 21);
    if (m <= GEMV_MAXR && dst_cs == 1 && rhs_cs == 1 && rhs_rs > 0) {
        // Row-major B: the GEMV reads it in place, nothing to pack.
        f16 *d = (f16 *)dst;
        const f16 *b = (const f16 *)rhs;
        float alpha = read_dst ? (float)f16_from_bits(alpha_bits) : 0.0f;
        float beta = (float)f16_from_bits(beta_bits);
        f16 *abc = apack_scratch(ep_cmul(ep_cmul(m, per_tile), sizeof(f16)));
        if (!abc) return -1;
        for (size_t r = 0; r < m; r++)
            bcast_row(abc + r * per_tile, (const f16 *)lhs + (long)r * lhs_rs, lhs_cs, k);
        size_t n_tiles = (n + 31) / 32, G_CHUNK = 16, g_chunks = (n_tiles + G_CHUNK - 1) / G_CHUNK;
        if (!big || g_chunks < 3) {
            run_gemv_rm(d, dst_rs, abc, m, b, rhs_rs, n, k, 0, n_tiles, alpha, beta, read_dst);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t nt0 = ci * G_CHUNK, nt1 = nt0 + G_CHUNK < n_tiles ? nt0 + G_CHUNK : n_tiles;
          run_gemv_rm(d, dst_rs, abc, m, b + nt0 * 32, rhs_rs, n, k, nt0, nt1, alpha, beta,
                      read_dst);
        });
        return 0;
    }
    if (big) {
        f16 *d = (f16 *)dst;
        const f16 *a = (const f16 *)lhs, *b = (const f16 *)rhs;
        float alpha = read_dst ? (float)f16_from_bits(alpha_bits) : 0.0f;
        float beta = (float)f16_from_bits(beta_bits);
        if (m <= GEMV_MAXR && dst_cs == 1)
            return run_gemv_cols(d, dst_rs, a, lhs_rs, lhs_cs, m, n, k, alpha, beta, read_dst, b,
                                 rhs_rs, rhs_cs);
        if (use_cols(m, n, ep_cmul(ep_cmul(m, k), sizeof(f16))))
            return run_cols(d, dst_cs, dst_rs, a, lhs_rs, lhs_cs, m, n, k, alpha, beta, read_dst, b,
                            rhs_rs, rhs_cs);
        return run_panels(d, dst_cs, dst_rs, a, lhs_rs, lhs_cs, m, n, k, alpha, beta, read_dst,
                          NULL, (size_t)1 << 16, ^(f16 *out, size_t t0, size_t t1) {
                            pack_b_range(out, b, rhs_rs, rhs_cs, n, k, t0, t1, per_tile);
                          });
    }
    size_t n_tiles_pad = (((n + 31) / 32) + 1) & ~(size_t)1;
    f16 *b_pack = (f16 *)calloc(ep_cmul(n_tiles_pad, per_tile), sizeof(f16));
    if (!b_pack) return -1;
    packb(b_pack, (const f16 *)rhs, n, k, rhs_rs, rhs_cs, (n + 31) / 32, per_tile);
    int rc = gemm_sme_f16f16_run_packed(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs,
                                        (const uint16_t *)b_pack, alpha_bits, beta_bits, NULL);
    free(b_pack);
    return rc;
}

// Batched f16: `count` independent same-shape GEMMs in ONE streaming session, so
#include "gemm_f16f16_batched.h"

// The 4-bit-resident path, kept in its own file (it needs `packa` and the f16
// typedef, so it is included rather than compiled separately).
#include "gemm_f16f16_q4.h"
