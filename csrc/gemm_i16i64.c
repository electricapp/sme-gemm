// SME i16 GEMM driver (Apple M5+, FEAT_SME_I16I64, SMOPA i16*i16 -> i64).
// SMOPA accumulates 4 K-values per instruction into a 64-bit accumulator:
// ZA64[i,j] += sum_{s=0..3} A[i,4p+s]*B[4p+s,j]. The 32-lane i16 operand packs
// 8 rows x 4 K-slices, panels are [ceil(K/4), 32] 4-way interleaved, and a
// 16x16 super-tile is four 8x8 i64 ZA quadrants. Output is raw i64.

#include "epilogue.h"
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// Scalar tail for one (partial) p-group: dst[p*32 + 4*i + s] = src[i, 4p+s].
static inline void pack_band_scalar_p(int16_t *dst, const int16_t *src, long lane_stride,
                                      long depth_stride, size_t k, size_t valid, size_t p) {
    for (size_t i = 0; i < 8; i++)
        for (size_t s = 0; s < 4; s++) {
            size_t d = 4 * p + s;
            dst[p * 32 + 4 * i + s] =
                (i < valid && d < k) ? src[(long)i * lane_stride + (long)d * depth_stride] : 0;
        }
}

// Pack one 8-wide band into [ceil(K/4), 32]: dst[p*32 + 4*i + s] = src[i, 4p+s].
// NEON fast paths for the full-8-lane case (the 4-way SMOPA interleave):
//   * depth contiguous (row-major A): the 4 s-values are one 64-bit word per (p,i).
//   * lanes contiguous (row-major B): LD4 of 4 depth-vectors of 8 u16, ST4 to
//     interleave them straight into the layout.
static void pack_band(int16_t *dst, const int16_t *src, long lane_stride, long depth_stride,
                      size_t k, size_t valid) {
    size_t kp4 = (k + 3) / 4;
    size_t pfull = (k & ~(size_t)3) / 4; // full groups of 4 depths

    if (valid == 8 && depth_stride == 1) {
        for (size_t p = 0; p < pfull; p++) {
            const int16_t *sp = src + 4 * (long)p;
            int16_t *d = dst + p * 32;
            for (size_t i = 0; i < 8; i++)
                __builtin_memcpy(d + 4 * i, sp + (long)i * lane_stride, 8);
        }
    } else if (valid == 8 && lane_stride == 1) {
        for (size_t p = 0; p < pfull; p++) {
            uint16x8x4_t v;
            v.val[0] = vld1q_u16((const uint16_t *)(src + (long)(4 * p + 0) * depth_stride));
            v.val[1] = vld1q_u16((const uint16_t *)(src + (long)(4 * p + 1) * depth_stride));
            v.val[2] = vld1q_u16((const uint16_t *)(src + (long)(4 * p + 2) * depth_stride));
            v.val[3] = vld1q_u16((const uint16_t *)(src + (long)(4 * p + 3) * depth_stride));
            vst4q_u16((uint16_t *)(dst + p * 32), v);
        }
    } else {
        pfull = 0;
    }
    for (size_t p = pfull; p < kp4; p++)
        pack_band_scalar_p(dst, src, lane_stride, depth_stride, k, valid, p);
}

// One K-step's MOPAs for the live ZA64 quadrants of the 16x16 super-tile
// (0=lo-M x lo-N, 1=lo-M x hi-N, 2=hi-M x lo-N, 3=hi-M x hi-N; 8-wide bands).
// NARROW-N (nc <= 8) leaves the hi-N band all zero-pad -> za1/za3 dead; NARROW-M
// (mr <= 8) leaves the hi-M band all-pad -> za2/za3 dead. Issuing only the live
// quadrants halves SMOPA issue on those small-m shapes -- and since i16 SMOPA is
// MOPA-issue-bound (unlike the bandwidth-bound f32 path) this is a real win.
// The store reads ZA by nc/mr predicates, so it is unaffected; the dispatch is
// hoisted out of the K-loop.
#define I16_STEP_FULL(al, ah, bl, bh)                                                              \
    svmopa_za64_s16_m(0, pb, pb, al, bl);                                                          \
    svmopa_za64_s16_m(1, pb, pb, al, bh);                                                          \
    svmopa_za64_s16_m(2, pb, pb, ah, bl);                                                          \
    svmopa_za64_s16_m(3, pb, pb, ah, bh)
#define I16_STEP_NARROW_N(al, ah, bl)                                                              \
    svmopa_za64_s16_m(0, pb, pb, al, bl);                                                          \
    svmopa_za64_s16_m(2, pb, pb, ah, bl)
#define I16_STEP_NARROW_M(al, bl, bh)                                                              \
    svmopa_za64_s16_m(0, pb, pb, al, bl);                                                          \
    svmopa_za64_s16_m(1, pb, pb, al, bh)

// m == 1: SME2 multi-vector SDOT into ZA64 vector groups instead of SMOPA;
// see run_gemv in gemm_f16f16.c. 8-column bands of 4-deep groups, two per
// 16-wide tile; `abc` is [kp4][32]: A[4p..4p+3] repeated across the vector.
// Past one row the SMOPA is cheaper.
__arm_locally_streaming __arm_new("za") static void run_gemv(void *dst_v, const int16_t *abc,
                                                             const int16_t *b_pack, size_t n,
                                                             size_t kp4, size_t nt_lo,
                                                             size_t nt_hi, const ep_dq_f32 *dq) {
    svbool_t p64 = svptrue_b64();
    svbool_t p32 = svptrue_b32();
    svcount_t pn = svptrue_c16();
    size_t per_band = ep_cmul(kp4, 32);
    size_t pfull = kp4 & ~(size_t)3;
    svcount_t pt = svwhilelt_c16((uint64_t)0, (uint64_t)((kp4 - pfull) * 32), 4);
    for (size_t nt = nt_lo; nt < nt_hi; nt += 2) {
        size_t T = nt_hi - nt < 2 ? 2 : 4; // bands
        const int16_t *b = b_pack + 2 * nt * per_band;
        svzero_za();
        if (T == 4) {
            for (size_t p = 0; p < pfull; p += 4) {
                svint16x4_t A = svld1_s16_x4(pn, abc + p * 32);
                svdot_za64_s16_vg1x4(0, svld1_s16_x4(pn, b + p * 32), A);
                svdot_za64_s16_vg1x4(1, svld1_s16_x4(pn, b + per_band + p * 32), A);
                svdot_za64_s16_vg1x4(2, svld1_s16_x4(pn, b + 2 * per_band + p * 32), A);
                svdot_za64_s16_vg1x4(3, svld1_s16_x4(pn, b + 3 * per_band + p * 32), A);
            }
        } else {
            for (size_t t = 0; t < T; t++)
                for (size_t p = 0; p < pfull; p += 4)
                    svdot_za64_s16_vg1x4((uint32_t)t, svld1_s16_x4(pn, b + t * per_band + p * 32),
                                         svld1_s16_x4(pn, abc + p * 32));
        }
        if (pfull < kp4) {
            svint16x4_t A = svld1_s16_x4(pt, abc + pfull * 32);
            for (size_t t = 0; t < T; t++)
                svdot_za64_s16_vg1x4((uint32_t)t, svld1_s16_x4(pt, b + t * per_band + pfull * 32), A);
        }
        for (size_t t = 0; t < T; t += 2) {
            size_t n0 = (nt + t / 2) * 16;
            size_t nc = n - n0 < 16 ? n - n0 : 16;
            svint64x4_t ql = svread_za64_s64_vg1x4((uint32_t)t);
            svint64x4_t qh = svread_za64_s64_vg1x4((uint32_t)(t + 1));
            svint64_t lo = svadd_x(p64, svadd_x(p64, svget4(ql, 0), svget4(ql, 1)),
                                   svadd_x(p64, svget4(ql, 2), svget4(ql, 3)));
            svint64_t hi = svadd_x(p64, svadd_x(p64, svget4(qh, 0), svget4(qh, 1)),
                                   svadd_x(p64, svget4(qh, 2), svget4(qh, 3)));
            if (dq) {
                float *fdst = (float *)dst_v;
                svbool_t pst = svwhilelt_b32((uint32_t)0, (uint32_t)nc);
                svfloat32_t sc =
                    dq->scale_n ? svld1_f32(pst, dq->scale_n + n0) : svdup_n_f32(dq->scale);
                svfloat32_t v = svmul_f32_x(
                    p32, svuzp1_f32(svcvt_f32_s64_x(p64, lo), svcvt_f32_s64_x(p64, hi)), sc);
                v = ep_apply_nodes_f32(p32, pst, v, dq->nodes, dq->n_nodes, 1, 0, n0, 0);
                svst1_f32(pst, fdst + n0, v);
            } else {
                int64_t *rp = (int64_t *)dst_v + n0;
                svst1_s64(svwhilelt_b64((uint64_t)0, (uint64_t)(nc < 8 ? nc : 8)), rp, lo);
                if (nc > 8) svst1_s64(svwhilelt_b64((uint64_t)0, (uint64_t)(nc - 8)), rp + 8, hi);
            }
        }
    }
}

// abc[p*32 + 4j + s] = A[0, 4p+s], zero past k.
static void bcast_row(int16_t *abc, const int16_t *a, long lhs_cs, size_t k) {
    size_t kp4 = (k + 3) / 4;
    for (size_t p = 0; p < kp4; p++) {
        int16_t g[4];
        for (size_t s = 0; s < 4; s++) {
            size_t d = 4 * p + s;
            g[s] = d < k ? a[(long)d * lhs_cs] : 0;
        }
        uint64_t w;
        __builtin_memcpy(&w, g, 8);
        uint16x8_t v = vreinterpretq_u16_u64(vdupq_n_u64(w));
        uint16x8x4_t q = {{v, v, v, v}};
        vst1q_u16_x4((uint16_t *)(abc + p * 32), q);
    }
}

// L2 blocking budget in 16-wide super-tiles.
static size_t i16_budget(size_t kp4) {
    size_t tile_bytes = 2 * ep_cmul(kp4, 32) * sizeof(int16_t);
    size_t b = (size_t)(16u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    return b < 2 ? 2 : b;
}

// N-block width shared by the driver and run_streaming.
static size_t i16_nc_blk(size_t kp4, size_t n_tiles) {
    size_t nc = i16_budget(kp4) / 2;
    if (nc < 1) nc = 1;
    if (nc >= n_tiles) return n_tiles;
    size_t blocks = (n_tiles + nc - 1) / nc;
    return (n_tiles + blocks - 1) / blocks;
}

// [nt_lo, nt_hi) is the N-tile sub-range this invocation owns -- pass
// (0, n_tiles) for the whole problem, or a slice for the N-parallel flat-M path
// (see the driver). i16 N-tiles are independent (svzero_za per nt, no cross-tile
// state), so any chunk granularity is safe; the dq store writes disjoint columns.
__arm_locally_streaming __arm_new("za") static void run_streaming(
    void *dst_v, long dst_cs, long dst_rs, const int16_t *a_pack, const int16_t *b_pack, size_t m,
    size_t n, size_t mt_lo, size_t mt_hi, size_t nt_lo, size_t nt_hi, size_t kp4,
    const ep_dq_f32 *dq) {
    // dq == NULL: raw i64 output (the original path, bit-identical). dq set: read
    // the four i64 ZA64 quadrants into a scratch, dequantize -> f32 (scalar) and
    // write the f32 output buffer.
    int64_t *dst = (int64_t *)dst_v;
    svbool_t pb = svptrue_b16();
    svcount_t pn16 = svptrue_c16();
    svbool_t p64 = svptrue_b64();
    const svint64_t z64 = svdup_n_s64(0);
    size_t per_tile = ep_cmul(kp4, 32);
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);

    size_t budget = i16_budget(kp4);
    size_t nt_span = nt_hi - nt_lo;
    size_t nc_blk = i16_nc_blk(kp4, nt_span);
    size_t mc_blk = budget / 2;
    if (mc_blk < 1) mc_blk = 1;
    if (mc_blk > mt_hi - mt_lo) mc_blk = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc_blk) {
        size_t jc_end = jc + nc_blk < nt_hi ? jc + nc_blk : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc_blk) {
            size_t ic_end = ic + mc_blk < mt_hi ? ic + mc_blk : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const int16_t *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
                const int16_t *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
                size_t m0 = mt * 16;
                size_t mr = m - m0 < 16 ? m - m0 : 16;
                size_t mr_lo = mr < 8 ? mr : 8;
                size_t mr_hi = mr > 8 ? mr - 8 : 0;
                for (size_t nt = jc; nt < jc_end; nt++) {
                    const int16_t *b_lo = b_pack + (size_t)(2 * nt) * per_tile;
                    const int16_t *b_hi = b_pack + (size_t)(2 * nt + 1) * per_tile;
                    size_t n0 = nt * 16;
                    size_t nc = n - n0 < 16 ? n - n0 : 16;

                    svzero_za();
                    size_t p = 0;
                    if (nc <= 8) { // narrow-N: hi-N band is pad, za1/za3 dead
                        for (; p + 2 <= kp4; p += 2) {
                            svint16x2_t AL = svld1_s16_x2(pn16, a_lo + p * 32);
                            svint16x2_t AH = svld1_s16_x2(pn16, a_hi + p * 32);
                            svint16x2_t BL = svld1_s16_x2(pn16, b_lo + p * 32);
                            svint16_t al0 = svget2_s16(AL, 0), al1 = svget2_s16(AL, 1);
                            svint16_t ah0 = svget2_s16(AH, 0), ah1 = svget2_s16(AH, 1);
                            svint16_t bl0 = svget2_s16(BL, 0), bl1 = svget2_s16(BL, 1);
                            I16_STEP_NARROW_N(al0, ah0, bl0);
                            I16_STEP_NARROW_N(al1, ah1, bl1);
                        }
                        for (; p < kp4; p++) {
                            svint16_t al = svld1_s16(pb, a_lo + p * 32);
                            svint16_t ah = svld1_s16(pb, a_hi + p * 32);
                            svint16_t bl = svld1_s16(pb, b_lo + p * 32);
                            I16_STEP_NARROW_N(al, ah, bl);
                        }
                    } else if (mr <= 8) { // narrow-M: hi-M band is pad, za2/za3 dead
                        for (; p + 2 <= kp4; p += 2) {
                            svint16x2_t AL = svld1_s16_x2(pn16, a_lo + p * 32);
                            svint16x2_t BL = svld1_s16_x2(pn16, b_lo + p * 32);
                            svint16x2_t BH = svld1_s16_x2(pn16, b_hi + p * 32);
                            svint16_t al0 = svget2_s16(AL, 0), al1 = svget2_s16(AL, 1);
                            svint16_t bl0 = svget2_s16(BL, 0), bl1 = svget2_s16(BL, 1);
                            svint16_t bh0 = svget2_s16(BH, 0), bh1 = svget2_s16(BH, 1);
                            I16_STEP_NARROW_M(al0, bl0, bh0);
                            I16_STEP_NARROW_M(al1, bl1, bh1);
                        }
                        for (; p < kp4; p++) {
                            svint16_t al = svld1_s16(pb, a_lo + p * 32);
                            svint16_t bl = svld1_s16(pb, b_lo + p * 32);
                            svint16_t bh = svld1_s16(pb, b_hi + p * 32);
                            I16_STEP_NARROW_M(al, bl, bh);
                        }
                    } else {
                        for (; p + 2 <= kp4; p += 2) {
                            svint16x2_t AL = svld1_s16_x2(pn16, a_lo + p * 32);
                            svint16x2_t AH = svld1_s16_x2(pn16, a_hi + p * 32);
                            svint16x2_t BL = svld1_s16_x2(pn16, b_lo + p * 32);
                            svint16x2_t BH = svld1_s16_x2(pn16, b_hi + p * 32);
                            svint16_t al0 = svget2_s16(AL, 0), al1 = svget2_s16(AL, 1);
                            svint16_t ah0 = svget2_s16(AH, 0), ah1 = svget2_s16(AH, 1);
                            svint16_t bl0 = svget2_s16(BL, 0), bl1 = svget2_s16(BL, 1);
                            svint16_t bh0 = svget2_s16(BH, 0), bh1 = svget2_s16(BH, 1);
                            I16_STEP_FULL(al0, ah0, bl0, bh0);
                            I16_STEP_FULL(al1, ah1, bl1, bh1);
                        }
                        for (; p < kp4; p++) {
                            svint16_t al = svld1_s16(pb, a_lo + p * 32);
                            svint16_t ah = svld1_s16(pb, a_hi + p * 32);
                            svint16_t bl = svld1_s16(pb, b_lo + p * 32);
                            svint16_t bh = svld1_s16(pb, b_hi + p * 32);
                            I16_STEP_FULL(al, ah, bl, bh);
                        }
                    }

                    if (dq && row_major) {
                        // Vectorized fused dequant store (row-major): per row, read
                        // the two i64 ZA quadrants (cols 0-7, 8-15), convert to f32
                        // and uzp1-pack into one 16-lane f32 row, scale (per-tensor
                        // splat or per-N vector), run the f32 op-graph in-register,
                        // store -- no per-cell scalar bounce. (i64->f32 narrowing
                        // may lose precision past 24 bits, as the scalar path does.)
                        float *fdst = (float *)dst_v;
                        svbool_t p32 = svptrue_b32();
                        svbool_t pst = svwhilelt_b32((uint32_t)0, (uint32_t)nc);
                        svfloat32_t sc =
                            dq->scale_n ? svld1_f32(pst, dq->scale_n + n0) : svdup_n_f32(dq->scale);
                        const EpNode *nodes = dq->nodes;
                        uint32_t n_nodes = dq->n_nodes;
                        // ZA tile numbers are instruction immediates, so the lo/hi
                        // row halves are two fixed-tile loops (0,1 then 2,3).
                        for (size_t r = 0; r < mr_lo; r++) {
                            svfloat32_t lo =
                                svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 0, (uint32_t)r));
                            svfloat32_t hi =
                                svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 1, (uint32_t)r));
                            svfloat32_t row = svmul_f32_x(p32, svuzp1_f32(lo, hi), sc);
                            row = ep_apply_nodes_f32(p32, pst, row, nodes, n_nodes, 1, m0 + r, n0, m0);
                            svst1_f32(pst, fdst + (long)(m0 + r) * dst_rs + (long)n0, row);
                        }
                        for (size_t r = 8; r < mr; r++) {
                            uint32_t rr = (uint32_t)(r - 8);
                            svfloat32_t lo = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 2, rr));
                            svfloat32_t hi = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 3, rr));
                            svfloat32_t row = svmul_f32_x(p32, svuzp1_f32(lo, hi), sc);
                            row = ep_apply_nodes_f32(p32, pst, row, nodes, n_nodes, 1, m0 + r, n0, m0);
                            svst1_f32(pst, fdst + (long)(m0 + r) * dst_rs + (long)n0, row);
                        }
                    } else if (dq) {
                        // Col-major / strided dst: streaming mode has no scatter
                        // store, so only the strided write stays scalar -- the
                        // dequant and op-graph run vectorized into a contiguous
                        // row, as in the row-major arm above.
                        float *fdst = (float *)dst_v;
                        float row[16];
                        svbool_t p32 = svptrue_b32();
                        svbool_t pst = svwhilelt_b32((uint32_t)0, (uint32_t)nc);
                        svfloat32_t sc =
                            dq->scale_n ? svld1_f32(pst, dq->scale_n + n0) : svdup_n_f32(dq->scale);
                        const EpNode *nodes = dq->nodes;
                        uint32_t n_nodes = dq->n_nodes;
#define DQ_STRIDED_ROW_I16(TLO, THI, RR)                                                           \
    {                                                                                              \
        svfloat32_t vlo = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, TLO, RR));           \
        svfloat32_t vhi = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, THI, RR));           \
        svfloat32_t v = svmul_f32_x(p32, svuzp1_f32(vlo, vhi), sc);                                 \
        v = ep_apply_nodes_f32(p32, pst, v, nodes, n_nodes, 1, m0 + r, n0, m0);                     \
        svst1_f32(pst, row, v);                                                                    \
        for (size_t c = 0; c < nc; c++)                                                            \
            fdst[(long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs] = row[c];                       \
    }
                        for (size_t r = 0; r < mr_lo; r++) DQ_STRIDED_ROW_I16(0, 1, (uint32_t)r)
                        for (size_t r = 8; r < mr; r++) DQ_STRIDED_ROW_I16(2, 3, (uint32_t)(r - 8))
#undef DQ_STRIDED_ROW_I16
                    } else if (col_major) {
                        svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)mr_lo);
                        svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)mr_hi);
                        // Clamp the hi-band base: when the band is empty (phi
                        // all-false) no lane is stored, but forming col+8 past
                        // one-past-end is still UB (see gemm_f64.c).
                        for (size_t c = 0; c < (nc < 8 ? nc : 8); c++) {
                            int64_t *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            int64_t *colh = mr_hi ? col + 8 : col;
                            svst1_ver_za64(0, (uint32_t)c, plo, col);
                            svst1_ver_za64(2, (uint32_t)c, phi, colh);
                        }
                        for (size_t c = 8; c < nc; c++) {
                            uint32_t cc = (uint32_t)(c - 8);
                            int64_t *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            int64_t *colh = mr_hi ? col + 8 : col;
                            svst1_ver_za64(1, cc, plo, col);
                            svst1_ver_za64(3, cc, phi, colh);
                        }
                    } else if (row_major) {
                        svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)(nc < 8 ? nc : 8));
                        svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)(nc > 8 ? nc - 8 : 0));
                        int wide = nc > 8;
                        for (size_t r = 0; r < mr_lo; r++) {
                            int64_t *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            svst1_hor_za64(0, (uint32_t)r, plo, rp);
                            svst1_hor_za64(1, (uint32_t)r, phi, wide ? rp + 8 : rp);
                        }
                        for (size_t r = 8; r < mr; r++) {
                            uint32_t rr = (uint32_t)(r - 8);
                            int64_t *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            svst1_hor_za64(2, rr, plo, rp);
                            svst1_hor_za64(3, rr, phi, wide ? rp + 8 : rp);
                        }
                    } else {
                        int64_t scratch[16 * 16];
                        for (uint32_t r = 0; r < 8; r++) {
                            svst1_s64(p64, scratch + (size_t)r * 16,
                                      svread_hor_za64_s64_m(z64, p64, 0, r));
                            svst1_s64(p64, scratch + (size_t)r * 16 + 8,
                                      svread_hor_za64_s64_m(z64, p64, 1, r));
                            svst1_s64(p64, scratch + (size_t)(8 + r) * 16,
                                      svread_hor_za64_s64_m(z64, p64, 2, r));
                            svst1_s64(p64, scratch + (size_t)(8 + r) * 16 + 8,
                                      svread_hor_za64_s64_m(z64, p64, 3, r));
                        }
                        for (size_t r = 0; r < mr; r++)
                            for (size_t c = 0; c < nc; c++)
                                dst[(long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs] =
                                    scratch[r * 16 + c];
                    }
                }
            }
        }
    }
}

static _Thread_local int16_t *g_apack = NULL;
static _Thread_local size_t g_apack_cap = 0;
static int16_t *apack_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1; // k==0: return a real (tiny) buffer, not a
                               // history-dependent NULL on fresh threads
    if (g_apack_cap < bytes) {
        free(g_apack);
        g_apack = (int16_t *)malloc(bytes);
        g_apack_cap = g_apack ? bytes : 0;
    }
    return g_apack;
}

// malloc that never requests 0 bytes. With k==0 (per_tile==0) the B/A-pack
// sizes below are 0; a malloc(0) may return NULL, which the OOM checks would
// misread as a real allocation failure (spurious rc=-1 for a valid k==0 GEMM).
static void *xmalloc(size_t bytes) { return malloc(bytes ? bytes : 1); }

// Small-GEMM fast path: A pre-packed, B packed per-nt into a small scratch
// straight from row-major rhs (no full B-pack malloc/pass). See gemm_f64.c.
__arm_locally_streaming __arm_new("za") static void run_small(
    void *dst_v, long dst_cs, long dst_rs, const int16_t *a_pack, const int16_t *rhs, long rhs_rs,
    int16_t *b_scratch, size_t m, size_t n, size_t k, size_t mt_lo, size_t mt_hi, size_t kp4,
    const ep_dq_f32 *dq) {
    int64_t *dst = (int64_t *)dst_v;
    svbool_t pb = svptrue_b16();
    svcount_t pn16 = svptrue_c16();
    svbool_t p64 = svptrue_b64();
    const svint64_t z64 = svdup_n_s64(0);
    size_t per_tile = ep_cmul(kp4, 32);
    size_t n_tiles = (n + 15) / 16;
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);

    // N-tile OUTER so each B tile is packed once and reused across this chunk's
    // M-tiles; with mt inner it was repacked per (mt, nt). Each (mt, nt) is
    // independent (svzero_za + accumulate + store), so the order is free.
    for (size_t nt = 0; nt < n_tiles; nt++) {
        size_t n0 = nt * 16;
        size_t nc = n - n0 < 16 ? n - n0 : 16;
        size_t vc_lo = nc < 8 ? nc : 8;
        size_t vc_hi = nc > 8 ? nc - 8 : 0;
        int16_t *b_lo = b_scratch;
        int16_t *b_hi = b_scratch + per_tile;
        // Clamp the hi-band base when vc_hi==0 (no lane read): forming
        // rhs + n0 + 8 past one-past-end would be UB.
        pack_band(b_lo, rhs + (long)n0, 1, rhs_rs, k, vc_lo);
        pack_band(b_hi, vc_hi ? rhs + (long)n0 + 8 : rhs, 1, rhs_rs, k, vc_hi);
        for (size_t mt = mt_lo; mt < mt_hi; mt++) {
            const int16_t *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
            const int16_t *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
            size_t m0 = mt * 16;
            size_t mr = m - m0 < 16 ? m - m0 : 16;
            size_t mr_lo = mr < 8 ? mr : 8;
            size_t mr_hi = mr > 8 ? mr - 8 : 0;

            svzero_za();
            size_t p = 0;
            if (nc <= 8) { // narrow-N: hi-N band is pad, za1/za3 dead
                for (; p + 2 <= kp4; p += 2) {
                    svint16x2_t AL = svld1_s16_x2(pn16, a_lo + p * 32);
                    svint16x2_t AH = svld1_s16_x2(pn16, a_hi + p * 32);
                    svint16x2_t BL = svld1_s16_x2(pn16, b_lo + p * 32);
                    svint16_t al0 = svget2_s16(AL, 0), al1 = svget2_s16(AL, 1);
                    svint16_t ah0 = svget2_s16(AH, 0), ah1 = svget2_s16(AH, 1);
                    svint16_t bl0 = svget2_s16(BL, 0), bl1 = svget2_s16(BL, 1);
                    I16_STEP_NARROW_N(al0, ah0, bl0);
                    I16_STEP_NARROW_N(al1, ah1, bl1);
                }
                for (; p < kp4; p++) {
                    svint16_t al = svld1_s16(pb, a_lo + p * 32);
                    svint16_t ah = svld1_s16(pb, a_hi + p * 32);
                    svint16_t bl = svld1_s16(pb, b_lo + p * 32);
                    I16_STEP_NARROW_N(al, ah, bl);
                }
            } else if (mr <= 8) { // narrow-M: hi-M band is pad, za2/za3 dead
                for (; p + 2 <= kp4; p += 2) {
                    svint16x2_t AL = svld1_s16_x2(pn16, a_lo + p * 32);
                    svint16x2_t BL = svld1_s16_x2(pn16, b_lo + p * 32);
                    svint16x2_t BH = svld1_s16_x2(pn16, b_hi + p * 32);
                    svint16_t al0 = svget2_s16(AL, 0), al1 = svget2_s16(AL, 1);
                    svint16_t bl0 = svget2_s16(BL, 0), bl1 = svget2_s16(BL, 1);
                    svint16_t bh0 = svget2_s16(BH, 0), bh1 = svget2_s16(BH, 1);
                    I16_STEP_NARROW_M(al0, bl0, bh0);
                    I16_STEP_NARROW_M(al1, bl1, bh1);
                }
                for (; p < kp4; p++) {
                    svint16_t al = svld1_s16(pb, a_lo + p * 32);
                    svint16_t bl = svld1_s16(pb, b_lo + p * 32);
                    svint16_t bh = svld1_s16(pb, b_hi + p * 32);
                    I16_STEP_NARROW_M(al, bl, bh);
                }
            } else {
                for (; p + 2 <= kp4; p += 2) {
                    svint16x2_t AL = svld1_s16_x2(pn16, a_lo + p * 32);
                    svint16x2_t AH = svld1_s16_x2(pn16, a_hi + p * 32);
                    svint16x2_t BL = svld1_s16_x2(pn16, b_lo + p * 32);
                    svint16x2_t BH = svld1_s16_x2(pn16, b_hi + p * 32);
                    svint16_t al0 = svget2_s16(AL, 0), al1 = svget2_s16(AL, 1);
                    svint16_t ah0 = svget2_s16(AH, 0), ah1 = svget2_s16(AH, 1);
                    svint16_t bl0 = svget2_s16(BL, 0), bl1 = svget2_s16(BL, 1);
                    svint16_t bh0 = svget2_s16(BH, 0), bh1 = svget2_s16(BH, 1);
                    I16_STEP_FULL(al0, ah0, bl0, bh0);
                    I16_STEP_FULL(al1, ah1, bl1, bh1);
                }
                for (; p < kp4; p++) {
                    svint16_t al = svld1_s16(pb, a_lo + p * 32);
                    svint16_t ah = svld1_s16(pb, a_hi + p * 32);
                    svint16_t bl = svld1_s16(pb, b_lo + p * 32);
                    svint16_t bh = svld1_s16(pb, b_hi + p * 32);
                    I16_STEP_FULL(al, ah, bl, bh);
                }
            }

            if (dq && row_major) {
                // Vectorized fused dequant store (row-major), identical to
                // run_streaming's: read the two i64 ZA quadrants per row, convert
                // to f32, uzp1-pack into one 16-lane row, scale, run the f32
                // op-graph in-register, store. The scalar per-cell alternative
                // (ep_dequant_cell_i64) costs 2.3-3.4x here, the same as it does
                // on the >=2^25-flop run_streaming path.
                float *fdst = (float *)dst_v;
                svbool_t p32 = svptrue_b32();
                svbool_t pst = svwhilelt_b32((uint32_t)0, (uint32_t)nc);
                svfloat32_t sc =
                    dq->scale_n ? svld1_f32(pst, dq->scale_n + n0) : svdup_n_f32(dq->scale);
                const EpNode *nodes = dq->nodes;
                uint32_t n_nodes = dq->n_nodes;
                for (size_t r = 0; r < mr_lo; r++) {
                    svfloat32_t lo =
                        svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 0, (uint32_t)r));
                    svfloat32_t hi =
                        svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 1, (uint32_t)r));
                    svfloat32_t row = svmul_f32_x(p32, svuzp1_f32(lo, hi), sc);
                    row = ep_apply_nodes_f32(p32, pst, row, nodes, n_nodes, 1, m0 + r, n0, m0);
                    svst1_f32(pst, fdst + (long)(m0 + r) * dst_rs + (long)n0, row);
                }
                for (size_t r = 8; r < mr; r++) {
                    uint32_t rr = (uint32_t)(r - 8);
                    svfloat32_t lo = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 2, rr));
                    svfloat32_t hi = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 3, rr));
                    svfloat32_t row = svmul_f32_x(p32, svuzp1_f32(lo, hi), sc);
                    row = ep_apply_nodes_f32(p32, pst, row, nodes, n_nodes, 1, m0 + r, n0, m0);
                    svst1_f32(pst, fdst + (long)(m0 + r) * dst_rs + (long)n0, row);
                }
            } else if (dq) {
                // Strided dst: vectorized dequant + op-graph into a contiguous
                // row, scalar only for the strided write (no scatter in
                // streaming mode). Same as run_streaming's strided arm.
                float *fdst = (float *)dst_v;
                float row[16];
                svbool_t p32 = svptrue_b32();
                svbool_t pst = svwhilelt_b32((uint32_t)0, (uint32_t)nc);
                svfloat32_t sc =
                    dq->scale_n ? svld1_f32(pst, dq->scale_n + n0) : svdup_n_f32(dq->scale);
                const EpNode *nodes = dq->nodes;
                uint32_t n_nodes = dq->n_nodes;
#define DQ_STRIDED_ROW_I16(TLO, THI, RR)                                                           \
    {                                                                                              \
        svfloat32_t vlo = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, TLO, RR));           \
        svfloat32_t vhi = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, THI, RR));           \
        svfloat32_t v = svmul_f32_x(p32, svuzp1_f32(vlo, vhi), sc);                                 \
        v = ep_apply_nodes_f32(p32, pst, v, nodes, n_nodes, 1, m0 + r, n0, m0);                     \
        svst1_f32(pst, row, v);                                                                    \
        for (size_t c = 0; c < nc; c++)                                                            \
            fdst[(long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs] = row[c];                       \
    }
                for (size_t r = 0; r < mr_lo; r++) DQ_STRIDED_ROW_I16(0, 1, (uint32_t)r)
                for (size_t r = 8; r < mr; r++) DQ_STRIDED_ROW_I16(2, 3, (uint32_t)(r - 8))
#undef DQ_STRIDED_ROW_I16
            } else if (col_major) {
                svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)mr_lo);
                svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)mr_hi);
                // Clamp the hi-band base when it is empty (phi all-false): no
                // lane is stored, but forming col+8 past one-past-end is UB.
                for (size_t c = 0; c < vc_lo; c++) {
                    int64_t *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                    int64_t *colh = mr_hi ? col + 8 : col;
                    svst1_ver_za64(0, (uint32_t)c, plo, col);
                    svst1_ver_za64(2, (uint32_t)c, phi, colh);
                }
                for (size_t c = 8; c < nc; c++) {
                    uint32_t cc = (uint32_t)(c - 8);
                    int64_t *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                    int64_t *colh = mr_hi ? col + 8 : col;
                    svst1_ver_za64(1, cc, plo, col);
                    svst1_ver_za64(3, cc, phi, colh);
                }
            } else if (row_major) {
                svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)vc_lo);
                svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)vc_hi);
                for (size_t r = 0; r < mr_lo; r++) {
                    int64_t *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                    svst1_hor_za64(0, (uint32_t)r, plo, rp);
                    svst1_hor_za64(1, (uint32_t)r, phi, vc_hi ? rp + 8 : rp);
                }
                for (size_t r = 8; r < mr; r++) {
                    uint32_t rr = (uint32_t)(r - 8);
                    int64_t *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                    svst1_hor_za64(2, rr, plo, rp);
                    svst1_hor_za64(3, rr, phi, vc_hi ? rp + 8 : rp);
                }
            } else {
                int64_t scratch[16 * 16];
                for (uint32_t r = 0; r < 8; r++) {
                    svst1_s64(p64, scratch + (size_t)r * 16, svread_hor_za64_s64_m(z64, p64, 0, r));
                    svst1_s64(p64, scratch + (size_t)r * 16 + 8,
                              svread_hor_za64_s64_m(z64, p64, 1, r));
                    svst1_s64(p64, scratch + (size_t)(8 + r) * 16,
                              svread_hor_za64_s64_m(z64, p64, 2, r));
                    svst1_s64(p64, scratch + (size_t)(8 + r) * 16 + 8,
                              svread_hor_za64_s64_m(z64, p64, 3, r));
                }
                for (size_t r = 0; r < mr; r++)
                    for (size_t c = 0; c < nc; c++)
                        dst[(long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs] = scratch[r * 16 + c];
            }
        }
    }
}

// Shared i16 driver. `dq` NULL: raw i64 output to `dst` (bit-identical to the
// original path). `dq` set: `dst` is an f32 buffer; each i64 accumulator is
// dequantized (per-tensor `scale` or per-N `scale_n`) and the op-graph applied
// in f32 at the store (scalar path), f32 output. Riding the existing non-packed
// i16 run -- no new packed path.
size_t gemm_sme_i16i64_packed_b_elems(size_t n, size_t k);
void gemm_sme_i16i64_packb(int16_t *b_pack, const int16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs);
int gemm_sme_i16i64_run_packed_impl(size_t m, size_t n, size_t k, void *dst, long dst_cs,
                                    long dst_rs, const int16_t *lhs, long lhs_cs, long lhs_rs,
                                    const int16_t *b_pack, const ep_dq_f32 *dq);

static int gemm_sme_i16i64_run_impl(size_t m, size_t n, size_t k, void *dst, long dst_cs,
                                    long dst_rs, const int16_t *lhs, long lhs_cs, long lhs_rs,
                                    const int16_t *rhs, long rhs_cs, long rhs_rs,
                                    const ep_dq_f32 *dq) {
    if (m == 0 || n == 0) return 0;
    size_t kp4 = (k + 3) / 4;
    size_t m_tiles = (m + 15) / 16;
    size_t per_tile = ep_cmul(kp4, 32);

    // Small + row-major B: pack B per-tile into a small scratch (no full B-pack
    // malloc/pass), light dispatch. See gemm_f64.c.
#define PACKA_SMALL(AP, MT0, MT1)                                                                   \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                             \
        size_t r0 = st * 8;                                                                        \
        size_t vr = (r0 < m) ? ((m - r0 < 8) ? (m - r0) : 8) : 0;                                  \
        pack_band((AP) + st * per_tile, vr ? lhs + (long)r0 * lhs_rs : lhs, lhs_rs, lhs_cs, k, vr); \
    }
    if (rhs_cs == 1 && ep_flops(m, n, k) < (1u << 25)) {
        size_t sc_chunk = 2;
        size_t sc_n = (m_tiles + sc_chunk - 1) / sc_chunk;
        if (sc_n <= 1) {
            int16_t *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(int16_t)));
            int16_t *b_scratch = a_pack ? (int16_t *)xmalloc(ep_cmul(2 * per_tile, sizeof(int16_t))) : NULL;
            if (a_pack && b_scratch) {
                PACKA_SMALL(a_pack, 0, m_tiles);
                run_small(dst, dst_cs, dst_rs, a_pack, rhs, rhs_rs, b_scratch, m, n, k, 0, m_tiles,
                          kp4, dq);
                free(b_scratch);
                return 0;
            }
            free(b_scratch);
        } else {
            int16_t *a_pack = (int16_t *)xmalloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(int16_t)));
            // One contiguous per-chunk B scratch, allocated UP FRONT: workers
            // must not allocate, because a worker-side failure after other
            // chunks have already stored their rows would break the documented
            // "rc != 0 => C untouched" contract, and a shared __block fail flag
            // is a C11 data race. On failure fall through to the main path.
            int16_t *b_scratch_all =
                a_pack ? (int16_t *)xmalloc(ep_cmul(ep_cmul(sc_n * 2, per_tile), sizeof(int16_t))) : NULL;
            if (a_pack && b_scratch_all) {
                dispatch_apply(
                    sc_n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
                      size_t mt0 = ci * sc_chunk;
                      size_t mt1 = mt0 + sc_chunk < m_tiles ? mt0 + sc_chunk : m_tiles;
                      int16_t *b_scratch = b_scratch_all + ci * 2 * per_tile;
                      PACKA_SMALL(a_pack, mt0, mt1);
                      run_small(dst, dst_cs, dst_rs, a_pack, rhs, rhs_rs, b_scratch, m, n, k, mt0,
                                mt1, kp4, dq);
                    });
                free(b_scratch_all);
                free(a_pack);
                return 0;
            }
            free(b_scratch_all);
            free(a_pack);
        }
    }
#undef PACKA_SMALL

    int16_t *b_pack =
        (int16_t *)xmalloc(ep_cmul(gemm_sme_i16i64_packed_b_elems(n, k), sizeof(int16_t)));
    if (!b_pack) return -1;
    gemm_sme_i16i64_packb(b_pack, rhs, n, k, rhs_rs, rhs_cs);

    int rc = gemm_sme_i16i64_run_packed_impl(m, n, k, dst, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs,
                                             b_pack, dq);
    free(b_pack);
    return rc;
}

// Elements in a packed-B buffer for (n, k). Saturating (see gemm_i8i32.c).
size_t gemm_sme_i16i64_packed_b_elems(size_t n, size_t k) {
    return ep_cmul(2 * ((n + 15) / 16), ep_cmul((k + 3) / 4, 32));
}

// Pack B once into a caller-allocated buffer of gemm_sme_i16i64_packed_b_elems
// int16: [2*n_tiles][ceil(k/4), 32] 4-way-interleaved 8-wide bands.
// Row-major B, four full bands at once. One band alone reads 8 lanes = 16 bytes
// per (depth, band), so it uses a quarter of each 64-byte line and comes back for
// the rest on the next three bands -- and at large n the rows are far enough
// apart that the line is gone by then. Four bands together consume the whole line
// on first touch. (i8 does not need this: its bands are 16 lanes of one byte, so
// a single band already reads a natural chunk.)
static void pack_bands4_rowmajor(int16_t *dst, size_t per_tile, const int16_t *src,
                                 long depth_stride, size_t k) {
    size_t pfull = (k & ~(size_t)3) / 4;
    for (size_t p = 0; p < pfull; p++) {
        uint16x8_t w[4][4];
        for (size_t s = 0; s < 4; s++) {
            const uint16_t *row = (const uint16_t *)(src + (long)(4 * p + s) * depth_stride);
            for (size_t g = 0; g < 4; g++)
                w[s][g] = vld1q_u16(row + 8 * g);
        }
        for (size_t g = 0; g < 4; g++) {
            uint16x8x4_t v = {{w[0][g], w[1][g], w[2][g], w[3][g]}};
            vst4q_u16((uint16_t *)(dst + g * per_tile + p * 32), v);
        }
    }
    // K tail: the scalar per-group filler, per band.
    for (size_t g = 0; g < 4; g++)
        for (size_t p = pfull; p < (k + 3) / 4; p++)
            pack_band_scalar_p(dst + g * per_tile, src + 8 * (long)g, 1, depth_stride, k, 8, p);
}

void gemm_sme_i16i64_packb(int16_t *b_pack, const int16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs) {
    size_t per_tile = ep_cmul((k + 3) / 4, 32);
    size_t n_tiles = (n + 15) / 16;
    size_t st = 0;
    if (rhs_cs == 1) {
        for (; st + 4 <= 2 * n_tiles && (st + 4) * 8 <= n; st += 4)
            pack_bands4_rowmajor(b_pack + st * per_tile, per_tile, rhs + (long)st * 8, rhs_rs, k);
    }
    for (; st < 2 * n_tiles; st++) {
        size_t c0 = st * 8;
        size_t vc = (c0 < n) ? ((n - c0 < 8) ? (n - c0) : 8) : 0;
        // vc==0 bands are zero-filled without reading src; clamp the base.
        pack_band(b_pack + st * per_tile, vc ? rhs + (long)c0 * rhs_cs : rhs, rhs_cs, rhs_rs, k, vc);
    }
}

// Packed-B i16 GEMM (raw i64 when dq is NULL, fused dequant to f32 otherwise).
// No run_small arm -- that packs B per N-tile from a row-major rhs.
int gemm_sme_i16i64_run_packed_impl(size_t m, size_t n, size_t k, void *dst, long dst_cs,
                                    long dst_rs, const int16_t *lhs, long lhs_cs, long lhs_rs,
                                    const int16_t *b_pack, const ep_dq_f32 *dq) {
    if (m == 0 || n == 0) return 0;
    size_t kp4 = (k + 3) / 4;
    size_t m_tiles = (m + 15) / 16;
    size_t n_tiles = (n + 15) / 16;
    size_t per_tile = ep_cmul(kp4, 32);

#define PACKA_RANGE(AP, MT0, MT1)                                                                  \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                             \
        size_t r0 = st * 8;                                                                        \
        size_t vr = (r0 < m) ? ((m - r0 < 8) ? (m - r0) : 8) : 0;                                  \
        pack_band((AP) + st * per_tile, vr ? lhs + (long)r0 * lhs_rs : lhs, lhs_rs, lhs_cs, k, vr); \
    }

    size_t M_CHUNK = 2;
    size_t n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    int big = ep_flops(m, n, k) >= (1u << 21);
    if (m == 1 && dst_cs == 1) {
        int16_t *abc = apack_scratch(ep_cmul(per_tile, sizeof(int16_t)));
        if (!abc) return -1;
        bcast_row(abc, lhs, lhs_cs, k);
        size_t G_CHUNK = 4; // tiles per chunk
        size_t g_chunks = (n_tiles + G_CHUNK - 1) / G_CHUNK;
        if (!big || g_chunks < 3) {
            run_gemv(dst, abc, b_pack, n, kp4, 0, n_tiles, dq);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t nt0 = ci * G_CHUNK;
          size_t nt1 = nt0 + G_CHUNK < n_tiles ? nt0 + G_CHUNK : n_tiles;
          run_gemv(dst, abc, b_pack, n, kp4, nt0, nt1, dq);
        });
        return 0;
    }

    // Flat-M (few M-tiles) but large/wide: M is the only M-chunk axis, so the
    // M-parallel scheme below would run this on ONE cluster -- exactly the
    // quantized small-m shape (n/k large). Parallelize over N
    // instead: pack all of A once (cheap; m_tiles small) into a shared buffer and
    // hand N-tile chunks to both clusters. C columns are disjoint per chunk; A and
    // B (and dq) are read-only and shared. Needs >= 2 chunks to beat serial.
    size_t N_CHUNK = 4;
    size_t nn_chunks = (n_tiles + N_CHUNK - 1) / N_CHUNK;
    if (n_chunks <= 1 && big && nn_chunks >= 2) {
        int16_t *a_pack = (int16_t *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(int16_t)));
        if (a_pack) {
            PACKA_RANGE(a_pack, 0, m_tiles);
            dispatch_apply(nn_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
              size_t nt0 = ci * N_CHUNK;
              size_t nt1 = nt0 + N_CHUNK < n_tiles ? nt0 + N_CHUNK : n_tiles;
              run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, nt0, nt1, kp4, dq);
            });
            free(a_pack);
            return 0;
        }
        // malloc failed: fall through to the serial path.
    }
    if (n_chunks <= 1 || !big) {
        int16_t *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(int16_t)));
        if (!a_pack) return -1;
        PACKA_RANGE(a_pack, 0, m_tiles);
        run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, 0, n_tiles, kp4, dq);
        return 0;
    }
    int16_t *a_pack = (int16_t *)xmalloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(int16_t)));
    if (!a_pack) return -1;
    size_t nc_blk = i16_nc_blk(kp4, n_tiles);
    size_t n_blocks = (n_tiles + nc_blk - 1) / nc_blk;
    if (n_blocks > 1)
        dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t mt0 = ci * M_CHUNK;
          PACKA_RANGE(a_pack, mt0, mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles);
        });
    // One flat dispatch over (N-block, M-chunk), block-major: no barrier per
    // block, so an E-cluster worker (~9x slower at SME) only delays the last item.
    dispatch_apply(n_blocks * n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t it) {
      size_t ci = it % n_chunks, jc = it / n_chunks * nc_blk;
      size_t mt0 = ci * M_CHUNK;
      size_t mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
      size_t jc_end = jc + nc_blk < n_tiles ? jc + nc_blk : n_tiles;
      if (n_blocks == 1) PACKA_RANGE(a_pack, mt0, mt1);
      run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, mt0, mt1, jc, jc_end, kp4, dq);
    });
#undef PACKA_RANGE
    free(a_pack);
    return 0;
}

// Raw i64 i16 GEMM (unchanged ABI): thin wrapper over the impl with no dq.
int gemm_sme_i16i64_run(size_t m, size_t n, size_t k, int64_t *dst, long dst_cs, long dst_rs,
                        const int16_t *lhs, long lhs_cs, long lhs_rs, const int16_t *rhs,
                        long rhs_cs, long rhs_rs) {
    return gemm_sme_i16i64_run_impl(m, n, k, dst, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs,
                                    rhs_rs, NULL);
}

// Fused-dequant i16 GEMM: i16*i16 -> i64 accumulate, then the f32 op-graph
// D(f32) = nodes(scale*acc) written to `dst` (f32). Rides the existing non-packed
// i16 run; only the output stage differs.
int gemm_sme_i16i64_run_dequant(size_t m, size_t n, size_t k, float *dst, long dst_cs, long dst_rs,
                                const int16_t *lhs, long lhs_cs, long lhs_rs, const int16_t *rhs,
                                long rhs_cs, long rhs_rs, float scale, const float *scale_n,
                                uint32_t n_nodes, const EpNode *nodes) {
    ep_dq_f32 dq = {scale, scale_n, n_nodes, nodes};
    return gemm_sme_i16i64_run_impl(m, n, k, dst, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs,
                                    rhs_rs, &dq);
}

// Batched i16: `count` independent same-shape GEMMs in ONE streaming session, so
#include "gemm_i16i64_batched.h"
