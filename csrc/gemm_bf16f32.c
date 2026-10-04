// SME bf16 GEMM driver (Apple M4+, base FEAT_SME, widening bf16->fp32).
// Structurally identical to the f16->f32 widening driver: pair-interleaved
// [K/2, 32] packing, four independent ZA32 quadrants per 32x32 super-tile, one
// streaming session, vectorized epilogue. Only the element type and the
// MOPA/convert intrinsics differ (BFMOPA; bf16<->f32 conversions).
//
// Computes dst = alpha*dst + beta*(A @ B), bf16 in/out, fp32 accumulate.

#include "panel_ring.h"
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

// Saturating m*n*k flop estimate for the parallel-vs-serial threshold. m*n is
// bounded by the validated C size, but the *k step can overflow uint64 for
// absurd (unallocatable) shapes -- saturate rather than wrap so a huge problem
// never misclassifies as "small". (This kernel is standalone -- no epilogue.h --
// so the helper is local; it mirrors ep_flops there.)
static inline uint64_t ep_flops(size_t m, size_t n, size_t k) {
    uint64_t mn = (uint64_t)m * (uint64_t)n;
    if (k != 0 && mn > UINT64_MAX / (uint64_t)k) {
        return UINT64_MAX;
    }
    return mn * (uint64_t)k;
}

// a*b, or SIZE_MAX on overflow -> the following malloc/apack_scratch returns
// NULL, handled as OOM (Rust falls back). Local copy (no epilogue.h); mirrors
// ep_cmul there. Defense in depth for a direct C-FFI caller bypassing Rust.
static inline size_t ep_cmul(size_t a, size_t b) {
    size_t r;
    return __builtin_mul_overflow(a, b, &r) ? SIZE_MAX : r;
}

// bf16 -> fp32 widen of the low / high 16 lanes: bf16 is the top 16 bits of
// the fp32, so zero-extend the lane to 32 bits and shift left 16.
static inline svfloat32_t bf16_widen_lo(svbool_t p32, svbfloat16_t d) __arm_streaming {
    svuint32_t u = svunpklo_u32(svreinterpret_u16_bf16(d));
    return svreinterpret_f32_u32(svlsl_n_u32_x(p32, u, 16));
}
static inline svfloat32_t bf16_widen_hi(svbool_t p32, svbfloat16_t d) __arm_streaming {
    svuint32_t u = svunpkhi_u32(svreinterpret_u16_bf16(d));
    return svreinterpret_f32_u32(svlsl_n_u32_x(p32, u, 16));
}

// Both operands pack into the same [KP, 32] pair-interleaved band, so these
// three routines serve A and B alike; only which stride is which differs.
// See the f16f32 twin.
//
// 16 lanes contiguous per depth: ST2 byte-interleave the two s-depths.
static size_t pack_lane_contig(bf16 *dst, const bf16 *src, long depth_stride, size_t k) {
    for (size_t p = 0; p < k / 2; p++) {
        const uint16_t *s0 = (const uint16_t *)(src + (long)(2 * p) * depth_stride);
        const uint16_t *s1 = (const uint16_t *)(src + (long)(2 * p + 1) * depth_stride);
        uint16x8x2_t lo = {{vld1q_u16(s0), vld1q_u16(s1)}};
        uint16x8x2_t hi = {{vld1q_u16(s0 + 8), vld1q_u16(s1 + 8)}};
        vst2q_u16((uint16_t *)(dst + p * 32), lo);
        vst2q_u16((uint16_t *)(dst + p * 32 + 16), hi);
    }
    return k / 2;
}

// One 8-lane x 8-depth block: transpose, then zip the two depths of a group.
static inline void pack_block8(bf16 *dst, const bf16 *src, long lane_stride, size_t lg, size_t d0) {
    uint16x8_t r[8];
    for (int i = 0; i < 8; i++)
        r[i] = vld1q_u16((const uint16_t *)(src + (long)(lg + i) * lane_stride + (long)d0));
    transpose_8x8_u16(r);
    for (int t = 0; t < 4; t++) {
        uint16_t *o = (uint16_t *)(dst + (d0 / 2 + (size_t)t) * 32 + 2 * lg);
        vst1q_u16(o, vzip1q_u16(r[2 * t], r[2 * t + 1]));
        vst1q_u16(o + 8, vzip2q_u16(r[2 * t], r[2 * t + 1]));
    }
}

// Each lane contiguous in depth (row-major A, col-major B).
static size_t pack_depth_contig(bf16 *dst, const bf16 *src, long lane_stride, size_t k) {
    size_t kt = k & ~(size_t)7;
    for (size_t lg = 0; lg < 16; lg += 8)
        for (size_t d0 = 0; d0 < kt; d0 += 8)
            pack_block8(dst, src, lane_stride, lg, d0);
    for (size_t p = kt / 2; p < k / 2; p++) {
        bf16 *d = dst + p * 32;
        for (size_t i = 0; i < 16; i++)
            __builtin_memcpy(d + 2 * i, src + (long)i * lane_stride + 2 * (long)p, 4);
    }
    return k / 2;
}

// Scalar filler from p-group `p0` on: ragged bands, odd-K slices, and layouts
// with neither stride equal to 1. Zero past the lane and depth edges.
static void pack_a_scalar(bf16 *dst, const bf16 *a, long rs, long cs, size_t k, size_t valid_rows,
                          size_t p0) {
    bf16 zero = bf16_from_bits(0);
    for (size_t p = p0; p < (k + 1) / 2; p++)
        for (size_t i = 0; i < 16; i++) {
            size_t d0 = 2 * p, d1 = 2 * p + 1;
            dst[p * 32 + 2 * i + 0] =
                (i < valid_rows && d0 < k) ? a[(long)i * rs + (long)d0 * cs] : zero;
            dst[p * 32 + 2 * i + 1] =
                (i < valid_rows && d1 < k) ? a[(long)i * rs + (long)d1 * cs] : zero;
        }
}

// Pack one 16-row band of A: dst[p*32 + 2*i + s] = A[i, 2p+s]. See the f16f32
// twin for why both operands share pack_{lane,depth}_contig.
static void pack_a_band(bf16 *dst, const bf16 *a, long rs, long cs, size_t k, size_t valid_rows) {
    if (valid_rows != 16) {
        pack_a_scalar(dst, a, rs, cs, k, valid_rows, 0);
        return;
    }
    if (cs == 1) { // row-major A: each row runs along depth
        pack_a_scalar(dst, a, rs, cs, k, valid_rows, pack_depth_contig(dst, a, rs, k));
        return;
    }
    if (rs == 1) { // col-major A: the 16 rows of a depth are contiguous
        pack_a_scalar(dst, a, rs, cs, k, valid_rows, pack_lane_contig(dst, a, cs, k));
        return;
    }
    pack_a_scalar(dst, a, rs, cs, k, valid_rows, 0);
}

// Scalar filler from p-group `p0` on: edge bands, odd-K slices, no-fast-path
// layouts. Zero past the column and depth edges.
static void pack_b_scalar(bf16 *dst, const bf16 *b, long rs, long cs, size_t k, size_t valid_cols,
                          size_t p0) {
    bf16 zero = bf16_from_bits(0);
    for (size_t p = p0; p < (k + 1) / 2; p++)
        for (size_t j = 0; j < 16; j++) {
            size_t d0 = 2 * p, d1 = 2 * p + 1;
            dst[p * 32 + 2 * j + 0] =
                (j < valid_cols && d0 < k) ? b[(long)d0 * rs + (long)j * cs] : zero;
            dst[p * 32 + 2 * j + 1] =
                (j < valid_cols && d1 < k) ? b[(long)d1 * rs + (long)j * cs] : zero;
        }
}

// Pack one 16-col band of B into [KP, 32] pair-interleaved:
//   dst[p*32 + 2*j + s] = B[2p+s, j]   (j in 0..15, s in 0..1)
static void pack_b_band(bf16 *dst, const bf16 *b, long rs, long cs, size_t k, size_t valid_cols) {
    if (valid_cols != 16) {
        pack_b_scalar(dst, b, rs, cs, k, valid_cols, 0);
        return;
    }
    if (cs == 1) {
        pack_b_scalar(dst, b, rs, cs, k, valid_cols, pack_lane_contig(dst, b, rs, k));
        return;
    }
    if (rs == 1) {
        pack_b_scalar(dst, b, rs, cs, k, valid_cols, pack_depth_contig(dst, b, cs, k));
        return;
    }
    pack_b_scalar(dst, b, rs, cs, k, valid_cols, 0);
}

static inline void store_vec(svbool_t p32, svbool_t pst, bf16 *ptr, svfloat32_t lo, svfloat32_t hi,
                             int read_dst, svfloat32_t va) __arm_streaming {
    if (read_dst) {
        svbfloat16_t d = svld1_bf16(pst, ptr);
        lo = svmla_x(p32, lo, bf16_widen_lo(p32, d), va);
        hi = svmla_x(p32, hi, bf16_widen_hi(p32, d), va);
    }
    svbfloat16_t row = svuzp1_bf16(svcvt_bf16_f32_x(p32, lo), svcvt_bf16_f32_x(p32, hi));
    svst1_bf16(pst, ptr, row);
}

// One K-step's MOPAs for the live ZA32 quadrants of the 32x32 super-tile
// (0=lo-M x lo-N, 1=lo-M x hi-N, 2=hi-M x lo-N, 3=hi-M x hi-N). NARROW-N
// (ncols <= 16) leaves the hi-N band all zero-pad -> za1/za3 dead; NARROW-M
// (mrows <= 16) leaves the hi-M band all-pad -> za2/za3 dead. Skipping the dead
// quadrants halves MOPA issue on small-m shapes (the store reads ZA by
// ncols/mrows predicates, so it is unaffected); the dispatch is hoisted out of
// the K-loop. Like f32 this is largely bandwidth-bound at tiny N, so the
// wall-clock win is modest, but the dead work is removed with no downside.
#define BF16W_STEP_FULL(al, ah, bl, bh)                                                            \
    svmopa_za32_bf16_m(0, p16, p16, al, bl);                                                       \
    svmopa_za32_bf16_m(1, p16, p16, al, bh);                                                       \
    svmopa_za32_bf16_m(2, p16, p16, ah, bl);                                                       \
    svmopa_za32_bf16_m(3, p16, p16, ah, bh)
#define BF16W_STEP_NARROW_N(al, ah, bl)                                                            \
    svmopa_za32_bf16_m(0, p16, p16, al, bl);                                                       \
    svmopa_za32_bf16_m(2, p16, p16, ah, bl)
#define BF16W_STEP_NARROW_M(al, bl, bh)                                                            \
    svmopa_za32_bf16_m(0, p16, p16, al, bl);                                                       \
    svmopa_za32_bf16_m(1, p16, p16, al, bh)

// L2 blocking budget in 32-wide super-tiles.
static size_t bf16w_budget(size_t kp) {
    size_t tile_bytes = 2 * ep_cmul(kp, 32) * sizeof(bf16);
    size_t b = (size_t)(16u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    return b < 2 ? 2 : b;
}

// N-block width shared by the driver and sme_run_streaming.
static size_t bf16w_nc_blk(size_t kp, size_t n_tiles) {
    size_t nc = bf16w_budget(kp) / 2;
    if (nc < 1) nc = 1;
    if (nc >= n_tiles) return n_tiles;
    size_t blocks = (n_tiles + nc - 1) / nc;
    return (n_tiles + blocks - 1) / blocks;
}

// [nt_lo, nt_hi) is the N-tile sub-range this invocation owns -- pass
// (0, n_tiles) for the whole problem, or a slice for the N-parallel flat-M path
// (see the driver); b_pack points at super-tile nt_lo's bands. Each N-tile is
// independent (svzero_za + accumulate + store per (mt,nt), no cross-N-tile
// state), so any chunk granularity is safe.
__arm_locally_streaming __arm_new("za") static void sme_run_streaming(
    bf16 *dst, long dst_cs, long dst_rs, const bf16 *a_pack, const bf16 *b_pack, size_t m, size_t n,
    size_t mt_lo, size_t mt_hi, size_t nt_lo, size_t nt_hi, size_t kp, float alpha, float beta,
    int read_dst) {
    svbool_t p16 = svptrue_b16();
    svbool_t p32 = svptrue_b32();
    const svfloat32_t z32 = svdup_n_f32(0.0f);
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    size_t per_tile = ep_cmul(kp, 32);
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);

    // nc_blk/mc_blk count 32-wide SUPER-tiles (= 2 packed 16-lane bands each), so
    // tile_bytes is the full super-tile and the budget constant equals the total
    // resident bytes (A-block + B-block). See gemm_f32.c. A band-sized tile_bytes would
    // instead double the working set.
    size_t budget = bf16w_budget(kp);
    size_t nt_span = nt_hi - nt_lo;
    size_t nc_blk = bf16w_nc_blk(kp, nt_span);
    size_t mc_blk = budget / 2;
    if (mc_blk < 1) mc_blk = 1;
    if (mc_blk > mt_hi - mt_lo) mc_blk = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc_blk) {
        size_t jc_end = jc + nc_blk < nt_hi ? jc + nc_blk : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc_blk) {
            size_t ic_end = ic + mc_blk < mt_hi ? ic + mc_blk : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const bf16 *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
                const bf16 *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
                size_t m0 = mt * 32;
                size_t mrows = (m - m0 < 32) ? (m - m0) : 32;
                for (size_t nt = jc; nt < jc_end; nt++) {
                    const bf16 *b_lo = b_pack + (size_t)(2 * (nt - nt_lo)) * per_tile;
                    const bf16 *b_hi = b_lo + per_tile;
                    size_t n0 = nt * 32;
                    size_t ncols = (n - n0 < 32) ? (n - n0) : 32;

                    svzero_za();
                    size_t p = 0;
                    if (ncols <= 16) { // narrow-N: hi-N band is pad, za1/za3 dead
                        for (; p + 2 <= kp; p += 2) {
                            svbfloat16_t al0 = svld1_bf16(p16, a_lo + p * 32);
                            svbfloat16_t ah0 = svld1_bf16(p16, a_hi + p * 32);
                            svbfloat16_t bl0 = svld1_bf16(p16, b_lo + p * 32);
                            svbfloat16_t al1 = svld1_bf16(p16, a_lo + (p + 1) * 32);
                            svbfloat16_t ah1 = svld1_bf16(p16, a_hi + (p + 1) * 32);
                            svbfloat16_t bl1 = svld1_bf16(p16, b_lo + (p + 1) * 32);
                            BF16W_STEP_NARROW_N(al0, ah0, bl0);
                            BF16W_STEP_NARROW_N(al1, ah1, bl1);
                        }
                        for (; p < kp; p++) {
                            svbfloat16_t al = svld1_bf16(p16, a_lo + p * 32);
                            svbfloat16_t ah = svld1_bf16(p16, a_hi + p * 32);
                            svbfloat16_t bl = svld1_bf16(p16, b_lo + p * 32);
                            BF16W_STEP_NARROW_N(al, ah, bl);
                        }
                    } else if (mrows <= 16) { // narrow-M: hi-M band is pad, za2/za3 dead
                        for (; p + 2 <= kp; p += 2) {
                            svbfloat16_t al0 = svld1_bf16(p16, a_lo + p * 32);
                            svbfloat16_t bl0 = svld1_bf16(p16, b_lo + p * 32);
                            svbfloat16_t bh0 = svld1_bf16(p16, b_hi + p * 32);
                            svbfloat16_t al1 = svld1_bf16(p16, a_lo + (p + 1) * 32);
                            svbfloat16_t bl1 = svld1_bf16(p16, b_lo + (p + 1) * 32);
                            svbfloat16_t bh1 = svld1_bf16(p16, b_hi + (p + 1) * 32);
                            BF16W_STEP_NARROW_M(al0, bl0, bh0);
                            BF16W_STEP_NARROW_M(al1, bl1, bh1);
                        }
                        for (; p < kp; p++) {
                            svbfloat16_t al = svld1_bf16(p16, a_lo + p * 32);
                            svbfloat16_t bl = svld1_bf16(p16, b_lo + p * 32);
                            svbfloat16_t bh = svld1_bf16(p16, b_hi + p * 32);
                            BF16W_STEP_NARROW_M(al, bl, bh);
                        }
                    } else {
                        for (; p + 2 <= kp; p += 2) {
                            svbfloat16_t al0 = svld1_bf16(p16, a_lo + p * 32);
                            svbfloat16_t ah0 = svld1_bf16(p16, a_hi + p * 32);
                            svbfloat16_t bl0 = svld1_bf16(p16, b_lo + p * 32);
                            svbfloat16_t bh0 = svld1_bf16(p16, b_hi + p * 32);
                            svbfloat16_t al1 = svld1_bf16(p16, a_lo + (p + 1) * 32);
                            svbfloat16_t ah1 = svld1_bf16(p16, a_hi + (p + 1) * 32);
                            svbfloat16_t bl1 = svld1_bf16(p16, b_lo + (p + 1) * 32);
                            svbfloat16_t bh1 = svld1_bf16(p16, b_hi + (p + 1) * 32);
                            BF16W_STEP_FULL(al0, ah0, bl0, bh0);
                            BF16W_STEP_FULL(al1, ah1, bl1, bh1);
                        }
                        for (; p < kp; p++) {
                            svbfloat16_t al = svld1_bf16(p16, a_lo + p * 32);
                            svbfloat16_t ah = svld1_bf16(p16, a_hi + p * 32);
                            svbfloat16_t bl = svld1_bf16(p16, b_lo + p * 32);
                            svbfloat16_t bh = svld1_bf16(p16, b_hi + p * 32);
                            BF16W_STEP_FULL(al, ah, bl, bh);
                        }
                    }

                    if (col_major) {
                        svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)mrows);
                        size_t nlo = ncols < 16 ? ncols : 16;
                        for (size_t c = 0; c < nlo; c++) {
                            bf16 *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            svfloat32_t lo =
                                svmul_x(p32, svread_ver_za32_m(z32, p32, 0, (uint32_t)c), vb);
                            svfloat32_t hi =
                                svmul_x(p32, svread_ver_za32_m(z32, p32, 2, (uint32_t)c), vb);
                            store_vec(p32, pst, col, lo, hi, read_dst, va);
                        }
                        for (size_t c = 16; c < ncols; c++) {
                            uint32_t cc = (uint32_t)(c - 16);
                            bf16 *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            svfloat32_t lo = svmul_x(p32, svread_ver_za32_m(z32, p32, 1, cc), vb);
                            svfloat32_t hi = svmul_x(p32, svread_ver_za32_m(z32, p32, 3, cc), vb);
                            store_vec(p32, pst, col, lo, hi, read_dst, va);
                        }
                    } else if (row_major) {
                        svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                        size_t mlo = mrows < 16 ? mrows : 16;
                        for (size_t r = 0; r < mlo; r++) {
                            bf16 *rowp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            svfloat32_t lo =
                                svmul_x(p32, svread_hor_za32_m(z32, p32, 0, (uint32_t)r), vb);
                            svfloat32_t hi =
                                svmul_x(p32, svread_hor_za32_m(z32, p32, 1, (uint32_t)r), vb);
                            store_vec(p32, pst, rowp, lo, hi, read_dst, va);
                        }
                        for (size_t r = 16; r < mrows; r++) {
                            uint32_t rr = (uint32_t)(r - 16);
                            bf16 *rowp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            svfloat32_t lo = svmul_x(p32, svread_hor_za32_m(z32, p32, 2, rr), vb);
                            svfloat32_t hi = svmul_x(p32, svread_hor_za32_m(z32, p32, 3, rr), vb);
                            store_vec(p32, pst, rowp, lo, hi, read_dst, va);
                        }
                    } else {
                        float scratch[32 * 32];
                        svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)16);
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
                        for (size_t r = 0; r < mrows; r++) {
                            for (size_t c = 0; c < ncols; c++) {
                                float ab = scratch[r * 32 + c] * beta;
                                bf16 *cell =
                                    dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                                *cell = (bf16)(read_dst ? alpha * (float)(*cell) + ab : ab);
                            }
                        }
                    }
                }
            }
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

// Row-major B read in place for one M-tile; see gemm_f16f32.c.
__arm_locally_streaming __arm_new("za") static size_t
    run_direct(bf16 *dst, long dst_rs, const bf16 *a_pack, const bf16 *b, long b_rs, size_t m,
               size_t n, size_t k, size_t mt, size_t st, size_t nst, bf16 *buf, int keep,
               float alpha, float beta, int read_dst) {
    svbool_t p16 = svptrue_b16(), p32 = svptrue_b32();
    const svbfloat16_t z16 = svdup_n_bf16((bf16)0.0f);
    const svfloat32_t z32 = svdup_n_f32(0.0f);
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    size_t kp = (k + 1) / 2, per_tile = ep_cmul(kp, 32);
    const bf16 *a_lo = a_pack + 2 * mt * per_tile, *a_hi = a_lo + per_tile;
    size_t m0 = mt * 32, mr = m - m0 < 32 ? m - m0 : 32;
    const bf16 *b0 = b + st * 32;
    svcount_t pc = svwhilelt_c16((uint64_t)st * 32, (uint64_t)n, 2); // nst 1: lanes past st unused
    int four = mr <= 16 && nst == 2;
    svzero_za();
    for (size_t p = 0; p < kp; p++) {
        const bf16 *r = b0 + (long)(2 * p) * b_rs;
        svbfloat16x2_t R0 = svld1_bf16_x2(pc, r);
        svbfloat16x2_t R1 = 2 * p + 1 < k ? svld1_bf16_x2(pc, r + b_rs) : svcreate2(z16, z16);
        svbfloat16_t bl = svzip1_bf16(svget2(R0, 0), svget2(R1, 0));
        svbfloat16_t bh = svzip2_bf16(svget2(R0, 0), svget2(R1, 0));
        svbfloat16_t al = svld1_bf16(p16, a_lo + p * 32);
        if (keep) {
            svst1_bf16(p16, buf + p * 32, bl);
            svst1_bf16(p16, buf + per_tile + p * 32, bh);
        }
        if (nst == 2) {
            svbfloat16_t cl = svzip1_bf16(svget2(R0, 1), svget2(R1, 1));
            svbfloat16_t ch = svzip2_bf16(svget2(R0, 1), svget2(R1, 1));
            if (four) {
                svmopa_za32_bf16_m(0, p16, p16, al, bl);
                svmopa_za32_bf16_m(1, p16, p16, al, bh);
                svmopa_za32_bf16_m(2, p16, p16, al, cl);
                svmopa_za32_bf16_m(3, p16, p16, al, ch);
                if (keep) {
                    svst1_bf16(p16, buf + 2 * per_tile + p * 32, cl);
                    svst1_bf16(p16, buf + 3 * per_tile + p * 32, ch);
                }
                continue;
            }
            svst1_bf16(p16, buf + 2 * per_tile + p * 32, cl);
            svst1_bf16(p16, buf + 3 * per_tile + p * 32, ch);
        }
        if (mr > 16) {
            svbfloat16_t ah = svld1_bf16(p16, a_hi + p * 32);
            BF16W_STEP_FULL(al, ah, bl, bh);
        } else {
            BF16W_STEP_NARROW_M(al, bl, bh);
        }
    }
#define STORE_HALF(t_lo, t_hi, r_lo, r_hi, c0)                                                     \
    for (size_t r = (r_lo); r < (r_hi); r++) {                                                     \
        uint32_t s_ = (uint32_t)(r - (r_lo));                                                      \
        svfloat32_t lo = svmul_x(p32, svread_hor_za32_m(z32, p32, t_lo, s_), vb);                  \
        svfloat32_t hi = svmul_x(p32, svread_hor_za32_m(z32, p32, t_hi, s_), vb);                  \
        store_vec(p32, pst, dst + (long)(m0 + r) * dst_rs + (long)(c0), lo, hi, read_dst, va);     \
    }
#define STORE_PST(c0) svwhilelt_b16((uint64_t)0, (uint64_t)(n - (c0) < 32 ? n - (c0) : 32))
    {
        svbool_t pst = STORE_PST(st * 32);
        STORE_HALF(0, 1, 0, mr < 16 ? mr : 16, st * 32)
        if (!four) STORE_HALF(2, 3, 16, mr > 16 ? mr : 16, st * 32)
    }
    if (four && (st + 1) * 32 < n) {
        svbool_t pst = STORE_PST((st + 1) * 32);
        STORE_HALF(2, 3, 0, mr, (st + 1) * 32)
    }
#undef STORE_PST
#undef STORE_HALF
    return four ? 2 : 1;
}

// Per-worker column items; see gemm_f16f32.c.
static int run_cols(bf16 *dst, long dst_cs, long dst_rs, const bf16 *a, long lhs_rs, long lhs_cs,
                    size_t m, size_t n, size_t k, float alpha, float beta, int read_dst,
                    const bf16 *b, long rhs_rs, long rhs_cs) {
    size_t kp = (k + 1) / 2, per_tile = ep_cmul(kp, 32);
    size_t m_tiles = (m + 31) / 32, n_tiles = (n + 31) / 32;
    size_t NC = 2; // super-tiles per item
    size_t M_CHUNK = 2, n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    size_t n_items = (n_tiles + NC - 1) / NC, n_units = n_chunks + n_items;
    size_t W = n_units < 12 ? n_units : 12;
    size_t panel = ep_cmul(2 * NC, per_tile);
    _Atomic size_t *ctr;
    bf16 *pool = (bf16 *)ring_pool(2 + n_chunks, ep_cmul(ep_cmul(W, panel), sizeof(bf16)), &ctr);
    bf16 *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(bf16)));
    if (!pool || !a_pack) return -1;
    _Atomic size_t *cur = ctr, *a_cnt = ctr + 1, *a_done = ctr + 2;
    // Row-major B and C: multiply straight from B's rows. Never more than ~6%
    // behind the pack below and up to 3x ahead at small m (16x4096x4096: 1.52
    // against 0.52 TF/s), so it is not gated on stride the way f16's is.
    int direct = rhs_cs == 1 && dst_cs == 1 && rhs_rs > 0;
    dispatch_apply(W, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t w) {
      bf16 *bt = pool + w * panel;
      for (size_t i; (i = atomic_fetch_add_explicit(cur, 1, memory_order_relaxed)) < n_units;) {
          if (i < n_chunks) {
              size_t mt1 = (i + 1) * M_CHUNK < m_tiles ? (i + 1) * M_CHUNK : m_tiles;
              for (size_t st = 2 * i * M_CHUNK; st < 2 * mt1; st++) {
                  size_t r0 = st * 16, vr = r0 < m ? (m - r0 < 16 ? m - r0 : 16) : 0;
                  pack_a_band(a_pack + st * per_tile, vr ? a + (long)r0 * lhs_rs : a, lhs_rs,
                              lhs_cs, k, vr);
              }
              atomic_store_explicit(&a_done[i], 1, memory_order_release);
              atomic_fetch_add_explicit(a_cnt, 1, memory_order_release);
              continue;
          }
          size_t t0 = (i - n_chunks) * NC, t1 = t0 + NC < n_tiles ? t0 + NC : n_tiles, mt_lo = 0;
          if (direct) {
              // M-tile 0 multiplies straight from B's rows; the bands it captures
              // feed the packed kernel for whatever it left.
              while (!atomic_load_explicit(&a_done[0], memory_order_acquire))
                  __builtin_arm_yield();
              size_t done = run_direct(dst, dst_rs, a_pack, b, rhs_rs, m, n, k, 0, t0, t1 - t0, bt,
                                       m_tiles > 1, alpha, beta, read_dst);
              if (done < t1 - t0)
                  sme_run_streaming(dst, 1, dst_rs, a_pack, bt + 2 * per_tile, m, n, 0, 1, t0 + 1,
                                    t1, kp, alpha, beta, read_dst);
              mt_lo = 1;
          } else {
              for (size_t st = 2 * t0; st < 2 * t1; st++) {
                  size_t c0 = st * 16, vc = c0 < n ? (n - c0 < 16 ? n - c0 : 16) : 0;
                  pack_b_band(bt + (st - 2 * t0) * per_tile, vc ? b + (long)c0 * rhs_cs : b, rhs_rs,
                              rhs_cs, k, vc);
              }
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
              sme_run_streaming(dst, dst_cs, dst_rs, a_pack, bt, m, n, mt0, mt1, t0, t1, kp, alpha,
                                beta, read_dst);
          }
      }
    });
    return 0;
}

// Large A: B is built per N-block into a panel ring (panel_ring.h) and shared by
// every M-chunk, as gemm_f32.c's run_panels.
static int run_panels(bf16 *dst, long dst_cs, long dst_rs, const bf16 *a, long lhs_rs, long lhs_cs,
                      size_t m, size_t n, size_t k, float alpha, float beta, int read_dst,
                      const bf16 *b, long rhs_rs, long rhs_cs) {
    size_t kp = (k + 1) / 2, per_tile = ep_cmul(kp, 32), tile = 2 * per_tile;
    size_t m_tiles = (m + 31) / 32, n_tiles = (n + 31) / 32;
    size_t nc_blk = ((size_t)4 << 20) / ep_cmul(tile, sizeof(bf16)); // 4 MB B-blocks
    if (nc_blk < 1) nc_blk = 1;
    if (nc_blk > n_tiles) nc_blk = n_tiles;
    size_t n_blocks = (n_tiles + nc_blk - 1) / nc_blk;
    nc_blk = (n_tiles + n_blocks - 1) / n_blocks; // even split
    size_t M_CHUNK = 2, n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    size_t n_sub = (RING_ITEMS + n_chunks - 1) / n_chunks;
    if (n_sub > nc_blk) n_sub = nc_blk;
    size_t ns = (nc_blk + n_sub - 1) / n_sub; // super-tiles per subrange
    n_sub = (nc_blk + ns - 1) / ns;
    size_t n_slots = n_blocks < RING_SLOTS ? n_blocks : RING_SLOTS;
    size_t dt = tile ? ((size_t)1 << 16) / tile : 1; // super-tiles per build unit
    if (dt < 1) dt = 1;
    if (dt > nc_blk) dt = nc_blk;
    size_t n_d = (nc_blk + dt - 1) / dt;
    size_t slot_elems = ep_cmul(nc_blk, tile);
    _Atomic size_t *ctr;
    bf16 *pool = (bf16 *)ring_pool(2 * n_blocks + 2 + n_chunks,
                                   ep_cmul(ep_cmul(n_slots, slot_elems), sizeof(bf16)), &ctr);
    bf16 *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(bf16)));
    if (!pool || !a_pack) return -1;
    _Atomic size_t *a_done = ctr + 2 * n_blocks + 2;
    void (^pack_a)(size_t, size_t) = ^(size_t mt0, size_t mt1) {
      for (size_t st = 2 * mt0; st < 2 * mt1; st++) {
          size_t r0 = st * 16, vr = r0 < m ? (m - r0 < 16 ? m - r0 : 16) : 0;
          pack_a_band(a_pack + st * per_tile, vr ? a + (long)r0 * lhs_rs : a, lhs_rs, lhs_cs, k,
                      vr);
      }
    };
    if (n_chunks <= 1) pack_a(0, m_tiles);

    ring_run(
        n_blocks, n_d, n_chunks * n_sub, ctr,
        ^(size_t sd, size_t j) {
          size_t lo = sd * nc_blk, hi = lo + nc_blk < n_tiles ? lo + nc_blk : n_tiles;
          size_t t0 = lo + j * dt, t1 = t0 + dt < hi ? t0 + dt : hi;
          bf16 *slot = pool + (sd % RING_SLOTS) * slot_elems;
          for (size_t st = 2 * t0; st < 2 * t1; st++) {
              size_t c0 = st * 16, vc = c0 < n ? (n - c0 < 16 ? n - c0 : 16) : 0;
              pack_b_band(slot + (st - 2 * lo) * per_tile, vc ? b + (long)c0 * rhs_cs : b, rhs_rs,
                          rhs_cs, k, vc);
          }
        },
        ^(size_t sg, size_t j) {
          size_t lo = sg * nc_blk, hi = lo + nc_blk < n_tiles ? lo + nc_blk : n_tiles;
          size_t ci = j / n_sub, sj = j % n_sub;
          size_t mt0 = ci * M_CHUNK, mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
          size_t n0 = lo + sj * ns < hi ? lo + sj * ns : hi, n1 = n0 + ns < hi ? n0 + ns : hi;
          // Block 0's first item per M-chunk packs that chunk of A.
          if (n_chunks > 1 && sg == 0 && sj == 0) {
              pack_a(mt0, mt1);
              atomic_store_explicit(&a_done[ci], 1, memory_order_release);
          } else if (n_chunks > 1) {
              while (!atomic_load_explicit(&a_done[ci], memory_order_acquire))
                  __builtin_arm_yield();
          }
          if (n0 < n1)
              sme_run_streaming(dst, dst_cs, dst_rs, a_pack,
                                pool + (sg % RING_SLOTS) * slot_elems + (n0 - lo) * tile, m, n, mt0,
                                mt1, n0, n1, kp, alpha, beta, read_dst);
        });
    return 0;
}

int gemm_sme_bf16f32_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                         int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                         const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                         uint16_t beta_bits) {
    if (m == 0 || n == 0) return 0;
    float alpha = read_dst ? (float)bf16_from_bits(alpha_bits) : 0.0f;
    float beta = (float)bf16_from_bits(beta_bits);
    // Build B inside the parallel GEMM rather than packing it all first: per-worker
    // columns while A is cache-sized, the shared panel ring past that.
    if (ep_flops(m, n, k) >= (1u << 21)) {
        bf16 *d = (bf16 *)dst;
        const bf16 *a = (const bf16 *)lhs, *b = (const bf16 *)rhs;
        if (use_cols(m, n, ep_cmul(ep_cmul(m, k), sizeof(bf16))))
            return run_cols(d, dst_cs, dst_rs, a, lhs_rs, lhs_cs, m, n, k, alpha, beta, read_dst, b,
                            rhs_rs, rhs_cs);
        return run_panels(d, dst_cs, dst_rs, a, lhs_rs, lhs_cs, m, n, k, alpha, beta, read_dst, b,
                          rhs_rs, rhs_cs);
    }

    size_t kp = (k + 1) / 2;
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    size_t per_tile = ep_cmul(kp, 32);

    bf16 *b_pack = (bf16 *)malloc(ep_cmul(ep_cmul(2 * n_tiles, per_tile), sizeof(bf16)));
    if (!b_pack) return -1;

    const bf16 *a = (const bf16 *)lhs;
    const bf16 *b = (const bf16 *)rhs;
    for (size_t st = 0; st < 2 * n_tiles; st++) {
        size_t c0 = st * 16;
        size_t vc = (c0 < n) ? ((n - c0 < 16) ? (n - c0) : 16) : 0;
        // vc==0 bands are zero-filled without reading src; clamp the base so no
        // out-of-bounds pointer is even formed (UB without a deref).
        pack_b_band(b_pack + st * per_tile, vc ? b + (long)c0 * rhs_cs : b, rhs_rs, rhs_cs, k, vc);
    }

#define PACKA_RANGE(AP, MT0, MT1)                                                                  \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                            \
        size_t r0 = st * 16;                                                                       \
        size_t vr = (r0 < m) ? ((m - r0 < 16) ? (m - r0) : 16) : 0;                                \
        pack_a_band((AP) + st * per_tile, vr ? a + (long)r0 * lhs_rs : a, lhs_rs, lhs_cs, k, vr);  \
    }
    // Finer chunks let dispatch_apply work-steal to balance the slower E-cluster.
    size_t M_CHUNK = 2;
    size_t n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    int big = ep_flops(m, n, k) >= (1u << 21);

    // Flat-M (few M-tiles) but large/wide: M is the only M-chunk axis, so the
    // M-parallel scheme below would run this on ONE cluster. Parallelize over N
    // instead -- pack all of A once (cheap; m_tiles is small) into a shared buffer
    // and hand N-tile chunks to both clusters. C columns are disjoint per chunk;
    // A and B are read-only and shared. The widening super-tile carries no
    // cross-N-tile state, so any chunk granularity is safe. Needs >= 2 chunks to
    // beat the serial path. (Mirrors the f32 driver.)
    size_t N_CHUNK = 4;
    size_t nn_chunks = (n_tiles + N_CHUNK - 1) / N_CHUNK;
    if (n_chunks <= 1 && big && nn_chunks >= 2) {
        bf16 *a_pack = (bf16 *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(bf16)));
        if (a_pack) {
            PACKA_RANGE(a_pack, 0, m_tiles);
            dispatch_apply(nn_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
              size_t nt0 = ci * N_CHUNK;
              size_t nt1 = nt0 + N_CHUNK < n_tiles ? nt0 + N_CHUNK : n_tiles;
              sme_run_streaming((bf16 *)dst, dst_cs, dst_rs, a_pack, b_pack + 2 * nt0 * per_tile, m,
                                n, 0, m_tiles, nt0, nt1, kp, alpha, beta, read_dst);
            });
            free(a_pack);
            free(b_pack);
            return 0;
        }
        // malloc failed: fall through to the serial path.
    }
    if (n_chunks <= 1 || !big) {
        bf16 *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(bf16)));
        if (!a_pack) {
            free(b_pack);
            return -1;
        }
        PACKA_RANGE(a_pack, 0, m_tiles);
        sme_run_streaming((bf16 *)dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, 0, n_tiles,
                          kp, alpha, beta, read_dst);
        free(b_pack);
        return 0;
    }
    bf16 *a_pack = (bf16 *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(bf16)));
    if (!a_pack) {
        free(b_pack);
        return -1;
    }
    size_t nc_blk = bf16w_nc_blk(kp, n_tiles);
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
      sme_run_streaming((bf16 *)dst, dst_cs, dst_rs, a_pack, b_pack + 2 * jc * per_tile, m, n, mt0,
                        mt1, jc, jc_end, kp, alpha, beta, read_dst);
    });
#undef PACKA_RANGE

    free(a_pack);
    free(b_pack);
    return 0;
}
