// Dedicated SME f16 GEMM driver (Apple M4+, base FEAT_SME, widening
// f16->fp32). This BYPASSES gemm's generic per-tile microkernel path. It is
// the high-throughput design: pre-pack both operands into the pair-interleaved
// [K/2, 32] layout (so the hot loop is load->FMOPA with NO zip), accumulate a
// 32x32 super-tile across all of K into four INDEPENDENT ZA32 quadrants (no
// RAW between consecutive MOPAs), run the whole tile grid in ONE streaming
// session, and narrow/combine/store with a vectorized epilogue.
//
// Computes (gemm contract):  dst = alpha*dst + beta*(A @ B)
//   A is m x k, B is k x n, with arbitrary row/col strides.
//   alpha_status: 0 overwrite (dst not read), 1 add, 2 scale-and-add.
//
// Single pass over the full K (no kc chunking), so alpha multiplies the
// original dst exactly once. Intended to run single-threaded -- the SME unit
// is shared per cluster; the Rust caller routes multi-threaded / small
// problems to the per-core NEON path.

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

// Both operands pack into the same [KP, 32] pair-interleaved band -- 16 lanes,
// two depths per group, dst[p*32 + 2*lane + s] = src[lane, 2p+s]. A's lanes are
// rows and B's are columns, so the two only differ in which stride is which, and
// the three routines below cover both:
//
//   lanes contiguous at each depth  -> pack_lane_contig   (row-major B, col-major A)
//   each lane contiguous in depth   -> pack_depth_contig  (row-major A, col-major B)
//   anything else / ragged bands    -> the scalar fillers
//
// Each returns the first p-group it did NOT fill, which the scalar filler takes.

// 16 lanes contiguous per depth: load the two s-depths and ST2 byte-interleave
// straight into the layout.
static size_t pack_lane_contig(f16 *dst, const f16 *src, long depth_stride, size_t k) {
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

// One 8-lane x 8-depth block: transpose so each vector is one depth across the
// 8 lanes, then zip the two depths of a p-group -- that zip IS the pair
// interleave the layout wants.
static inline void pack_block8(f16 *dst, const f16 *src, long lane_stride, size_t lg, size_t d0) {
    uint16x8_t r[8];
    for (int i = 0; i < 8; i++)
        r[i] = vld1q_u16((const uint16_t *)(src + (long)(lg + i) * lane_stride + (long)d0));
    transpose_8x8_u16(r); // r[t] = depth d0+t across the 8 lanes
    for (int t = 0; t < 4; t++) {
        uint16_t *o = (uint16_t *)(dst + (d0 / 2 + (size_t)t) * 32 + 2 * lg);
        vst1q_u16(o, vzip1q_u16(r[2 * t], r[2 * t + 1]));
        vst1q_u16(o + 8, vzip2q_u16(r[2 * t], r[2 * t + 1]));
    }
}

// Each lane contiguous in depth. The 32-bit-gather alternative issues 16 strided
// loads per p-group; blocking through the transpose measured ~1.4-2x over it.
static size_t pack_depth_contig(f16 *dst, const f16 *src, long lane_stride, size_t k) {
    size_t kt = k & ~(size_t)7; // whole 8-depth blocks
    for (size_t lg = 0; lg < 16; lg += 8)
        for (size_t d0 = 0; d0 < kt; d0 += 8)
            pack_block8(dst, src, lane_stride, lg, d0);
    // Depth remainder below 8: the 32-bit gather, still cheaper than scalar.
    for (size_t p = kt / 2; p < k / 2; p++) {
        f16 *d = dst + p * 32;
        for (size_t i = 0; i < 16; i++)
            __builtin_memcpy(d + 2 * i, src + (long)i * lane_stride + 2 * (long)p, 4);
    }
    return k / 2;
}

// Scalar filler from p-group `p0` on: ragged bands, odd-K slices, and layouts
// with neither stride equal to 1. Zero past the lane and depth edges.
static void pack_a_scalar(f16 *dst, const f16 *a, long rs, long cs, size_t k, size_t valid_rows,
                          size_t p0) {
    for (size_t p = p0; p < (k + 1) / 2; p++)
        for (size_t i = 0; i < 16; i++) {
            size_t d0 = 2 * p, d1 = 2 * p + 1;
            f16 v0 = (i < valid_rows && d0 < k) ? a[(long)i * rs + (long)d0 * cs] : (f16)0.0f;
            f16 v1 = (i < valid_rows && d1 < k) ? a[(long)i * rs + (long)d1 * cs] : (f16)0.0f;
            dst[p * 32 + 2 * i + 0] = v0;
            dst[p * 32 + 2 * i + 1] = v1;
        }
}

static void pack_b_scalar(f16 *dst, const f16 *b, long rs, long cs, size_t k, size_t valid_cols,
                          size_t p0) {
    for (size_t p = p0; p < (k + 1) / 2; p++)
        for (size_t j = 0; j < 16; j++) {
            size_t d0 = 2 * p, d1 = 2 * p + 1;
            f16 v0 = (j < valid_cols && d0 < k) ? b[(long)d0 * rs + (long)j * cs] : (f16)0.0f;
            f16 v1 = (j < valid_cols && d1 < k) ? b[(long)d1 * rs + (long)j * cs] : (f16)0.0f;
            dst[p * 32 + 2 * j + 0] = v0;
            dst[p * 32 + 2 * j + 1] = v1;
        }
}

// Pack one 16-row band of A: dst[p*32 + 2*i + s] = A[i, 2p+s].
static void pack_a_band(f16 *dst, const f16 *a, long rs, long cs, size_t k, size_t valid_rows) {
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

// Pack one 16-col band of B: dst[p*32 + 2*j + s] = B[2p+s, j].
static void pack_b_band(f16 *dst, const f16 *b, long rs, long cs, size_t k, size_t valid_cols) {
    if (valid_cols != 16) {
        pack_b_scalar(dst, b, rs, cs, k, valid_cols, 0);
        return;
    }
    if (cs == 1) { // row-major B: the 16 cols of a depth are contiguous
        pack_b_scalar(dst, b, rs, cs, k, valid_cols, pack_lane_contig(dst, b, rs, k));
        return;
    }
    if (rs == 1) { // col-major B (Q@K^T): each column runs along depth
        pack_b_scalar(dst, b, rs, cs, k, valid_cols, pack_depth_contig(dst, b, cs, k));
        return;
    }
    pack_b_scalar(dst, b, rs, cs, k, valid_cols, 0);
}

// Widen the low / high 16 f16 lanes of `d` (lanes 0..15 / 16..31) to f32.
// svcvt reads even lanes, so duplicate-zip first.
static inline svfloat32_t widen_lo(svbool_t p32, svfloat16_t d) __arm_streaming {
    return svcvt_f32_x(p32, svzip1_f16(d, d));
}
static inline svfloat32_t widen_hi(svbool_t p32, svfloat16_t d) __arm_streaming {
    return svcvt_f32_x(p32, svzip2_f16(d, d));
}

// Given the two fp32 halves already scaled by beta (lo = M0..15 / N0..15,
// hi = the other 16), optionally add alpha*dst, narrow to a contiguous
// 32-lane f16 vector, and store under `pst`. `ptr` is contiguous in the
// stored direction (dst_rs==1 column, or dst_cs==1 row).
static inline void store_vec(svbool_t p32, svbool_t pst, f16 *ptr, svfloat32_t lo, svfloat32_t hi,
                             int read_dst, svfloat32_t va) __arm_streaming {
    if (read_dst) {
        svfloat16_t d = svld1_f16(pst, ptr);
        lo = svmla_x(p32, lo, widen_lo(p32, d), va);
        hi = svmla_x(p32, hi, widen_hi(p32, d), va);
    }
    svfloat16_t row = svuzp1_f16(svcvt_f16_x(p32, lo), svcvt_f16_x(p32, hi));
    svst1_f16(pst, ptr, row);
}

// One K-step's MOPAs for the live ZA32 quadrants of the 32x32 super-tile
// (0=lo-M x lo-N, 1=lo-M x hi-N, 2=hi-M x lo-N, 3=hi-M x hi-N). NARROW-N
// (ncols <= 16) leaves the hi-N band all zero-pad -> za1/za3 dead; NARROW-M
// (mrows <= 16) leaves the hi-M band all-pad -> za2/za3 dead. Skipping the dead
// quadrants halves MOPA issue on decode/GEMV shapes (the store reads ZA by
// ncols/mrows predicates, so it is unaffected); the dispatch is hoisted out of
// the K-loop. Like f32 this path is largely B-/A-bandwidth-bound at tiny N, so
// the wall-clock win is modest, but the dead work is removed with no downside.
#define F16W_STEP_FULL(al, ah, bl, bh)                                                             \
    svmopa_za32_f16_m(0, p16, p16, al, bl);                                                        \
    svmopa_za32_f16_m(1, p16, p16, al, bh);                                                        \
    svmopa_za32_f16_m(2, p16, p16, ah, bl);                                                        \
    svmopa_za32_f16_m(3, p16, p16, ah, bh)
#define F16W_STEP_NARROW_N(al, ah, bl)                                                             \
    svmopa_za32_f16_m(0, p16, p16, al, bl);                                                        \
    svmopa_za32_f16_m(2, p16, p16, ah, bl)
#define F16W_STEP_NARROW_M(al, bl, bh)                                                             \
    svmopa_za32_f16_m(0, p16, p16, al, bl);                                                        \
    svmopa_za32_f16_m(1, p16, p16, al, bh)

// L2 blocking budget in 32-wide super-tiles.
static size_t f16w_budget(size_t kp) {
    size_t tile_bytes = 2 * ep_cmul(kp, 32) * sizeof(f16);
    size_t b = (size_t)(16u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    return b < 2 ? 2 : b;
}

// N-block width shared by the driver and sme_run_streaming.
static size_t f16w_nc_blk(size_t kp, size_t n_tiles) {
    size_t nc = f16w_budget(kp) / 2;
    if (nc < 1) nc = 1;
    if (nc >= n_tiles) return n_tiles;
    size_t blocks = (n_tiles + nc - 1) / nc;
    return (n_tiles + blocks - 1) / blocks;
}

// The streaming compute: one ZA lifetime for the whole tile grid.
// [nt_lo, nt_hi) is the N-tile sub-range this invocation owns -- pass
// (0, n_tiles) for the whole problem, or a slice for the N-parallel flat-M path
// (see the driver). Each N-tile is independent (svzero_za + accumulate +
// store per (mt,nt), no cross-N-tile state), so any chunk granularity is safe.
__arm_locally_streaming __arm_new("za") static void sme_run_streaming(
    f16 *dst, long dst_cs, long dst_rs, const f16 *a_pack, const f16 *b_pack, size_t m, size_t n,
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

    // Cache blocking (BLIS jc->ic): B-block + A-block resident in L2. nc_blk /
    // mc_blk count 32-wide SUPER-tiles (= 2 packed 16-lane bands each), so
    // tile_bytes is the full super-tile and the budget constant equals the total
    // resident bytes (A-block + B-block) -- matching gemm_f32.c. A band-sized tile_bytes would
    // instead double the working set.
    size_t budget = f16w_budget(kp);
    size_t nt_span = nt_hi - nt_lo;
    size_t nc_blk = f16w_nc_blk(kp, nt_span);
    size_t mc_blk = budget / 2;
    if (mc_blk < 1) mc_blk = 1;
    if (mc_blk > mt_hi - mt_lo) mc_blk = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc_blk) {
        size_t jc_end = jc + nc_blk < nt_hi ? jc + nc_blk : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc_blk) {
            size_t ic_end = ic + mc_blk < mt_hi ? ic + mc_blk : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const f16 *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
                const f16 *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
                size_t m0 = mt * 32;
                size_t mrows = (m - m0 < 32) ? (m - m0) : 32;
                for (size_t nt = jc; nt < jc_end; nt++) {
                    const f16 *b_lo = b_pack + (size_t)(2 * nt) * per_tile;
                    const f16 *b_hi = b_pack + (size_t)(2 * nt + 1) * per_tile;
                    size_t n0 = nt * 32;
                    size_t ncols = (n - n0 < 32) ? (n - n0) : 32;

                    // Accumulate the 32x32 super-tile: 4 independent ZA32 quadrants,
                    // no zip (panels pre-interleaved), no RAW (distinct tiles). K-loop
                    // unrolled by 2 to run loads ahead of the MOPAs.
                    svzero_za();
                    size_t p = 0;
                    if (ncols <= 16) { // narrow-N: hi-N band is pad, za1/za3 dead
                        for (; p + 2 <= kp; p += 2) {
                            svfloat16_t al0 = svld1_f16(p16, a_lo + p * 32);
                            svfloat16_t ah0 = svld1_f16(p16, a_hi + p * 32);
                            svfloat16_t bl0 = svld1_f16(p16, b_lo + p * 32);
                            svfloat16_t al1 = svld1_f16(p16, a_lo + (p + 1) * 32);
                            svfloat16_t ah1 = svld1_f16(p16, a_hi + (p + 1) * 32);
                            svfloat16_t bl1 = svld1_f16(p16, b_lo + (p + 1) * 32);
                            F16W_STEP_NARROW_N(al0, ah0, bl0);
                            F16W_STEP_NARROW_N(al1, ah1, bl1);
                        }
                        for (; p < kp; p++) {
                            svfloat16_t al = svld1_f16(p16, a_lo + p * 32);
                            svfloat16_t ah = svld1_f16(p16, a_hi + p * 32);
                            svfloat16_t bl = svld1_f16(p16, b_lo + p * 32);
                            F16W_STEP_NARROW_N(al, ah, bl);
                        }
                    } else if (mrows <= 16) { // narrow-M: hi-M band is pad, za2/za3 dead
                        for (; p + 2 <= kp; p += 2) {
                            svfloat16_t al0 = svld1_f16(p16, a_lo + p * 32);
                            svfloat16_t bl0 = svld1_f16(p16, b_lo + p * 32);
                            svfloat16_t bh0 = svld1_f16(p16, b_hi + p * 32);
                            svfloat16_t al1 = svld1_f16(p16, a_lo + (p + 1) * 32);
                            svfloat16_t bl1 = svld1_f16(p16, b_lo + (p + 1) * 32);
                            svfloat16_t bh1 = svld1_f16(p16, b_hi + (p + 1) * 32);
                            F16W_STEP_NARROW_M(al0, bl0, bh0);
                            F16W_STEP_NARROW_M(al1, bl1, bh1);
                        }
                        for (; p < kp; p++) {
                            svfloat16_t al = svld1_f16(p16, a_lo + p * 32);
                            svfloat16_t bl = svld1_f16(p16, b_lo + p * 32);
                            svfloat16_t bh = svld1_f16(p16, b_hi + p * 32);
                            F16W_STEP_NARROW_M(al, bl, bh);
                        }
                    } else {
                        for (; p + 2 <= kp; p += 2) {
                            svfloat16_t al0 = svld1_f16(p16, a_lo + p * 32);
                            svfloat16_t ah0 = svld1_f16(p16, a_hi + p * 32);
                            svfloat16_t bl0 = svld1_f16(p16, b_lo + p * 32);
                            svfloat16_t bh0 = svld1_f16(p16, b_hi + p * 32);
                            svfloat16_t al1 = svld1_f16(p16, a_lo + (p + 1) * 32);
                            svfloat16_t ah1 = svld1_f16(p16, a_hi + (p + 1) * 32);
                            svfloat16_t bl1 = svld1_f16(p16, b_lo + (p + 1) * 32);
                            svfloat16_t bh1 = svld1_f16(p16, b_hi + (p + 1) * 32);
                            F16W_STEP_FULL(al0, ah0, bl0, bh0);
                            F16W_STEP_FULL(al1, ah1, bl1, bh1);
                        }
                        for (; p < kp; p++) {
                            svfloat16_t al = svld1_f16(p16, a_lo + p * 32);
                            svfloat16_t ah = svld1_f16(p16, a_hi + p * 32);
                            svfloat16_t bl = svld1_f16(p16, b_lo + p * 32);
                            svfloat16_t bh = svld1_f16(p16, b_hi + p * 32);
                            F16W_STEP_FULL(al, ah, bl, bh);
                        }
                    }

                    if (col_major) {
                        // One contiguous (dst_rs==1) M-column per N-col: read ZA
                        // VERTICALLY so the 32 M-values land contiguous. The ZA tile
                        // index must be a constant, so split N0..15 (tiles 0,2) from
                        // N16..31 (tiles 1,3).
                        svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)mrows);
                        size_t nlo = ncols < 16 ? ncols : 16;
                        for (size_t c = 0; c < nlo; c++) {
                            f16 *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            svfloat32_t lo =
                                svmul_x(p32, svread_ver_za32_m(z32, p32, 0, (uint32_t)c), vb);
                            svfloat32_t hi =
                                svmul_x(p32, svread_ver_za32_m(z32, p32, 2, (uint32_t)c), vb);
                            store_vec(p32, pst, col, lo, hi, read_dst, va);
                        }
                        for (size_t c = 16; c < ncols; c++) {
                            uint32_t cc = (uint32_t)(c - 16);
                            f16 *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                            svfloat32_t lo = svmul_x(p32, svread_ver_za32_m(z32, p32, 1, cc), vb);
                            svfloat32_t hi = svmul_x(p32, svread_ver_za32_m(z32, p32, 3, cc), vb);
                            store_vec(p32, pst, col, lo, hi, read_dst, va);
                        }
                    } else if (row_major) {
                        // One contiguous (dst_cs==1) N-row per M-row: read ZA
                        // HORIZONTALLY. Split M0..15 (tiles 0,1) from M16..31 (2,3).
                        svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                        size_t mlo = mrows < 16 ? mrows : 16;
                        for (size_t r = 0; r < mlo; r++) {
                            f16 *rowp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            svfloat32_t lo =
                                svmul_x(p32, svread_hor_za32_m(z32, p32, 0, (uint32_t)r), vb);
                            svfloat32_t hi =
                                svmul_x(p32, svread_hor_za32_m(z32, p32, 1, (uint32_t)r), vb);
                            store_vec(p32, pst, rowp, lo, hi, read_dst, va);
                        }
                        for (size_t r = 16; r < mrows; r++) {
                            uint32_t rr = (uint32_t)(r - 16);
                            f16 *rowp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                            svfloat32_t lo = svmul_x(p32, svread_hor_za32_m(z32, p32, 2, rr), vb);
                            svfloat32_t hi = svmul_x(p32, svread_hor_za32_m(z32, p32, 3, rr), vb);
                            store_vec(p32, pst, rowp, lo, hi, read_dst, va);
                        }
                    } else {
                        // Fully general strides: extract horizontally to a small
                        // scratch, scalar combine. Rare; correctness over speed.
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
                                f16 *cell = dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                                *cell = (f16)(read_dst ? alpha * (float)(*cell) + ab : ab);
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

int gemm_sme_f16f32_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                        int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                        const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                        uint16_t beta_bits) {
    if (m == 0 || n == 0) return 0;
    float alpha = read_dst ? (float)f16_from_bits(alpha_bits) : 0.0f;
    float beta = (float)f16_from_bits(beta_bits);

    size_t kp = (k + 1) / 2;
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    size_t a_subtiles = 2 * m_tiles; // 16-row bands
    size_t b_subtiles = 2 * n_tiles; // 16-col bands
    size_t per_tile = ep_cmul(kp, 32);

    f16 *b_pack = (f16 *)malloc(ep_cmul(ep_cmul(b_subtiles, per_tile), sizeof(f16)));
    if (!b_pack) return -1;

    const f16 *a = (const f16 *)lhs;
    const f16 *b = (const f16 *)rhs;
    for (size_t st = 0; st < b_subtiles; st++) {
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
        /* vr==0 bands are zero-filled without reading src; clamp the base so no   */              \
        /* out-of-bounds pointer is even formed (UB without a deref), as pack_b_band does. */      \
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
        f16 *a_pack = (f16 *)malloc(ep_cmul(ep_cmul(a_subtiles, per_tile), sizeof(f16)));
        if (a_pack) {
            PACKA_RANGE(a_pack, 0, m_tiles);
            dispatch_apply(nn_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
              size_t nt0 = ci * N_CHUNK;
              size_t nt1 = nt0 + N_CHUNK < n_tiles ? nt0 + N_CHUNK : n_tiles;
              sme_run_streaming((f16 *)dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, nt0,
                                nt1, kp, alpha, beta, read_dst);
            });
            free(a_pack);
            free(b_pack);
            return 0;
        }
        // malloc failed: fall through to the serial path.
    }
    if (n_chunks <= 1 || !big) {
        f16 *a_pack = apack_scratch(ep_cmul(ep_cmul(a_subtiles, per_tile), sizeof(f16)));
        if (!a_pack) {
            free(b_pack);
            return -1;
        }
        PACKA_RANGE(a_pack, 0, m_tiles);
        sme_run_streaming((f16 *)dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, 0, n_tiles,
                          kp, alpha, beta, read_dst);
        free(b_pack);
        return 0;
    }
    f16 *a_pack = (f16 *)malloc(ep_cmul(ep_cmul(a_subtiles, per_tile), sizeof(f16)));
    if (!a_pack) {
        free(b_pack);
        return -1;
    }
    size_t nc_blk = f16w_nc_blk(kp, n_tiles);
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
      sme_run_streaming((f16 *)dst, dst_cs, dst_rs, a_pack, b_pack, m, n, mt0, mt1, jc, jc_end, kp,
                        alpha, beta, read_dst);
    });
#undef PACKA_RANGE

    free(a_pack);
    free(b_pack);
    return 0;
}
