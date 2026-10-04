// SME f32 GEMM driver (Apple M4+, FMOPA single-precision). 32x32 super-tile =
// four 16x16 ZA32 quadrants; one K-value per MOPA (no widening, no zip). fp32
// in/out, so the epilogue is a direct ZA->memory store (pure C=A@B) or a
// read/scale/store. Packing is plain [k, 16] per 16-wide band.
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

#include "attention.h"
#include "epilogue.h"
#include "panel_ring.h"

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
// the live quadrants halves MOPA issue on those small-m shapes (the store,
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

// Resident-bytes budget for the cache blocking below, in 32-wide super-tiles
// (2 packed bands each), so `budget * super_bytes` is the constant. f32 panels
// are 2x the bytes of f16, so this path needs a larger one than the f16 driver
// (8 MB, where the tile IS the super-tile).
#define F32_CACHE_BUDGET_BYTES (24u * 1024 * 1024)

// Largest B (n*k*4 bytes) the small path may read unpacked; justified at its use
// site in gemm_sme_f32_run.
#define SME_F32_DIRECT_B_BYTES ((size_t)14 * 1024 * 1024)

// Flop count below which the direct-B path stays on one cluster (justified at
// its use site in gemm_sme_f32_run).
#define SME_F32_SERIAL_FLOPS ((uint64_t)1 << 22)

// Largest A (in elements) the ZA-transpose pack takes; justified at its use site
// in gemm_sme_f32_run.
#define SME_F32_ZA_PACK_ELEMS ((size_t)1 << 20)

// Workers for the direct-B path's dynamic M-chunk claiming. Deliberately one
// fewer than the number of chunks: hand every worker exactly one and the call
// ends on the slowest, which is an E-cluster worker every time. Leaving a chunk
// spare is what lets a P-cluster worker come back for it. Below four chunks
// there is not enough left to steal and holding a worker back only
// under-subscribes the machine.
#define SME_F32_MAX_WORKERS 8
static size_t f32_workers(size_t chunks) {
    if (chunks <= 3) return chunks;
    size_t w = chunks - 1;
    return w > SME_F32_MAX_WORKERS ? SME_F32_MAX_WORKERS : w;
}

static size_t f32_budget(size_t k) {
    size_t tile_bytes = 2 * ep_cmul(k, 16) * sizeof(float);
    size_t b = (size_t)F32_CACHE_BUDGET_BYTES / (tile_bytes ? tile_bytes : 1);
    return b < 2 ? 2 : b;
}

// N-block width, shared by the driver's outer jc loop and run_streaming's inner
// one so the two agree on block boundaries. Rounded down to an even split of
// n_tiles: the budget is a ceiling, not a target, and a ragged last block (64
// tiles at width 48 -> 48+16) amortizes the A-block re-read over too little N.
static size_t f32_nc_blk(size_t k, size_t n_tiles) {
    size_t nc = f32_budget(k) / 2;
    if (nc < 1) nc = 1;
    if (nc >= n_tiles) return n_tiles;
    size_t blocks = (n_tiles + nc - 1) / nc;
    return (n_tiles + blocks - 1) / blocks;
}

// m <= 4: SME2 multi-vector FMLA into ZA vector groups instead of MOPA; see
// run_gemv in gemm_f16f16.c. 16-column bands, two per 32-wide tile; `abc` is
// [rows][k][16] broadcast A; b_pack points at tile nt_lo. The bound is FMLA
// issue (~3 cycles per vg1x4, measured), and the P-cluster's one SME unit is
// shared, so more threads do not raise it.
#define GEMV_MAXR 2 // f32 MOPA issues ~2x the FMLA rate, so the crossover is lower
__arm_locally_streaming __arm_new("za") static void run_gemv(float *dst, long dst_rs,
                                                             const float *abc, size_t rows,
                                                             const float *b_pack, size_t n,
                                                             size_t k, size_t nt_lo, size_t nt_hi,
                                                             float alpha, float beta, int read_dst,
                                                             const ep_desc_f32 *ep) {
    svbool_t p32 = svptrue_b32();
    svcount_t pn = svptrue_c32();
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    int has_ep = ep && ep->n_nodes > 0;
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t per_band = ep_cmul(k, 16); // also the stride between abc rows
    size_t kfull = k & ~(size_t)3;
    svcount_t pt = svwhilelt_c32((uint64_t)0, (uint64_t)((k - kfull) * 16), 4);
    // Two tiles (four bands) per pass, so rows x bands <= 16 vector groups.
    for (size_t nt = nt_lo; nt < nt_hi; nt += 2) {
        size_t T = nt_hi - nt < 2 ? 2 * (nt_hi - nt) : 4; // bands
        const float *b = b_pack + 2 * (nt - nt_lo) * per_band;
        svzero_za();
        if (T == 4) {
            for (size_t d = 0; d < kfull; d += 4) {
                svfloat32x4_t B0 = svld1_f32_x4(pn, b + d * 16);
                svfloat32x4_t B1 = svld1_f32_x4(pn, b + per_band + d * 16);
                svfloat32x4_t B2 = svld1_f32_x4(pn, b + 2 * per_band + d * 16);
                svfloat32x4_t B3 = svld1_f32_x4(pn, b + 3 * per_band + d * 16);
                for (size_t r = 0; r < rows; r++) {
                    svfloat32x4_t A = svld1_f32_x4(pn, abc + r * per_band + d * 16);
                    uint32_t w = (uint32_t)(4 * r);
                    svmla_za32_f32_vg1x4(w, B0, A);
                    svmla_za32_f32_vg1x4(w + 1, B1, A);
                    svmla_za32_f32_vg1x4(w + 2, B2, A);
                    svmla_za32_f32_vg1x4(w + 3, B3, A);
                }
            }
        } else {
            for (size_t t = 0; t < T; t++)
                for (size_t d = 0; d < kfull; d += 4) {
                    svfloat32x4_t B = svld1_f32_x4(pn, b + t * per_band + d * 16);
                    for (size_t r = 0; r < rows; r++)
                        svmla_za32_f32_vg1x4((uint32_t)(4 * r + t), B,
                                             svld1_f32_x4(pn, abc + r * per_band + d * 16));
                }
        }
        if (kfull < k)
            for (size_t t = 0; t < T; t++) {
                svfloat32x4_t B = svld1_f32_x4(pt, b + t * per_band + kfull * 16);
                for (size_t r = 0; r < rows; r++)
                    svmla_za32_f32_vg1x4((uint32_t)(4 * r + t), B,
                                         svld1_f32_x4(pt, abc + r * per_band + kfull * 16));
            }
        for (size_t r = 0; r < rows; r++)
            for (size_t t = 0; t < T; t += 2) {
                size_t n0 = (nt + t / 2) * 32;
                size_t nc = n - n0 < 32 ? n - n0 : 32;
                svbool_t plo = svwhilelt_b32((uint32_t)0, (uint32_t)(nc < 16 ? nc : 16));
                svbool_t phi = svwhilelt_b32((uint32_t)0, (uint32_t)(nc > 16 ? nc - 16 : 0));
                svfloat32x4_t ql = svread_za32_f32_vg1x4((uint32_t)(4 * r + t));
                svfloat32x4_t qh = svread_za32_f32_vg1x4((uint32_t)(4 * r + t + 1));
                svfloat32_t lo = svadd_x(p32, svadd_x(p32, svget4(ql, 0), svget4(ql, 1)),
                                         svadd_x(p32, svget4(ql, 2), svget4(ql, 3)));
                svfloat32_t hi = svadd_x(p32, svadd_x(p32, svget4(qh, 0), svget4(qh, 1)),
                                         svadd_x(p32, svget4(qh, 2), svget4(qh, 3)));
                float *rp = dst + (long)r * dst_rs + n0;
                float *rph = nc > 16 ? rp + 16 : rp;
                if (pure) {
                    svst1_f32(plo, rp, lo);
                    svst1_f32(phi, rph, hi);
                } else if (has_ep) {
#define EP_GEMV_RDL(s) lo
#define EP_GEMV_RDH(s) hi
                    EP_STORE_TILE_ROWMAJOR_F32(p32, plo, phi, rp, dst_rs, EP_GEMV_RDL, EP_GEMV_RDH, 0, 1,
                                               vb, va, read_dst, nodes, n_nodes, r, n0);
#undef EP_GEMV_RDL
#undef EP_GEMV_RDH
                } else {
                    lo = svmul_x(p32, lo, vb);
                    hi = svmul_x(p32, hi, vb);
                    if (read_dst) {
                        lo = svmla_x(p32, lo, svld1_f32(plo, rp), va);
                        hi = svmla_x(p32, hi, svld1_f32(phi, rph), va);
                    }
                    svst1_f32(plo, rp, lo);
                    svst1_f32(phi, rph, hi);
                }
            }
    }
}

// Row-major-B GEMV read in place (`b_rm` is column nt_lo*32, row stride b_rs).
// One load is four bands at one depth, so it takes the tuple-by-vector FMLA,
// which issues ~25% faster than run_gemv's tuple-by-tuple form; group (r, g, p)
// holds band group g's depth phase p. Loads go depth-major across the pass's
// band groups: down one band group at a power-of-two stride every load hits the
// same L2 bank (1.8x slower, measured). Each band still sums its four depth
// phases in run_gemv's order, so the two agree bit for bit.
//
// One pass's accumulate, inlined per row count (see gemm_f16f16.c's twin).
__attribute__((always_inline)) static inline void
gemv_rm_acc(size_t rows, const float *b, long b_rs, const float *abc, size_t per_band, size_t k,
            size_t G, svcount_t pc0, svcount_t pc1, svcount_t pc2,
            svcount_t pc3) __arm_streaming __arm_inout("za") {
    svbool_t p32 = svptrue_b32();
    svcount_t pn = svptrue_c32();
    size_t BG = 4 / rows, kfull = k & ~(size_t)3;
#define GEMV_FMLA(r, p, a_)                                                                        \
    if (rows > (r)) {                                                                              \
        uint32_t w = (uint32_t)(4 * (r) * BG + (p));                                               \
        svmla_single_za32_f32_vg1x4(w, B0, a_);                                                    \
        if (G > 1) svmla_single_za32_f32_vg1x4(w + 4, B1, a_);                                     \
        if (G > 2) svmla_single_za32_f32_vg1x4(w + 8, B2, a_);                                     \
        if (G > 3) svmla_single_za32_f32_vg1x4(w + 12, B3, a_);                                    \
    }
#define GEMV_ROW(p, row, AV)                                                                       \
    {                                                                                              \
        svfloat32x4_t B0 = svld1_f32_x4(pc0, (row)), B1 = B0, B2 = B0, B3 = B0;                    \
        if (G > 1) B1 = svld1_f32_x4(pc1, (row) + 64);                                             \
        if (G > 2) B2 = svld1_f32_x4(pc2, (row) + 128);                                            \
        if (G > 3) B3 = svld1_f32_x4(pc3, (row) + 192);                                            \
        GEMV_FMLA(0, p, AV(0))                                                                     \
        GEMV_FMLA(1, p, AV(1))                                                                     \
    }
    size_t d = 0;
    for (; d < kfull; d += 4) {
        const float *b0 = b + (long)d * b_rs;
        svfloat32x4_t A0 = svld1_f32_x4(pn, abc + d * 16), A1 = A0;
        if (rows > 1) A1 = svld1_f32_x4(pn, abc + per_band + d * 16);
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
#define AVT(r) svld1_f32(p32, abc + (r) * per_band + d * 16)
    for (; d < k; d++)
        GEMV_ROW(d - kfull, b + (long)d * b_rs, AVT)
#undef AVT
#undef GEMV_ROW
#undef GEMV_FMLA
}

__arm_locally_streaming __arm_new("za") static void run_gemv_rm(
    float *dst, long dst_rs, const float *abc, size_t rows, const float *b_rm, long b_rs, size_t n,
    size_t k, size_t nt_lo, size_t nt_hi, float alpha, float beta, int read_dst,
    const ep_desc_f32 *ep) {
    svbool_t p32 = svptrue_b32();
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    int has_ep = ep && ep->n_nodes > 0;
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t per_band = ep_cmul(k, 16); // the stride between abc rows
    size_t BG = 4 / rows;             // band groups (64 columns, two tiles) per pass
    for (size_t nt = nt_lo; nt < nt_hi; nt += 2 * BG) {
        size_t G = (nt_hi - nt + 1) / 2 < BG ? (nt_hi - nt + 1) / 2 : BG;
        const float *b = b_rm + (nt - nt_lo) * 32;
        svcount_t pc0 = svwhilelt_c32((uint64_t)nt * 32, (uint64_t)n, 4);
        svcount_t pc1 = svwhilelt_c32((uint64_t)nt * 32 + 64, (uint64_t)n, 4);
        svcount_t pc2 = svwhilelt_c32((uint64_t)nt * 32 + 128, (uint64_t)n, 4);
        svcount_t pc3 = svwhilelt_c32((uint64_t)nt * 32 + 192, (uint64_t)n, 4);
        svzero_za();
        if (rows == 1)
            gemv_rm_acc(1, b, b_rs, abc, per_band, k, G, pc0, pc1, pc2, pc3);
        else
            gemv_rm_acc(2, b, b_rs, abc, per_band, k, G, pc0, pc1, pc2, pc3);
        for (size_t r = 0; r < rows; r++)
            for (size_t g = 0; g < G; g++) {
                uint32_t w = (uint32_t)(4 * (r * BG + g));
                svfloat32x4_t Q0 = svread_za32_f32_vg1x4(w), Q1 = svread_za32_f32_vg1x4(w + 1);
                svfloat32x4_t Q2 = svread_za32_f32_vg1x4(w + 2), Q3 = svread_za32_f32_vg1x4(w + 3);
                size_t T = 2 * (nt_hi - nt) - 4 * g < 4 ? 2 * (nt_hi - nt) - 4 * g : 4; // bands
                size_t ntg = nt + 2 * g;
            // Tile i = bands 2i, 2i+1, each its four phase partials in run_gemv's order.
#define GEMV_BAND(j)                                                                               \
    svadd_x(p32, svadd_x(p32, svget4(Q0, j), svget4(Q1, j)),                                       \
            svadd_x(p32, svget4(Q2, j), svget4(Q3, j)))
#define GEMV_ST(i)                                                                                 \
    if (2 * i < T) {                                                                               \
        size_t n0 = (ntg + i) * 32;                                                                \
        size_t nc = n - n0 < 32 ? n - n0 : 32;                                                     \
        svbool_t plo = svwhilelt_b32((uint32_t)0, (uint32_t)(nc < 16 ? nc : 16));                  \
        svbool_t phi = svwhilelt_b32((uint32_t)0, (uint32_t)(nc > 16 ? nc - 16 : 0));              \
        svfloat32_t lo = GEMV_BAND(2 * i), hi = GEMV_BAND(2 * i + 1);                              \
        float *rp = dst + (long)r * dst_rs + n0;                                                   \
        float *rph = nc > 16 ? rp + 16 : rp;                                                       \
        if (pure) {                                                                                \
            svst1_f32(plo, rp, lo);                                                                \
            svst1_f32(phi, rph, hi);                                                               \
        } else if (has_ep) {                                                                       \
            EP_STORE_TILE_ROWMAJOR_F32(p32, plo, phi, rp, dst_rs, EP_GEMV_RDL, EP_GEMV_RDH, 0, 1,  \
                                       vb, va, read_dst, nodes, n_nodes, r, n0);                   \
        } else {                                                                                   \
            lo = svmul_x(p32, lo, vb);                                                             \
            hi = svmul_x(p32, hi, vb);                                                             \
            if (read_dst) {                                                                        \
                lo = svmla_x(p32, lo, svld1_f32(plo, rp), va);                                     \
                hi = svmla_x(p32, hi, svld1_f32(phi, rph), va);                                    \
            }                                                                                      \
            svst1_f32(plo, rp, lo);                                                                \
            svst1_f32(phi, rph, hi);                                                               \
        }                                                                                          \
    }
#define EP_GEMV_RDL(s) lo
#define EP_GEMV_RDH(s) hi
                GEMV_ST(0)
                GEMV_ST(1)
#undef EP_GEMV_RDL
#undef EP_GEMV_RDH
#undef GEMV_ST
#undef GEMV_BAND
            }
    }
}

// [nt_lo, nt_hi) is the N-tile sub-range this invocation owns -- pass
// (0, n_tiles) for the whole problem, or a slice for the N-parallel flat-M path
// (see the driver); b_pack points at super-tile nt_lo's bands. f32 N-tiles are
// independent (no dual-tile lockstep), so any chunk granularity is safe.
__arm_locally_streaming __arm_new("za") static void run_streaming(
    float *dst, long dst_cs, long dst_rs, const float *a_pack, const float *b_pack, size_t m,
    size_t n, size_t mt_lo, size_t mt_hi, size_t nt_lo, size_t nt_hi, size_t k, float alpha,
    float beta, int read_dst, const ep_desc_f32 *ep) {
    svbool_t p32 = svptrue_b32();
    const svfloat32_t z32 = svdup_n_f32(0.0f);
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    svcount_t pn32 = svptrue_c32();
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

    // Cache blocking (BLIS jc->ic). nc_blk/mc_blk count 32-wide SUPER-tiles
    // (= 2 packed bands each). mc_blk is capped at the M-chunk (M_CHUNK
    // super-tiles), so in the parallel path only nc_blk grows. The jc loop here
    // only blocks within one M-chunk; cross-chunk B reuse comes from the driver
    // running this once per N-block (see gemm_sme_f32_run_packed).
    size_t budget = f32_budget(k);
    size_t nt_span = nt_hi - nt_lo;
    size_t nc_blk = f32_nc_blk(k, nt_span);
    if (has_reduce) nc_blk = nt_span;
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
                    const float *b_lo = b_pack + (size_t)(2 * (nt - nt_lo)) * per_tile;
                    const float *b_hi = b_lo + per_tile;
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
                        // One multi-vector LD1W per band: 4 depth-vectors a load.
                        for (; d + 4 <= k; d += 4) {
                            svfloat32x4_t AL = svld1_f32_x4(pn32, a_lo + d * 16);
                            svfloat32x4_t AH = svld1_f32_x4(pn32, a_hi + d * 16);
                            svfloat32x4_t BL = svld1_f32_x4(pn32, b_lo + d * 16);
                            svfloat32x4_t BH = svld1_f32_x4(pn32, b_hi + d * 16);
                            F32_STEP_FULL(svget4(AL, 0), svget4(AH, 0), svget4(BL, 0),
                                          svget4(BH, 0));
                            F32_STEP_FULL(svget4(AL, 1), svget4(AH, 1), svget4(BL, 1),
                                          svget4(BH, 1));
                            F32_STEP_FULL(svget4(AL, 2), svget4(AH, 2), svget4(BL, 2),
                                          svget4(BH, 2));
                            F32_STEP_FULL(svget4(AL, 3), svget4(AH, 3), svget4(BL, 3),
                                          svget4(BH, 3));
                        }
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

// Transpose row-major A into the packed [k, 16] layout using ZA itself: a
// vertical load writes a memory vector into a tile COLUMN, a horizontal store
// reads a tile ROW back out, so 16 of each transpose a 16x16 block. That is 32
// instructions per 256 floats against the NEON path's 16 loads, 16 stores and
// ~32 shuffles -- the NEON transpose is issue-bound at ~1 instruction per float,
// and this is the only way past that. Only tile 0 is touched, and the caller's
// accumulator tiles are dead at this point (this runs before the K-loop).
__arm_locally_streaming __arm_new("za") static void packa_sme(float *a_pack, const float *lhs,
                                                              long lhs_rs, size_t m, size_t k,
                                                              size_t mt_lo, size_t mt_hi) {
    svbool_t pg = svptrue_b32();
    size_t per_tile = ep_cmul(k, 16);
    for (size_t st = 2 * mt_lo; st < 2 * mt_hi; st++) {
        size_t r0 = st * 16;
        size_t vr = (r0 < m) ? ((m - r0 < 16) ? (m - r0) : 16) : 0;
        float *dst = a_pack + st * per_tile;
        size_t d = 0;
        // Four depth-blocks in flight, one per ZA tile: every store of a tile
        // depends on all sixteen loads into it, so a single tile leaves that
        // latency exposed.
        for (; d + 64 <= k; d += 64) {
            if (vr < 16) svzero_za();
            for (size_t i = 0; i < vr; i++) {
                const float *src = lhs + (long)(r0 + i) * lhs_rs + (long)d;
                svld1_ver_za32(0, (uint32_t)i, pg, src);
                svld1_ver_za32(1, (uint32_t)i, pg, src + 16);
                svld1_ver_za32(2, (uint32_t)i, pg, src + 32);
                svld1_ver_za32(3, (uint32_t)i, pg, src + 48);
            }
            for (uint32_t r = 0; r < 16; r++) {
                svst1_hor_za32(0, r, pg, dst + (d + r) * 16);
                svst1_hor_za32(1, r, pg, dst + (d + 16 + r) * 16);
                svst1_hor_za32(2, r, pg, dst + (d + 32 + r) * 16);
                svst1_hor_za32(3, r, pg, dst + (d + 48 + r) * 16);
            }
        }
        for (; d + 16 <= k; d += 16) {
            if (vr < 16) svzero_za();
            for (size_t i = 0; i < vr; i++)
                svld1_ver_za32(0, (uint32_t)i, pg, lhs + (long)(r0 + i) * lhs_rs + (long)d);
            for (uint32_t r = 0; r < 16; r++)
                svst1_hor_za32(0, r, pg, dst + (d + r) * 16);
        }
        for (; d < k; d++)
            for (size_t i = 0; i < 16; i++)
                dst[d * 16 + i] = (i < vr) ? lhs[(long)(r0 + i) * lhs_rs + (long)d] : 0.0f;
    }
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

// GEMV operand: abc[r][d*16 + j] = A[r, d], in the per-thread A scratch.
static float *bcast_rows(const float *a, long rs, long cs, size_t m, size_t k) {
    size_t per_band = ep_cmul(k, 16);
    float *abc = apack_scratch(ep_cmul(ep_cmul(m, per_band), sizeof(float)));
    if (!abc) return NULL;
    for (size_t r = 0; r < m; r++)
        for (size_t d = 0; d < k; d++) {
            float32x4_t v = vdupq_n_f32(a[(long)r * rs + (long)d * cs]);
            float32x4x4_t q = {{v, v, v, v}};
            vst1q_f32_x4(abc + r * per_band + d * 16, q);
        }
    return abc;
}

// Pack super-tiles [t0, t1) of B into consecutive band pairs at dst.
static void pack_b_range(float *dst, const float *b, long rs, long cs, size_t n, size_t k,
                         size_t t0, size_t t1) {
    size_t per_tile = ep_cmul(k, 16);
    for (size_t st = 2 * t0; st < 2 * t1; st++) {
        size_t c0 = st * 16, vc = c0 < n ? (n - c0 < 16 ? n - c0 : 16) : 0;
        pack_band(dst + (st - 2 * t0) * per_tile, vc ? b + (long)c0 * cs : b, cs, rs, k, vc);
    }
}

static void pack_a_range(float *a_pack, const float *a, long rs, long cs, size_t m, size_t k,
                         size_t mt0, size_t mt1) {
    size_t per_tile = ep_cmul(k, 16);
    for (size_t st = 2 * mt0; st < 2 * mt1; st++) {
        size_t r0 = st * 16, vr = r0 < m ? (m - r0 < 16 ? m - r0 : 16) : 0;
        pack_band(a_pack + st * per_tile, vr ? a + (long)r0 * rs : a, rs, cs, k, vr);
    }
}

// Row-major B read in place for super-tile nt (`b` is B's column 0, row stride
// b_rs): its two bands are one 128-byte line per row, so one x2 load per depth
// feeds the MOPAs and nothing is packed. With `cap` set the pass also writes the
// bands out packed ([2][k][16]) for the packed kernel to take the other
// M-tiles, and prefetches B 16 rows ahead: the SME loads alone keep too few
// misses in flight (128x8192x4096 1.45 -> 2.05 TF/s; neither the 16-bit kernels
// nor a one-M-tile pass gain from it). See gemm_f16f16.c's twin.
// Row-major C, no epilogue.
__arm_locally_streaming __arm_new("za") static void run_direct(float *dst, long dst_rs,
                                                               const float *a_pack, const float *b,
                                                               long b_rs, size_t m, size_t n,
                                                               size_t k, size_t mt_lo, size_t mt_hi,
                                                               size_t nt, float *cap, float alpha,
                                                               float beta, int read_dst) {
    svbool_t p32 = svptrue_b32();
    svcount_t pn = svptrue_c32();
    const svfloat32_t z32 = svdup_n_f32(0.0f);
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    int pure = (beta == 1.0f && !read_dst);
    size_t per_tile = ep_cmul(k, 16), kfull = k & ~(size_t)3;
    const float *bt = b + nt * 32;
    svcount_t pc = svwhilelt_c32((uint64_t)nt * 32, (uint64_t)n, 2);
    size_t n0 = nt * 32, nc = n - n0 < 32 ? n - n0 : 32;
    svbool_t plo = svwhilelt_b32((uint32_t)0, (uint32_t)(nc < 16 ? nc : 16));
    svbool_t phi = svwhilelt_b32((uint32_t)0, (uint32_t)(nc > 16 ? nc - 16 : 0));
    for (size_t mt = mt_lo; mt < mt_hi; mt++, cap = NULL) {
        const float *a_lo = a_pack + 2 * mt * per_tile, *a_hi = a_lo + per_tile;
        size_t m0 = mt * 32, mr = m - m0 < 32 ? m - m0 : 32;
        svzero_za();
#define DIRECT_B(row, d)                                                                           \
    if (cap) __builtin_prefetch((row) + 16 * b_rs);                                                \
    svfloat32x2_t B_ = svld1_f32_x2(pc, (row));                                                    \
    if (cap) {                                                                                     \
        svst1_f32(p32, cap + (d) * 16, svget2(B_, 0));                                             \
        svst1_f32(p32, cap + per_tile + (d) * 16, svget2(B_, 1));                                  \
    }
#define DIRECT_FULL(al, ah, row, d)                                                                \
    {                                                                                              \
        DIRECT_B(row, d)                                                                           \
        F32_STEP_FULL(al, ah, svget2(B_, 0), svget2(B_, 1));                                       \
    }
#define DIRECT_NARROW(al, row, d)                                                                  \
    {                                                                                              \
        DIRECT_B(row, d)                                                                           \
        F32_STEP_NARROW_M(al, svget2(B_, 0), svget2(B_, 1));                                       \
    }
        size_t d = 0;
        if (mr > 16) {
            for (; d < kfull; d += 4) {
                svfloat32x4_t AL = svld1_f32_x4(pn, a_lo + d * 16);
                svfloat32x4_t AH = svld1_f32_x4(pn, a_hi + d * 16);
                const float *r = bt + (long)d * b_rs;
                DIRECT_FULL(svget4(AL, 0), svget4(AH, 0), r, d)
                DIRECT_FULL(svget4(AL, 1), svget4(AH, 1), r + b_rs, d + 1)
                DIRECT_FULL(svget4(AL, 2), svget4(AH, 2), r + 2 * b_rs, d + 2)
                DIRECT_FULL(svget4(AL, 3), svget4(AH, 3), r + 3 * b_rs, d + 3)
            }
            for (; d < k; d++)
                DIRECT_FULL(svld1_f32(p32, a_lo + d * 16), svld1_f32(p32, a_hi + d * 16),
                            bt + (long)d * b_rs, d)
        } else {
            for (; d < kfull; d += 4) {
                svfloat32x4_t AL = svld1_f32_x4(pn, a_lo + d * 16);
                const float *r = bt + (long)d * b_rs;
                DIRECT_NARROW(svget4(AL, 0), r, d)
                DIRECT_NARROW(svget4(AL, 1), r + b_rs, d + 1)
                DIRECT_NARROW(svget4(AL, 2), r + 2 * b_rs, d + 2)
                DIRECT_NARROW(svget4(AL, 3), r + 3 * b_rs, d + 3)
            }
            for (; d < k; d++)
                DIRECT_NARROW(svld1_f32(p32, a_lo + d * 16), bt + (long)d * b_rs, d)
        }
#undef DIRECT_NARROW
#undef DIRECT_FULL
#undef DIRECT_B
        float *row = dst + (long)m0 * dst_rs + (long)n0;
        for (size_t r = 0; r < mr; r++, row += dst_rs) {
            uint32_t s = (uint32_t)(r & 15);
            if (pure && r < 16) {
                svst1_hor_za32(0, s, plo, row);
                svst1_hor_za32(1, s, phi, row + 16);
            } else if (pure) {
                svst1_hor_za32(2, s, plo, row);
                svst1_hor_za32(3, s, phi, row + 16);
            } else {
                svfloat32_t lo = r < 16 ? svread_hor_za32_f32_m(z32, p32, 0, s)
                                        : svread_hor_za32_f32_m(z32, p32, 2, s);
                svfloat32_t hi = r < 16 ? svread_hor_za32_f32_m(z32, p32, 1, s)
                                        : svread_hor_za32_f32_m(z32, p32, 3, s);
                lo = svmul_x(p32, lo, vb);
                hi = svmul_x(p32, hi, vb);
                if (read_dst) {
                    lo = svmla_x(p32, lo, svld1_f32(plo, row), va);
                    hi = svmla_x(p32, hi, svld1_f32(phi, row + 16), va);
                }
                svst1_f32(plo, row, lo);
                svst1_f32(phi, row + 16, hi);
            }
        }
    }
}

// Per-worker column items, as gemm_f16f16.c's run_cols: each worker packs its own
// two super-tiles of B and multiplies them while they are in cache; the first
// units pack A by M-chunk.
static int run_cols(float *dst, long dst_cs, long dst_rs, const float *a, long lhs_rs, long lhs_cs,
                    size_t m, size_t n, size_t k, float alpha, float beta, int read_dst,
                    const float *b, long rhs_rs, long rhs_cs, const ep_desc_f32 *ep) {
    size_t per_tile = ep_cmul(k, 16);
    size_t m_tiles = (m + 31) / 32, n_tiles = (n + 31) / 32;
    // Row-major B and C, no epilogue: multiply straight from B's rows (run_direct).
    // Unlike f16 this wins at power-of-two strides too (128x8192x4096: 2.1
    // against 1.7 TF/s through the pack).
    int direct = rhs_cs == 1 && dst_cs == 1 && rhs_rs > 0 && !(ep && ep->n_nodes);
    // Super-tiles per item. Packed, wider items lose 10-50% at small m; direct,
    // one pass down B per item beats two adjacent ones (0.9-0.95x of packed
    // against 0.8-0.87x).
    size_t NC = direct ? 1 : 2;
    size_t M_CHUNK = 2, n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    size_t n_items = (n_tiles + NC - 1) / NC, n_units = n_chunks + n_items;
    size_t W = n_units < 12 ? n_units : 12;
    size_t panel = ep_cmul(2 * NC, per_tile);
    _Atomic size_t *ctr;
    float *pool = (float *)ring_pool(2 + n_chunks, ep_cmul(ep_cmul(W, panel), sizeof(float)), &ctr);
    float *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
    if (!pool || !a_pack) return -1;
    _Atomic size_t *cur = ctr, *a_cnt = ctr + 1, *a_done = ctr + 2;
    dispatch_apply(W, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t w) {
      float *bt = pool + w * panel;
      for (size_t i; (i = atomic_fetch_add_explicit(cur, 1, memory_order_relaxed)) < n_units;) {
          if (i < n_chunks) {
              size_t mt1 = (i + 1) * M_CHUNK < m_tiles ? (i + 1) * M_CHUNK : m_tiles;
              pack_a_range(a_pack, a, lhs_rs, lhs_cs, m, k, i * M_CHUNK, mt1);
              atomic_store_explicit(&a_done[i], 1, memory_order_release);
              atomic_fetch_add_explicit(a_cnt, 1, memory_order_release);
              continue;
          }
          size_t t0 = (i - n_chunks) * NC, t1 = t0 + NC < n_tiles ? t0 + NC : n_tiles, mt_lo = 0;
          if (direct) {
              // M-tile 0 multiplies straight from B's rows, capturing the bands
              // when more M-tiles follow; those run the packed kernel.
              while (!atomic_load_explicit(&a_done[0], memory_order_acquire))
                  __builtin_arm_yield();
              for (size_t t = t0; t < t1; t++)
                  run_direct(dst, dst_rs, a_pack, b, rhs_rs, m, n, k, 0, 1, t,
                             m_tiles > 1 ? bt + 2 * (t - t0) * per_tile : NULL, alpha, beta,
                             read_dst);
              mt_lo = 1;
          } else {
              pack_b_range(bt, b, rhs_rs, rhs_cs, n, k, t0, t1);
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
              run_streaming(dst, dst_cs, dst_rs, a_pack, bt, m, n, mt0, mt1, t0, t1, k, alpha, beta,
                            read_dst, ep);
          }
      }
    });
    return 0;
}

// Large A: B is built per N-block into a panel ring (panel_ring.h) and shared by
// every M-chunk, as gemm_f16f16.c's run_panels.
static int run_panels(float *dst, long dst_cs, long dst_rs, const float *a, long lhs_rs,
                      long lhs_cs, size_t m, size_t n, size_t k, float alpha, float beta,
                      int read_dst, const float *b, long rhs_rs, long rhs_cs,
                      const ep_desc_f32 *ep) {
    size_t per_tile = ep_cmul(k, 16), tile = 2 * per_tile;
    size_t m_tiles = (m + 31) / 32, n_tiles = (n + 31) / 32;
    // 4 MB B-blocks, as f16: three ring slots of the run_packed-sized 12 MB block
    // overflow L2 (0.93-0.96x at 4096^3 and 4096x11008x4096, against 0.98-0.99).
    size_t nc_blk = ((size_t)4 << 20) / ep_cmul(tile, sizeof(float));
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
    float *pool = (float *)ring_pool(2 * n_blocks + 2 + n_chunks,
                                     ep_cmul(ep_cmul(n_slots, slot_elems), sizeof(float)), &ctr);
    float *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
    if (!pool || !a_pack) return -1;
    _Atomic size_t *a_done = ctr + 2 * n_blocks + 2;
    if (n_chunks <= 1) pack_a_range(a_pack, a, lhs_rs, lhs_cs, m, k, 0, m_tiles);

    ring_run(
        n_blocks, n_d, n_chunks * n_sub, ctr,
        ^(size_t sd, size_t j) {
          size_t lo = sd * nc_blk, hi = lo + nc_blk < n_tiles ? lo + nc_blk : n_tiles;
          size_t t0 = lo + j * dt, t1 = t0 + dt < hi ? t0 + dt : hi;
          if (t0 < t1)
              pack_b_range(pool + (sd % RING_SLOTS) * slot_elems + (t0 - lo) * tile, b, rhs_rs,
                           rhs_cs, n, k, t0, t1);
        },
        ^(size_t sg, size_t j) {
          size_t lo = sg * nc_blk, hi = lo + nc_blk < n_tiles ? lo + nc_blk : n_tiles;
          size_t ci = j / n_sub, sj = j % n_sub;
          size_t mt0 = ci * M_CHUNK, mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
          size_t n0 = lo + sj * ns < hi ? lo + sj * ns : hi, n1 = n0 + ns < hi ? n0 + ns : hi;
          // Block 0's first item per M-chunk packs that chunk of A.
          if (n_chunks > 1 && sg == 0 && sj == 0) {
              pack_a_range(a_pack, a, lhs_rs, lhs_cs, m, k, mt0, mt1);
              atomic_store_explicit(&a_done[ci], 1, memory_order_release);
          } else if (n_chunks > 1) {
              while (!atomic_load_explicit(&a_done[ci], memory_order_acquire))
                  __builtin_arm_yield();
          }
          if (n0 < n1)
              run_streaming(dst, dst_cs, dst_rs, a_pack,
                            pool + (sg % RING_SLOTS) * slot_elems + (n0 - lo) * tile, m, n, mt0,
                            mt1, n0, n1, k, alpha, beta, read_dst, ep);
        });
    return 0;
}

int gemm_sme_f32_run(size_t m, size_t n, size_t k, float *dst, long dst_cs, long dst_rs,
                     int read_dst, const float *lhs, long lhs_cs, long lhs_rs, const float *rhs,
                     long rhs_cs, long rhs_rs, float alpha, float beta, const ep_desc_f32 *ep) {
    if (m == 0 || n == 0) return 0;
    if (!read_dst) alpha = 0.0f;

    size_t m_tiles = (m + 31) / 32;
    size_t per_tile = ep_cmul(k, 16);

    if (m <= GEMV_MAXR && dst_cs == 1 && rhs_cs == 1 && rhs_rs > 0 && !ep_has_reduce(ep)) {
        // Row-major B: the GEMV reads it in place, nothing to pack.
        float *abc = bcast_rows(lhs, lhs_rs, lhs_cs, m, k);
        if (!abc) return -1;
        size_t n_tiles = (n + 31) / 32, G_CHUNK = 8, g_chunks = (n_tiles + G_CHUNK - 1) / G_CHUNK;
        if (ep_flops(m, n, k) < (1u << 21) || g_chunks < 3) {
            run_gemv_rm(dst, dst_rs, abc, m, rhs, rhs_rs, n, k, 0, n_tiles, alpha, beta, read_dst,
                        ep);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t nt0 = ci * G_CHUNK, nt1 = nt0 + G_CHUNK < n_tiles ? nt0 + G_CHUNK : n_tiles;
          run_gemv_rm(dst, dst_rs, abc, m, rhs + nt0 * 32, rhs_rs, n, k, nt0, nt1, alpha, beta,
                      read_dst, ep);
        });
        return 0;
    }

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
    // Direct-B path: read B straight from the row-major rhs rather than pay a
    // full read and write of it to pack before any MOPA issues. B is then
    // re-read per M-tile, so what decides this is B's FOOTPRINT, not the problem
    // size -- 8192x1024x1024 and 2048^3 have the same flop count and opposite
    // winners. Blocking the path over N does not lift the limit: a block still
    // walks k depths at stride n, covering B's whole address range whatever its
    // byte size, and packing is what makes that contiguous.
    // run_small has no reduction block; reductions take run_streaming.
    size_t b_budget = SME_F32_DIRECT_B_BYTES / sizeof(float);
    if (rhs_cs == 1 && !ep_has_reduce(ep) && ep_cmul(n, k) <= b_budget) {
        size_t sc_chunk = 2;
        size_t sc_n = (m_tiles + sc_chunk - 1) / sc_chunk;
        // Below this the dispatch costs more than the second cluster earns.
        // Absolute work is the signal, not chunk count: 128^3 and 128x512x512
        // both split two ways and want opposite answers.
        if (sc_n <= 1 || ep_flops(m, n, k) < SME_F32_SERIAL_FLOPS) {
            float *a_pack = apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
            if (a_pack) {
                PACKA_SMALL(a_pack, 0, m_tiles);
                run_small(dst, dst_cs, dst_rs, a_pack, rhs, rhs_rs, m, n, k, 0, m_tiles, alpha,
                          beta, read_dst, ep);
                return 0;
            }
        } else {
            float *a_pack = (float *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
            if (a_pack) {
                // Chunks are CLAIMED off a shared cursor rather than handed out
                // one per worker, so the P-cluster takes more of them than the
                // E-cluster instead of the call ending on the slower half. Each
                // worker still packs the chunk it is about to compute, which
                // keeps the NEON pack overlapping MOPA issue -- splitting the
                // pack into its own pass to allow finer scheduling costs more
                // than the balance it buys.
                _Atomic size_t cursor = 0;
                // Blocks capture by const value, so capture the POINTER: taking
                // &cursor inside the block would give each worker its own.
                _Atomic size_t *cur = &cursor;
                // ZA-transpose the A-pack while A is small enough to stay in
                // cache; once it is DRAM-sized the NEON transpose's wider
                // memory-level parallelism wins back more than the instruction
                // count costs (0.91x at 16384x512x512, where A is 32 MB).
                int sme_pack = lhs_cs == 1 &&
                               ep_cmul(m, k) <= SME_F32_ZA_PACK_ELEMS;
                dispatch_apply(f32_workers(sc_n),
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
                                     if (sme_pack) {
                                         packa_sme(a_pack, lhs, lhs_rs, m, k, mt0, mt1);
                                     } else {
                                         PACKA_SMALL(a_pack, mt0, mt1);
                                     }
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

    // Build B inside the parallel GEMM rather than packing it all first. A
    // reduction spans all of N, so it keeps the one-block path below.
    if (m > GEMV_MAXR && !ep_has_reduce(ep) && ep_flops(m, n, k) >= (1u << 21)) {
        if (use_cols(m, n, ep_cmul(ep_cmul(m, k), sizeof(float))))
            return run_cols(dst, dst_cs, dst_rs, lhs, lhs_rs, lhs_cs, m, n, k, alpha, beta,
                            read_dst, rhs, rhs_rs, rhs_cs, ep);
        return run_panels(dst, dst_cs, dst_rs, lhs, lhs_rs, lhs_cs, m, n, k, alpha, beta, read_dst,
                          rhs, rhs_rs, rhs_cs, ep);
    }

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
    if (m <= GEMV_MAXR && dst_cs == 1 && !ep_has_reduce(ep)) {
        float *abc = bcast_rows(lhs, lhs_rs, lhs_cs, m, k);
        if (!abc) return -1;
        size_t G_CHUNK = 4; // tiles per chunk
        size_t g_chunks = (n_tiles + G_CHUNK - 1) / G_CHUNK;
        if (!big || g_chunks < 3) {
            run_gemv(dst, dst_rs, abc, m, b_pack, n, k, 0, n_tiles, alpha, beta, read_dst, ep);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t nt0 = ci * G_CHUNK;
          size_t nt1 = nt0 + G_CHUNK < n_tiles ? nt0 + G_CHUNK : n_tiles;
          run_gemv(dst, dst_rs, abc, m, b_pack + 2 * nt0 * per_tile, n, k, nt0, nt1, alpha, beta,
                   read_dst, ep);
        });
        return 0;
    }

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
              run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack + 2 * nt0 * per_tile, m, n, 0,
                            m_tiles, nt0, nt1, k, alpha, beta, read_dst, ep);
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

    // N-block loop OUTSIDE the M dispatch, so one B-block serves every M-chunk
    // while it is still L2-resident. Nested the other way each chunk sweeps all
    // of B and B leaves L2 before the next chunk reaches it, re-streaming packed
    // B from DRAM once per chunk -- 64 chunks x 64 MB at 4096^3 -- and the inner
    // blocking cannot help because it only ever sees one chunk. A reduction
    // epilogue spans all of N, so it keeps a single block.
    size_t nc_blk = ep_has_reduce(ep) ? n_tiles : f32_nc_blk(k, n_tiles);
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
      run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack + 2 * jc * per_tile, m, n, mt0, mt1, jc,
                    jc_end, k, alpha, beta, read_dst, ep);
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
