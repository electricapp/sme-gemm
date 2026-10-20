// SME f32 GEMM driver (Apple M4+, FMOPA single-precision). 32x32 super-tile =
// four 16x16 ZA32 quadrants; one K-value per MOPA (no widening, no zip). fp32
// in/out, so the epilogue is a direct ZA->memory store (pure C=A@B) or a
// read/scale/store. Packing is plain [k, 16] per 16-wide band.
//
// Computes dst = alpha*dst + beta*(A @ B).

#include "sme_gemm_api.h"

#include <arm_neon.h>
#include <arm_sme.h>
#include <atomic>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "epilogue.h"
#include "gemm_float.h"
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

// m <= 4: SME2 multi-vector FMLA into ZA vector groups instead of MOPA; see
// run_gemv in gemm_half.h. 16-column bands, two per 32-wide tile; `abc` is
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
// One pass's accumulate, inlined per row count (see gemm_half.h's twin).
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

// GEMV operand: abc[r][d*16 + j] = A[r, d], in the per-thread A scratch.
static float *bcast_rows(const float *a, long rs, long cs, size_t m, size_t k) {
    size_t per_band = ep_cmul(k, 16);
    float *abc = apack_scratch<float>(ep_cmul(ep_cmul(m, per_band), sizeof(float)));
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
// nor a one-M-tile pass gain from it). See gemm_half.h's twin.
// Row-major C, no epilogue.
__arm_locally_streaming __arm_new("za") static void run_direct(float *dst, long dst_rs,
                                                               const float *a_pack, const float *b,
                                                               long b_rs, size_t m, size_t n,
                                                               size_t k, size_t mt_lo, size_t mt_hi,
                                                               size_t nt, float *cap, float alpha,
                                                               float beta, int read_dst) {
    using A = Mopa<float, float>;
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
        quad_mopa<A, true, true>(p32, al, ah, svget2(B_, 0), svget2(B_, 1));                       \
    }
#define DIRECT_NARROW(al, row, d)                                                                  \
    {                                                                                              \
        DIRECT_B(row, d)                                                                           \
        quad_mopa<A, false, true>(p32, al, al, svget2(B_, 0), svget2(B_, 1));                      \
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

// Per-worker column items, as gemm_half.h's run_cols: each worker packs its own
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
    std::atomic<size_t> *ctr;
    float *pool = (float *)ring_pool(2 + n_chunks, ep_cmul(ep_cmul(W, panel), sizeof(float)), &ctr);
    float *a_pack = apack_scratch<float>(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
    if (!pool || !a_pack) return -1;
    std::atomic<size_t> *cur = ctr, *a_cnt = ctr + 1, *a_done = ctr + 2;
    dispatch_apply(W, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t w) {
      float *bt = pool + w * panel;
      for (size_t i; (i = cur->fetch_add(1, std::memory_order_relaxed)) < n_units;) {
          if (i < n_chunks) {
              size_t mt1 = (i + 1) * M_CHUNK < m_tiles ? (i + 1) * M_CHUNK : m_tiles;
              pack_a_range(a_pack, a, lhs_rs, lhs_cs, m, k, i * M_CHUNK, mt1);
              a_done[i].store(1, std::memory_order_release);
              a_cnt->fetch_add(1, std::memory_order_release);
              continue;
          }
          size_t t0 = (i - n_chunks) * NC, t1 = t0 + NC < n_tiles ? t0 + NC : n_tiles, mt_lo = 0;
          if (direct) {
              // M-tile 0 multiplies straight from B's rows, capturing the bands
              // when more M-tiles follow; those run the packed kernel.
              while (!a_done[0].load(std::memory_order_acquire))
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
          int all = a_cnt->load(std::memory_order_acquire) == n_chunks;
          for (size_t ci = 0; ci < (all ? 1 : n_chunks); ci++) {
              size_t mt0 = all ? 0 : ci * M_CHUNK;
              size_t mt1 = all ? m_tiles : (mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles);
              if (mt0 < mt_lo) mt0 = mt_lo;
              if (mt0 >= mt1) continue;
              if (!all)
                  while (!a_done[ci].load(std::memory_order_acquire))
                      __builtin_arm_yield();
              run_streaming(dst, dst_cs, dst_rs, a_pack, bt, m, n, mt0, mt1, t0, t1, k, alpha, beta,
                            read_dst, ep);
          }
      }
    });
    return 0;
}

// Large A: B is built per N-block into a panel ring (panel_ring.h) and shared by
// every M-chunk, as gemm_half.h's run_panels.
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
    std::atomic<size_t> *ctr;
    float *pool = (float *)ring_pool(2 * n_blocks + 2 + n_chunks,
                                     ep_cmul(ep_cmul(n_slots, slot_elems), sizeof(float)), &ctr);
    float *a_pack = apack_scratch<float>(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
    if (!pool || !a_pack) return -1;
    std::atomic<size_t> *a_done = ctr + 2 * n_blocks + 2;
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
              a_done[ci].store(1, std::memory_order_release);
          } else if (n_chunks > 1) {
              while (!a_done[ci].load(std::memory_order_acquire))
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

    if (m <= GEMV_MAXR && dst_cs == 1 && rhs_cs == 1 && rhs_rs > 0) {
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
    size_t b_budget = SME_F32_DIRECT_B_BYTES / sizeof(float);
    if (rhs_cs == 1 && ep_cmul(n, k) <= b_budget) {
        size_t sc_chunk = 2;
        size_t sc_n = (m_tiles + sc_chunk - 1) / sc_chunk;
        // Below this the dispatch costs more than the second cluster earns.
        // Absolute work is the signal, not chunk count: 128^3 and 128x512x512
        // both split two ways and want opposite answers.
        if (sc_n <= 1 || ep_flops(m, n, k) < SME_F32_SERIAL_FLOPS) {
            float *a_pack =
                apack_scratch<float>(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
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
                std::atomic<size_t> cursor = 0;
                // Blocks capture by const value, so capture the POINTER: taking
                // &cursor inside the block would give each worker its own.
                std::atomic<size_t> *cur = &cursor;
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
                                     size_t ci = cur->fetch_add(1, std::memory_order_relaxed);
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

    // Build B inside the parallel GEMM rather than packing it all first.
    if (m > GEMV_MAXR && ep_flops(m, n, k) >= (1u << 21)) {
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

// Elements in a packed-B buffer for (n, k). Saturating (see gemm_i8i32.cc).
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
    if (m <= GEMV_MAXR && dst_cs == 1) {
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
    if (n_chunks <= 1 && big && nn_chunks >= 2) {
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
        float *a_pack =
            apack_scratch<float>(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
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
    // blocking cannot help because it only ever sees one chunk.
    size_t nc_blk = float_nc_blk(k, n_tiles);
    size_t n_blocks = (n_tiles + nc_blk - 1) / nc_blk;
    if (n_blocks > 1)
        dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t mt0 = ci * M_CHUNK;
          PACKA_RANGE(a_pack, mt0, mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles);
        });
    // Flat dispatch over (N-block, M-chunk), block-major; see gemm_i16i64.cc.
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

// Row-major C = nodes(beta*A@B), B pre-packed, plus each finished row's sum and
// max. Rows are independent, so the M-chunks run in parallel, each across all
// of N.
int gemm_sme_f32_rows_packed(size_t m, size_t n, size_t k, float *dst, const float *lhs,
                             const float *b_pack, float beta, const ep_desc_f32 *ep, float *row_sum,
                             float *row_max) {
    if (m == 0 || n == 0) return 0;
    size_t m_tiles = (m + 31) / 32, per_tile = ep_cmul(k, 16);
    size_t M_CHUNK = 2, n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    RowStats<float> stats = {row_sum, row_max};
    if (n_chunks <= 1 || ep_flops(m, n, k) < (1u << 21)) {
        float *a_pack =
            apack_scratch<float>(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
        if (!a_pack) return -1;
        pack_a_range(a_pack, lhs, (long)k, 1, m, k, 0, m_tiles);
        run_rows(dst, a_pack, b_pack, m, n, 0, m_tiles, k, beta, ep, stats);
        return 0;
    }
    float *a_pack = (float *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(float)));
    if (!a_pack) return -1;
    dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
      size_t mt0 = ci * M_CHUNK, mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
      pack_a_range(a_pack, lhs, (long)k, 1, m, k, mt0, mt1);
      run_rows(dst, a_pack, b_pack, m, n, mt0, mt1, k, beta, ep, stats);
    });
    free(a_pack);
    return 0;
}

// gemm_sme_f32_rows_packed with B unpacked: B is packed once, then shared.
int gemm_sme_f32_rows(size_t m, size_t n, size_t k, float *dst, const float *lhs, const float *rhs,
                      float beta, const ep_desc_f32 *ep, float *row_sum, float *row_max) {
    if (m == 0 || n == 0) return 0;
    float *b_pack = (float *)malloc(ep_cmul(gemm_sme_f32_packed_b_elems(n, k), sizeof(float)));
    if (!b_pack) return -1;
    gemm_sme_f32_packb(b_pack, rhs, n, k, (long)n, 1);
    int rc = gemm_sme_f32_rows_packed(m, n, k, dst, lhs, b_pack, beta, ep, row_sum, row_max);
    free(b_pack);
    return rc;
}

// C = softmax_rows(A @ B), B pre-packed. One GEMM (whose store also produces the
// per-row maxima) followed by the parallel NEON softmax pass in attention.cc.
int gemm_sme_f32_softmax(size_t m, size_t n, size_t k, float *dst, const float *lhs,
                         const float *b_pack, float *row_max) {
    int rc = gemm_sme_f32_rows_packed(m, n, k, dst, lhs, b_pack, 1.0f, NULL, NULL, row_max);
    if (rc != 0 || m == 0 || n == 0) return rc;
    attn_softmax_rows_f32(dst, m, n, (long)n, row_max);
    return 0;
}

// Batched f32: `count` same-shape row-major GEMMs in one streaming session, with
// an optional epilogue applied to each (gemm_float.h's run_batched_ep).
int gemm_sme_f32_batched_ep(size_t count, size_t m, size_t n, size_t k, float *dst,
                            const float *lhs, const float *rhs, const ep_desc_f32 *ep) {
    return run_batched_ep(count, m, n, k, dst, lhs, rhs, ep);
}

int gemm_sme_f32_batched(size_t count, size_t m, size_t n, size_t k, float *dst, const float *lhs,
                         const float *rhs) {
    return run_batched_ep(count, m, n, k, dst, lhs, rhs, (const ep_desc_f32 *)NULL);
}
