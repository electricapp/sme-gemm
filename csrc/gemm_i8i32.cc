// SME i8 GEMM driver (Apple M4+, base FEAT_SME, SMOPA i8*i8 -> i32).
//
// SMOPA accumulates 4 K-values per instruction (a 32-bit accumulator holds the
// sum of four int8 products): ZA32[i,j] += sum_{s=0..3} A[i,4p+s]*B[4p+s,j].
// So the 64-lane i8 operand packs 16 rows x 4 K-slices, the panels are
// [ceil(K/4), 64] 4-way interleaved, and a 32x32 super-tile is four 16x16 i32
// ZA quadrants. Output is raw i32 or, with a dequant descriptor, f32; the
// kernels shared with i16 are in gemm_int.h.

#include "sme_gemm_api.h"

#include "epilogue.h"
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <type_traits>

// Scalar tail for one (partial) p-group: fills dst[p*64 + 4*i + s] with the
// 4-interleaved source, zero past the K and lane edges.
static inline void pack_band_scalar_p(int8_t *dst, const int8_t *src, long lane_stride,
                                      long depth_stride, size_t k, size_t valid, size_t p) {
    for (size_t i = 0; i < 16; i++)
        for (size_t s = 0; s < 4; s++) {
            size_t d = 4 * p + s;
            dst[p * 64 + 4 * i + s] =
                (i < valid && d < k) ? src[(long)i * lane_stride + (long)d * depth_stride] : 0;
        }
}

// Pack one 16-wide band into [ceil(K/4), 64]: dst[p*64 + 4*i + s] = src[i, 4p+s].
// Two NEON fast paths for the full-16-lane case (the SMOPA 4-way interleave was
// the dominant small/medium-tile cost as a scalar byte loop):
//   * depth contiguous (row-major A, depth_stride==1): the 4 s-values are one
//     32-bit word per (p,i) -- a strided 32-bit gather.
//   * lanes contiguous (row-major B, lane_stride==1): load 4 depth-vectors of
//     16 bytes and ST4 to interleave them byte-wise straight into the layout.
static void pack_band(int8_t *dst, const int8_t *src, long lane_stride, long depth_stride, size_t k,
                      size_t valid) {
    size_t kp4 = (k + 3) / 4;
    size_t pfull = (k & ~(size_t)3) / 4; // full groups of 4 depths

    if (valid == 16 && depth_stride == 1) {
        for (size_t p = 0; p < pfull; p++) {
            const int8_t *sp = src + 4 * (long)p;
            int8_t *d = dst + p * 64;
            for (size_t i = 0; i < 16; i++)
                __builtin_memcpy(d + 4 * i, sp + (long)i * lane_stride, 4);
        }
    } else if (valid == 16 && lane_stride == 1) {
        for (size_t p = 0; p < pfull; p++) {
            uint8x16x4_t v;
            v.val[0] = vld1q_u8((const uint8_t *)(src + (long)(4 * p + 0) * depth_stride));
            v.val[1] = vld1q_u8((const uint8_t *)(src + (long)(4 * p + 1) * depth_stride));
            v.val[2] = vld1q_u8((const uint8_t *)(src + (long)(4 * p + 2) * depth_stride));
            v.val[3] = vld1q_u8((const uint8_t *)(src + (long)(4 * p + 3) * depth_stride));
            vst4q_u8((uint8_t *)(dst + p * 64), v);
        }
    } else {
        pfull = 0; // no fast path -- fall through to scalar for all p
    }
    for (size_t p = pfull; p < kp4; p++)
        pack_band_scalar_p(dst, src, lane_stride, depth_stride, k, valid, p);
}

#include "gemm_int.h"

using I8 = IntMopa<int8_t>;

// Narrow-N (n <= 16: a single ZA32 N-band) row-major fast path. The 32x32
// super-tile's right N-band (tiles 1,3, cols 16-31) is all pad when n <= 16, so
// half the MOPAs are wasted (n=16 measured ~half of n=32 per flop). Pair two
// M-tiles against the single N-band instead: za0/za2 = A[mt] (rows 0-15 / 16-31),
// za1/za3 = A[mt+1] -- all four tiles cols 0..n, every MOPA productive. Row-major
// only; covers raw i32 AND fused dequant (the dequantized f32 output). The
// single N-tile sits at column 0, so n0 == 0.
template <class Out>
__arm_locally_streaming __arm_new("za") static void run_narrow_rowmajor(
    Out out, long dst_rs, const int8_t *a_pack, const int8_t *b_pack, size_t m, size_t n,
    size_t mt_lo, size_t mt_hi, size_t kp4) {
    svbool_t pb = svptrue_b8();
    svcount_t pn8 = svptrue_c8();
    size_t per_tile = ep_cmul(kp4, 64);
    svbool_t pst = svwhilelt_b32((uint32_t)0, (uint32_t)n); // n <= 16: the single N-band
    const int8_t *b_lo = b_pack;                            // N-tile 0, band 0 (cols 0..15)

    for (size_t mt = mt_lo; mt < mt_hi; mt += 2) {
        const int8_t *a0_lo = a_pack + (size_t)(2 * mt) * per_tile;
        const int8_t *a0_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
        int have1 = (mt + 1 < mt_hi);
        const int8_t *a1_lo = have1 ? a_pack + (size_t)(2 * (mt + 1)) * per_tile : a0_lo;
        const int8_t *a1_hi = have1 ? a_pack + (size_t)(2 * (mt + 1) + 1) * per_tile : a0_hi;
        size_t m0_0 = mt * 32;
        size_t mr0 = m - m0_0 < 32 ? m - m0_0 : 32;
        size_t mr0_lo = mr0 < 16 ? mr0 : 16, mr0_hi = mr0 > 16 ? mr0 - 16 : 0;
        size_t m0_1 = (mt + 1) * 32;
        size_t mr1 = have1 ? (m - m0_1 < 32 ? m - m0_1 : 32) : 0;
        size_t mr1_lo = mr1 < 16 ? mr1 : 16, mr1_hi = mr1 > 16 ? mr1 - 16 : 0;

        // za0 = A[mt] lo @ B, za2 = A[mt] hi @ B, za1 = A[mt+1] lo @ B,
        // za3 = A[mt+1] hi @ B -- four distinct tiles, single shared N-band.
        svzero_za();
        size_t p = 0;
        for (; p + 2 <= kp4; p += 2) {
            svint8x2_t A0L = svld1_s8_x2(pn8, a0_lo + p * 64);
            svint8x2_t A0H = svld1_s8_x2(pn8, a0_hi + p * 64);
            svint8x2_t A1L = svld1_s8_x2(pn8, a1_lo + p * 64);
            svint8x2_t A1H = svld1_s8_x2(pn8, a1_hi + p * 64);
            svint8x2_t B = svld1_s8_x2(pn8, b_lo + p * 64);
            svint8_t b0 = svget2_s8(B, 0), b1 = svget2_s8(B, 1);
            I8::mopa<0>(pb, svget2_s8(A0L, 0), b0);
            I8::mopa<2>(pb, svget2_s8(A0H, 0), b0);
            I8::mopa<1>(pb, svget2_s8(A1L, 0), b0);
            I8::mopa<3>(pb, svget2_s8(A1H, 0), b0);
            I8::mopa<0>(pb, svget2_s8(A0L, 1), b1);
            I8::mopa<2>(pb, svget2_s8(A0H, 1), b1);
            I8::mopa<1>(pb, svget2_s8(A1L, 1), b1);
            I8::mopa<3>(pb, svget2_s8(A1H, 1), b1);
        }
        for (; p < kp4; p++) {
            svint8_t b0 = svld1_s8(pb, b_lo + p * 64);
            I8::mopa<0>(pb, svld1_s8(pb, a0_lo + p * 64), b0);
            I8::mopa<2>(pb, svld1_s8(pb, a0_hi + p * 64), b0);
            I8::mopa<1>(pb, svld1_s8(pb, a1_lo + p * 64), b0);
            I8::mopa<3>(pb, svld1_s8(pb, a1_hi + p * 64), b0);
        }

        if constexpr (std::is_same_v<Out, DequantOut>) {
            // Per row: read the i32 ZA slice, convert, scale, op-graph, store.
            const ep_dq_f32 *dq = out.dq;
            svfloat32_t sc = dq->scale_n ? svld1_f32(pst, dq->scale_n) : svdup_n_f32(dq->scale);
            dq_rows<I8, 0, 0, false>(out.c, 1, dst_rs, m0_0, 0, 0, mr0_lo, n, pst, pst, sc, sc,
                                     dq->nodes, dq->n_nodes);
            dq_rows<I8, 2, 2, false>(out.c, 1, dst_rs, m0_0 + 16, 0, 0, mr0_hi, n, pst, pst, sc, sc,
                                     dq->nodes, dq->n_nodes);
            dq_rows<I8, 1, 1, false>(out.c, 1, dst_rs, m0_1, 0, 0, mr1_lo, n, pst, pst, sc, sc,
                                     dq->nodes, dq->n_nodes);
            dq_rows<I8, 3, 3, false>(out.c, 1, dst_rs, m0_1 + 16, 0, 0, mr1_hi, n, pst, pst, sc, sc,
                                     dq->nodes, dq->n_nodes);
        } else {
            int32_t *dst = out.c;
            for (size_t r = 0; r < mr0_lo; r++)
                svst1_hor_za32(0, (uint32_t)r, pst, dst + (long)(m0_0 + r) * dst_rs);
            for (size_t r = 0; r < mr0_hi; r++)
                svst1_hor_za32(2, (uint32_t)r, pst, dst + (long)(m0_0 + 16 + r) * dst_rs);
            for (size_t r = 0; r < mr1_lo; r++)
                svst1_hor_za32(1, (uint32_t)r, pst, dst + (long)(m0_1 + r) * dst_rs);
            for (size_t r = 0; r < mr1_hi; r++)
                svst1_hor_za32(3, (uint32_t)r, pst, dst + (long)(m0_1 + 16 + r) * dst_rs);
        }
    }
}

// m == 1: SME2 multi-vector SDOT into ZA vector groups, 16-column bands.
// `abc` is [kp4][64]: A[4p..4p+3] repeated across the vector.
#define GEMV_T 4
#define GEMV_MAXR 4 // rows; past this the MOPA path does fewer ZA ops
template <class Out>
__arm_locally_streaming __arm_new("za") static void run_gemv(Out out, long dst_rs,
                                                             const int8_t *abc, size_t rows,
                                                             const int8_t *b_pack, size_t n,
                                                             size_t kp4, size_t bd_lo,
                                                             size_t bd_hi) {
    svbool_t p32 = svptrue_b32();
    svcount_t pn = svptrue_c8();
    size_t per_band = ep_cmul(kp4, 64); // also the stride between abc rows
    size_t pfull = kp4 & ~(size_t)3;
    // K tail: predicated loads zero the dead groups.
    svcount_t pt = svwhilelt_c8((uint64_t)0, (uint64_t)((kp4 - pfull) * 64), 4);
    for (size_t bd = bd_lo; bd < bd_hi; bd += GEMV_T) {
        size_t T = bd_hi - bd < GEMV_T ? bd_hi - bd : GEMV_T;
        const int8_t *b = b_pack + bd * per_band;
        svzero_za();
        if (T == GEMV_T) {
            for (size_t p = 0; p < pfull; p += 4) {
                svint8x4_t B0 = svld1_s8_x4(pn, b + p * 64);
                svint8x4_t B1 = svld1_s8_x4(pn, b + per_band + p * 64);
                svint8x4_t B2 = svld1_s8_x4(pn, b + 2 * per_band + p * 64);
                svint8x4_t B3 = svld1_s8_x4(pn, b + 3 * per_band + p * 64);
                for (size_t r = 0; r < rows; r++) {
                    svint8x4_t A = svld1_s8_x4(pn, abc + r * per_band + p * 64);
                    uint32_t w = (uint32_t)(4 * r);
                    svdot_za32_s8_vg1x4(w, B0, A);
                    svdot_za32_s8_vg1x4(w + 1, B1, A);
                    svdot_za32_s8_vg1x4(w + 2, B2, A);
                    svdot_za32_s8_vg1x4(w + 3, B3, A);
                }
            }
        } else {
            for (size_t t = 0; t < T; t++)
                for (size_t p = 0; p < pfull; p += 4) {
                    svint8x4_t B = svld1_s8_x4(pn, b + t * per_band + p * 64);
                    for (size_t r = 0; r < rows; r++)
                        svdot_za32_s8_vg1x4((uint32_t)(4 * r + t), B,
                                            svld1_s8_x4(pn, abc + r * per_band + p * 64));
                }
        }
        if (pfull < kp4)
            for (size_t t = 0; t < T; t++) {
                svint8x4_t B = svld1_s8_x4(pt, b + t * per_band + pfull * 64);
                for (size_t r = 0; r < rows; r++)
                    svdot_za32_s8_vg1x4((uint32_t)(4 * r + t), B,
                                        svld1_s8_x4(pt, abc + r * per_band + pfull * 64));
            }
        for (size_t r = 0; r < rows; r++)
            for (size_t t = 0; t < T; t++) {
                size_t c0 = (bd + t) * 16;
                size_t nc = n - c0 < 16 ? n - c0 : 16;
                svbool_t pst = svwhilelt_b32((uint64_t)0, (uint64_t)nc);
                svint32x4_t q = svread_za32_s32_vg1x4((uint32_t)(4 * r + t));
                svint32_t acc = svadd_s32_x(p32, svadd_s32_x(p32, svget4(q, 0), svget4(q, 1)),
                                            svadd_s32_x(p32, svget4(q, 2), svget4(q, 3)));
                if constexpr (std::is_same_v<Out, DequantOut>) {
                    const ep_dq_f32 *dq = out.dq;
                    float *fdst = out.c + (long)r * dst_rs;
                    svfloat32_t sc =
                        dq->scale_n ? svld1_f32(pst, dq->scale_n + c0) : svdup_n_f32(dq->scale);
                    svfloat32_t v = svmul_f32_x(p32, svcvt_f32_s32_x(p32, acc), sc);
                    v = ep_apply_nodes_f32(p32, pst, v, dq->nodes, dq->n_nodes, 1, r, c0, 0);
                    svst1_f32(pst, fdst + c0, v);
                } else {
                    svst1_s32(pst, out.c + (long)r * dst_rs + c0, acc);
                }
            }
    }
}

// abc[p*64 + 4j + s] = A[0, 4p+s], zero past k.
static void bcast_row(int8_t *abc, const int8_t *a, long lhs_cs, size_t k) {
    size_t kp4 = (k + 3) / 4;
    for (size_t p = 0; p < kp4; p++) {
        uint8_t g[4];
        for (size_t s = 0; s < 4; s++) {
            size_t d = 4 * p + s;
            g[s] = d < k ? (uint8_t)a[(long)d * lhs_cs] : 0;
        }
        uint32_t w;
        __builtin_memcpy(&w, g, 4);
        uint8x16_t v = vreinterpretq_u8_u32(vdupq_n_u32(w));
        uint8x16x4_t q = {{v, v, v, v}};
        vst1q_u8_x4((uint8_t *)abc + p * 64, q);
    }
}

// Number of int8 elements in a packed-B buffer for (n, k).
// Saturating: SIZE_MAX on overflow. This sizes the caller's buffer while packb
// writes the full panel regardless, so a WRAPPED size is a heap overflow, not a
// short read. SIZE_MAX makes the allocation fail instead, which callers handle.
size_t gemm_sme_i8i32_packed_b_elems(size_t n, size_t k) {
    size_t kp4 = (k + 3) / 4;
    size_t n_tiles = (n + 31) / 32;
    return ep_cmul(ep_cmul(2 * n_tiles, kp4), 64);
}

// Pack B once into a caller-allocated buffer of gemm_sme_i8i32_packed_b_elems
// int8. Reuse across many GEMMs (quantized weights packed once).
void gemm_sme_i8i32_packb(int8_t *b_pack, const int8_t *rhs, size_t n, size_t k, long rhs_rs,
                          long rhs_cs) {
    size_t kp4 = (k + 3) / 4;
    size_t n_tiles = (n + 31) / 32;
    size_t per_tile = ep_cmul(kp4, 64);
    for (size_t st = 0; st < 2 * n_tiles; st++) {
        size_t c0 = st * 16;
        size_t vc = (c0 < n) ? ((n - c0 < 16) ? (n - c0) : 16) : 0;
        // vc==0 bands are zero-filled without reading src; clamp the base so no
        // out-of-bounds pointer is even formed (UB without a deref).
        pack_band(b_pack + st * per_tile, vc ? rhs + (long)c0 * rhs_cs : rhs, rhs_cs, rhs_rs, k, vc);
    }
}

// Per-thread reusable A-pack scratch. Small/medium GEMMs call repeatedly; a
// malloc/free per call is a measurable fraction of a tiny problem's time. The
// buffer grows monotonically and lives for the thread's lifetime (each GCD
// worker keeps its own, so the parallel path is unaffected).
static thread_local int8_t *g_apack = NULL;
static thread_local size_t g_apack_cap = 0;
static int8_t *apack_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1; // k==0: return a real (tiny) buffer, not a
                               // history-dependent NULL on fresh threads
    if (g_apack_cap < bytes) {
        free(g_apack);
        g_apack = (int8_t *)malloc(bytes);
        g_apack_cap = g_apack ? bytes : 0;
    }
    return g_apack;
}

// Same for the NON-packed entry's B panel, which gemm_sme_i8i32_run builds
// itself, which is what makes the per-call malloc/free avoidable. Only the
// calling thread resizes or frees it; the dispatch_apply workers just READ it
// and finish before the call returns, so handing them a pointer into the
// owner's TLS is safe.
static thread_local int8_t *g_bpack = NULL;
static thread_local size_t g_bpack_cap = 0;
static int8_t *bpack_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1; // k==0 / n==0: a real (tiny) buffer, never a
                               // history-dependent NULL on fresh threads
    if (g_bpack_cap < bytes) {
        free(g_bpack);
        g_bpack = (int8_t *)malloc(bytes);
        g_bpack_cap = g_bpack ? bytes : 0;
    }
    return g_bpack;
}

// Packed-B i8 GEMM: caller supplies B pre-packed via gemm_sme_i8i32_packb.
// Only A is packed here -- the quantized-inference pattern.
template <class Out>
static int run_packed(size_t m, size_t n, size_t k, Out out, long dst_cs, long dst_rs,
                      const int8_t *lhs, long lhs_cs, long lhs_rs, const int8_t *b_pack) {
    if (m == 0 || n == 0) return 0;
    size_t kp4 = (k + 3) / 4;
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    size_t per_tile = ep_cmul(kp4, 64);

#define PACKA_RANGE(AP, MT0, MT1)                                                                  \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                            \
        size_t r0 = st * 16;                                                                       \
        size_t vr = (r0 < m) ? ((m - r0 < 16) ? (m - r0) : 16) : 0;                                \
        pack_band((AP) + st * per_tile, vr ? lhs + (long)r0 * lhs_rs : lhs, lhs_rs, lhs_cs, k, vr); \
    }

    // n<=16 (single ZA32 N-band) row-major: pair two M-tiles against the one
    // N-band (run_narrow_rowmajor) instead of wasting tiles 1,3 on pad. Same
    // M-chunk parallelism; just a different inner loop.
    int narrow = (n <= 16 && dst_cs == 1);
    // Two tiles per chunk -- fine enough for dispatch_apply to work-steal across
    // the asymmetric clusters (2 beats 4 by ~5% geomean), and EVEN so
    // run_narrow_rowmajor's M-tile pairing never idles za1/za3.
    size_t M_CHUNK = 2;
    size_t n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    int big = ep_flops(m, n, k) >= (1u << 21);
    if (m <= GEMV_MAXR && dst_cs == 1) {
        int8_t *abc = apack_scratch(ep_cmul(ep_cmul(m, kp4), 64));
        if (!abc) return -1;
        for (size_t r = 0; r < m; r++)
            bcast_row(abc + r * kp4 * 64, lhs + (long)r * lhs_rs, lhs_cs, k);
        size_t bands = (n + 15) / 16;
        size_t G_CHUNK = 8; // bands per chunk (four 32-wide N-tiles)
        size_t g_chunks = (bands + G_CHUNK - 1) / G_CHUNK;
        if (!big || g_chunks < 3) {
            run_gemv(out, dst_rs, abc, m, b_pack, n, kp4, 0, bands);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t b0 = ci * G_CHUNK;
          size_t b1 = b0 + G_CHUNK < bands ? b0 + G_CHUNK : bands;
          run_gemv(out, dst_rs, abc, m, b_pack, n, kp4, b0, b1);
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
        int8_t *a_pack = (int8_t *)malloc(ep_cmul(2 * m_tiles, per_tile));
        if (a_pack) {
            PACKA_RANGE(a_pack, 0, m_tiles);
            dispatch_apply(nn_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
              size_t nt0 = ci * N_CHUNK;
              size_t nt1 = nt0 + N_CHUNK < n_tiles ? nt0 + N_CHUNK : n_tiles;
              run_streaming(out, dst_cs, dst_rs, a_pack, b_pack + 2 * nt0 * per_tile, m, n, 0,
                            m_tiles, nt0, nt1, kp4);
            });
            free(a_pack);
            return 0;
        }
        // malloc failed: fall through to the serial path.
    }
    if (n_chunks <= 1 || !big) {
        // Serial: reuse the per-thread scratch (no per-call malloc/free).
        int8_t *a_pack = apack_scratch(ep_cmul(2 * m_tiles, per_tile));
        if (!a_pack) return -1;
        PACKA_RANGE(a_pack, 0, m_tiles);
        if (narrow)
            run_narrow_rowmajor(out, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, kp4);
        else
            run_streaming(out, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, 0, n_tiles, kp4);
        return 0;
    }
    int8_t *a_pack = (int8_t *)malloc(ep_cmul(2 * m_tiles, per_tile));
    if (!a_pack) return -1;
    if (narrow) {
        dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t mt0 = ci * M_CHUNK;
          size_t mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
          PACKA_RANGE(a_pack, mt0, mt1);
          run_narrow_rowmajor(out, dst_rs, a_pack, b_pack, m, n, mt0, mt1, kp4);
        });
    } else {
        size_t nc_blk = int_nc_blk(kp4, n_tiles);
        size_t n_blocks = (n_tiles + nc_blk - 1) / nc_blk;
        if (n_blocks > 1)
            dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
              size_t mt0 = ci * M_CHUNK;
              PACKA_RANGE(a_pack, mt0, mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles);
            });
        // Flat dispatch over (N-block, M-chunk), block-major; see gemm_i16i64.cc.
        dispatch_apply(n_blocks * n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0),
                       ^(size_t it) {
                         size_t ci = it % n_chunks, jc = it / n_chunks * nc_blk;
                         size_t mt0 = ci * M_CHUNK;
                         size_t mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
                         size_t jc_end = jc + nc_blk < n_tiles ? jc + nc_blk : n_tiles;
                         if (n_blocks == 1) PACKA_RANGE(a_pack, mt0, mt1);
                         run_streaming(out, dst_cs, dst_rs, a_pack, b_pack + 2 * jc * per_tile, m,
                                       n, mt0, mt1, jc, jc_end, kp4);
                       });
    }
#undef PACKA_RANGE

    free(a_pack);
    return 0;
}

// Raw i32 packed GEMM.
int gemm_sme_i8i32_run_packed(size_t m, size_t n, size_t k, int32_t *dst, long dst_cs, long dst_rs,
                              const int8_t *lhs, long lhs_cs, long lhs_rs, const int8_t *b_pack) {
    return run_packed(m, n, k, RawOut<int32_t>{dst}, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, b_pack);
}

// Fused-dequant packed GEMM: i8*i8 -> i32 accumulate, then the f32 op-graph
// D(f32) = nodes(scale*acc) written to `dst` (f32). `b_pack` is the same packed-B
// as the raw path; only the output stage differs.
int gemm_sme_i8i32_run_packed_dequant(size_t m, size_t n, size_t k, float *dst, long dst_cs,
                                      long dst_rs, const int8_t *lhs, long lhs_cs, long lhs_rs,
                                      const int8_t *b_pack, float scale, const float *scale_n,
                                      uint32_t n_nodes, const EpNode *nodes) {
    ep_dq_f32 dq = {scale, scale_n, n_nodes, nodes};
    return run_packed(m, n, k, DequantOut{dst, &dq}, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, b_pack);
}

int gemm_sme_i8i32_run(size_t m, size_t n, size_t k, int32_t *dst, long dst_cs, long dst_rs,
                       const int8_t *lhs, long lhs_cs, long lhs_rs, const int8_t *rhs, long rhs_cs,
                       long rhs_rs) {
    if (m == 0 || n == 0) return 0;
    // Per-thread reusable B panel: no malloc/free per call (see bpack_scratch).
    int8_t *b_pack = bpack_scratch(gemm_sme_i8i32_packed_b_elems(n, k));
    if (!b_pack) return -1;
    gemm_sme_i8i32_packb(b_pack, rhs, n, k, rhs_rs, rhs_cs);
    return gemm_sme_i8i32_run_packed(m, n, k, dst, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, b_pack);
}

// Batched i8 GEMM: `count` independent row-major C_i(i32) = A_i @ B_i (same
// shape), one streaming-mode session (gemm_int.h's run_batched_impl).
int gemm_sme_i8i32_batched(size_t count, size_t m, size_t n, size_t k, int32_t *dst,
                           const int8_t *lhs, const int8_t *rhs) {
    return run_batched_impl(count, m, n, k, RawOut<int32_t>{dst}, lhs, rhs);
}

// Batched i8 GEMM with fused dequant + op-graph applied to EACH item: the i32
// accumulator is scaled (per-tensor `scale`, or per-N `scale_n`) into f32 and the
// op-graph (`nodes`) is applied, all in-register. f32 output to `dst`. Same
// scale/nodes/operands for every item. NULL nodes / n_nodes==0 = pure dequant.
int gemm_sme_i8i32_batched_dequant(size_t count, size_t m, size_t n, size_t k, float *dst,
                                   const int8_t *lhs, const int8_t *rhs, float scale,
                                   const float *scale_n, uint32_t n_nodes, const EpNode *nodes) {
    ep_dq_f32 dq = {scale, scale_n, n_nodes, nodes};
    return run_batched_impl(count, m, n, k, DequantOut{dst, &dq}, lhs, rhs);
}
