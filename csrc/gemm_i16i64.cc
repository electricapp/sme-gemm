// SME i16 GEMM driver (Apple M5+, FEAT_SME_I16I64, SMOPA i16*i16 -> i64).
// SMOPA accumulates 4 K-values per instruction into a 64-bit accumulator:
// ZA64[i,j] += sum_{s=0..3} A[i,4p+s]*B[4p+s,j]. The 32-lane i16 operand packs
// 8 rows x 4 K-slices, panels are [ceil(K/4), 32] 4-way interleaved, and a
// 16x16 super-tile is four 8x8 i64 ZA quadrants. Output is raw i64 or, with a
// dequant descriptor, f32; the kernels shared with i8 are in gemm_int.h.

#include "sme_gemm_api.h"

#include "epilogue.h"
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <type_traits>

#include "gemm_int.h"

using I16 = IntMopa<int16_t>;

// m == 1: SME2 multi-vector SDOT into ZA64 vector groups instead of SMOPA;
// see run_gemv in gemm_half.h. 8-column bands of 4-deep groups, two per
// 16-wide tile; `abc` is [kp4][32]: A[4p..4p+3] repeated across the vector.
// Past one row the SMOPA is cheaper.
template <class Out>
__arm_locally_streaming __arm_new("za") static void run_gemv(Out out, const int16_t *abc,
                                                             const int16_t *b_pack, size_t n,
                                                             size_t kp4, size_t nt_lo,
                                                             size_t nt_hi) {
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
            if constexpr (std::is_same_v<Out, DequantOut>) {
                const ep_dq_f32 *dq = out.dq;
                svbool_t pst = svwhilelt_b32((uint32_t)0, (uint32_t)nc);
                svfloat32_t sc =
                    dq->scale_n ? svld1_f32(pst, dq->scale_n + n0) : svdup_n_f32(dq->scale);
                svfloat32_t v = svmul_f32_x(
                    p32, svuzp1_f32(svcvt_f32_s64_x(p64, lo), svcvt_f32_s64_x(p64, hi)), sc);
                v = ep_apply_nodes_f32(p32, pst, v, dq->nodes, dq->n_nodes, 1, 0, n0, 0);
                svst1_f32(pst, out.c + n0, v);
            } else {
                int64_t *rp = out.c + n0;
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

static thread_local int16_t *g_apack = NULL;
static thread_local size_t g_apack_cap = 0;
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

// Small-GEMM fast path: A pre-packed, B packed per N-tile into a small scratch
// straight from row-major rhs (no full B-pack malloc/pass). N-tile OUTER so each
// B tile is packed once and reused across this chunk's M-tiles; with mt inner it
// was repacked per (mt, nt). Each (mt, nt) is independent, so the order is free.
template <class Out>
__arm_locally_streaming __arm_new("za") static void run_small(
    Out out, long dst_cs, long dst_rs, const int16_t *a_pack, const int16_t *rhs, long rhs_rs,
    int16_t *b_scratch, size_t m, size_t n, size_t k, size_t mt_lo, size_t mt_hi, size_t kp4) {
    size_t per_tile = ep_cmul(kp4, 32);
    size_t n_tiles = (n + 15) / 16;
    for (size_t nt = 0; nt < n_tiles; nt++) {
        size_t n0 = nt * 16;
        size_t nc = n - n0 < 16 ? n - n0 : 16;
        size_t vc_lo = nc < 8 ? nc : 8;
        size_t vc_hi = nc > 8 ? nc - 8 : 0;
        int16_t *b_lo = b_scratch;
        int16_t *b_hi = b_scratch + per_tile;
        // Clamp the hi-band base when vc_hi==0 (no lane read): forming
        // rhs + n0 + 8 past one-past-end would be UB.
        pack_band<I16>(b_lo, rhs + (long)n0, 1, rhs_rs, k, vc_lo);
        pack_band<I16>(b_hi, vc_hi ? rhs + (long)n0 + 8 : rhs, 1, rhs_rs, k, vc_hi);
        for (size_t mt = mt_lo; mt < mt_hi; mt++) {
            const int16_t *a_lo = a_pack + 2 * mt * per_tile, *a_hi = a_lo + per_tile;
            size_t m0 = mt * 16, mr = m - m0 < 16 ? m - m0 : 16;
            svzero_za();
            quad_accumulate<I16, KSteps<2>>(kp4, a_lo, a_hi, b_lo, b_hi, mr > 8, nc > 8);
            int_store<I16>(out, dst_cs, dst_rs, m0, n0, mr, nc);
        }
    }
}

// Elements in a packed-B buffer for (n, k). Saturating (see gemm_i8i32.cc).
size_t gemm_sme_i16i64_packed_b_elems(size_t n, size_t k) {
    return ep_cmul(2 * ((n + 15) / 16), ep_cmul((k + 3) / 4, 32));
}

// Pack B once into a caller-allocated buffer of gemm_sme_i16i64_packed_b_elems
// int16: [2*n_tiles][ceil(k/4), 32] 4-way-interleaved 8-wide bands.
void gemm_sme_i16i64_packb(int16_t *b_pack, const int16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs) {
    pack_bands<I16>(b_pack, rhs, rhs_cs, rhs_rs, n, k, 0, 2 * ((n + 15) / 16));
}

// Packed-B i16 GEMM. No run_small arm -- that packs B per N-tile from a
// row-major rhs.
template <class Out>
static int run_packed(size_t m, size_t n, size_t k, Out out, long dst_cs, long dst_rs,
                      const int16_t *lhs, long lhs_cs, long lhs_rs, const int16_t *b_pack) {
    if (m == 0 || n == 0) return 0;
    size_t kp4 = (k + 3) / 4;
    size_t m_tiles = (m + 15) / 16;
    size_t n_tiles = (n + 15) / 16;
    size_t per_tile = ep_cmul(kp4, 32);

#define PACKA_RANGE(AP, MT0, MT1)                                                                  \
    pack_bands<I16>((AP) + 2 * (MT0) * per_tile, lhs, lhs_rs, lhs_cs, m, k, 2 * (MT0), 2 * (MT1))

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
            run_gemv(out, abc, b_pack, n, kp4, 0, n_tiles);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t nt0 = ci * G_CHUNK;
          size_t nt1 = nt0 + G_CHUNK < n_tiles ? nt0 + G_CHUNK : n_tiles;
          run_gemv(out, abc, b_pack, n, kp4, nt0, nt1);
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
              run_streaming(out, dst_cs, dst_rs, a_pack, b_pack + 2 * nt0 * per_tile, m, n, 0,
                            m_tiles, nt0, nt1, kp4);
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
        run_streaming(out, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, 0, n_tiles, kp4);
        return 0;
    }
    int16_t *a_pack = (int16_t *)xmalloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(int16_t)));
    if (!a_pack) return -1;
    size_t nc_blk = int_nc_blk(kp4, n_tiles);
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
      run_streaming(out, dst_cs, dst_rs, a_pack, b_pack + 2 * jc * per_tile, m, n, mt0, mt1, jc,
                    jc_end, kp4);
    });
#undef PACKA_RANGE
    free(a_pack);
    return 0;
}

// Unpacked-B i16 GEMM: the small path packs B per N-tile, the rest packs it once
// and runs the packed driver.
template <class Out>
static int run_impl(size_t m, size_t n, size_t k, Out out, long dst_cs, long dst_rs,
                    const int16_t *lhs, long lhs_cs, long lhs_rs, const int16_t *rhs, long rhs_cs,
                    long rhs_rs) {
    if (m == 0 || n == 0) return 0;
    size_t kp4 = (k + 3) / 4;
    size_t m_tiles = (m + 15) / 16;
    size_t per_tile = ep_cmul(kp4, 32);

    // Small + row-major B: pack B per-tile into a small scratch (no full B-pack
    // malloc/pass), light dispatch. See gemm_f64.cc.
#define PACKA_SMALL(AP, MT0, MT1)                                                                  \
    pack_bands<I16>((AP) + 2 * (MT0) * per_tile, lhs, lhs_rs, lhs_cs, m, k, 2 * (MT0), 2 * (MT1))
    if (rhs_cs == 1 && ep_flops(m, n, k) < (1u << 25)) {
        size_t sc_chunk = 2;
        size_t sc_n = (m_tiles + sc_chunk - 1) / sc_chunk;
        if (sc_n <= 1) {
            int16_t *a_pack =
                apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(int16_t)));
            int16_t *b_scratch =
                a_pack ? (int16_t *)xmalloc(ep_cmul(2 * per_tile, sizeof(int16_t))) : NULL;
            if (a_pack && b_scratch) {
                PACKA_SMALL(a_pack, 0, m_tiles);
                run_small(out, dst_cs, dst_rs, a_pack, rhs, rhs_rs, b_scratch, m, n, k, 0, m_tiles,
                          kp4);
                free(b_scratch);
                return 0;
            }
            free(b_scratch);
        } else {
            int16_t *a_pack =
                (int16_t *)xmalloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(int16_t)));
            // One contiguous per-chunk B scratch, allocated UP FRONT: workers
            // must not allocate, because a worker-side failure after other
            // chunks have already stored their rows would break the documented
            // "rc != 0 => C untouched" contract, and a shared __block fail flag
            // is a C11 data race. On failure fall through to the main path.
            int16_t *b_scratch_all =
                a_pack ? (int16_t *)xmalloc(ep_cmul(ep_cmul(sc_n * 2, per_tile), sizeof(int16_t)))
                       : NULL;
            if (a_pack && b_scratch_all) {
                dispatch_apply(sc_n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                               ^(size_t ci) {
                                 size_t mt0 = ci * sc_chunk;
                                 size_t mt1 = mt0 + sc_chunk < m_tiles ? mt0 + sc_chunk : m_tiles;
                                 int16_t *b_scratch = b_scratch_all + ci * 2 * per_tile;
                                 PACKA_SMALL(a_pack, mt0, mt1);
                                 run_small(out, dst_cs, dst_rs, a_pack, rhs, rhs_rs, b_scratch, m,
                                           n, k, mt0, mt1, kp4);
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
    int rc = run_packed(m, n, k, out, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, b_pack);
    free(b_pack);
    return rc;
}

int gemm_sme_i16i64_run(size_t m, size_t n, size_t k, int64_t *dst, long dst_cs, long dst_rs,
                        const int16_t *lhs, long lhs_cs, long lhs_rs, const int16_t *rhs,
                        long rhs_cs, long rhs_rs) {
    return run_impl(m, n, k, RawOut<int64_t>{dst}, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs,
                    rhs_rs);
}

// Fused-dequant i16 GEMM: i16*i16 -> i64 accumulate, then the f32 op-graph
// D(f32) = nodes(scale*acc) written to `dst` (f32).
int gemm_sme_i16i64_run_dequant(size_t m, size_t n, size_t k, float *dst, long dst_cs, long dst_rs,
                                const int16_t *lhs, long lhs_cs, long lhs_rs, const int16_t *rhs,
                                long rhs_cs, long rhs_rs, float scale, const float *scale_n,
                                uint32_t n_nodes, const EpNode *nodes) {
    ep_dq_f32 dq = {scale, scale_n, n_nodes, nodes};
    return run_impl(m, n, k, DequantOut{dst, &dq}, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs,
                    rhs_rs);
}

int gemm_sme_i16i64_run_packed(size_t m, size_t n, size_t k, int64_t *dst, long dst_cs, long dst_rs,
                               const int16_t *lhs, long lhs_cs, long lhs_rs,
                               const int16_t *b_pack) {
    return run_packed(m, n, k, RawOut<int64_t>{dst}, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, b_pack);
}

int gemm_sme_i16i64_run_packed_dequant(size_t m, size_t n, size_t k, float *dst, long dst_cs,
                                       long dst_rs, const int16_t *lhs, long lhs_cs, long lhs_rs,
                                       const int16_t *b_pack, float scale, const float *scale_n,
                                       uint32_t n_nodes, const EpNode *nodes) {
    ep_dq_f32 dq = {scale, scale_n, n_nodes, nodes};
    return run_packed(m, n, k, DequantOut{dst, &dq}, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, b_pack);
}

// Batched i16 GEMM: `count` independent row-major C_i(i64) = A_i @ B_i (same
// shape), one streaming-mode session (gemm_int.h's run_batched_impl). M5-only.
int gemm_sme_i16i64_batched(size_t count, size_t m, size_t n, size_t k, int64_t *dst,
                            const int16_t *lhs, const int16_t *rhs) {
    return run_batched_impl(count, m, n, k, RawOut<int64_t>{dst}, lhs, rhs);
}

// Batched i16 GEMM with fused dequant + op-graph applied to EACH item: the i64
// accumulator is scaled (per-tensor `scale`, or per-N `scale_n`) into f32 and the
// op-graph (`nodes`) is applied. f32 output to `fdst`. Same scale/nodes/operands
// for every item. NULL nodes / n_nodes==0 = pure dequant. M5-only.
int gemm_sme_i16i64_batched_dequant(size_t count, size_t m, size_t n, size_t k, float *fdst,
                                    const int16_t *lhs, const int16_t *rhs, float scale,
                                    const float *scale_n, uint32_t n_nodes, const EpNode *nodes) {
    ep_dq_f32 dq = {scale, scale_n, n_nodes, nodes};
    return run_batched_impl(count, m, n, k, DequantOut{fdst, &dq}, lhs, rhs);
}
