// SME f64 GEMM driver (Apple M5+, FEAT_SME_F64F64, FMOPA double-precision).
// SVL=512 gives 8 f64 lanes, so a 16x16 super-tile is four 8x8 ZA64 quadrants;
// one K-value per MOPA. Packing is plain [k, 8] per 8-wide band.
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

// Pack one 8-wide band into [k, 8]: dst[d*8 + i] = src[i, d]. NEON fast paths
// for the full-8-lane case: lanes contiguous (B) -> copy per depth; depth
// contiguous (row-major A) -> 2x2 f64 transpose.
static void pack_band(double *dst, const double *src, long lane_stride, long depth_stride, size_t k,
                      size_t valid) {
    if (valid == 8 && lane_stride == 1) {
        for (size_t d = 0; d < k; d++)
            memcpy(dst + d * 8, src + (long)d * depth_stride, 8 * sizeof(double));
        return;
    }
    if (valid == 8 && depth_stride == 1) {
        // Depth-outer so the four lane groups of one depth fill its 64-byte line
        // consecutively; lane-group-outer walks the whole panel four times. Same
        // reasoning as the f32 twin.
        size_t kf = k & ~(size_t)1;
        for (size_t d = 0; d < kf; d += 2) {
            for (size_t rg = 0; rg < 8; rg += 2) {
                float64x2_t r0 = vld1q_f64(src + (long)(rg + 0) * lane_stride + d);
                float64x2_t r1 = vld1q_f64(src + (long)(rg + 1) * lane_stride + d);
                vst1q_f64(dst + (d + 0) * 8 + rg, vtrn1q_f64(r0, r1));
                vst1q_f64(dst + (d + 1) * 8 + rg, vtrn2q_f64(r0, r1));
            }
        }
        for (size_t d = kf; d < k; d++)
            for (size_t i = 0; i < 8; i++)
                dst[d * 8 + i] = src[(long)i * lane_stride + d];
        return;
    }
    for (size_t d = 0; d < k; d++)
        for (size_t i = 0; i < 8; i++)
            dst[d * 8 + i] =
                (i < valid) ? src[(long)i * lane_stride + (long)d * depth_stride] : 0.0;
}

// Largest B (n*k*8 bytes) the direct-B path may read unpacked, and the flop
// count below which it stays on one cluster; see gemm_f32.cc. The budget is well
// under f32's because an f64 N-tile is 8 lanes, not 32: unpacked B yields 64
// useful bytes per n*8 stride against f32's 128 per n*4, so an equal footprint
// costs twice the address span.
#define SME_F64_DIRECT_B_BYTES ((size_t)4 * 1024 * 1024)
#define SME_F64_SERIAL_FLOPS ((uint64_t)1 << 22)

// Workers for the dynamic M-chunk claiming; see gemm_f32.cc.
#define SME_F64_MAX_WORKERS 8
static size_t f64_workers(size_t chunks) {
    if (chunks <= 3) return chunks;
    size_t w = chunks - 1;
    return w > SME_F64_MAX_WORKERS ? SME_F64_MAX_WORKERS : w;
}

// m == 1: SME2 multi-vector FMLA into ZA vector groups instead of MOPA; see
// run_gemv in gemm_half.h. 8-column bands, two per 16-wide tile; `abc` is
// [k][8] broadcast A. Past one row the 8x8 MOPA is cheaper.
__arm_locally_streaming __arm_new("za") static void run_gemv(double *dst, const double *abc,
                                                             const double *b_pack, size_t n,
                                                             size_t k, size_t nt_lo, size_t nt_hi,
                                                             double alpha, double beta,
                                                             int read_dst, const ep_desc_f64 *ep) {
    svbool_t p64 = svptrue_b64();
    svcount_t pn = svptrue_c64();
    const svfloat64_t va = svdup_n_f64(alpha);
    const svfloat64_t vb = svdup_n_f64(beta);
    int has_ep = ep && ep->n_nodes > 0;
    int pure = (beta == 1.0 && !read_dst && !has_ep);
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t per_band = ep_cmul(k, 8);
    size_t kfull = k & ~(size_t)3;
    svcount_t pt = svwhilelt_c64((uint64_t)0, (uint64_t)((k - kfull) * 8), 4);
    for (size_t nt = nt_lo; nt < nt_hi; nt += 2) {
        size_t T = nt_hi - nt < 2 ? 2 : 4; // bands
        const double *b = b_pack + 2 * nt * per_band;
        svzero_za();
        if (T == 4) {
            for (size_t d = 0; d < kfull; d += 4) {
                svfloat64x4_t A = svld1_f64_x4(pn, abc + d * 8);
                svmla_za64_f64_vg1x4(0, svld1_f64_x4(pn, b + d * 8), A);
                svmla_za64_f64_vg1x4(1, svld1_f64_x4(pn, b + per_band + d * 8), A);
                svmla_za64_f64_vg1x4(2, svld1_f64_x4(pn, b + 2 * per_band + d * 8), A);
                svmla_za64_f64_vg1x4(3, svld1_f64_x4(pn, b + 3 * per_band + d * 8), A);
            }
        } else {
            for (size_t t = 0; t < T; t++)
                for (size_t d = 0; d < kfull; d += 4)
                    svmla_za64_f64_vg1x4((uint32_t)t, svld1_f64_x4(pn, b + t * per_band + d * 8),
                                         svld1_f64_x4(pn, abc + d * 8));
        }
        if (kfull < k) {
            svfloat64x4_t A = svld1_f64_x4(pt, abc + kfull * 8);
            for (size_t t = 0; t < T; t++)
                svmla_za64_f64_vg1x4((uint32_t)t, svld1_f64_x4(pt, b + t * per_band + kfull * 8), A);
        }
        for (size_t t = 0; t < T; t += 2) {
            size_t n0 = (nt + t / 2) * 16;
            size_t nc = n - n0 < 16 ? n - n0 : 16;
            svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)(nc < 8 ? nc : 8));
            svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)(nc > 8 ? nc - 8 : 0));
            svfloat64x4_t ql = svread_za64_f64_vg1x4((uint32_t)t);
            svfloat64x4_t qh = svread_za64_f64_vg1x4((uint32_t)(t + 1));
            svfloat64_t lo = svadd_x(p64, svadd_x(p64, svget4(ql, 0), svget4(ql, 1)),
                                     svadd_x(p64, svget4(ql, 2), svget4(ql, 3)));
            svfloat64_t hi = svadd_x(p64, svadd_x(p64, svget4(qh, 0), svget4(qh, 1)),
                                     svadd_x(p64, svget4(qh, 2), svget4(qh, 3)));
            double *rp = dst + n0;
            double *rph = nc > 8 ? rp + 8 : rp;
            if (pure) {
                svst1_f64(plo, rp, lo);
                svst1_f64(phi, rph, hi);
            } else if (has_ep) {
#define EP_GEMV_RDL(s) lo
#define EP_GEMV_RDH(s) hi
                EP_STORE_TILE_ROWMAJOR_F64(p64, plo, phi, rp, 0, EP_GEMV_RDL, EP_GEMV_RDH, 0, 1, vb,
                                           va, read_dst, nodes, n_nodes, 0, n0);
#undef EP_GEMV_RDL
#undef EP_GEMV_RDH
            } else {
                lo = svmul_x(p64, lo, vb);
                hi = svmul_x(p64, hi, vb);
                if (read_dst) {
                    lo = svmla_x(p64, lo, svld1_f64(plo, rp), va);
                    hi = svmla_x(p64, hi, svld1_f64(phi, rph), va);
                }
                svst1_f64(plo, rp, lo);
                svst1_f64(phi, rph, hi);
            }
        }
    }
}

// malloc that never requests 0 bytes. With k==0 (per_tile==0) the pack sizes
// below are 0; a malloc(0) may return NULL, which the OOM checks would misread
// as a real allocation failure (spurious rc=-1 for a valid k==0 GEMM).
static void *xmalloc(size_t bytes) { return malloc(bytes ? bytes : 1); }

int gemm_sme_f64_run(size_t m, size_t n, size_t k, double *dst, long dst_cs, long dst_rs,
                     int read_dst, const double *lhs, long lhs_cs, long lhs_rs, const double *rhs,
                     long rhs_cs, long rhs_rs, double alpha, double beta, const ep_desc_f64 *ep) {
    if (m == 0 || n == 0) return 0;
    if (!read_dst) alpha = 0.0;

    size_t m_tiles = (m + 15) / 16;
    size_t per_tile = ep_cmul(k, 8);

    // Small + row-major B: direct B load (no B pack), light dispatch. See gemm_f32.cc.
#define PACKA_SMALL(AP, MT0, MT1)                                                                  \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                             \
        size_t r0 = st * 8;                                                                        \
        size_t vr = (r0 < m) ? ((m - r0 < 8) ? (m - r0) : 8) : 0;                                  \
        pack_band((AP) + st * per_tile, vr ? lhs + (long)r0 * lhs_rs : lhs, lhs_rs, lhs_cs, k, vr); \
    }
    // Direct-B gate is B's footprint, and the serial gate a flop floor; see
    // gemm_f32.cc for why.
    size_t b_budget = SME_F64_DIRECT_B_BYTES / sizeof(double);
    if (rhs_cs == 1 && ep_cmul(n, k) <= b_budget) {
        size_t sc_chunk = 2;
        size_t sc_n = (m_tiles + sc_chunk - 1) / sc_chunk;
        if (sc_n <= 1 || ep_flops(m, n, k) < SME_F64_SERIAL_FLOPS) {
            double *a_pack =
                apack_scratch<double>(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
            if (a_pack) {
                PACKA_SMALL(a_pack, 0, m_tiles);
                run_small(dst, dst_cs, dst_rs, a_pack, rhs, rhs_rs, m, n, k, 0, m_tiles, alpha, beta,
                          read_dst, ep);
                return 0;
            }
        } else {
            double *a_pack = (double *)xmalloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
            if (a_pack) {
                // Chunks claimed off a shared cursor, packing kept with compute;
                // see gemm_sme_f32_run for why.
                std::atomic<size_t> cursor = 0;
                std::atomic<size_t> *cur = &cursor;
                dispatch_apply(f64_workers(sc_n),
                               dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                               ^(size_t w) {
                                 (void)w;
                                 for (;;) {
                                     size_t ci = cur->fetch_add(1, std::memory_order_relaxed);
                                     size_t mt0 = ci * sc_chunk;
                                     if (mt0 >= m_tiles) break;
                                     size_t mt1 =
                                         mt0 + sc_chunk < m_tiles ? mt0 + sc_chunk : m_tiles;
                                     PACKA_SMALL(a_pack, mt0, mt1);
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

    double *b_pack = (double *)xmalloc(ep_cmul(gemm_sme_f64_packed_b_elems(n, k), sizeof(double)));
    if (!b_pack) return -1;
    gemm_sme_f64_packb(b_pack, rhs, n, k, rhs_rs, rhs_cs);

    int rc = gemm_sme_f64_run_packed(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs,
                                     b_pack, alpha, beta, ep);
    free(b_pack);
    return rc;
}

// Elements in a packed-B buffer for (n, k). Saturating (see gemm_i8i32.cc).
size_t gemm_sme_f64_packed_b_elems(size_t n, size_t k) {
    return ep_cmul(2 * ((n + 15) / 16), ep_cmul(k, 8));
}

// Pack B once into a caller-allocated buffer of gemm_sme_f64_packed_b_elems
// doubles: [2*n_tiles][k, 8] bands, the layout run_streaming consumes.
void gemm_sme_f64_packb(double *b_pack, const double *rhs, size_t n, size_t k, long rhs_rs,
                        long rhs_cs) {
    size_t per_tile = ep_cmul(k, 8);
    size_t n_tiles = (n + 15) / 16;
    for (size_t st = 0; st < 2 * n_tiles; st++) {
        size_t c0 = st * 8;
        size_t vc = (c0 < n) ? ((n - c0 < 8) ? (n - c0) : 8) : 0;
        // vc==0 bands are zero-filled without reading src; clamp the base.
        pack_band(b_pack + st * per_tile, vc ? rhs + (long)c0 * rhs_cs : rhs, rhs_cs, rhs_rs, k, vc);
    }
}

// Packed-B f64 GEMM: caller supplies B pre-packed via gemm_sme_f64_packb. No
// run_small arm -- that reads B straight from a row-major rhs.
int gemm_sme_f64_run_packed(size_t m, size_t n, size_t k, double *dst, long dst_cs, long dst_rs,
                            int read_dst, const double *lhs, long lhs_cs, long lhs_rs,
                            const double *b_pack, double alpha, double beta,
                            const ep_desc_f64 *ep) {
    if (m == 0 || n == 0) return 0;
    if (!read_dst) alpha = 0.0;
    size_t m_tiles = (m + 15) / 16;
    size_t n_tiles = (n + 15) / 16;
    size_t per_tile = ep_cmul(k, 8);

#define PACKA_RANGE(AP, MT0, MT1)                                                                  \
    for (size_t st = 2 * (MT0); st < 2 * (MT1); st++) {                                             \
        size_t r0 = st * 8;                                                                        \
        size_t vr = (r0 < m) ? ((m - r0 < 8) ? (m - r0) : 8) : 0;                                  \
        /* vr==0 bands are zero-filled without reading src; clamp the base so no   */               \
        /* out-of-bounds pointer is even formed (UB without a deref), as the B band above. */       \
        pack_band((AP) + st * per_tile, vr ? lhs + (long)r0 * lhs_rs : lhs, lhs_rs, lhs_cs, k, vr); \
    }

    size_t M_CHUNK = 2;
    size_t n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    int big = ep_flops(m, n, k) >= (1u << 21);
    if (m == 1 && dst_cs == 1) {
        double *abc = apack_scratch<double>(ep_cmul(per_tile, sizeof(double)));
        if (!abc) return -1;
        for (size_t d = 0; d < k; d++) {
            float64x2_t v = vdupq_n_f64(lhs[(long)d * lhs_cs]);
            float64x2x4_t q = {{v, v, v, v}};
            vst1q_f64_x4(abc + d * 8, q);
        }
        size_t G_CHUNK = 4; // tiles per chunk
        size_t g_chunks = (n_tiles + G_CHUNK - 1) / G_CHUNK;
        if (!big || g_chunks < 3) {
            run_gemv(dst, abc, b_pack, n, k, 0, n_tiles, alpha, beta, read_dst, ep);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t nt0 = ci * G_CHUNK;
          size_t nt1 = nt0 + G_CHUNK < n_tiles ? nt0 + G_CHUNK : n_tiles;
          run_gemv(dst, abc, b_pack, n, k, nt0, nt1, alpha, beta, read_dst, ep);
        });
        return 0;
    }

    // Flat-M (few M-tiles) but large/wide: M is the only M-chunk axis, so the
    // M-parallel scheme below would run this on ONE cluster. Parallelize over N
    // instead -- pack all of A once (cheap; m_tiles is small) into a shared buffer
    // and hand N-tile chunks to both clusters. C columns are disjoint per chunk;
    // A and B are read-only and shared. f64 N-tiles carry no cross-tile state, so
    // any chunk granularity is safe. Needs >= 2 chunks to beat the serial path.
    // (Mirrors the f32 driver.)
    size_t N_CHUNK = 4;
    size_t nn_chunks = (n_tiles + N_CHUNK - 1) / N_CHUNK;
    if (n_chunks <= 1 && big && nn_chunks >= 2) {
        double *a_pack = (double *)malloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
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
        double *a_pack =
            apack_scratch<double>(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
        if (!a_pack) return -1;
        PACKA_RANGE(a_pack, 0, m_tiles);
        run_streaming(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, 0, n_tiles, k, alpha,
                      beta, read_dst, ep);
        return 0;
    }
    double *a_pack = (double *)xmalloc(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(double)));
    if (!a_pack) return -1;

    // N-block loop OUTSIDE the M dispatch, so one B-block serves every M-chunk
    // while it is still L2-resident; see gemm_f32.cc for the traffic argument.
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

// Batched f64: `count` same-shape row-major GEMMs in one streaming session, with
// an optional epilogue applied to each (gemm_float.h's run_batched_ep).
int gemm_sme_f64_batched_ep(size_t count, size_t m, size_t n, size_t k, double *dst,
                            const double *lhs, const double *rhs, const ep_desc_f64 *ep) {
    return run_batched_ep(count, m, n, k, dst, lhs, rhs, ep);
}

int gemm_sme_f64_batched(size_t count, size_t m, size_t n, size_t k, double *dst, const double *lhs,
                         const double *rhs) {
    return run_batched_ep(count, m, n, k, dst, lhs, rhs, (const ep_desc_f64 *)NULL);
}
