// SME widening 16-bit GEMMs (Apple M4+, base FEAT_SME): f16 x f16 and bf16 x
// bf16 accumulated in f32 (FMOPA / BFMOPA) and narrowed back on the store. Both
// operands pre-pack into the pair-interleaved [K/2, 32] layout, so the hot loop
// is load -> MOPA with no zip; a 32x32 super-tile accumulates across all of K in
// four independent ZA32 quadrants (no RAW between consecutive MOPAs); the whole
// tile grid runs in one streaming session. The two element types differ only in
// their f32 conversions (F16 and Bf16 below); everything else is one template.
//
// Computes dst = alpha*dst + beta*(A @ B): A is m x k, B is k x n, any strides.
// One pass over the full K (no kc chunking), so alpha multiplies the original
// dst exactly once.

#include "sme_gemm_api.h"

#include "panel_ring.h"
#include "sme_tile.h"
#include "transpose16.h"
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// f16 widens through svcvt, which reads the even lanes: duplicate-zip first.
struct F16 {
    using T = __fp16;
    static SME_INLINE svfloat32_t widen_lo(svbool_t p32, svfloat16_t d) __arm_streaming {
        return svcvt_f32_x(p32, svzip1(d, d));
    }
    static SME_INLINE svfloat32_t widen_hi(svbool_t p32, svfloat16_t d) __arm_streaming {
        return svcvt_f32_x(p32, svzip2(d, d));
    }
    static SME_INLINE svfloat16_t narrow(svbool_t p32, svfloat32_t lo,
                                         svfloat32_t hi) __arm_streaming {
        return svuzp1(svcvt_f16_x(p32, lo), svcvt_f16_x(p32, hi));
    }
};

// bf16 is the top half of an f32: widen by zero-extending and shifting left 16.
struct Bf16 {
    using T = __bf16;
    static SME_INLINE svfloat32_t widen_lo(svbool_t p32, svbfloat16_t d) __arm_streaming {
        return svreinterpret_f32(svlsl_n_u32_x(p32, svunpklo(svreinterpret_u16(d)), 16));
    }
    static SME_INLINE svfloat32_t widen_hi(svbool_t p32, svbfloat16_t d) __arm_streaming {
        return svreinterpret_f32(svlsl_n_u32_x(p32, svunpkhi(svreinterpret_u16(d)), 16));
    }
    static SME_INLINE svbfloat16_t narrow(svbool_t p32, svfloat32_t lo,
                                          svfloat32_t hi) __arm_streaming {
        return svuzp1(svcvt_bf16_x(p32, lo), svcvt_bf16_x(p32, hi));
    }
};

template <class E> static float from_bits(uint16_t b) {
    typename E::T h;
    memcpy(&h, &b, sizeof h);
    return (float)h;
}

// Both operands pack into the same [KP, 32] pair-interleaved band -- 16 lanes,
// two depths per group, dst[p*32 + 2*lane + s] = src[lane, 2p+s]. A's lanes are
// rows and B's are columns, so the two only differ in which stride is which.
// Packing moves 16-bit patterns, so these serve f16 and bf16 alike:
//
//   lanes contiguous at each depth  -> pack_lane_contig   (row-major B, col-major A)
//   each lane contiguous in depth   -> pack_depth_contig  (row-major A, col-major B)
//   anything else / ragged bands    -> pack_scalar
//
// Each fast path returns the first p-group it did NOT fill, which pack_scalar takes.

// 16 lanes contiguous per depth: load the two s-depths and ST2 interleave
// straight into the layout.
static size_t pack_lane_contig(uint16_t *dst, const uint16_t *src, long depth_stride, size_t k) {
    for (size_t p = 0; p < k / 2; p++) {
        const uint16_t *s0 = src + (long)(2 * p) * depth_stride, *s1 = s0 + depth_stride;
        uint16x8x2_t lo = {{vld1q_u16(s0), vld1q_u16(s1)}};
        uint16x8x2_t hi = {{vld1q_u16(s0 + 8), vld1q_u16(s1 + 8)}};
        vst2q_u16(dst + p * 32, lo);
        vst2q_u16(dst + p * 32 + 16, hi);
    }
    return k / 2;
}

// One 8-lane x 8-depth block: transpose so each vector is one depth across the
// 8 lanes, then zip the two depths of a p-group -- that zip IS the pair
// interleave the layout wants.
static inline void pack_block8(uint16_t *dst, const uint16_t *src, long lane_stride, size_t lg,
                               size_t d0) {
    uint16x8_t r[8];
    for (int i = 0; i < 8; i++)
        r[i] = vld1q_u16(src + (long)(lg + i) * lane_stride + (long)d0);
    transpose_8x8_u16(r); // r[t] = depth d0+t across the 8 lanes
    for (int t = 0; t < 4; t++) {
        uint16_t *o = dst + (d0 / 2 + (size_t)t) * 32 + 2 * lg;
        vst1q_u16(o, vzip1q_u16(r[2 * t], r[2 * t + 1]));
        vst1q_u16(o + 8, vzip2q_u16(r[2 * t], r[2 * t + 1]));
    }
}

// Each lane contiguous in depth. The 32-bit-gather alternative issues 16 strided
// loads per p-group; blocking through the transpose measured ~1.4-2x over it.
static size_t pack_depth_contig(uint16_t *dst, const uint16_t *src, long lane_stride, size_t k) {
    size_t kt = k & ~(size_t)7; // whole 8-depth blocks
    for (size_t lg = 0; lg < 16; lg += 8)
        for (size_t d0 = 0; d0 < kt; d0 += 8)
            pack_block8(dst, src, lane_stride, lg, d0);
    // Depth remainder below 8: the 32-bit gather, still cheaper than scalar.
    for (size_t p = kt / 2; p < k / 2; p++)
        for (size_t i = 0; i < 16; i++)
            memcpy(dst + p * 32 + 2 * i, src + (long)i * lane_stride + 2 * (long)p, 4);
    return k / 2;
}

// Scalar filler from p-group `p0` on: ragged bands, odd-K slices, and layouts
// with neither stride equal to 1. Zero past the lane and depth edges.
static void pack_scalar(uint16_t *dst, const uint16_t *src, long lane_stride, long depth_stride,
                        size_t k, size_t valid, size_t p0) {
    for (size_t p = p0; p < (k + 1) / 2; p++)
        for (size_t i = 0; i < 16; i++)
            for (size_t s = 0; s < 2; s++) {
                size_t d = 2 * p + s;
                dst[p * 32 + 2 * i + s] =
                    i < valid && d < k ? src[(long)i * lane_stride + (long)d * depth_stride] : 0;
            }
}

// Pack 16-lane bands [s0, s1) of an operand with `lanes` lanes into consecutive
// [KP, 32] bands at dst. Bands past the edge zero-fill without reading src (the
// base is clamped, so no out-of-bounds pointer is even formed).
static void pack_bands(void *dst_v, const void *src_v, long lane_stride, long depth_stride,
                       size_t lanes, size_t k, size_t s0, size_t s1) {
    uint16_t *dst = (uint16_t *)dst_v;
    const uint16_t *src = (const uint16_t *)src_v;
    size_t per_tile = ep_cmul((k + 1) / 2, 32);
    for (size_t st = s0; st < s1; st++, dst += per_tile) {
        size_t l0 = st * 16, v = l0 < lanes ? (lanes - l0 < 16 ? lanes - l0 : 16) : 0;
        const uint16_t *s = v ? src + (long)l0 * lane_stride : src;
        size_t p0 = 0;
        if (v == 16 && depth_stride == 1)
            p0 = pack_depth_contig(dst, s, lane_stride, k);
        else if (v == 16 && lane_stride == 1)
            p0 = pack_lane_contig(dst, s, depth_stride, k);
        pack_scalar(dst, s, lane_stride, depth_stride, k, v, p0);
    }
}

// Given the two f32 halves of one 32-lane C slice, already scaled by beta, add
// alpha*C when read_dst, narrow, and store under `pst`. `ptr` is contiguous in
// the stored direction (a column of col-major C or a row of row-major C).
template <class E>
SME_INLINE void store_vec(svbool_t p32, svbool_t pst, typename E::T *ptr, svfloat32_t lo,
                          svfloat32_t hi, int read_dst, svfloat32_t va) __arm_streaming {
    if (read_dst) {
        Vec<typename E::T> d = svld1(pst, ptr);
        lo = svmla_x(p32, lo, E::widen_lo(p32, d), va);
        hi = svmla_x(p32, hi, E::widen_hi(p32, d), va);
    }
    svst1(pst, ptr, E::narrow(p32, lo, hi));
}

// L2 blocking budget in 32-wide super-tiles.
static size_t widen_budget(size_t kp) {
    size_t tile_bytes = 2 * ep_cmul(kp, 32) * sizeof(uint16_t);
    size_t b = (size_t)(16u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    return b < 2 ? 2 : b;
}

// N-block width for run_streaming's jc loop.
static size_t widen_nc_blk(size_t kp, size_t n_tiles) {
    size_t nc = widen_budget(kp) / 2;
    if (nc < 1) nc = 1;
    if (nc >= n_tiles) return n_tiles;
    size_t blocks = (n_tiles + nc - 1) / nc;
    return (n_tiles + blocks - 1) / blocks;
}

// The streaming compute: one ZA lifetime for a tile range. b_pack points at
// super-tile nt_lo's bands; [nt_lo, nt_hi) is the N-tile sub-range this
// invocation owns. Each super-tile is independent (zero, accumulate, store), so
// any chunk granularity is safe.
template <class E>
__arm_locally_streaming __arm_new("za") static void run_streaming(
    typename E::T *dst, long dst_cs, long dst_rs, const typename E::T *a_pack,
    const typename E::T *b_pack, size_t m, size_t n, size_t mt_lo, size_t mt_hi, size_t nt_lo,
    size_t nt_hi, size_t kp, float alpha, float beta, int read_dst) {
    using T = typename E::T;
    using A = Mopa<T, float>;
    svbool_t p32 = svptrue_b32();
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    size_t per_tile = ep_cmul(kp, 32);

    // Cache blocking (BLIS jc->ic): B-block + A-block resident in L2. nc_blk /
    // mc_blk count 32-wide SUPER-tiles (= 2 packed 16-lane bands each), so
    // tile_bytes is the full super-tile and the budget constant equals the total
    // resident bytes (A-block + B-block) -- matching gemm_f32.cc. A band-sized
    // tile_bytes would instead double the working set.
    size_t budget = widen_budget(kp);
    size_t nc_blk = widen_nc_blk(kp, nt_hi - nt_lo);
    size_t mc_blk = budget / 2;
    if (mc_blk < 1) mc_blk = 1;
    if (mc_blk > mt_hi - mt_lo) mc_blk = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc_blk) {
        size_t jc_end = jc + nc_blk < nt_hi ? jc + nc_blk : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc_blk) {
            size_t ic_end = ic + mc_blk < mt_hi ? ic + mc_blk : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const T *a_lo = a_pack + 2 * mt * per_tile, *a_hi = a_lo + per_tile;
                size_t m0 = mt * 32, mr = m - m0 < 32 ? m - m0 : 32;
                for (size_t nt = jc; nt < jc_end; nt++) {
                    const T *b_lo = b_pack + 2 * (nt - nt_lo) * per_tile, *b_hi = b_lo + per_tile;
                    size_t n0 = nt * 32, nc = n - n0 < 32 ? n - n0 : 32;

                    svzero_za();
                    quad_accumulate<A, KSteps<2>>(kp, a_lo, a_hi, b_lo, b_hi, mr > 16, nc > 16);

                    if (dst_rs == 1) {
                        // Col-major C: read ZA vertically so a column's 32
                        // M-values land contiguous.
                        svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)mr);
                        quad_cols<16>(
                            nc, [&] [[gnu::always_inline]] (
                                    auto zl, auto zh, uint32_t s,
                                    size_t c) __arm_streaming __arm_in("za") {
                                store_vec<E>(p32, pst, dst + (long)(n0 + c) * dst_cs + (long)m0,
                                             svmul_x(p32, za_col<zl, float>(s), vb),
                                             svmul_x(p32, za_col<zh, float>(s), vb), read_dst, va);
                            });
                    } else if (dst_cs == 1) {
                        // Row-major C: read ZA horizontally.
                        svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)nc);
                        quad_rows<16>(
                            mr, [&] [[gnu::always_inline]] (
                                    auto zl, auto zh, uint32_t s,
                                    size_t r) __arm_streaming __arm_in("za") {
                                store_vec<E>(p32, pst, dst + (long)(m0 + r) * dst_rs + (long)n0,
                                             svmul_x(p32, za_row<zl, float>(s), vb),
                                             svmul_x(p32, za_row<zh, float>(s), vb), read_dst, va);
                            });
                    } else {
                        // Fully general strides: through a scratch, scalar combine.
                        // Rare; correctness over speed.
                        float scratch[32 * 32];
                        quad_to_scratch(scratch);
                        for (size_t r = 0; r < mr; r++)
                            for (size_t c = 0; c < nc; c++) {
                                float ab = scratch[r * 32 + c] * beta;
                                T *cell = dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                                *cell = (T)(read_dst ? alpha * (float)*cell + ab : ab);
                            }
                    }
                }
            }
        }
    }
}

// Per-thread reusable A-pack scratch (no per-call malloc/free on the serial
// small/medium path). Grows monotonically, lives for the thread's lifetime.
static thread_local void *g_apack = NULL;
static thread_local size_t g_apack_cap = 0;
static void *apack_scratch(size_t bytes) {
    if (bytes == 0)
        bytes = 1; // k==0: return a real (tiny) buffer, not a
                   // history-dependent NULL on fresh threads
    if (g_apack_cap < bytes) {
        free(g_apack);
        g_apack = malloc(bytes);
        g_apack_cap = g_apack ? bytes : 0;
    }
    return g_apack;
}

// M-tile mt against row-major B read in place (`b` is B's column 0, row stride
// b_rs), super-tiles [st, st + nst), nst <= 2. Rows 2p and 2p+1 zip into the
// pair-interleaved bands the widening MOPA takes, so nothing is packed. Every
// load is a full 128-byte line across both super-tiles: one A band (rows <= 16)
// multiplies all four bands at once; otherwise the pass multiplies super-tile st
// and writes st+1's bands to buf[2..3] for the packed kernel. `keep` also writes
// st's bands to buf[0..1], for later M-tiles. Returns how many super-tiles it
// multiplied. Row-major C.
template <class E>
__arm_locally_streaming __arm_new("za") static size_t
    run_direct(typename E::T *dst, long dst_rs, const typename E::T *a_pack, const typename E::T *b,
               long b_rs, size_t m, size_t n, size_t k, size_t mt, size_t st, size_t nst,
               typename E::T *buf, int keep, float alpha, float beta, int read_dst) {
    using T = typename E::T;
    using A = Mopa<T, float>;
    svbool_t p16 = svptrue_b16(), p32 = svptrue_b32();
    const Vec<T> z16 = Sve<T>::zero();
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    size_t kp = (k + 1) / 2, per_tile = ep_cmul(kp, 32);
    const T *a_lo = a_pack + 2 * mt * per_tile, *a_hi = a_lo + per_tile;
    size_t m0 = mt * 32, mr = m - m0 < 32 ? m - m0 : 32;
    const T *b0 = b + st * 32;
    svcount_t pc = svwhilelt_c16((uint64_t)st * 32, (uint64_t)n, 2); // nst 1: lanes past st unused
    int four = mr <= 16 && nst == 2;
    svzero_za();
    for (size_t p = 0; p < kp; p++) {
        const T *r = b0 + (long)(2 * p) * b_rs;
        auto R0 = svld1_x2(pc, r);
        auto R1 = 2 * p + 1 < k ? svld1_x2(pc, r + b_rs) : svcreate2(z16, z16);
        Vec<T> bl = svzip1(svget2(R0, 0), svget2(R1, 0));
        Vec<T> bh = svzip2(svget2(R0, 0), svget2(R1, 0));
        Vec<T> al = svld1(p16, a_lo + p * 32);
        if (keep) {
            svst1(p16, buf + p * 32, bl);
            svst1(p16, buf + per_tile + p * 32, bh);
        }
        if (nst == 2) {
            Vec<T> cl = svzip1(svget2(R0, 1), svget2(R1, 1));
            Vec<T> ch = svzip2(svget2(R0, 1), svget2(R1, 1));
            if (four) {
                A::template mopa<0>(p16, al, bl);
                A::template mopa<1>(p16, al, bh);
                A::template mopa<2>(p16, al, cl);
                A::template mopa<3>(p16, al, ch);
                if (keep) {
                    svst1(p16, buf + 2 * per_tile + p * 32, cl);
                    svst1(p16, buf + 3 * per_tile + p * 32, ch);
                }
                continue;
            }
            svst1(p16, buf + 2 * per_tile + p * 32, cl);
            svst1(p16, buf + 3 * per_tile + p * 32, ch);
        }
        if (mr > 16)
            quad_mopa<A, true, true>(p16, al, svld1(p16, a_hi + p * 32), bl, bh);
        else
            quad_mopa<A, false, true>(p16, al, al, bl, bh);
    }
    // Rows [r0, r1) of the tile pair (zl, zh) to the 32 columns from c0.
    auto store_rows = [&] [[gnu::always_inline]] (auto zl, auto zh, size_t r0, size_t r1,
                                                  size_t c0) __arm_streaming __arm_in("za") {
        svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)(n - c0 < 32 ? n - c0 : 32));
        for (size_t r = r0; r < r1; r++) {
            uint32_t s = (uint32_t)(r - r0);
            store_vec<E>(p32, pst, dst + (long)(m0 + r) * dst_rs + (long)c0,
                         svmul_x(p32, za_row<zl, float>(s), vb),
                         svmul_x(p32, za_row<zh, float>(s), vb), read_dst, va);
        }
    };
    store_rows(Tile<0>{}, Tile<1>{}, 0, mr < 16 ? mr : 16, st * 32);
    if (!four) store_rows(Tile<2>{}, Tile<3>{}, 16, mr > 16 ? mr : 16, st * 32);
    if (four && (st + 1) * 32 < n) store_rows(Tile<2>{}, Tile<3>{}, 0, mr, (st + 1) * 32);
    return four ? 2 : 1;
}

// Per-worker column items, as gemm_f16f16.cc's run_cols: each worker packs its
// own two super-tiles of B and multiplies them while they are in cache; the
// first units pack A by M-chunk.
template <class E>
static int run_cols(typename E::T *dst, long dst_cs, long dst_rs, const typename E::T *a,
                    long lhs_rs, long lhs_cs, size_t m, size_t n, size_t k, float alpha, float beta,
                    int read_dst, const typename E::T *b, long rhs_rs, long rhs_cs) {
    using T = typename E::T;
    size_t kp = (k + 1) / 2, per_tile = ep_cmul(kp, 32);
    size_t m_tiles = (m + 31) / 32, n_tiles = (n + 31) / 32;
    size_t NC = 2; // super-tiles per item
    size_t M_CHUNK = 2, n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    size_t n_items = (n_tiles + NC - 1) / NC, n_units = n_chunks + n_items;
    size_t W = n_units < 12 ? n_units : 12;
    size_t panel = ep_cmul(2 * NC, per_tile);
    std::atomic<size_t> *ctr;
    T *pool = (T *)ring_pool(2 + n_chunks, ep_cmul(ep_cmul(W, panel), sizeof(T)), &ctr);
    T *a_pack = (T *)apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(T)));
    if (!pool || !a_pack) return -1;
    std::atomic<size_t> *cur = ctr, *a_cnt = ctr + 1, *a_done = ctr + 2;
    // Row-major B and C: multiply straight from B's rows. Never more than ~6%
    // behind the pack below and up to 3x ahead at small m (16x4096x4096: 1.52
    // against 0.52 TF/s), so it is not gated on stride the way f16f16's is.
    int direct = rhs_cs == 1 && dst_cs == 1 && rhs_rs > 0;
    dispatch_apply(W, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t w) {
      T *bt = pool + w * panel;
      for (size_t i; (i = cur->fetch_add(1, std::memory_order_relaxed)) < n_units;) {
          if (i < n_chunks) {
              size_t mt0 = i * M_CHUNK, mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
              pack_bands(a_pack + 2 * mt0 * per_tile, a, lhs_rs, lhs_cs, m, k, 2 * mt0, 2 * mt1);
              a_done[i].store(1, std::memory_order_release);
              a_cnt->fetch_add(1, std::memory_order_release);
              continue;
          }
          size_t t0 = (i - n_chunks) * NC, t1 = t0 + NC < n_tiles ? t0 + NC : n_tiles, mt_lo = 0;
          if (direct) {
              // M-tile 0 multiplies straight from B's rows; the bands it captures
              // feed the packed kernel for whatever it left.
              while (!a_done[0].load(std::memory_order_acquire))
                  __builtin_arm_yield();
              size_t done = run_direct<E>(dst, dst_rs, a_pack, b, rhs_rs, m, n, k, 0, t0, t1 - t0,
                                          bt, m_tiles > 1, alpha, beta, read_dst);
              if (done < t1 - t0)
                  run_streaming<E>(dst, 1, dst_rs, a_pack, bt + 2 * per_tile, m, n, 0, 1, t0 + 1,
                                   t1, kp, alpha, beta, read_dst);
              mt_lo = 1;
          } else {
              pack_bands(bt, b, rhs_cs, rhs_rs, n, k, 2 * t0, 2 * t1);
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
              run_streaming<E>(dst, dst_cs, dst_rs, a_pack, bt, m, n, mt0, mt1, t0, t1, kp, alpha,
                               beta, read_dst);
          }
      }
    });
    return 0;
}

// Large A: B is built per N-block into a panel ring (panel_ring.h) and shared by
// every M-chunk, as gemm_f32.cc's run_panels.
template <class E>
static int run_panels(typename E::T *dst, long dst_cs, long dst_rs, const typename E::T *a,
                      long lhs_rs, long lhs_cs, size_t m, size_t n, size_t k, float alpha,
                      float beta, int read_dst, const typename E::T *b, long rhs_rs, long rhs_cs) {
    using T = typename E::T;
    size_t kp = (k + 1) / 2, per_tile = ep_cmul(kp, 32), tile = 2 * per_tile;
    size_t m_tiles = (m + 31) / 32, n_tiles = (n + 31) / 32;
    size_t nc_blk = ((size_t)4 << 20) / ep_cmul(tile, sizeof(T)); // 4 MB B-blocks
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
    T *pool = (T *)ring_pool(2 * n_blocks + 2 + n_chunks,
                             ep_cmul(ep_cmul(n_slots, slot_elems), sizeof(T)), &ctr);
    T *a_pack = (T *)apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(T)));
    if (!pool || !a_pack) return -1;
    std::atomic<size_t> *a_done = ctr + 2 * n_blocks + 2;
    if (n_chunks <= 1) pack_bands(a_pack, a, lhs_rs, lhs_cs, m, k, 0, 2 * m_tiles);

    ring_run(
        n_blocks, n_d, n_chunks * n_sub, ctr,
        ^(size_t sd, size_t j) {
          size_t lo = sd * nc_blk, hi = lo + nc_blk < n_tiles ? lo + nc_blk : n_tiles;
          size_t t0 = lo + j * dt, t1 = t0 + dt < hi ? t0 + dt : hi;
          T *slot = pool + (sd % RING_SLOTS) * slot_elems;
          pack_bands(slot + (t0 - lo) * tile, b, rhs_cs, rhs_rs, n, k, 2 * t0, 2 * t1);
        },
        ^(size_t sg, size_t j) {
          size_t lo = sg * nc_blk, hi = lo + nc_blk < n_tiles ? lo + nc_blk : n_tiles;
          size_t ci = j / n_sub, sj = j % n_sub;
          size_t mt0 = ci * M_CHUNK, mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
          size_t n0 = lo + sj * ns < hi ? lo + sj * ns : hi, n1 = n0 + ns < hi ? n0 + ns : hi;
          // Block 0's first item per M-chunk packs that chunk of A.
          if (n_chunks > 1 && sg == 0 && sj == 0) {
              pack_bands(a_pack + 2 * mt0 * per_tile, a, lhs_rs, lhs_cs, m, k, 2 * mt0, 2 * mt1);
              a_done[ci].store(1, std::memory_order_release);
          } else if (n_chunks > 1) {
              while (!a_done[ci].load(std::memory_order_acquire))
                  __builtin_arm_yield();
          }
          if (n0 < n1)
              run_streaming<E>(dst, dst_cs, dst_rs, a_pack,
                               pool + (sg % RING_SLOTS) * slot_elems + (n0 - lo) * tile, m, n, mt0,
                               mt1, n0, n1, kp, alpha, beta, read_dst);
        });
    return 0;
}

template <class E>
static int run(size_t m, size_t n, size_t k, uint16_t *dst_bits, long dst_cs, long dst_rs,
               int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs, const uint16_t *rhs,
               long rhs_cs, long rhs_rs, uint16_t alpha_bits, uint16_t beta_bits) {
    using T = typename E::T;
    if (m == 0 || n == 0) return 0;
    float alpha = read_dst ? from_bits<E>(alpha_bits) : 0.0f;
    float beta = from_bits<E>(beta_bits);
    T *dst = (T *)dst_bits;
    const T *a = (const T *)lhs, *b = (const T *)rhs;
    // Build B inside the parallel GEMM rather than packing it all first: per-worker
    // columns while A is cache-sized, the shared panel ring past that.
    if (ep_flops(m, n, k) >= (1u << 21)) {
        if (use_cols(m, n, ep_cmul(ep_cmul(m, k), sizeof(T))))
            return run_cols<E>(dst, dst_cs, dst_rs, a, lhs_rs, lhs_cs, m, n, k, alpha, beta,
                               read_dst, b, rhs_rs, rhs_cs);
        return run_panels<E>(dst, dst_cs, dst_rs, a, lhs_rs, lhs_cs, m, n, k, alpha, beta, read_dst,
                             b, rhs_rs, rhs_cs);
    }
    // Small: both operands packed whole, one streaming pass on this thread.
    size_t kp = (k + 1) / 2, per_tile = ep_cmul(kp, 32);
    size_t m_tiles = (m + 31) / 32, n_tiles = (n + 31) / 32;
    T *b_pack = (T *)malloc(ep_cmul(ep_cmul(2 * n_tiles, per_tile), sizeof(T)));
    T *a_pack = (T *)apack_scratch(ep_cmul(ep_cmul(2 * m_tiles, per_tile), sizeof(T)));
    if (!b_pack || !a_pack) {
        free(b_pack);
        return -1;
    }
    pack_bands(b_pack, b, rhs_cs, rhs_rs, n, k, 0, 2 * n_tiles);
    pack_bands(a_pack, a, lhs_rs, lhs_cs, m, k, 0, 2 * m_tiles);
    run_streaming<E>(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, 0, m_tiles, 0, n_tiles, kp, alpha,
                     beta, read_dst);
    free(b_pack);
    return 0;
}

int gemm_sme_f16f32_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                        int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                        const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                        uint16_t beta_bits) {
    return run<F16>(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs,
                    rhs_rs, alpha_bits, beta_bits);
}

int gemm_sme_bf16f32_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                         int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                         const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                         uint16_t beta_bits) {
    return run<Bf16>(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs,
                     rhs_rs, alpha_bits, beta_bits);
}
