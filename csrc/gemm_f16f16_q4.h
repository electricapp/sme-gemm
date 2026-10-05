// f16 Q4 path: C(f16) = A(f16) @ dequant(B), B kept 4-bit resident.
// Included at the end of gemm_f16f16.c for its `f16` typedef, `packa` and the
// dense kernels; the bf16 twin is gemm_b16b16_q4.h.
//
// B is resident band-major: bands of Q4_BAND = 128 columns, and for each depth a
// band holds 64 bytes of nibbles, column c of the band at nibble c (two a byte,
// low first). One 64-byte load and one LUTI4 x4 give one depth of the band's
// four 32-column tiles, so the GEMV multiplies all four by a single A value with
// the indexed FMLA, and the MOPA path unpacks a row of four panels at once.
// Scales (and mins) are [band][ceil(k/block)][128] f16, one per (column,
// K-block). `bshift` is log2 of the K-block size (5 => 32, the Q4_0/Q4_1
// default). `mins` is NULL for the scale-only form (Q4_0), else the matching
// offsets of the affine form (Q4_1 / llama.cpp *_1): w = scale*code + min.
#ifndef SME_GEMM_F16F16_Q4_H
#define SME_GEMM_F16F16_Q4_H

#include <stdatomic.h>

#include "neon_act.h"

#define Q4_BAND 128

// Hooks for a GEMV chained to work on other threads (src/mlp.rs). `done` is
// raised (release) to the last output column of each pass once its outputs are
// stored; before each K-block the kernel waits (acquire) until `ready` covers
// the block's depths of A. Either may be NULL.
typedef struct {
    _Atomic size_t *done;
    const _Atomic size_t *ready;
} q4_chain;

// Code c -> f16(sign-extended c); 16-bit LUTI4 reads the low half of 32-bit entries.
static uint32_t g_q4_lut[16] __attribute__((aligned(64)));
static void q4_lut_init(void) {
    for (int c = 0; c < 16; c++) {
        f16 v = (f16)(float)(c < 8 ? c : c - 16);
        uint16_t bits;
        __builtin_memcpy(&bits, &v, 2);
        g_q4_lut[c] = bits;
    }
}

// LUTI4 unpack of tiles [t0, t1) into consecutive [k][32] f16 panels at `out`.
// Per depth, one LUTI4 gives the band's four tiles and one FMLA adds code*scale
// onto ZA holding zero or the min, so every weight rounds once, as
// dequant_q4_with does; eight depths fill eight ZA groups between stores.
__arm_locally_streaming __arm_new("za", "zt0") static void unpack_q4_f16(
    f16 *out, const uint8_t *nibbles, const f16 *scales, const f16 *mins, size_t k, unsigned bshift,
    size_t t0, size_t t1, size_t nib_per_band, size_t sc_per_band) {
    svldr_zt(0, g_q4_lut);
    svbool_t p8 = svptrue_b8(), p16 = svptrue_b16();
    svcount_t pn = svptrue_c16();
    size_t per_tile = k * 32;
    for (size_t b = t0 / 4; b * 4 < t1; b++) {
        const uint8_t *nb = nibbles + b * nib_per_band;
        const f16 *sc = scales + b * sc_per_band;
        const f16 *mn = mins ? mins + b * sc_per_band : NULL;
        // The band's four panels, NULL outside [t0, t1).
        f16 *o[4];
        for (size_t v = 0; v < 4; v++) {
            size_t t = b * 4 + v;
            o[v] = t >= t0 && t < t1 ? out + (t - t0) * per_tile : NULL;
        }
        for (size_t d0 = 0; d0 < k; d0 += 8) {
            size_t rows = k - d0 < 8 ? k - d0 : 8;
            if (!mn) svzero_za();
            // One K-block's scales (and mins) serve all eight depths when the block
            // is at least 8 deep; smaller blocks reload them a depth.
            size_t bk0 = d0 >> bshift;
            svfloat16x4_t s4 = svld1_f16_x4(pn, sc + bk0 * Q4_BAND);
            svfloat16x4_t m4 = mn ? svld1_f16_x4(pn, mn + bk0 * Q4_BAND) : s4;
            for (size_t i = 0; i < rows; i++) {
                size_t d = d0 + i, bk = d >> bshift;
                if (bk != bk0) {
                    bk0 = bk;
                    s4 = svld1_f16_x4(pn, sc + bk * Q4_BAND);
                    if (mn) m4 = svld1_f16_x4(pn, mn + bk * Q4_BAND);
                }
                if (mn) svwrite_za16_f16_vg1x4((uint32_t)i, m4);
                svmla_za16_f16_vg1x4((uint32_t)i,
                                     svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + d * 64), 0), s4);
            }
            for (size_t i = 0; i < rows; i++) {
                svfloat16x4_t q = svread_za16_f16_vg1x4((uint32_t)i);
                size_t off = (d0 + i) * 32;
                if (o[0]) svst1_f16(p16, o[0] + off, svget4(q, 0));
                if (o[1]) svst1_f16(p16, o[1] + off, svget4(q, 1));
                if (o[2]) svst1_f16(p16, o[2] + off, svget4(q, 2));
                if (o[3]) svst1_f16(p16, o[3] + off, svget4(q, 3));
            }
        }
    }
}

// Q4 MOPA path (m > Q4_GEMV_MAXR): each N-block of B is unpacked once into a
// ring slot and shared by every M-chunk (run_panels).
static int run_q4_panels(f16 *dst, const f16 *a, const uint8_t *nibbles, const f16 *scales,
                         const f16 *mins, size_t m, size_t n, size_t k, unsigned bshift,
                         const ep_desc16 *ep) {
    size_t nib_per_band = ep_cmul(k, 64);
    size_t sc_per_band = ep_cmul((k + ((size_t)1 << bshift) - 1) >> bshift, Q4_BAND);
    return run_panels(dst, 1, (long)n, a, (long)k, 1, m, n, k, 0.0f, 1.0f, 0, ep, (size_t)1 << 18,
                      ^(f16 *out, size_t t0, size_t t1) {
                        unpack_q4_f16(out, nibbles, scales, mins, k, bshift, t0, t1, nib_per_band,
                                      sc_per_band);
                      });
}

// Small-m Q4 GEMV. Per depth and band, one LUTI4 expands the band's four tiles
// and one indexed FMLA per row multiplies them by that row's A value, lane i of
// a 16-byte replicated load of A -- no broadcast of A is ever built, and the
// indexed form issues at twice the tuple-by-tuple rate (0.24 vs 0.48 ns on M5).
// Raw codes accumulate per K-block in groups 0-3 and 8-11; at each block edge one
// tuple FMLA folds acc*scale (+min*sum(A)) into the totals 4 above, and one
// ZERO {za0-3.d} clears the block sums (group w lies in ZA64 tile w%8).
#define Q4_ACC(q) ((uint32_t)((q) < 4 ? (q) : (q) + 4))
// Up to here the GEMV beats the MOPA path, whose panel unpack costs as much SME
// time as a 32-row M-tile (measured 1.45x at m=5, 1.06x at 7, 0.96x at 8).
#define Q4_GEMV_MAXR 7
// A's lines are usually fresh from the caller's core, and the unit's first read
// of each waits on that core (~0.15 us over k=384, ~0.37 us over k=1536 when
// met one by one inside the loop). Loading it all up front, four vectors at a
// time with no result used, overlaps those waits; the last load is predicated
// to the end of A. The library's own producers store with STNP (neon_act.h),
// which leaves nothing to wait on.
#define Q4_TOUCH(p, bytes)                                                                         \
    for (uint64_t o_ = 0; o_ < (uint64_t)(bytes); o_ += 4 * svcntb())                              \
    __asm__ volatile("whilelt pn8.b, %0, %1, vlx4\n\tld1b {z0.b-z3.b}, pn8/z, [%2, %0]" ::"r"(o_), \
                     "r"((uint64_t)(bytes)), "r"(p)                                                \
                     : "z0", "z1", "z2", "z3", "p8", "memory")

// One row: depth i of the step at `d`, for each of BB bands. BB is a constant in
// each expansion, so the band loop unrolls (a runtime count spills).
// With 4 bands or fewer, odd depths accumulate in a second group per band (4
// above), so each depth still has twice the independent FMLA chains to issue.
#define Q4_D1(i, BB)                                                                               \
    for (size_t j = 0; j < (BB); j++)                                                              \
    svmla_lane_za16_f16_vg1x4(                                                                     \
        Q4_ACC(j + ((BB) <= 4 ? 4 * ((i) & 1) : 0)),                                               \
        svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + j * nib_per_band + (d + (i)) * 64), 0), aq,    \
        (i))
// Depths 2p and 2p+1 of BB bands: one 128-byte load per band covers both (a
// whole cache line; a 64-byte SME load costs as much as a full line), then the
// two depths' FMLAs interleave across bands. Bands past BB are never touched.
#define Q4_L2(j, BB)                                                                               \
    if ((BB) > (j)) v##j = svld1_u8_x2(pc, q + (j) * nib_per_band)
#define Q4_F2(j, i, BB)                                                                            \
    if ((BB) > (j))                                                                                \
    svmla_lane_za16_f16_vg1x4(Q4_ACC((j) + ((BB) <= 4 ? 4 * ((i) & 1) : 0)),                       \
                              svluti4_lane_zt_f16_x4(0, svget2(v##j, (i) & 1), 0), aq, (i))
#define Q4_F2B(i, BB)                                                                              \
    Q4_F2(0, i, BB);                                                                               \
    Q4_F2(1, i, BB);                                                                               \
    Q4_F2(2, i, BB);                                                                               \
    Q4_F2(3, i, BB);                                                                               \
    Q4_F2(4, i, BB);                                                                               \
    Q4_F2(5, i, BB);                                                                               \
    Q4_F2(6, i, BB);                                                                               \
    Q4_F2(7, i, BB)
#define Q4_P2(p, BB)                                                                               \
    do {                                                                                           \
        svuint8x2_t v0, v1, v2, v3, v4, v5, v6, v7;                                                \
        const uint8_t *q = nb + (d + 2 * (p)) * 64;                                                \
        Q4_L2(0, BB);                                                                              \
        Q4_L2(1, BB);                                                                              \
        Q4_L2(2, BB);                                                                              \
        Q4_L2(3, BB);                                                                              \
        Q4_L2(4, BB);                                                                              \
        Q4_L2(5, BB);                                                                              \
        Q4_L2(6, BB);                                                                              \
        Q4_L2(7, BB);                                                                              \
        Q4_F2B(2 * (p), BB);                                                                       \
        Q4_F2B(2 * (p) + 1, BB);                                                                   \
    } while (0)
#define Q4_STEP1(BB)                                                                               \
    do {                                                                                           \
        if (L == 8) {                                                                              \
            Q4_P2(0, BB);                                                                          \
            Q4_P2(1, BB);                                                                          \
            Q4_P2(2, BB);                                                                          \
            Q4_P2(3, BB);                                                                          \
        } else {                                                                                   \
            Q4_D1(0, BB);                                                                          \
            if (L > 1) Q4_D1(1, BB);                                                               \
            if (L > 2) Q4_D1(2, BB);                                                               \
            if (L > 3) Q4_D1(3, BB);                                                               \
            if (L > 4) Q4_D1(4, BB);                                                               \
            if (L > 5) Q4_D1(5, BB);                                                               \
            if (L > 6) Q4_D1(6, BB);                                                               \
        }                                                                                          \
    } while (0)
// Several rows: depth i for B bands and every row.
#define Q4_DR(i)                                                                                   \
    if ((i) < L)                                                                                   \
        for (size_t j = 0; j < B; j++) {                                                           \
            svfloat16x4_t c4 = svluti4_lane_zt_f16_x4(                                             \
                0, svld1_u8(p8, nb + j * nib_per_band + (d + (i)) * 64), 0);                       \
            for (size_t r = 0; r < rows; r++)                                                      \
                svmla_lane_za16_f16_vg1x4(Q4_ACC(r * B + j), c4, svld1rq_f16(pa, a + r * k + d),   \
                                          (i));                                                    \
        }

__arm_locally_streaming __arm_new("za", "zt0") static void run_q4_gemv(
    f16 *dst, const f16 *a, const f16 *asum, size_t rows, const uint8_t *nibbles, const f16 *scales,
    const f16 *mins, size_t n, size_t k, size_t b_lo, size_t b_hi, size_t nib_per_band,
    size_t sc_per_band, unsigned bshift, const uint32_t *lut, const ep_desc16 *ep,
    const q4_chain *chain) {
    svbool_t p8 = svptrue_b8(), p16 = svptrue_b16();
    svcount_t pn = svptrue_c16(), pc = svptrue_c8();
    const EpNode *nodes = ep ? ep->nodes : NULL;
    uint32_t n_nodes = ep ? ep->n_nodes : 0;
    svfloat16_t vb = svdup_n_f16((f16)1.0f);
    svfloat16_t va = svdup_n_f16((f16)0.0f);
    svldr_zt(0, lut);
    _Atomic size_t *done = chain ? chain->done : NULL;
    const _Atomic size_t *ready = chain ? chain->ready : NULL;
    if (!ready) Q4_TOUCH(a, rows * k * sizeof(f16)); // gated A is not written yet
    size_t block = (size_t)1 << bshift;
    size_t nbk = (k + block - 1) >> bshift;
    // Rows x bands <= 8 block-sum groups; spread the bands evenly over passes.
    size_t bb = 8 / rows, nbands = b_hi - b_lo;
    size_t passes = (nbands + bb - 1) / bb, per = (nbands + passes - 1) / passes;
    // Core-side atomics stall streaming code (on M5 a load ~20 ns, a release
    // store ~60 ns, waiting for the unit's stores): `ready` is read again only
    // past the depths it last covered, and a pass's columns are published one
    // K-block into the next pass, by when its stores have long completed.
    size_t avail = ready ? 0 : k, pending = 0;
    for (size_t b0 = b_lo; b0 < b_hi; b0 += per) {
        size_t B = b_hi - b0 < per ? b_hi - b0 : per;
        const uint8_t *nb = nibbles + b0 * nib_per_band;
        svzero_za();
        for (size_t bk = 0; bk < nbk; bk++) {
            size_t d1 = ((bk + 1) << bshift) < k ? (bk + 1) << bshift : k;
            while (avail < d1)
                avail = atomic_load_explicit(ready, memory_order_acquire);
            if (bk == 1 && pending) {
                atomic_store_explicit(done, pending, memory_order_release);
                pending = 0;
            }
            for (size_t d = bk << bshift; d < d1; d += 8) {
                size_t L = d1 - d < 8 ? d1 - d : 8;
                // Lanes at or past k load as zero; past-k depths are never run.
                svbool_t pa = svwhilelt_b16((uint64_t)d, (uint64_t)k);
                if (rows == 1) {
                    svfloat16_t aq = svld1rq_f16(pa, a + d);
                    switch (B) {
                        case 8:
                            Q4_STEP1(8);
                            break;
                        case 7:
                            Q4_STEP1(7);
                            break;
                        case 6:
                            Q4_STEP1(6);
                            break;
                        case 5:
                            Q4_STEP1(5);
                            break;
                        case 4:
                            Q4_STEP1(4);
                            break;
                        case 3:
                            Q4_STEP1(3);
                            break;
                        case 2:
                            Q4_STEP1(2);
                            break;
                        default:
                            Q4_STEP1(1);
                            break;
                    }
                } else {
                    Q4_DR(0) Q4_DR(1) Q4_DR(2) Q4_DR(3) Q4_DR(4) Q4_DR(5) Q4_DR(6) Q4_DR(7)
                }
            }
            for (size_t r = 0; r < rows; r++)
                for (size_t j = 0; j < B; j++) {
                    uint32_t g = Q4_ACC(r * B + j);
                    size_t s0 = (b0 + j) * sc_per_band + bk * Q4_BAND;
                    svfloat16x4_t s4 = svld1_f16_x4(pn, scales + s0);
                    svmla_za16_f16_vg1x4(g + 4, svread_za16_f16_vg1x4(g), s4);
                    if (rows == 1 && B <= 4)
                        svmla_za16_f16_vg1x4(g + 4, svread_za16_f16_vg1x4(Q4_ACC(j + 4)), s4);
                    if (mins)
                        svmla_single_za16_f16_vg1x4(g + 4, svld1_f16_x4(pn, mins + s0),
                                                    svdup_n_f16(asum[r * nbk + bk]));
                }
            svzero_mask_za(0x0F);
        }
        for (size_t r = 0; r < rows; r++)
            for (size_t j = 0; j < B; j++) {
                svfloat16x4_t q = svread_za16_f16_vg1x4(Q4_ACC(r * B + j) + 4);
#define Q4_ST(v)                                                                                   \
    do {                                                                                           \
        size_t n0 = (b0 + j) * Q4_BAND + (v) * 32;                                                 \
        if (n0 < n) {                                                                              \
            size_t ncols = n - n0 < 32 ? n - n0 : 32;                                              \
            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);                            \
            svfloat16_t acc = svget4(q, v);                                                        \
            f16 *out = dst + r * n + n0;                                                           \
            if (!n_nodes) {                                                                        \
                svst1_f16(pst, out, acc);                                                          \
            } else {                                                                               \
                EP_STORE_TILE_ROWMAJOR_F16(p16, pst, out, (long)n, EP_Q4G_RD, 1, vb, va, 0, nodes, \
                                           n_nodes, r, n0);                                        \
            }                                                                                      \
        }                                                                                          \
    } while (0)
#define EP_Q4G_RD(s) acc
                Q4_ST(0);
                Q4_ST(1);
                Q4_ST(2);
                Q4_ST(3);
#undef EP_Q4G_RD
#undef Q4_ST
            }
        if (done) {
            size_t c1 = (b0 + B) * Q4_BAND;
            pending = c1 < n ? c1 : n;
        }
    }
    // The last pass is published by the caller once out of streaming mode,
    // where a release store costs nothing (gemm_sme_f16f16_q4_chained).
}
#undef Q4_D1
#undef Q4_L2
#undef Q4_F2
#undef Q4_F2B
#undef Q4_P2
#undef Q4_STEP1
#undef Q4_DR

// Q4 GEMM: C(f16) = A(f16) @ dequant(B), B 4-bit resident band-major (see the
// top of this file); f16 exists only one N-block panel at a time on the MOPA
// path. `block` is the K-block size (power of two; 32 = Q4_0/Q4_1). `ep` is the
// fused f16 op-graph applied in-register at the store, or NULL. `chain` (or
// NULL) holds the progress hooks of a chained call, which runs as one call.
static int q4_core(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                   const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                   size_t block, const ep_desc16 *ep, const q4_chain *chain) {
    if (m == 0 || n == 0) return 0;
    if (block == 0 || (block & (block - 1)) != 0) return -1; // power of two only
    unsigned bshift = 0;
    while (((size_t)1 << bshift) < block)
        bshift++;
    static dispatch_once_t lut_once;
    dispatch_once_f(&lut_once, NULL, (dispatch_function_t)q4_lut_init);
    size_t n_bands = (n + Q4_BAND - 1) / Q4_BAND;
    size_t nib_per_band = ep_cmul(k, 64);
    size_t nbk = (k + block - 1) / block;
    size_t sc_per_band = ep_cmul(nbk, Q4_BAND);

    if (m <= Q4_GEMV_MAXR && block >= 4) {
        const f16 *a = (const f16 *)lhs;
        // The affine form's per-row, per-block sums of A. Shares the dense path's
        // per-thread A-pack buffer (this header is included into gemm_f16f16.c):
        // only one gemm call per thread is ever in flight.
        f16 *asum = NULL;
        if (mins) {
            asum = apack_scratch(ep_cmul(ep_cmul(m, nbk), sizeof(f16)));
            if (!asum) return -1;
            for (size_t i = 0; i < m * nbk; i++) {
                size_t r = i / nbk, b = i % nbk;
                float acc = 0.0f;
                size_t d1 = (b + 1) * block < k ? (b + 1) * block : k;
                for (size_t d = b * block; d < d1; d++)
                    acc += (float)a[r * k + d];
                asum[i] = (f16)acc;
            }
        }
        const f16 *sc = (const f16 *)scales, *mn = (const f16 *)mins;
        // Bands a dispatched chunk: one pass's worth, but small enough that the
        // E-cluster's slower unit does not hold the call up on a big chunk.
        size_t chunk = 8 / m < 2 ? 8 / m : 2;
        size_t g_chunks = (n_bands + chunk - 1) / chunk;
        if (chain || ep_flops(m, n, k) < (1u << 21) || g_chunks < 3) {
            run_q4_gemv((f16 *)dst, a, asum, m, nibbles, sc, mn, n, k, 0, n_bands, nib_per_band,
                        sc_per_band, bshift, g_q4_lut, ep, chain);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t b0 = ci * chunk;
          size_t b1 = b0 + chunk < n_bands ? b0 + chunk : n_bands;
          run_q4_gemv((f16 *)dst, a, asum, m, nibbles, sc, mn, n, k, b0, b1, nib_per_band,
                      sc_per_band, bshift, g_q4_lut, ep, NULL);
        });
        return 0;
    }
    // The MOPA path reads all of A up front and stores C at the end.
    if (chain && chain->ready)
        while (atomic_load_explicit(chain->ready, memory_order_acquire) < k) {}
    return run_q4_panels((f16 *)dst, (const f16 *)lhs, nibbles, (const f16 *)scales,
                         (const f16 *)mins, m, n, k, bshift, ep);
}

// A trailing gelu/silu/sigmoid/tanh runs as a NEON pass over the output after
// the kernel (neon_act.h): in a streaming epilogue it costs ~2.5 ns an output.
int gemm_sme_f16f16_q4(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                       const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                       size_t block, const ep_desc16 *ep) {
    ep_desc16 rest;
    uint32_t act = na_split(ep, &rest);
    const ep_desc16 *kep = na_kernel_ep(ep, act, &rest);
    int rc = q4_core(m, n, k, dst, lhs, nibbles, scales, mins, block, kep, NULL);
    if (rc == 0 && act) na_post_f16(dst, m, n, (long)n, 1, act);
    return rc;
}

// gemm_sme_f16f16_q4 as one link of a chain (src/mlp.rs): raises *done as
// output columns are stored and waits on *ready for A's depths (either NULL).
// Nothing is split off: a trailing activation in ep runs in the kernel.
int gemm_sme_f16f16_q4_chained(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                               const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                               size_t block, const ep_desc16 *ep, _Atomic size_t *done,
                               const _Atomic size_t *ready) {
    q4_chain chain = {done, ready};
    int rc = q4_core(m, n, k, dst, lhs, nibbles, scales, mins, block, ep, &chain);
    if (rc == 0 && done) atomic_store_explicit(done, n, memory_order_release);
    return rc;
}

#endif // SME_GEMM_F16F16_Q4_H
