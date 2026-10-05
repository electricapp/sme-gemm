// f16 Q4 path: C(f16) = A(f16) @ dequant(B), B kept 4-bit resident.
// Included at the end of gemm_f16f16.c for its `f16` typedef, `packa` and the
// dense kernels; the bf16 twin is gemm_b16b16_q4.h.
//
// B is packed Q4 tiles ([k,16] nibbles, column-order 2/byte) with per-(column,
// K-block) scales. `bshift` is log2 of the K-block size (5 => 32, the Q4_0/Q4_1
// default). `mins` is NULL for scale-only (Q4_0) or the matching offsets for the
// affine form (Q4_1 / llama.cpp *_1): w = scale*code + min.
#ifndef SME_GEMM_F16F16_Q4_H
#define SME_GEMM_F16F16_Q4_H

#include "neon_act.h"

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

// LUTI4 unpack of Q4 tiles [t0, t1) into consecutive [k][32] f16 tiles at `out`.
// ZT0 maps code -> value and one FMLA per 4 rows adds code*scale onto ZA holding
// zero or the min, so every weight rounds once, as dequant_q4_with does.
__arm_locally_streaming __arm_new("za", "zt0") static void unpack_q4_f16(
    f16 *out, const uint8_t *nibbles, const f16 *scales, const f16 *mins, size_t k, unsigned bshift,
    size_t t0, size_t t1, size_t nib_per_tile, size_t sc_per_tile) {
    svldr_zt(0, g_q4_lut);
    svbool_t p16 = svptrue_b16();
    svcount_t pn = svptrue_c16(), pn8 = svptrue_c8();
    size_t per_tile = k * 32, block = (size_t)1 << bshift, blast = (k - 1) >> bshift;
    svfloat16_t z = svdup_n_f16((f16)0.0f);
    svfloat16x4_t z4 = svcreate4(z, z, z, z);
    for (size_t t = t0; t < t1; t++) {
        const uint8_t *nb = nibbles + t * nib_per_tile;
        const f16 *sc = scales + t * sc_per_tile;
        const f16 *mn = mins ? mins + t * sc_per_tile : NULL;
        f16 *o = out + (t - t0) * per_tile;
        if (block < 4) {
            // Each row of a 4-row group has its own scale (and min) row; rows past
            // k unpack zero nibbles and are not stored, so clamp their block.
            for (size_t d = 0, bk = 0; d < k; d += 4, bk += 4 >> bshift) {
                size_t b1 = bk + (1 >> bshift), b2 = bk + (2 >> bshift), b3 = bk + (3 >> bshift);
                b1 = b1 < blast ? b1 : blast, b2 = b2 < blast ? b2 : blast,
                b3 = b3 < blast ? b3 : blast;
#define Q4_ROWS(v)                                                                                 \
    svcreate4(svld1_f16(p16, v + bk * 32), svld1_f16(p16, v + b1 * 32),                            \
              svld1_f16(p16, v + b2 * 32), svld1_f16(p16, v + b3 * 32))
                uint64_t rows = k - d < 4 ? k - d : 4;
                svuint8_t idx = svld1_u8(svwhilelt_b8((uint64_t)0, rows * 16), nb + d * 16);
                svwrite_za16_f16_vg1x4(0, mn ? Q4_ROWS(mn) : z4);
                svmla_za16_f16_vg1x4(0, svluti4_lane_zt_f16_x4(0, idx, 0), Q4_ROWS(sc));
#undef Q4_ROWS
                svst1_f16_x4(svwhilelt_c16((uint64_t)0, rows * 32, 4), o + d * 32,
                             svread_za16_f16_vg1x4(0));
            }
            continue;
        }
        // A block counter, not d0 >> bshift: Apple clang 21 crashes on that address.
        for (size_t d0 = 0, bk = 0; d0 < k; d0 += block, bk++) {
            svfloat16_t s = svld1_f16(p16, sc + bk * 32);
            svfloat16_t mv = mn ? svld1_f16(p16, mn + bk * 32) : z;
            svfloat16x4_t m4 = svcreate4(mv, mv, mv, mv);
            size_t d1 = d0 + block < k ? d0 + block : k, d = d0;
            // 32 rows = 8 groups: one ZERO (or 8 min writes), nibbles 4 vectors a load.
            for (; d + 32 <= d1; d += 32) {
                if (mn)
                    for (uint32_t g = 0; g < 8; g++)
                        svwrite_za16_f16_vg1x4(g, m4);
                else
                    svzero_za();
                svuint8x4_t n0 = svld1_u8_x4(pn8, nb + d * 16);
                svuint8x4_t n1 = svld1_u8_x4(pn8, nb + d * 16 + 256);
#define Q4_DEC(g, v) svmla_single_za16_f16_vg1x4(g, svluti4_lane_zt_f16_x4(0, v, 0), s)
                Q4_DEC(0, svget4(n0, 0)), Q4_DEC(1, svget4(n0, 1));
                Q4_DEC(2, svget4(n0, 2)), Q4_DEC(3, svget4(n0, 3));
                Q4_DEC(4, svget4(n1, 0)), Q4_DEC(5, svget4(n1, 1));
                Q4_DEC(6, svget4(n1, 2)), Q4_DEC(7, svget4(n1, 3));
#undef Q4_DEC
                for (uint32_t g = 0; g < 8; g++)
                    svst1_f16_x4(pn, o + (d + 4 * g) * 32, svread_za16_f16_vg1x4(g));
            }
            for (; d < d1; d += 4) {
                uint64_t rows = d1 - d < 4 ? d1 - d : 4;
                svuint8_t idx = svld1_u8(svwhilelt_b8((uint64_t)0, rows * 16), nb + d * 16);
                svwrite_za16_f16_vg1x4(0, m4);
                svmla_single_za16_f16_vg1x4(0, svluti4_lane_zt_f16_x4(0, idx, 0), s);
                svst1_f16_x4(svwhilelt_c16((uint64_t)0, rows * 32, 4), o + d * 32,
                             svread_za16_f16_vg1x4(0));
            }
        }
    }
}

// Q4 MOPA path (m > Q4_GEMV_MAXR): each N-block of B is unpacked once into a
// ring slot and shared by every M-chunk (run_panels).
static int run_q4_panels(f16 *dst, const f16 *a, const uint8_t *nibbles, const f16 *scales,
                         const f16 *mins, size_t m, size_t n, size_t k, unsigned bshift,
                         const ep_desc16 *ep) {
    size_t nib_per_tile = ep_cmul(k, 16);
    size_t sc_per_tile = ep_cmul((k + ((size_t)1 << bshift) - 1) >> bshift, 32);
    return run_panels(dst, 1, (long)n, a, (long)k, 1, m, n, k, 0.0f, 1.0f, 0, ep, (size_t)1 << 18,
                      ^(f16 *out, size_t t0, size_t t1) {
                        unpack_q4_f16(out, nibbles, scales, mins, k, bshift, t0, t1, nib_per_tile,
                                      sc_per_tile);
                      });
}

// Small-m Q4 GEMV: LUTI4 expands nibbles from ZT0, FMLA accumulates raw codes per
// K-block in groups 0-3, and each block folds acc*scale (+min*sum(A)) into the
// totals in groups 4-7. Group w lives in ZA64 tile w%8, so ZERO {0-3} clears acc.
#define Q4_ACC(q) ((uint32_t)((q) < 4 ? (q) : (q) + 4))
// Up to here the GEMV beats the MOPA path, whose panel unpack costs as much SME
// time as a 32-row M-tile (measured 1.45x at m=5, 1.06x at 7, 0.96x at 8).
#define Q4_GEMV_MAXR 7
__arm_locally_streaming __arm_new("za", "zt0") static void run_q4_gemv(
    f16 *dst, const f16 *abc, const f16 *asum, size_t rows, const uint8_t *nibbles,
    const f16 *scales, const f16 *mins, size_t n, size_t k, size_t nt_lo, size_t nt_hi,
    size_t nib_per_tile, size_t sc_per_tile, unsigned bshift, const uint32_t *lut,
    const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    svbool_t p8 = svptrue_b8();
    svcount_t pn = svptrue_c16();
    const svfloat16_t z16 = svdup_n_f16((f16)0.0f);
    const svfloat16x4_t zero4 = svcreate4(z16, z16, z16, z16);
    const EpNode *nodes = ep ? ep->nodes : NULL;
    uint32_t n_nodes = ep ? ep->n_nodes : 0;
    svfloat16_t vb = svdup_n_f16((f16)1.0f);
    svfloat16_t va = svdup_n_f16((f16)0.0f);
    svldr_zt(0, lut);
    size_t block = (size_t)1 << bshift;
    size_t nbk = (k + block - 1) >> bshift;
    // Rows x tiles <= 8: block sums in groups 0-3 and 8-11 (ZA64 tiles 0-3),
    // totals 4 above each (tiles 4-7). Past 4 rows a pass covers one tile.
    size_t TT = rows <= 2 ? 4 : rows <= 4 ? 2 : 1;
    size_t per_row = ep_cmul(k, 32);
    for (size_t nt = nt_lo; nt < nt_hi; nt += TT) {
        size_t T = nt_hi - nt < TT ? nt_hi - nt : TT;
        const uint8_t *nb = nibbles + nt * nib_per_tile;
        const f16 *sc = scales + nt * sc_per_tile;
        const f16 *mn = mins ? mins + nt * sc_per_tile : NULL;
        svzero_za();
        for (size_t bk = 0; bk < nbk; bk++) {
            size_t d = bk << bshift;
            size_t d1 = d + block < k ? d + block : k;
            if (rows == 1 && T == 4) {
                for (; d + 4 <= d1; d += 4) {
                    svfloat16x4_t A = svld1_f16_x4(pn, abc + d * 32);
                    svmla_za16_f16_vg1x4(
                        0, svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + d * 16), 0), A);
                    svmla_za16_f16_vg1x4(
                        1, svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + nib_per_tile + d * 16), 0),
                        A);
                    svmla_za16_f16_vg1x4(
                        2,
                        svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + 2 * nib_per_tile + d * 16), 0),
                        A);
                    svmla_za16_f16_vg1x4(
                        3,
                        svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + 3 * nib_per_tile + d * 16), 0),
                        A);
                }
            } else {
                for (; d + 4 <= d1; d += 4)
                    for (size_t t = 0; t < T; t++) {
                        svfloat16x4_t C =
                            svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + t * nib_per_tile + d * 16), 0);
                        for (size_t r = 0; r < rows; r++)
                            svmla_za16_f16_vg1x4(Q4_ACC(r * T + t), C,
                                                  svld1_f16_x4(pn, abc + r * per_row + d * 32));
                    }
            }
            if (d < d1) {
                // K tail: nibble 0 maps to 0 and A is zero past k.
                svcount_t pt = svwhilelt_c16((uint64_t)0, (uint64_t)((d1 - d) * 32), 4);
                svbool_t pb = svwhilelt_b8((uint64_t)0, (uint64_t)((d1 - d) * 16));
                for (size_t t = 0; t < T; t++) {
                    svfloat16x4_t C =
                        svluti4_lane_zt_f16_x4(0, svld1_u8(pb, nb + t * nib_per_tile + d * 16), 0);
                    for (size_t r = 0; r < rows; r++)
                        svmla_za16_f16_vg1x4(Q4_ACC(r * T + t), C,
                                              svld1_f16_x4(pt, abc + r * per_row + d * 32));
                }
            }
            for (size_t r = 0; r < rows; r++)
                for (size_t t = 0; t < T; t++) {
                    uint32_t g = Q4_ACC(r * T + t);
                    svmla_single_za16_f16_vg1x4(g + 4, svread_za16_f16_vg1x4(g),
                                                 svld1_f16(p16, sc + t * sc_per_tile + bk * 32));
                    if (mn)
                        svmla_single_za16_f16_vg1x4(
                            g + 4,
                            svset4_f16(zero4, 0, svld1_f16(p16, mn + t * sc_per_tile + bk * 32)),
                            svdup_n_f16(asum[r * nbk + bk]));
                }
            svzero_mask_za(0x0F);
        }
        for (size_t r = 0; r < rows; r++)
            for (size_t t = 0; t < T; t++) {
                size_t n0 = (nt + t) * 32;
                size_t ncols = n - n0 < 32 ? n - n0 : 32;
                svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                svfloat16x4_t q = svread_za16_f16_vg1x4(Q4_ACC(r * T + t) + 4);
                svfloat16_t acc = svadd_f16_x(p16, svadd_f16_x(p16, svget4(q, 0), svget4(q, 1)),
                                         svadd_f16_x(p16, svget4(q, 2), svget4(q, 3)));
                f16 *out = dst + r * n + n0;
                if (!n_nodes) {
                    svst1_f16(pst, out, acc);
                } else {
#define EP_Q4G_RD(s) acc
                    EP_STORE_TILE_ROWMAJOR_F16(p16, pst, out, (long)n, EP_Q4G_RD, 1, vb, va, 0, nodes,
                                               n_nodes, r, n0);
#undef EP_Q4G_RD
                }
            }
    }
}

// Depths d..d+3 of one A row, each broadcast across a vector: one 16-byte
// replicating load and four indexed DUPs. Lanes at or past k load as zero,
// which the K tail relies on.
#define Q4_A4(ar, d, k)                                                                            \
    ({                                                                                             \
        svfloat16_t q_ = svld1rq_f16(svwhilelt_b16((uint64_t)(d), (uint64_t)(k)), (ar) + (d));     \
        svcreate4(svdup_lane_f16(q_, 0), svdup_lane_f16(q_, 1), svdup_lane_f16(q_, 2),             \
                  svdup_lane_f16(q_, 3));                                                          \
    })
// One-row Q4 GEMV (run_q4_gemv at m = 1). It builds A's broadcasts in the kernel
// (Q4_A4) instead of reading a [k][32] broadcast, whose build and per-pass
// reloads dominate small calls (~1.5-1.8x at 1024^2 and under). The DUPs are
// streaming ALU work, so they only pay shared across a pass of 8 tiles; m > 1
// passes fewer tiles per broadcast and keeps the materialized form.
__arm_locally_streaming __arm_new("za", "zt0") static void run_q4_gemv1(
    f16 *dst, const f16 *a, const f16 *asum, const uint8_t *nibbles, const f16 *scales,
    const f16 *mins, size_t n, size_t k, size_t nt_lo, size_t nt_hi, size_t nib_per_tile,
    size_t sc_per_tile, unsigned bshift, const uint32_t *lut, const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    svbool_t p8 = svptrue_b8();
    const svfloat16_t z16 = svdup_n_f16((f16)0.0f);
    const svfloat16x4_t zero4 = svcreate4(z16, z16, z16, z16);
    const EpNode *nodes = ep ? ep->nodes : NULL;
    uint32_t n_nodes = ep ? ep->n_nodes : 0;
    svfloat16_t vb = svdup_n_f16((f16)1.0f);
    svfloat16_t va = svdup_n_f16((f16)0.0f);
    svldr_zt(0, lut);
    size_t block = (size_t)1 << bshift;
    size_t nbk = (k + block - 1) >> bshift;
    // One row takes 8 tiles a pass (block sums in groups 0-3 and 8-11, totals 4
    // above each), so each Q4_A4 is shared by all 8.
    const size_t TT = 8;
    for (size_t nt = nt_lo; nt < nt_hi; nt += TT) {
        size_t T = nt_hi - nt < TT ? nt_hi - nt : TT;
        const uint8_t *nb = nibbles + nt * nib_per_tile;
        const f16 *sc = scales + nt * sc_per_tile;
        const f16 *mn = mins ? mins + nt * sc_per_tile : NULL;
        svzero_za();
        for (size_t bk = 0; bk < nbk; bk++) {
            size_t d = bk << bshift;
            size_t d1 = d + block < k ? d + block : k;
            if (T == 8) {
                for (; d + 4 <= d1; d += 4) {
                    svfloat16x4_t A = Q4_A4(a, d, k);
#define Q4_T1(t)                                                                                   \
    svmla_za16_f16_vg1x4(                                                                          \
        Q4_ACC(t), svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + (t) * nib_per_tile + d * 16), 0),   \
        A)
                    Q4_T1(0), Q4_T1(1), Q4_T1(2), Q4_T1(3);
                    Q4_T1(4), Q4_T1(5), Q4_T1(6), Q4_T1(7);
#undef Q4_T1
                }
            } else {
                for (; d + 4 <= d1; d += 4) {
                    svfloat16x4_t A = Q4_A4(a, d, k);
                    for (size_t t = 0; t < T; t++)
                        svmla_za16_f16_vg1x4(
                            Q4_ACC(t),
                            svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + t * nib_per_tile + d * 16),
                                                   0),
                            A);
                }
            }
            if (d < d1) {
                // K tail: nibble 0 maps to 0 and A is zero past k.
                svbool_t pb = svwhilelt_b8((uint64_t)0, (uint64_t)((d1 - d) * 16));
                for (size_t t = 0; t < T; t++) {
                    svfloat16x4_t C =
                        svluti4_lane_zt_f16_x4(0, svld1_u8(pb, nb + t * nib_per_tile + d * 16), 0);
                    svmla_za16_f16_vg1x4(Q4_ACC(t), C, Q4_A4(a, d, k));
                }
            }
            for (size_t t = 0; t < T; t++) {
                uint32_t g = Q4_ACC(t);
                svmla_single_za16_f16_vg1x4(g + 4, svread_za16_f16_vg1x4(g),
                                            svld1_f16(p16, sc + t * sc_per_tile + bk * 32));
                if (mn)
                    svmla_single_za16_f16_vg1x4(
                        g + 4, svset4_f16(zero4, 0, svld1_f16(p16, mn + t * sc_per_tile + bk * 32)),
                        svdup_n_f16(asum[bk]));
            }
            svzero_mask_za(0x0F);
        }
        for (size_t t = 0; t < T; t++) {
            size_t n0 = (nt + t) * 32;
            size_t ncols = n - n0 < 32 ? n - n0 : 32;
            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
            svfloat16x4_t q = svread_za16_f16_vg1x4(Q4_ACC(t) + 4);
            svfloat16_t acc = svadd_f16_x(p16, svadd_f16_x(p16, svget4(q, 0), svget4(q, 1)),
                                          svadd_f16_x(p16, svget4(q, 2), svget4(q, 3)));
            f16 *out = dst + n0;
            if (!n_nodes) {
                svst1_f16(pst, out, acc);
            } else {
#define EP_Q4G_RD(s) acc
                EP_STORE_TILE_ROWMAJOR_F16(p16, pst, out, (long)n, EP_Q4G_RD, 1, vb, va, 0, nodes,
                                           n_nodes, 0, n0);
#undef EP_Q4G_RD
            }
        }
    }
}

// Q4 GEMM: C(f16) = A(f16) @ dequant(B). B stays 4-bit resident (nibbles
// tile-major [n_tiles][k][16 bytes], column-order 2/byte) plus scales
// ([n_tiles][ceil(k/block)][32] f16); f16 exists only one N-block panel at a time.
// `block` is the K-block size (power of two; 32 = Q4_0/Q4_1). `mins` is NULL for
// the scale-only form, else the matching offsets: w = scale*code + min. `ep` is
// the fused f16 op-graph applied in-register at the store, or NULL.
static int q4_core(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                   const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                   size_t block, const ep_desc16 *ep) {
    if (m == 0 || n == 0) return 0;
    if (block == 0 || (block & (block - 1)) != 0) return -1; // power of two only
    unsigned bshift = 0;
    while (((size_t)1 << bshift) < block) bshift++;
    static dispatch_once_t lut_once;
    dispatch_once_f(&lut_once, NULL, (dispatch_function_t)q4_lut_init);
    size_t n_tiles = (n + 31) / 32;
    size_t nib_per_tile = ep_cmul(k, 16);
    size_t nbk = (k + block - 1) / block;
    size_t sc_per_tile = ep_cmul(nbk, 32);

    // Shares the dense path's per-thread A-pack buffer (this header is included
    // into gemm_f16f16.c): only one gemm call per thread is ever in flight, so
    // the two uses cannot overlap.
    if (m <= Q4_GEMV_MAXR && block >= 4) {
        size_t nbk_s = mins ? m * nbk : 0;
        // [m][k][32] broadcast A (m > 1; one row broadcasts in the kernel), then
        // the per-row, per-block sums of A.
        size_t per_row = m > 1 ? ep_cmul(k, 32) : 0;
        f16 *abc = apack_scratch(ep_cmul(ep_cmul(m, per_row) + nbk_s + 1, sizeof(f16)));
        if (!abc) return -1;
        const f16 *a = (const f16 *)lhs;
        if (m > 1)
            for (size_t r = 0; r < m; r++)
                bcast_row(abc + r * per_row, a + r * k, 1, k);
        f16 *asum = abc + m * per_row;
        for (size_t i = 0; i < nbk_s; i++) {
            size_t r = i / nbk, b = i % nbk;
            float acc = 0.0f;
            size_t d1 = (b + 1) * block < k ? (b + 1) * block : k;
            for (size_t d = b * block; d < d1; d++) acc += (float)a[r * k + d];
            asum[i] = (f16)acc;
        }
        const f16 *sc = (const f16 *)scales, *mn = (const f16 *)mins;
        size_t G_CHUNK = m == 1 ? 8 : 4; // tiles per chunk: one GEMV pass
        size_t g_chunks = (n_tiles + G_CHUNK - 1) / G_CHUNK;
        if (ep_flops(m, n, k) < (1u << 21) || g_chunks < 3) {
            if (m == 1)
                run_q4_gemv1((f16 *)dst, a, asum, nibbles, sc, mn, n, k, 0, n_tiles, nib_per_tile,
                             sc_per_tile, bshift, g_q4_lut, ep);
            else
                run_q4_gemv((f16 *)dst, abc, asum, m, nibbles, sc, mn, n, k, 0, n_tiles,
                            nib_per_tile, sc_per_tile, bshift, g_q4_lut, ep);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t nt0 = ci * G_CHUNK;
          size_t nt1 = nt0 + G_CHUNK < n_tiles ? nt0 + G_CHUNK : n_tiles;
          if (m == 1)
              run_q4_gemv1((f16 *)dst, a, asum, nibbles, sc, mn, n, k, nt0, nt1, nib_per_tile,
                           sc_per_tile, bshift, g_q4_lut, ep);
          else
              run_q4_gemv((f16 *)dst, abc, asum, m, nibbles, sc, mn, n, k, nt0, nt1, nib_per_tile,
                          sc_per_tile, bshift, g_q4_lut, ep);
        });
        return 0;
    }
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
    int rc = q4_core(m, n, k, dst, lhs, nibbles, scales, mins, block, kep);
    if (rc == 0 && act) NA_NEON_PHASE(na_post_f16(dst, m, n, (long)n, 1, act));
    return rc;
}

#endif // SME_GEMM_F16F16_Q4_H
