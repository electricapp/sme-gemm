// bf16 Q4 path: C(bf16) = A(bf16) @ dequant(B), B kept 4-bit resident.
// Included at the end of gemm_b16b16.c for its `bf16` typedef, `packa` and the
// dense kernels; the f16 twin is gemm_f16f16_q4.h, which describes the
// band-major resident layout both read. Scales and mins stay f16 -- that is the
// GGUF Q4_0/Q4_1 on-disk type, and it carries more mantissa than bf16, so one
// Q4Weights serves both compute paths.
#ifndef SME_GEMM_B16B16_Q4_H
#define SME_GEMM_B16B16_Q4_H

#include "neon_act.h"

#define Q4_BAND 128

// Code c -> f16(sign-extended c), for unpack_q4_bf16's 16-bit LUTI4 (low half
// of each 32-bit entry).
static uint32_t g_q4_lut_h[16] __attribute__((aligned(64)));
static void q4_lut_h_init(void) {
    for (int c = 0; c < 16; c++) {
        ep_f16 v = (ep_f16)(float)(c < 8 ? c : c - 16);
        uint16_t bits;
        __builtin_memcpy(&bits, &v, 2);
        g_q4_lut_h[c] = bits;
    }
}

// LUTI4 unpack of tiles [t0, t1) into consecutive [k][32] bf16 panels at `out`;
// the bf16 twin of unpack_q4_f16. Per depth, FMLAL widens code*scale for the
// band's four tiles exactly into f32 ZA on top of zero or the min (one rounding,
// to f32), and BFCVTN rounds that to bf16, as dequant_q4_bf16_with does. FMLAL
// writes each vector's even and odd columns to groups w and w+1, so the min is
// seeded the same way and BFCVTN interleaves them back.
__arm_locally_streaming __arm_new("za", "zt0") static void unpack_q4_bf16(
    bf16 *out, const uint8_t *nibbles, const ep_f16 *scales, const ep_f16 *mins, size_t k,
    unsigned bshift, size_t t0, size_t t1, size_t nib_per_band, size_t sc_per_band) {
    svldr_zt(0, g_q4_lut_h);
    svbool_t p8 = svptrue_b8(), p16 = svptrue_b16(), p32 = svptrue_b32();
    svcount_t pn = svptrue_c16();
    size_t per_tile = k * 32;
    for (size_t b = t0 / 4; b * 4 < t1; b++) {
        const uint8_t *nb = nibbles + b * nib_per_band;
        const ep_f16 *sc = scales + b * sc_per_band;
        const ep_f16 *mn = mins ? mins + b * sc_per_band : NULL;
        bf16 *o[4];
        for (size_t v = 0; v < 4; v++) {
            size_t t = b * 4 + v;
            o[v] = t >= t0 && t < t1 ? out + (t - t0) * per_tile : NULL;
        }
        for (size_t d0 = 0; d0 < k; d0 += 8) {
            size_t rows = k - d0 < 8 ? k - d0 : 8;
            if (!mn) svzero_za();
            for (size_t i = 0; i < rows; i++) {
                size_t d = d0 + i, bk = d >> bshift;
                uint32_t w = (uint32_t)(2 * i);
                if (mn) {
                    svfloat16x4_t m4 = svld1_f16_x4(pn, mn + bk * Q4_BAND);
#define Q4_EV(v) svcvt_f32_f16_x(p32, svget4(m4, v))
#define Q4_OD(v) svcvtlt_f32_f16_x(p32, svget4(m4, v))
                    svwrite_za32_f32_vg1x4(w, svcreate4(Q4_EV(0), Q4_EV(1), Q4_EV(2), Q4_EV(3)));
                    svwrite_za32_f32_vg1x4(w + 1,
                                           svcreate4(Q4_OD(0), Q4_OD(1), Q4_OD(2), Q4_OD(3)));
#undef Q4_EV
#undef Q4_OD
                }
                svmla_za32_f16_vg2x4(w, svluti4_lane_zt_f16_x4(0, svld1_u8(p8, nb + d * 64), 0),
                                     svld1_f16_x4(pn, sc + bk * Q4_BAND));
            }
            for (size_t i = 0; i < rows; i++) {
                uint32_t w = (uint32_t)(2 * i);
                svfloat32x4_t ev = svread_za32_f32_vg1x4(w), od = svread_za32_f32_vg1x4(w + 1);
                size_t off = (d0 + i) * 32;
#define Q4_ST(v)                                                                                   \
    if (o[v])                                                                                      \
        svst1_bf16(p16, o[v] + off, svcvtn_bf16_f32_x2(svcreate2(svget4(ev, v), svget4(od, v))));
                Q4_ST(0) Q4_ST(1) Q4_ST(2) Q4_ST(3)
#undef Q4_ST
            }
        }
    }
}

// Q4 MOPA path; the bf16 twin of gemm_f16f16_q4.h's run_q4_panels.
static int run_q4_panels_bf16(bf16 *dst, const bf16 *a, const uint8_t *nibbles,
                              const ep_f16 *scales, const ep_f16 *mins, size_t m, size_t n,
                              size_t k, unsigned bshift, const ep_desc16 *ep) {
    size_t nib_per_band = ep_cmul(k, 64);
    size_t sc_per_band = ep_cmul((k + ((size_t)1 << bshift) - 1) >> bshift, Q4_BAND);
    return run_panels(dst, 1, (long)n, a, (long)k, 1, m, n, k, 0.0f, 1.0f, 0, ep, (size_t)1 << 18,
                      ^(bf16 *out, size_t t0, size_t t1) {
                        unpack_q4_bf16(out, nibbles, scales, mins, k, bshift, t0, t1, nib_per_band,
                                       sc_per_band);
                      });
}

// Small-m Q4 GEMV; see run_q4_gemv in gemm_f16f16_q4.h. Scales are the bf16 copies.
#define Q4_ACC(q) ((uint32_t)((q) < 4 ? (q) : (q) + 4))
// Up to here the GEMV beats the MOPA path, whose panel unpack costs as much SME
// time as a 32-row M-tile (measured 1.45x at m=5, 1.06x at 7, 0.96x at 8).
#define Q4_GEMV_MAXR 7
// Pulls A's lines in up front; see the f16 kernel.
#define Q4_TOUCH(p, bytes)                                                                         \
    for (uint64_t o_ = 0; o_ < (uint64_t)(bytes); o_ += 4 * svcntb())                              \
    __asm__ volatile("whilelt pn8.b, %0, %1, vlx4\n\tld1b {z0.b-z3.b}, pn8/z, [%2, %0]" ::"r"(o_), \
                     "r"((uint64_t)(bytes)), "r"(p)                                                \
                     : "z0", "z1", "z2", "z3", "p8", "memory")

// As the f16 kernel: with 4 bands or fewer, odd depths go to a second group.
#define Q4_D1(i, BB)                                                                               \
    for (size_t j = 0; j < (BB); j++)                                                              \
    svmla_lane_za16_bf16_vg1x4(                                                                    \
        Q4_ACC(j + ((BB) <= 4 ? 4 * ((i) & 1) : 0)),                                               \
        svluti4_lane_zt_bf16_x4(0, svld1_u8(p8, nb + j * nib_per_band + (d + (i)) * 64), 0), aq,   \
        (i))
// Depth pairs from one 128-byte load per band, as the f16 kernel.
#define Q4_L2(j, BB)                                                                               \
    if ((BB) > (j)) v##j = svld1_u8_x2(pc, q + (j) * nib_per_band)
#define Q4_F2(j, i, BB)                                                                            \
    if ((BB) > (j))                                                                                \
    svmla_lane_za16_bf16_vg1x4(Q4_ACC((j) + ((BB) <= 4 ? 4 * ((i) & 1) : 0)),                      \
                               svluti4_lane_zt_bf16_x4(0, svget2(v##j, (i) & 1), 0), aq, (i))
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
#define Q4_DR(i)                                                                                   \
    if ((i) < L)                                                                                   \
        for (size_t j = 0; j < B; j++) {                                                           \
            svbfloat16x4_t c4 = svluti4_lane_zt_bf16_x4(                                           \
                0, svld1_u8(p8, nb + j * nib_per_band + (d + (i)) * 64), 0);                       \
            for (size_t r = 0; r < rows; r++)                                                      \
                svmla_lane_za16_bf16_vg1x4(Q4_ACC(r * B + j), c4, svld1rq_bf16(pa, a + r * k + d), \
                                           (i));                                                   \
        }

__arm_locally_streaming __arm_new("za", "zt0") static void run_q4_gemv_bf16(
    bf16 *dst, const bf16 *a, const bf16 *asum, size_t rows, const uint8_t *nibbles,
    const bf16 *scales, const bf16 *mins, size_t n, size_t k, size_t b_lo, size_t b_hi,
    size_t nib_per_band, size_t sc_per_band, unsigned bshift, const uint32_t *lut,
    const ep_desc16 *ep) {
    svbool_t p8 = svptrue_b8(), p16 = svptrue_b16();
    svcount_t pn = svptrue_c16(), pc = svptrue_c8();
    const EpNode *nodes = ep ? ep->nodes : NULL;
    uint32_t n_nodes = ep ? ep->n_nodes : 0;
    svbfloat16_t vb = svdup_n_bf16((bf16)1.0f);
    svbfloat16_t va = svdup_n_bf16((bf16)0.0f);
    svldr_zt(0, lut);
    Q4_TOUCH(a, rows * k * sizeof(bf16));
    size_t block = (size_t)1 << bshift;
    size_t nbk = (k + block - 1) >> bshift;
    size_t bb = 8 / rows, nbands = b_hi - b_lo;
    size_t passes = (nbands + bb - 1) / bb, per = (nbands + passes - 1) / passes;
    for (size_t b0 = b_lo; b0 < b_hi; b0 += per) {
        size_t B = b_hi - b0 < per ? b_hi - b0 : per;
        const uint8_t *nb = nibbles + b0 * nib_per_band;
        svzero_za();
        for (size_t bk = 0; bk < nbk; bk++) {
            size_t d1 = ((bk + 1) << bshift) < k ? (bk + 1) << bshift : k;
            for (size_t d = bk << bshift; d < d1; d += 8) {
                size_t L = d1 - d < 8 ? d1 - d : 8;
                svbool_t pa = svwhilelt_b16((uint64_t)d, (uint64_t)k);
                if (rows == 1) {
                    svbfloat16_t aq = svld1rq_bf16(pa, a + d);
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
                    svbfloat16x4_t s4 = svld1_bf16_x4(pn, scales + s0);
                    svmla_za16_bf16_vg1x4(g + 4, svread_za16_bf16_vg1x4(g), s4);
                    if (rows == 1 && B <= 4)
                        svmla_za16_bf16_vg1x4(g + 4, svread_za16_bf16_vg1x4(Q4_ACC(j + 4)), s4);
                    if (mins)
                        svmla_single_za16_bf16_vg1x4(g + 4, svld1_bf16_x4(pn, mins + s0),
                                                     svdup_n_bf16(asum[r * nbk + bk]));
                }
            svzero_mask_za(0x0F);
        }
        for (size_t r = 0; r < rows; r++)
            for (size_t j = 0; j < B; j++) {
                svbfloat16x4_t q = svread_za16_bf16_vg1x4(Q4_ACC(r * B + j) + 4);
#define Q4_ST(v)                                                                                   \
    do {                                                                                           \
        size_t n0 = (b0 + j) * Q4_BAND + (v) * 32;                                                 \
        if (n0 < n) {                                                                              \
            size_t ncols = n - n0 < 32 ? n - n0 : 32;                                              \
            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);                            \
            svbfloat16_t acc = svget4(q, v);                                                       \
            bf16 *out = dst + r * n + n0;                                                          \
            if (!n_nodes) {                                                                        \
                svst1_bf16(pst, out, acc);                                                         \
            } else {                                                                               \
                EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, out, (long)n, EP_Q4G_RD, 1, vb, va, 0,       \
                                            nodes, n_nodes, r, n0);                                \
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
    }
}
#undef Q4_D1
#undef Q4_L2
#undef Q4_F2
#undef Q4_F2B
#undef Q4_P2
#undef Q4_STEP1
#undef Q4_DR

// Code c -> bf16(sign-extended c); 16-bit LUTI4 reads the low half of 32-bit entries.
static uint32_t g_q4_lut_bf16[16] __attribute__((aligned(64)));
static void q4_lut_bf16_init(void) {
    for (int c = 0; c < 16; c++) {
        bf16 v = (bf16)(float)(c < 8 ? c : c - 16);
        uint16_t bits;
        __builtin_memcpy(&bits, &v, 2);
        g_q4_lut_bf16[c] = bits;
    }
}

// Q4 GEMM, bf16 accumulation (M5+ FEAT_SME_B16B16): the bf16 twin of
// gemm_sme_f16f16_q4, over the same band-major 4-bit weights and f16 scales.
// `block` is the K-block size (power of two; 32 = Q4_0/Q4_1). `mins` is NULL for
// the scale-only form, else the matching offsets: w = scale*code + min.
// `scales_bf16`/`mins_bf16` are bf16 copies for the GEMV.
static int q4_bf16_core(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                        const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                        const uint16_t *scales_bf16, const uint16_t *mins_bf16, size_t block,
                        const ep_desc16 *ep) {
    if (m == 0 || n == 0) return 0;
    if (block == 0 || (block & (block - 1)) != 0) return -1; // power of two only
    unsigned bshift = 0;
    while (((size_t)1 << bshift) < block)
        bshift++;
    size_t n_bands = (n + Q4_BAND - 1) / Q4_BAND;
    size_t nib_per_band = ep_cmul(k, 64);
    size_t nbk = (k + block - 1) / block;
    size_t sc_per_band = ep_cmul(nbk, Q4_BAND);

    if (m <= Q4_GEMV_MAXR && block >= 4 && scales_bf16 && (!mins || mins_bf16)) {
        static dispatch_once_t lut_once;
        dispatch_once_f(&lut_once, NULL, (dispatch_function_t)q4_lut_bf16_init);
        const bf16 *a = (const bf16 *)lhs;
        // The affine form's per-row, per-block sums of A, in the dense path's
        // per-thread A-pack buffer.
        bf16 *asum = NULL;
        if (mins) {
            asum = apack_scratch(ep_cmul(ep_cmul(m, nbk), sizeof(bf16)));
            if (!asum) return -1;
            for (size_t i = 0; i < m * nbk; i++) {
                size_t r = i / nbk, b = i % nbk;
                float acc = 0.0f;
                size_t d1 = (b + 1) * block < k ? (b + 1) * block : k;
                for (size_t d = b * block; d < d1; d++)
                    acc += (float)a[r * k + d];
                asum[i] = (bf16)acc;
            }
        }
        const bf16 *sc = (const bf16 *)scales_bf16;
        const bf16 *mn = mins ? (const bf16 *)mins_bf16 : NULL;
        // Bands a dispatched chunk; see gemm_f16f16_q4.h.
        size_t chunk = 8 / m < 2 ? 8 / m : 2;
        size_t g_chunks = (n_bands + chunk - 1) / chunk;
        if (ep_flops(m, n, k) < (1u << 21) || g_chunks < 3) {
            run_q4_gemv_bf16((bf16 *)dst, a, asum, m, nibbles, sc, mn, n, k, 0, n_bands,
                             nib_per_band, sc_per_band, bshift, g_q4_lut_bf16, ep);
            return 0;
        }
        dispatch_apply(g_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t b0 = ci * chunk;
          size_t b1 = b0 + chunk < n_bands ? b0 + chunk : n_bands;
          run_q4_gemv_bf16((bf16 *)dst, a, asum, m, nibbles, sc, mn, n, k, b0, b1, nib_per_band,
                           sc_per_band, bshift, g_q4_lut_bf16, ep);
        });
        return 0;
    }
    // Shares the dense path's per-thread A-pack buffer; see gemm_f16f16_q4.h.
    static dispatch_once_t luth_once;
    dispatch_once_f(&luth_once, NULL, (dispatch_function_t)q4_lut_h_init);
    return run_q4_panels_bf16((bf16 *)dst, (const bf16 *)lhs, nibbles, (const ep_f16 *)scales,
                              (const ep_f16 *)mins, m, n, k, bshift, ep);
}

// A trailing gelu/silu/sigmoid/tanh runs as a NEON pass over the output after
// the kernel (neon_act.h): in a streaming epilogue it costs ~2.5 ns an output.
int gemm_sme_b16b16_q4(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                       const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                       const uint16_t *scales_bf16, const uint16_t *mins_bf16, size_t block,
                       const ep_desc16 *ep) {
    ep_desc16 rest;
    uint32_t act = na_split(ep, &rest);
    const ep_desc16 *kep = na_kernel_ep(ep, act, &rest);
    int rc =
        q4_bf16_core(m, n, k, dst, lhs, nibbles, scales, mins, scales_bf16, mins_bf16, block, kep);
    if (rc == 0 && act) na_post_bf16(dst, m, n, (long)n, 1, act);
    return rc;
}

#endif // SME_GEMM_B16B16_Q4_H
