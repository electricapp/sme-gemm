// 1- and 2-bit weights on the f16 MOPA path: C(f16) = A(f16) @ dequant(B), B
// resident in gemm_q2.cc's band-major layout. Included at the end of
// gemm_f16f16.cc for its `f16` typedef and gemm_half.h's run_panels.
//
// Each N-block of B is unpacked into [k][32] f16 panels in a ring slot and shared
// by every M-chunk (run_panels), so the products are the eager dequant plus the
// dense f16 GEMM: a 1-bit weight is its scale, sign flipped where the bit is
// clear (exact), a 2-bit one f16(scale * codebook[code]), rounded once. The
// unpack is NEON, off the SME unit the GEMM keeps busy, and stores with STNP so
// the unit's first read of a panel line has no core to wait on (neon_act.h).
//
// The layout (gemm_q2.cc): bands of LB_BAND = 64 columns, a band's codes for
// each quad of depths in 32 bytes (1-bit: column c's four signs in nibble c, low
// nibble first, depth 4q+i at bit i, set for +scale) or 64 (2-bit: byte c holds
// column c's four codes, depth 4q+i at bits 2i), k padded to a multiple of 16.
// Scales are [band][nbk][64] f16, column c of a block's 64 at
// 32(c/32) + 2(c%16) + (c%32)/16, so one VLD2 of a tile's 32 deinterleaves them
// into columns 0-7 and 16-23, the next into 8-15 and 24-31.
#ifndef SME_GEMM_F16F16_LOWBIT_H
#define SME_GEMM_F16F16_LOWBIT_H

#include "neon_act.h"

#define LB_BAND 64

// The scales of tile t's columns 8v..8v+7 for K-block blk, bits XOR `flip`.
static inline void lb_scales(uint16x8_t s[4], const uint16_t *sc, size_t blk, uint16_t flip) {
    uint16x8x2_t lo = vld2q_u16(sc + blk * LB_BAND), hi = vld2q_u16(sc + blk * LB_BAND + 16);
    uint16x8_t f = vdupq_n_u16(flip);
    s[0] = veorq_u16(lo.val[0], f);
    s[1] = veorq_u16(hi.val[0], f);
    s[2] = veorq_u16(lo.val[1], f);
    s[3] = veorq_u16(hi.val[1], f);
}

// One depth of a 1-bit tile: the negated scales with the sign flipped back
// where tb (bit i of each nibble, as the f16 sign byte) is set.
static inline void lb_q1_row(f16 *row, const uint8x16_t ix[4], const uint16x8_t ns[4],
                             uint8x16_t tb) {
    uint16x8_t v[4];
    for (int c = 0; c < 4; c++)
        v[c] = veorq_u16(ns[c], vreinterpretq_u16_u8(vqtbl1q_u8(tb, ix[c])));
    na_stnp16(row, vreinterpretq_f16_u16(v[0]), vreinterpretq_f16_u16(v[1]));
    na_stnp16(row + 16, vreinterpretq_f16_u16(v[2]), vreinterpretq_f16_u16(v[3]));
}

// 1-bit tiles [t0, t1) into consecutive [k][32] f16 panels at `out`. Per quad,
// each column's nibble goes to the high byte of a u16 lane of TBL indices (the
// low byte 0xff, which TBL reads as 0); per depth, one TBL per 8 columns gives
// 0x8000 where the sign is set, XORed onto -scale.
static void unpack_q1_f16(f16 *out, const uint8_t *signs, const f16 *scales, size_t k,
                          unsigned bshift, size_t t0, size_t t1, size_t band_bytes,
                          size_t sc_per_band) {
    uint8x16_t tb[4];
    for (int i = 0; i < 4; i++) {
        uint8_t t[16];
        for (int e = 0; e < 16; e++)
            t[e] = e >> i & 1 ? 0x80 : 0;
        tb[i] = vld1q_u8(t);
    }
    const uint8x16_t ff = vdupq_n_u8(0xff), lo4 = vdupq_n_u8(0x0f);
    for (size_t t = t0; t < t1; t++) {
        const uint8_t *sg = signs + (t / 2) * band_bytes + 16 * (t % 2);
        const uint16_t *sc = (const uint16_t *)scales + (t / 2) * sc_per_band + 32 * (t % 2);
        f16 *o = out + (t - t0) * k * 32;
        uint16x8_t ns[4];
        size_t blk = SIZE_MAX;
        for (size_t d0 = 0; d0 < k; d0 += 4) {
            if (d0 >> bshift != blk) {
                blk = d0 >> bshift;
                lb_scales(ns, sc, blk, 0x8000);
            }
            uint8x16_t b = vld1q_u8(sg + d0 * 8);
            uint8x16_t lo = vandq_u8(b, lo4), hi = vshrq_n_u8(b, 4);
            uint8x16_t n0 = vzip1q_u8(lo, hi), n1 = vzip2q_u8(lo, hi);
            uint8x16_t ix[4] = {vzip1q_u8(ff, n0), vzip2q_u8(ff, n0), vzip1q_u8(ff, n1),
                                vzip2q_u8(ff, n1)};
            f16 *row = o + d0 * 32;
            if (k - d0 >= 4) {
                lb_q1_row(row, ix, ns, tb[0]);
                lb_q1_row(row + 32, ix, ns, tb[1]);
                lb_q1_row(row + 64, ix, ns, tb[2]);
                lb_q1_row(row + 96, ix, ns, tb[3]);
            } else {
                for (size_t i = 0; i < k - d0; i++)
                    lb_q1_row(row + 32 * i, ix, ns, tb[i]);
            }
        }
    }
}

// One depth of a 2-bit tile: per 16 columns, codes c (as bytes) look up the
// codebook's low and high f16 bytes, zipped into lanes and scaled.
static inline void lb_q2_row(f16 *row, uint8x16_t c0, uint8x16_t c1, uint8x16_t tlo, uint8x16_t thi,
                             const uint16x8_t s[4]) {
    uint8x16_t c[2] = {c0, c1};
    float16x8_t v[4];
    for (int h = 0; h < 2; h++) {
        uint8x16_t lo = vqtbl1q_u8(tlo, c[h]), hi = vqtbl1q_u8(thi, c[h]);
        v[2 * h] =
            vmulq_f16(vreinterpretq_f16_u8(vzip1q_u8(lo, hi)), vreinterpretq_f16_u16(s[2 * h]));
        v[2 * h + 1] =
            vmulq_f16(vreinterpretq_f16_u8(vzip2q_u8(lo, hi)), vreinterpretq_f16_u16(s[2 * h + 1]));
    }
    na_stnp16(row, v[0], v[1]);
    na_stnp16(row + 16, v[2], v[3]);
}

// 2-bit tiles [t0, t1) into consecutive [k][32] f16 panels at `out`: each
// weight one f16 multiply of its codebook value by its scale.
static void unpack_q2_f16(f16 *out, const uint8_t *codes, const f16 *scales, const int8_t *codebook,
                          size_t k, unsigned bshift, size_t t0, size_t t1, size_t band_bytes,
                          size_t sc_per_band) {
    uint8_t lo_b[16] = {0}, hi_b[16] = {0};
    for (int v = 0; v < 4; v++) {
        f16 h = (f16)(float)codebook[v];
        uint16_t bits;
        memcpy(&bits, &h, 2);
        lo_b[v] = (uint8_t)bits;
        hi_b[v] = (uint8_t)(bits >> 8);
    }
    const uint8x16_t tlo = vld1q_u8(lo_b), thi = vld1q_u8(hi_b), three = vdupq_n_u8(3);
    for (size_t t = t0; t < t1; t++) {
        const uint8_t *cd = codes + (t / 2) * band_bytes + 32 * (t % 2);
        const uint16_t *sc = (const uint16_t *)scales + (t / 2) * sc_per_band + 32 * (t % 2);
        f16 *o = out + (t - t0) * k * 32;
        uint16x8_t s[4];
        size_t blk = SIZE_MAX;
        for (size_t d0 = 0; d0 < k; d0 += 4) {
            if (d0 >> bshift != blk) {
                blk = d0 >> bshift;
                lb_scales(s, sc, blk, 0);
            }
            uint8x16_t b0 = vld1q_u8(cd + d0 * 16), b1 = vld1q_u8(cd + d0 * 16 + 16);
            f16 *row = o + d0 * 32;
            lb_q2_row(row, vandq_u8(b0, three), vandq_u8(b1, three), tlo, thi, s);
            if (k - d0 > 1)
                lb_q2_row(row + 32, vandq_u8(vshrq_n_u8(b0, 2), three),
                          vandq_u8(vshrq_n_u8(b1, 2), three), tlo, thi, s);
            if (k - d0 > 2)
                lb_q2_row(row + 64, vandq_u8(vshrq_n_u8(b0, 4), three),
                          vandq_u8(vshrq_n_u8(b1, 4), three), tlo, thi, s);
            if (k - d0 > 3) lb_q2_row(row + 96, vshrq_n_u8(b0, 6), vshrq_n_u8(b1, 6), tlo, thi, s);
        }
    }
}

// bits 1: `codes` are signs and `codebook` unused; bits 2: codes and codebook.
static int lowbit_panels(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                         const uint8_t *codes, const uint16_t *scales, unsigned bits,
                         const int8_t *codebook, size_t block, const ep_desc16 *ep) {
    if (m == 0 || n == 0) return 0;
    if (k == 0 || block < 16 || (block & (block - 1)) != 0) return -1;
    unsigned bshift = (unsigned)__builtin_ctzl(block);
    size_t kp = (k + 15) & ~(size_t)15, nbk = (k + block - 1) >> bshift;
    size_t band_bytes = ep_cmul(kp, 8 * bits), sc_per_band = ep_cmul(nbk, LB_BAND);
    const f16 *a = (const f16 *)lhs, *sc = (const f16 *)scales;
    if (bits == 1)
        return run_panels((f16 *)dst, 1, (long)n, a, (long)k, 1, m, n, k, 0.0f, 1.0f, 0, ep,
                          (size_t)1 << 18, ^(f16 *out, size_t t0, size_t t1) {
                            unpack_q1_f16(out, codes, sc, k, bshift, t0, t1, band_bytes,
                                          sc_per_band);
                          });
    return run_panels((f16 *)dst, 1, (long)n, a, (long)k, 1, m, n, k, 0.0f, 1.0f, 0, ep,
                      (size_t)1 << 18, ^(f16 *out, size_t t0, size_t t1) {
                        unpack_q2_f16(out, codes, sc, codebook, k, bshift, t0, t1, band_bytes,
                                      sc_per_band);
                      });
}

// A trailing gelu/silu/sigmoid/tanh runs as a NEON pass after the kernel, as
// for gemm_sme_f16f16_q4.
static int lowbit_entry(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                        const uint8_t *codes, const uint16_t *scales, unsigned bits,
                        const int8_t *codebook, size_t block, const ep_desc16 *ep) {
    ep_desc16 rest;
    uint32_t act = na_split(ep, &rest);
    const ep_desc16 *kep = na_kernel_ep(ep, act, &rest);
    int rc = lowbit_panels(m, n, k, dst, lhs, codes, scales, bits, codebook, block, kep);
    if (rc == 0 && act) na_post_f16(dst, m, n, (long)n, 1, act);
    return rc;
}

int gemm_sme_f16f16_q1(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                       const uint8_t *signs, const uint16_t *scales, size_t block,
                       const ep_desc16 *ep) {
    return lowbit_entry(m, n, k, dst, lhs, signs, scales, 1, NULL, block, ep);
}

int gemm_sme_f16f16_q2(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                       const uint8_t *codes, const uint16_t *scales, const int8_t *codebook,
                       size_t block, const ep_desc16 *ep) {
    return lowbit_entry(m, n, k, dst, lhs, codes, scales, 2, codebook, block, ep);
}

#endif // SME_GEMM_F16F16_LOWBIT_H
