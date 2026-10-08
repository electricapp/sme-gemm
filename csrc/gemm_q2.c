// 2-bit path: C(f16) = A(f16) @ dequant(B), B kept 2-bit resident, int8
// products (SME2 LUTI2 + SDOT; MOVAZ from SME2p1). Apple M5+.
//
// B is resident band-major: bands of Q2_BAND = 64 columns, and for each group
// of four depths (a quad) a band holds 64 bytes, byte c holding column c's four
// 2-bit codes (depth 4q+i at bits 2i). One 64-byte load and one LUTI2 x4 .B
// turn a quad into 256 int8 weights -- four vectors, vector v holding columns
// 16v..16v+15 four depths each -- which is exactly the first operand of one
// indexed SDOT vg1x4. k is padded to kp, a multiple of 16; codes past k are
// never weighted (A is zero there). `lut` holds the four int8 values the codes
// map to: LUTI2 .B reads the low byte of ZT0's 32-bit entries 0-3.
// Scales are [band][nbk][64] f32, one per (column, K-block).
//
// A is quantized per (row, K-block) before the kernel: s = amax/127 and
// q = round_even(a * 127/amax), so every product is an exact int32 sum. At each
// block edge the sum folds into an f32 total as total + f32(sum) * (scale*s),
// fused, block by block in K order. Nothing in a column's result depends on how
// rows, bands or threads are split, and src/kernels/q2 computes the same bits.
#include "epilogue.h"
#include "neon_act.h"
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef __fp16 f16;

#define Q2_BAND 64

// Per-thread scratch for the quantized A and its scales. Grows monotonically;
// dispatched chunks read the caller's buffer while the caller waits.
static _Thread_local uint8_t *g_q2_scratch = NULL;
static _Thread_local size_t g_q2_cap = 0;
static uint8_t *q2_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (g_q2_cap < bytes) {
        free(g_q2_scratch);
        g_q2_scratch = (uint8_t *)malloc(bytes);
        g_q2_cap = g_q2_scratch ? bytes : 0;
    }
    return g_q2_scratch;
}

// |a| max over n halves; NaN if any is NaN (FMAX propagates it).
static float q2_amax(const f16 *a, size_t n) {
    float16x8_t m8 = vdupq_n_f16((f16)0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        m8 = vmaxq_f16(m8, vabsq_f16(vld1q_f16(a + i)));
    float m = (float)vmaxvq_f16(m8);
    for (; i < n; i++) {
        float x = fabsf((float)a[i]);
        m = (x != x || m != m) ? NAN : (x > m ? x : m);
    }
    return m;
}

// Row-major m x k f16 -> m rows of kp int8 (zero past k) and m*nbk f32 scales.
static void q2_quant(int8_t *a8, float *as, const f16 *a, size_t m, size_t k, size_t kp,
                     size_t block, size_t nbk) {
    for (size_t r = 0; r < m; r++) {
        const f16 *ar = a + r * k;
        int8_t *qr = a8 + r * kp;
        for (size_t b = 0; b < nbk; b++) {
            size_t d0 = b * block, d1 = d0 + block < k ? d0 + block : k;
            float amax = q2_amax(ar + d0, d1 - d0);
            float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            as[r * nbk + b] = amax / 127.0f;
            size_t d = d0;
            for (; d + 8 <= d1; d += 8) {
                float16x8_t h = vld1q_f16(ar + d);
                int32x4_t lo = vcvtnq_s32_f32(vmulq_n_f32(vcvt_f32_f16(vget_low_f16(h)), inv));
                int32x4_t hi = vcvtnq_s32_f32(vmulq_n_f32(vcvt_high_f32_f16(h), inv));
                vst1_s8(qr + d, vqmovn_s16(vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))));
            }
            for (; d < d1; d++) {
                float q = __builtin_roundevenf((float)ar[d] * inv);
                qr[d] = (int8_t)(q != q ? 0.0f : q > 127.0f ? 127.0f : q < -128.0f ? -128.0f : q);
            }
        }
        memset(qr + k, 0, kp - k);
    }
}

// The unit's first read of a line the core just stored waits on that core;
// loading A up front, four vectors at a time with no result used, overlaps
// those waits (as Q4_TOUCH in gemm_f16f16_q4.h).
#define Q2_TOUCH(p, bytes)                                                                         \
    for (uint64_t o_ = 0; o_ < (uint64_t)(bytes); o_ += 4 * svcntb())                              \
    __asm__ volatile("whilelt pn8.b, %0, %1, vlx4\n\tld1b {z0.b-z3.b}, pn8/z, [%2, %0]" ::"r"(o_), \
                     "r"((uint64_t)(bytes)), "r"(p)                                                \
                     : "z0", "z1", "z2", "z3", "p8", "memory")

typedef struct {
    f16 *dst;
    const int8_t *a8;
    const float *as;
    const uint8_t *codes;
    const float *scales;
    const uint32_t *lut;
    const ep_desc16 *ep;
    size_t n, kp, nbk, band_bytes, sc_per_band;
    unsigned bshift;
} q2_call;

// Bands in a pass of `rows` rows: what fits ZA (below) at full chain count.
static inline size_t q2_cap(size_t rows) {
    return rows == 1 ? 3 : rows == 2 ? 2 : 1;
}
#define Q2_MAXR 5

// ZA groups of a pass with Q2R rows, Q2B bands and Q2C sum chains per (row,
// band), Q2R*Q2B*(2*Q2C + 1) <= 16: the f32 totals of row r, band j in group
// r*Q2B + j; the int32 sums of set s (blocks alternate sets, so a block folds
// one step into the next, after its SDOTs have retired) and chain c (quads
// alternate chains, so consecutive SDOTs into a sum are Q2C*Q2R*Q2B apart).
#define Q2_TOT(r, j) ((uint32_t)((r) * Q2B + (j)))
#define Q2_SUM(r, j, c, s) ((uint32_t)(Q2R * Q2B + (((s) * Q2C + (c)) * Q2R + (r)) * Q2B + (j)))

#define Q2_LD(j)                                                                                   \
    do {                                                                                           \
        if (Q2B > (j)) {                                                                           \
            lo##j = svld1_u8_x2(pc, cb + (j) * band_bytes);                                        \
            hi##j = svld1_u8_x2(pc, cb + (j) * band_bytes + 128);                                  \
        }                                                                                          \
    } while (0)
// Quad i (0-3) of the step for band j: one LUTI2, one SDOT per row.
#define Q2_QJ(i, h, j)                                                                             \
    do {                                                                                           \
        if (Q2B > (j)) {                                                                           \
            svint8x4_t w_ = svluti2_lane_zt_s8_x4(0, svget2(h##j, (i) & 1), 0);                    \
            uint32_t g_ = Q2_SUM(0, j, (i) % Q2C, s);                                              \
            svdot_lane_za32_s8_vg1x4(g_, w_, aq0, (i));                                            \
            if (Q2R > 1) svdot_lane_za32_s8_vg1x4(g_ + Q2B, w_, aq1, (i));                         \
            if (Q2R > 2) svdot_lane_za32_s8_vg1x4(g_ + 2 * Q2B, w_, aq2, (i));                     \
            if (Q2R > 3) svdot_lane_za32_s8_vg1x4(g_ + 3 * Q2B, w_, aq3, (i));                     \
            if (Q2R > 4) svdot_lane_za32_s8_vg1x4(g_ + 4 * Q2B, w_, aq4, (i));                     \
        }                                                                                          \
    } while (0)
#define Q2_Q(i, h)                                                                                 \
    do {                                                                                           \
        Q2_QJ(i, h, 0);                                                                            \
        Q2_QJ(i, h, 1);                                                                            \
        Q2_QJ(i, h, 2);                                                                            \
    } while (0)

// Folds block `blk`'s sums (set fs) into the totals; MOVAZ leaves them zeroed.
#define Q2_FOLD(blk, fs)                                                                           \
    do {                                                                                           \
        for (size_t r_ = 0; r_ < Q2R; r_++) {                                                      \
            svfloat32_t sa_ = svdup_n_f32(as[(r0 + r_) * nbk + (blk)]);                            \
            for (size_t j_ = 0; j_ < Q2B; j_++) {                                                  \
                svint32x4_t t_ = svreadz_za32_s32_vg1x4(Q2_SUM(r_, j_, 0, fs));                    \
                if (Q2C == 2) {                                                                    \
                    svint32x4_t u_ = svreadz_za32_s32_vg1x4(Q2_SUM(r_, j_, 1, fs));                \
                    t_ = svcreate4(svadd_s32_x(p32, svget4(t_, 0), svget4(u_, 0)),                 \
                                   svadd_s32_x(p32, svget4(t_, 1), svget4(u_, 1)),                 \
                                   svadd_s32_x(p32, svget4(t_, 2), svget4(u_, 2)),                 \
                                   svadd_s32_x(p32, svget4(t_, 3), svget4(u_, 3)));                \
                }                                                                                  \
                svfloat32x4_t w_ =                                                                 \
                    svld1_f32_x4(pc32, scales + (b0 + j_) * sc_per_band + (blk) * Q2_BAND);        \
                w_ = svcreate4(                                                                    \
                    svmul_f32_x(p32, svget4(w_, 0), sa_), svmul_f32_x(p32, svget4(w_, 1), sa_),    \
                    svmul_f32_x(p32, svget4(w_, 2), sa_), svmul_f32_x(p32, svget4(w_, 3), sa_));   \
                svmla_za32_f32_vg1x4(Q2_TOT(r_, j_), svcvt_f32_s32_x4(t_), w_);                    \
            }                                                                                      \
        }                                                                                          \
    } while (0)

// Stores 32 columns (half v of band j_) of row `row`, through the epilogue
// when there is one.
#define Q2_ST(v, h)                                                                                \
    do {                                                                                           \
        size_t n0 = (b0 + j_) * Q2_BAND + (v) * 32;                                                \
        if (n0 < n) {                                                                              \
            size_t ncols = n - n0 < 32 ? n - n0 : 32;                                              \
            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);                            \
            svfloat16_t acc = (h);                                                                 \
            f16 *out = dst + row * n + n0;                                                         \
            if (!n_nodes) {                                                                        \
                svst1_f16(pst, out, acc);                                                          \
            } else {                                                                               \
                EP_STORE_TILE_ROWMAJOR_F16(p16, pst, out, (long)n, EP_Q2_RD, 1, vb, va, 0, nodes,  \
                                           n_nodes, row, n0);                                      \
            }                                                                                      \
        }                                                                                          \
    } while (0)
#define EP_Q2_RD(s) acc

// One pass: rows r0.. (RR of them) by bands b0.. (BB of them), all of K.
#define Q2_PASS(RR, BB)                                                                            \
    do {                                                                                           \
        enum { Q2R = (RR), Q2B = (BB), Q2C = (RR) * (BB) * 5 <= 16 ? 2 : 1 };                      \
        svzero_za();                                                                               \
        for (size_t bk = 0; bk < nbk; bk++) {                                                      \
            uint32_t s = (uint32_t)(bk & 1);                                                       \
            size_t d0 = bk << bshift,                                                              \
                   d1 = d0 + ((size_t)1 << bshift) < kp ? d0 + ((size_t)1 << bshift) : kp;         \
            for (size_t d = d0; d < d1; d += 16) {                                                 \
                const int8_t *ar = a8 + r0 * kp + d;                                               \
                const uint8_t *cb = codes + b0 * band_bytes + d * 16;                              \
                svint8_t aq0 = svld1rq_s8(p8, ar), aq1, aq2, aq3, aq4;                             \
                if (Q2R > 1) aq1 = svld1rq_s8(p8, ar + kp);                                        \
                if (Q2R > 2) aq2 = svld1rq_s8(p8, ar + 2 * kp);                                    \
                if (Q2R > 3) aq3 = svld1rq_s8(p8, ar + 3 * kp);                                    \
                if (Q2R > 4) aq4 = svld1rq_s8(p8, ar + 4 * kp);                                    \
                svuint8x2_t lo0, hi0, lo1, hi1, lo2, hi2;                                          \
                Q2_LD(0);                                                                          \
                Q2_LD(1);                                                                          \
                Q2_LD(2);                                                                          \
                Q2_Q(0, lo);                                                                       \
                Q2_Q(1, lo);                                                                       \
                Q2_Q(2, hi);                                                                       \
                Q2_Q(3, hi);                                                                       \
                if (d == d0 && bk > 0) Q2_FOLD(bk - 1, s ^ 1);                                     \
            }                                                                                      \
        }                                                                                          \
        if (nbk) Q2_FOLD(nbk - 1, (uint32_t)((nbk - 1) & 1));                                      \
        for (size_t r_ = 0; r_ < Q2R; r_++)                                                        \
            for (size_t j_ = 0; j_ < Q2B; j_++) {                                                  \
                size_t row = r0 + r_;                                                              \
                svfloat32x4_t t_ = svread_za32_f32_vg1x4(Q2_TOT(r_, j_));                          \
                Q2_ST(0, svcvt_f16_f32_x2(svcreate2(svget4(t_, 0), svget4(t_, 1))));               \
                Q2_ST(1, svcvt_f16_f32_x2(svcreate2(svget4(t_, 2), svget4(t_, 3))));               \
            }                                                                                      \
    } while (0)

// Rows r0..r0+rows (at most Q2_MAXR) over bands [b_lo, b_hi), in balanced
// passes of up to q2_cap(rows) bands.
__arm_locally_streaming __arm_new("za", "zt0") static void run_q2_gemv(const q2_call *c, size_t r0,
                                                                       size_t rows, size_t b_lo,
                                                                       size_t b_hi) {
    f16 *dst = c->dst;
    const int8_t *a8 = c->a8;
    const float *as = c->as;
    const uint8_t *codes = c->codes;
    const float *scales = c->scales;
    size_t n = c->n, kp = c->kp, nbk = c->nbk, band_bytes = c->band_bytes;
    size_t sc_per_band = c->sc_per_band;
    unsigned bshift = c->bshift;
    const EpNode *nodes = c->ep ? c->ep->nodes : NULL;
    uint32_t n_nodes = c->ep ? c->ep->n_nodes : 0;
    svbool_t p8 = svptrue_b8(), p16 = svptrue_b16(), p32 = svptrue_b32();
    svcount_t pc = svptrue_c8(), pc32 = svptrue_c32();
    svfloat16_t vb = svdup_n_f16((f16)1.0f);
    svfloat16_t va = svdup_n_f16((f16)0.0f);
    svldr_zt(0, c->lut);
    Q2_TOUCH(a8 + r0 * kp, rows * kp);
    size_t cap = q2_cap(rows), nbands = b_hi - b_lo;
    size_t passes = (nbands + cap - 1) / cap, per = (nbands + passes - 1) / passes;
    for (size_t b0 = b_lo; b0 < b_hi; b0 += per) {
        size_t B = b_hi - b0 < per ? b_hi - b0 : per;
        switch (rows * 4 + B) {
            case 1 * 4 + 3:
                Q2_PASS(1, 3);
                break;
            case 1 * 4 + 2:
                Q2_PASS(1, 2);
                break;
            case 1 * 4 + 1:
                Q2_PASS(1, 1);
                break;
            case 2 * 4 + 2:
                Q2_PASS(2, 2);
                break;
            case 2 * 4 + 1:
                Q2_PASS(2, 1);
                break;
            case 3 * 4 + 1:
                Q2_PASS(3, 1);
                break;
            case 4 * 4 + 1:
                Q2_PASS(4, 1);
                break;
            default:
                Q2_PASS(5, 1);
                break;
        }
    }
}
#undef Q2_PASS
#undef Q2_ST
#undef EP_Q2_RD
#undef Q2_FOLD
#undef Q2_Q
#undef Q2_QJ
#undef Q2_LD
#undef Q2_SUM
#undef Q2_TOT

// The GEMM: rows in balanced groups of at most Q2_MAXR, each a GEMV over all
// bands; spread over the cores as (group, pass-sized band chunk) items unless
// the call is small. `block` is the K-block size (a power of two, >= 16).
static int q2_core(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                   const uint8_t *codes, const float *scales, const int8_t *codebook, size_t block,
                   const ep_desc16 *ep) {
    if (m == 0 || n == 0) return 0;
    if (block < 16 || (block & (block - 1)) != 0) return -1;
    unsigned bshift = 0;
    while (((size_t)1 << bshift) < block)
        bshift++;
    size_t kp = (k + 15) & ~(size_t)15, nbk = (k + block - 1) >> bshift;
    size_t n_bands = (n + Q2_BAND - 1) / Q2_BAND;
    size_t a_bytes = ep_cmul(m, kp), s_off = (a_bytes + 63) & ~(size_t)63;
    uint8_t *scratch = q2_scratch(s_off + ep_cmul(ep_cmul(m, nbk), sizeof(float)));
    if (!scratch) return -1;
    int8_t *a8 = (int8_t *)scratch;
    float *as = (float *)(void *)(scratch + s_off);
    q2_quant(a8, as, (const f16 *)lhs, m, k, kp, block, nbk);
    uint32_t lut[16] __attribute__((aligned(64))) = {0};
    for (int i = 0; i < 4; i++)
        lut[i] = (uint8_t)codebook[i];
    q2_call c = {.dst = (f16 *)dst,
                 .a8 = a8,
                 .as = as,
                 .codes = codes,
                 .scales = scales,
                 .lut = lut,
                 .ep = ep,
                 .n = n,
                 .kp = kp,
                 .nbk = nbk,
                 .band_bytes = ep_cmul(kp, 16),
                 .sc_per_band = ep_cmul(nbk, Q2_BAND),
                 .bshift = bshift};
    size_t groups = (m + Q2_MAXR - 1) / Q2_MAXR;
    size_t chunk = q2_cap((m + groups - 1) / groups);
    size_t chunks = (n_bands + chunk - 1) / chunk;
    if (ep_flops(m, n, k) < (1u << 21) || groups * chunks < 3) {
        for (size_t g = 0; g < groups; g++) {
            size_t g0 = m * g / groups, g1 = m * (g + 1) / groups;
            run_q2_gemv(&c, g0, g1 - g0, 0, n_bands);
        }
        return 0;
    }
    const q2_call *cp = &c;
    dispatch_apply(groups * chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
      size_t g = ci / chunks, b0 = ci % chunks * chunk;
      size_t g0 = m * g / groups, g1 = m * (g + 1) / groups;
      size_t b1 = b0 + chunk < n_bands ? b0 + chunk : n_bands;
      run_q2_gemv(cp, g0, g1 - g0, b0, b1);
    });
    return 0;
}

// A trailing gelu/silu/sigmoid/tanh runs as a NEON pass over the output after
// the kernel (neon_act.h), as on the Q4 path.
int gemm_sme_f16_q2(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                    const uint8_t *codes, const float *scales, const int8_t *codebook, size_t block,
                    const ep_desc16 *ep);
int gemm_sme_f16_q2(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                    const uint8_t *codes, const float *scales, const int8_t *codebook, size_t block,
                    const ep_desc16 *ep) {
    ep_desc16 rest;
    uint32_t act = na_split(ep, &rest);
    const ep_desc16 *kep = na_kernel_ep(ep, act, &rest);
    int rc = q2_core(m, n, k, dst, lhs, codes, scales, codebook, block, kep);
    if (rc == 0 && act) na_post_f16(dst, m, n, (long)n, 1, act);
    return rc;
}
