// Elementwise passes the Rust layer calls between kernels: plain NEON, on
// the calling thread or spread over the cores for large outputs.
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "neon_act.h"

void neon_glu_f16(__fp16 *out, const __fp16 *in, size_t m, size_t n, uint32_t act);
void neon_f32_to_f16(__fp16 *dst, const float *src, size_t n);
void neon_f16_to_f32(float *dst, const __fp16 *src, size_t n);
void neon_add_f16_to_f32(float *dst, const __fp16 *src, size_t n);
void neon_act_f16(__fp16 *dst, const __fp16 *src, size_t n, uint32_t act);
void neon_norm_row_f32(float *y, const float *x, const float *w, const float *b, size_t n,
                       float eps, int rms);
void neon_norm_row_f16(__fp16 *y, const float *x, const float *w, const float *b, size_t n,
                       float eps, int rms);

// Sum of x (or of (x - c)^2 when sq), 16 lanes at a time in four chains.
static float row_sum(const float *x, size_t n, float c, int sq) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = s0, s2 = s0, s3 = s0, vc = vdupq_n_f32(c);
    size_t i = 0;
#define TERM(o)                                                                                    \
    (sq ? vmulq_f32(vsubq_f32(vld1q_f32(x + i + (o)), vc), vsubq_f32(vld1q_f32(x + i + (o)), vc))  \
        : vld1q_f32(x + i + (o)))
    for (; i + 16 <= n; i += 16) {
        s0 = vaddq_f32(s0, TERM(0));
        s1 = vaddq_f32(s1, TERM(4));
        s2 = vaddq_f32(s2, TERM(8));
        s3 = vaddq_f32(s3, TERM(12));
    }
#undef TERM
    float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++)
        s += sq ? (x[i] - c) * (x[i] - c) : x[i];
    return s;
}

// One row of nn::layer_norm (rms = 0: (x - mean) / sqrt(var + eps) * w (+ b),
// mean and variance in two passes) or nn::rms_norm (rms = 1: x / sqrt(mean(x^2)
// + eps) * w); b may be NULL.
void neon_norm_row_f32(float *y, const float *x, const float *w, const float *b, size_t n,
                       float eps, int rms) {
    float mean = rms ? 0.0f : row_sum(x, n, 0.0f, 0) / (float)n;
    float r = 1.0f / sqrtf(row_sum(x, n, mean, 1) / (float)n + eps);
    float32x4_t vm = vdupq_n_f32(mean), vr = vdupq_n_f32(r);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vmulq_f32(vmulq_f32(vsubq_f32(vld1q_f32(x + i), vm), vr), vld1q_f32(w + i));
        vst1q_f32(y + i, b ? vaddq_f32(v, vld1q_f32(b + i)) : v);
    }
    for (; i < n; i++)
        y[i] = (x[i] - mean) * r * w[i] + (b ? b[i] : 0.0f);
}

// neon_norm_row_f32 rounded to f16 as it goes, stored non-temporally for the
// GEMV that reads it next: the same bits as the f32 row then neon_f32_to_f16,
// without the f32 row.
void neon_norm_row_f16(__fp16 *y, const float *x, const float *w, const float *b, size_t n,
                       float eps, int rms) {
    float mean = rms ? 0.0f : row_sum(x, n, 0.0f, 0) / (float)n;
    float r = 1.0f / sqrtf(row_sum(x, n, mean, 1) / (float)n + eps);
    float32x4_t vm = vdupq_n_f32(mean), vr = vdupq_n_f32(r);
    size_t i = 0;
#define NORM4(o)                                                                                   \
    (b ? vaddq_f32(vmulq_f32(vmulq_f32(vsubq_f32(vld1q_f32(x + i + (o)), vm), vr),                 \
                             vld1q_f32(w + i + (o))),                                              \
                   vld1q_f32(b + i + (o)))                                                         \
       : vmulq_f32(vmulq_f32(vsubq_f32(vld1q_f32(x + i + (o)), vm), vr), vld1q_f32(w + i + (o))))
    for (; i + 16 <= n; i += 16)
        na_stnp16(y + i, vcvt_high_f16_f32(vcvt_f16_f32(NORM4(0)), NORM4(4)),
                  vcvt_high_f16_f32(vcvt_f16_f32(NORM4(8)), NORM4(12)));
#undef NORM4
    for (; i < n; i++)
        y[i] = (__fp16)((x[i] - mean) * r * w[i] + (b ? b[i] : 0.0f));
}

// dst = act(src) over n contiguous values, dst stored non-temporally for the
// next kernel (src/mlp.rs, on a hidden layer as the GEMV produces it). Out of
// place on purpose: a line this core has just read stays in its cache when
// rewritten even by STNP, and an SME read from another core then waits on it.
void neon_act_f16(__fp16 *dst, const __fp16 *src, size_t n, uint32_t act) {
    size_t j = 0;
#define ACT8(h)                                                                                    \
    vcvt_high_f16_f32(vcvt_f16_f32(na_apply(vcvt_f32_f16(vget_low_f16(h)), act)),                  \
                      na_apply(vcvt_high_f32_f16(h), act))
    for (; j + 16 <= n; j += 16) {
        float16x8_t h0 = vld1q_f16(src + j), h1 = vld1q_f16(src + j + 8);
        na_stnp16(dst + j, ACT8(h0), ACT8(h1));
    }
    for (; j + 8 <= n; j += 8)
        vst1q_f16(dst + j, ACT8(vld1q_f16(src + j)));
#undef ACT8
    for (; j < n; j++)
        dst[j] = (__fp16)vgetq_lane_f32(na_apply(vdupq_n_f32((float)src[j]), act), 0);
}

// The f16 <-> f32 conversions at a model's edges, 16 values a step (FCVTN /
// FCVTL, round to nearest even, the same bits as a scalar FCVT). The f16 side
// is usually the next kernel's input, so it is stored non-temporally.
void neon_f32_to_f16(__fp16 *dst, const float *src, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float16x8_t lo =
            vcvt_high_f16_f32(vcvt_f16_f32(vld1q_f32(src + i)), vld1q_f32(src + i + 4));
        float16x8_t hi =
            vcvt_high_f16_f32(vcvt_f16_f32(vld1q_f32(src + i + 8)), vld1q_f32(src + i + 12));
        na_stnp16(dst + i, lo, hi);
    }
    for (; i < n; i++)
        dst[i] = (__fp16)src[i];
}

void neon_f16_to_f32(float *dst, const __fp16 *src, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float16x8_t lo = vld1q_f16(src + i), hi = vld1q_f16(src + i + 8);
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(lo)));
        vst1q_f32(dst + i + 4, vcvt_high_f32_f16(lo));
        vst1q_f32(dst + i + 8, vcvt_f32_f16(vget_low_f16(hi)));
        vst1q_f32(dst + i + 12, vcvt_high_f32_f16(hi));
    }
    for (; i < n; i++)
        dst[i] = (float)src[i];
}

// dst += src: an f16 matmul output added into an f32 residual stream.
void neon_add_f16_to_f32(float *dst, const __fp16 *src, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float16x8_t lo = vld1q_f16(src + i), hi = vld1q_f16(src + i + 8);
        vst1q_f32(dst + i, vaddq_f32(vld1q_f32(dst + i), vcvt_f32_f16(vget_low_f16(lo))));
        vst1q_f32(dst + i + 4, vaddq_f32(vld1q_f32(dst + i + 4), vcvt_high_f32_f16(lo)));
        vst1q_f32(dst + i + 8, vaddq_f32(vld1q_f32(dst + i + 8), vcvt_f32_f16(vget_low_f16(hi))));
        vst1q_f32(dst + i + 12, vaddq_f32(vld1q_f32(dst + i + 12), vcvt_high_f32_f16(hi)));
    }
    for (; i < n; i++)
        dst[i] += (float)src[i];
}

// One row of the gated pass: `in` holds 2n values, gate and up interleaved 32
// columns at a time (the last chunk w < 32 wide: w gate then w up).
static void glu_row(__fp16 *out, const __fp16 *in, size_t n, uint32_t act) {
    for (size_t c0 = 0; c0 < n; c0 += 32) {
        size_t w = n - c0 < 32 ? n - c0 : 32;
        const __fp16 *g = in + 2 * c0, *u = g + w;
        size_t j = 0;
#define GLU_LO(o)                                                                                  \
    vmulq_f32(na_apply(vcvt_f32_f16(vget_low_f16(vld1q_f16(g + j + (o)))), act),                   \
              vcvt_f32_f16(vget_low_f16(vld1q_f16(u + j + (o)))))
#define GLU_HI(o)                                                                                  \
    vmulq_f32(na_apply(vcvt_high_f32_f16(vld1q_f16(g + j + (o))), act),                            \
              vcvt_high_f32_f16(vld1q_f16(u + j + (o))))
#define GLU8(o) vcvt_high_f16_f32(vcvt_f16_f32(GLU_LO(o)), GLU_HI(o))
        // The output is the next matmul's input: non-temporal (neon_act.h).
        for (; j + 16 <= w; j += 16)
            na_stnp16(out + c0 + j, GLU8(0), GLU8(8));
        for (; j + 8 <= w; j += 8)
            vst1q_f16(out + c0 + j, GLU8(0));
#undef GLU8
#undef GLU_HI
#undef GLU_LO
        for (; j < w; j += 4) {
            float gb[4] = {0}, ub[4] = {0};
            size_t r = w - j < 4 ? w - j : 4;
            for (size_t t = 0; t < r; t++) gb[t] = (float)g[j + t], ub[t] = (float)u[j + t];
            vst1q_f32(gb, vmulq_f32(na_apply(vld1q_f32(gb), act), vld1q_f32(ub)));
            for (size_t t = 0; t < r; t++) out[c0 + j + t] = (__fp16)gb[t];
        }
    }
}

// out (m x n) = act(gate) * up over m rows of interleaved gate/up (m x 2n).
void neon_glu_f16(__fp16 *out, const __fp16 *in, size_t m, size_t n, uint32_t act) {
    size_t cells = m * n;
    size_t chunks = cells < NA_PAR_MIN ? 1 : cells / (NA_PAR_MIN / 2);
    if (chunks > m) chunks = m;
    if (chunks > NA_MAX_CHUNKS) chunks = NA_MAX_CHUNKS;
    if (chunks <= 1) {
        for (size_t i = 0; i < m; i++) glu_row(out + i * n, in + i * 2 * n, n, act);
        return;
    }
    dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
      for (size_t i = m * ci / chunks; i < m * (ci + 1) / chunks; i++)
          glu_row(out + i * n, in + i * 2 * n, n, act);
    });
}
