// Row-softmax and flash-attention support passes (Apple aarch64, f32).
//
// These use NO SME -- hence the attn_ prefix rather than the sme_ one the GEMM
// kernels carry. They are pure elementwise exp work, and measured on M5 over
// 16M f32: scalar libm ~19 ms, streaming-SVE with the epilogue's ep_exp_f32
// ~21 ms, plain NEON dispatched across all cores ~1.2 ms. Streaming-mode SVE
// ALU throughput is tuned for MOPA rather than this, and the SME unit is shared
// per cluster, so threading a streaming pass cannot help either.
//
// Rate: 2.14 cycles/element single-core, ~5.8x across the 4P+6E cluster. The
// per-core figure is ~70% of the NEON issue limit for the ~24 vector ops each
// 4-element group costs (max sweep + exp sweep + normalize sweep). DRAM traffic
// is 8 B/element -- a row stays L1-resident across the three sweeps -- so at
// 0.095 ns/element this draws ~84 GB/s of a ~150 GB/s machine: compute-bound,
// not bandwidth-bound.

#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "attention.h"

// Below this many elements the dispatch_apply round trip costs more than the
// work it spreads.
#define ATTN_PAR_MIN (1u << 16)
#define ATTN_MAX_CHUNKS 64

// exp(x) for four lanes, ~1e-7 relative: range-reduce to 2^n * 2^r with
// r in [-0.5, 0.5], degree-5 minimax for 2^r, then scale by exponent bits.
// The input clamp keeps n inside the f32 exponent range (a NaN input clamps to
// the low bound, so NaN does not propagate -- callers pre-screen non-finite
// maxima, which is what makes that safe here).
static inline float32x4_t nexp_f32(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.33654f)), vdupq_n_f32(88.72283f));
    float32x4_t t = vmulq_n_f32(x, 1.4426950408889634f); // x * log2(e)
    float32x4_t n = vrndnq_f32(t);
    float32x4_t r = vsubq_f32(t, n);
    // 2^r = sum (r*ln2)^i / i!
    float32x4_t p = vdupq_n_f32(1.3333558e-3f);
    p = vfmaq_f32(vdupq_n_f32(9.6181291e-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.5504109e-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtnq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

// Max of a contiguous row, maxNum semantics (NaN ignored); -INFINITY if empty.
static float row_maxv(const float *rp, size_t n) {
    float32x4_t vm = vdupq_n_f32(-INFINITY);
    size_t j = 0;
    for (; j + 4 <= n; j += 4) vm = vmaxnmq_f32(vm, vld1q_f32(rp + j));
    float mx = vmaxnmvq_f32(vm);
    for (; j < n; j++)
        if (rp[j] > mx) mx = rp[j];
    return mx;
}

// In-place x <- exp(x - mx) over a contiguous row, returning sum(x).
static float row_exp_sum(float *rp, size_t n, float mx) {
    float32x4_t vmx = vdupq_n_f32(mx);
    float32x4_t vs = vdupq_n_f32(0.0f);
    size_t j = 0;
    for (; j + 4 <= n; j += 4) {
        float32x4_t e = nexp_f32(vsubq_f32(vld1q_f32(rp + j), vmx));
        vst1q_f32(rp + j, e);
        vs = vaddq_f32(vs, e);
    }
    float sum = vaddvq_f32(vs);
    for (; j < n; j++) {
        float e = expf(rp[j] - mx);
        rp[j] = e;
        sum += e;
    }
    return sum;
}

static void row_scale(float *rp, size_t n, float s) {
    size_t j = 0;
    for (; j + 4 <= n; j += 4) vst1q_f32(rp + j, vmulq_n_f32(vld1q_f32(rp + j), s));
    for (; j < n; j++) rp[j] *= s;
}

// Split [0, m) into at most ATTN_MAX_CHUNKS row bands and run `body` on each,
// in parallel when there is enough work to pay for the dispatch.
//
// The chunk cap is deliberately well above the core count: over-decomposing lets
// dispatch_apply work-steal, which is what keeps the 6 E-cores contributing
// instead of straggling. Measured on M5 over a 4.19M-element softmax, capping
// chunks at the 4 P-cores costs 2.4x (739 us vs 308 us), and no size from the
// parallel threshold upward prefers a smaller cap.
static void attn_rows(size_t m, size_t work, void (^body)(size_t, size_t)) {
    if (m == 0) return;
    size_t want = m < ATTN_MAX_CHUNKS ? m : ATTN_MAX_CHUNKS;
    size_t rows = (m + want - 1) / want;
    size_t chunks = (m + rows - 1) / rows;
    if (chunks <= 1 || work < ATTN_PAR_MIN) {
        body(0, m);
        return;
    }
    dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
      size_t i0 = ci * rows;
      size_t i1 = i0 + rows < m ? i0 + rows : m;
      body(i0, i1);
    });
}

// Numerically-stable row softmax over an m x n row-major buffer. `row_max` may
// carry maxima a GEMM's fused reduction already produced; NULL computes them.
void attn_softmax_rows_f32(float *c, size_t m, size_t n, long rs, const float *row_max) {
    attn_rows(m, m * n, ^(size_t i0, size_t i1) {
      for (size_t i = i0; i < i1; i++) {
          float *rp = c + (long)i * rs;
          float mx = row_max ? row_max[i] : row_maxv(rp, n);
          // A row with no finite max (all -inf, or n == 0) would give inf - inf.
          if (!__builtin_isfinite(mx)) continue;
          float sum = row_exp_sum(rp, n, mx);
          if (sum != 0.0f) row_scale(rp, n, 1.0f / sum);
      }
    });
}

// One flash-attention key-block step, applied to the freshly computed score
// block `s` (m x bj, row stride s_rs) and the running output accumulator `o`
// (m x dv, row stride o_rs):
//
//   m_new = max(m_run, rowmax(s))
//   corr  = exp(m_run - m_new)          (0 on the first block, m_run = -inf)
//   s     = exp(s - m_new)
//   l_run = corr*l_run + rowsum(s)
//   o     = corr*o                      (the caller then does o += s @ V_block)
//
// `row_max` must be pre-filled with -INFINITY and `row_sum` with 0 before the
// first block.
void attn_flash_block_f32(float *s, size_t m, size_t bj, long s_rs, float *o, size_t dv, long o_rs,
                         float *row_max, float *row_sum) {
    attn_rows(m, m * bj, ^(size_t i0, size_t i1) {
      for (size_t i = i0; i < i1; i++) {
          float *sp = s + (long)i * s_rs;
          float prev = row_max[i];
          float mx = row_maxv(sp, bj);
          if (prev > mx) mx = prev;
          // Nothing finite seen yet: leave the running state alone.
          if (!__builtin_isfinite(mx)) continue;
          float corr = __builtin_isfinite(prev) ? expf(prev - mx) : 0.0f;
          row_sum[i] = row_sum[i] * corr + row_exp_sum(sp, bj, mx);
          row_max[i] = mx;
          if (corr != 1.0f) row_scale(o + (long)i * o_rs, dv, corr);
      }
    });
}

// Final flash normalization: o[i, :] /= l_run[i].
void attn_flash_finish_f32(float *o, size_t m, size_t dv, long o_rs, const float *row_sum) {
    attn_rows(m, m * dv, ^(size_t i0, size_t i1) {
      for (size_t i = i0; i < i1; i++) {
          float l = row_sum[i];
          row_scale(o + (long)i * o_rs, dv, l == 0.0f ? 0.0f : 1.0f / l);
      }
    });
}

// --- half-precision score blocks --------------------------------------------
// The GEMM writes scores in f16/bf16, so the passes below convert 4 lanes at a
// time, do all softmax arithmetic in f32, and convert back.

static inline float32x4_t ld4_f16(const __fp16 *p) { return vcvt_f32_f16(vld1_f16(p)); }
static inline void st4_f16(__fp16 *p, float32x4_t v) { vst1_f16(p, vcvt_f16_f32(v)); }
static inline float ld1_f16(const __fp16 *p) { return (float)*p; }
static inline void st1_f16(__fp16 *p, float x) { *p = (__fp16)x; }

static inline float32x4_t ld4_bf16(const uint16_t *p) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}

// f32 -> bf16, round to nearest even. The inputs here are exp() results and
// finite dot products, so the NaN quieting case does not arise.
static inline void st4_bf16(uint16_t *p, float32x4_t v) {
    uint32x4_t u = vreinterpretq_u32_f32(v);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
    vst1_u16(p, vshrn_n_u32(vaddq_u32(u, vaddq_u32(vdupq_n_u32(0x7fff), lsb)), 16));
}

static inline float ld1_bf16(const uint16_t *p) {
    uint32_t u = (uint32_t)*p << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

static inline void st1_bf16(uint16_t *p, float x) {
    uint32_t u;
    __builtin_memcpy(&u, &x, 4);
    *p = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

#define ATTN_FLASH_HALF_IMPL(TAG, T)                                                                \
    static float row_maxv_##TAG(const T *rp, size_t n, float scale) {                              \
        float32x4_t vm = vdupq_n_f32(-INFINITY);                                                   \
        size_t j = 0;                                                                              \
        for (; j + 4 <= n; j += 4)                                                                 \
            vm = vmaxnmq_f32(vm, vmulq_n_f32(ld4_##TAG(rp + j), scale));                           \
        float mx = vmaxnmvq_f32(vm);                                                               \
        for (; j < n; j++) {                                                                       \
            float x = ld1_##TAG(rp + j) * scale;                                                   \
            if (x > mx) mx = x;                                                                    \
        }                                                                                          \
        return mx;                                                                                 \
    }                                                                                              \
                                                                                                   \
    void attn_flash_block_##TAG(T *s, size_t m, size_t bj, long s_rs, float scale,                  \
                               float *row_max, float *row_sum, float *corr) {                      \
        attn_rows(m, m *bj, ^(size_t i0, size_t i1) {                                              \
          for (size_t i = i0; i < i1; i++) {                                                       \
              T *sp = s + (long)i * s_rs;                                                          \
              float prev = row_max[i];                                                             \
              float mx = row_maxv_##TAG(sp, bj, scale);                                            \
              if (prev > mx) mx = prev;                                                            \
              if (!__builtin_isfinite(mx)) {                                                       \
                  corr[i] = 1.0f;                                                                  \
                  continue;                                                                        \
              }                                                                                    \
              float c = __builtin_isfinite(prev) ? expf(prev - mx) : 0.0f;                          \
              float32x4_t vmx = vdupq_n_f32(mx), vs = vdupq_n_f32(0.0f);                            \
              size_t j = 0;                                                                        \
              for (; j + 4 <= bj; j += 4) {                                                        \
                  float32x4_t e = nexp_f32(vsubq_f32(vmulq_n_f32(ld4_##TAG(sp + j), scale), vmx)); \
                  st4_##TAG(sp + j, e);                                                            \
                  vs = vaddq_f32(vs, e);                                                           \
              }                                                                                    \
              float sum = vaddvq_f32(vs);                                                          \
              for (; j < bj; j++) {                                                                \
                  float e = expf(ld1_##TAG(sp + j) * scale - mx);                                  \
                  st1_##TAG(sp + j, e);                                                            \
                  sum += e;                                                                        \
              }                                                                                    \
              row_sum[i] = row_sum[i] * c + sum;                                                   \
              row_max[i] = mx;                                                                     \
              corr[i] = c;                                                                         \
          }                                                                                        \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    void attn_flash_accum_##TAG(float *acc, const T *t, size_t m, size_t dv, long t_rs,             \
                               const float *corr) {                                                \
        attn_rows(m, m *dv, ^(size_t i0, size_t i1) {                                              \
          for (size_t i = i0; i < i1; i++) {                                                       \
              float *ap = acc + i * dv;                                                            \
              const T *tp = t + (long)i * t_rs;                                                    \
              float32x4_t vc = vdupq_n_f32(corr[i]);                                               \
              size_t j = 0;                                                                        \
              for (; j + 4 <= dv; j += 4)                                                          \
                  vst1q_f32(ap + j, vfmaq_f32(ld4_##TAG(tp + j), vld1q_f32(ap + j), vc));           \
              for (; j < dv; j++) ap[j] = ap[j] * corr[i] + ld1_##TAG(tp + j);                     \
          }                                                                                        \
        });                                                                                        \
    }                                                                                              \
                                                                                                   \
    void attn_flash_finish_##TAG(T *out, const float *acc, size_t m, size_t dv, long out_rs,        \
                                const float *row_sum) {                                            \
        attn_rows(m, m *dv, ^(size_t i0, size_t i1) {                                              \
          for (size_t i = i0; i < i1; i++) {                                                       \
              float l = row_sum[i];                                                                \
              float inv = l == 0.0f ? 0.0f : 1.0f / l;                                             \
              const float *ap = acc + i * dv;                                                      \
              T *op = out + (long)i * out_rs;                                                      \
              size_t j = 0;                                                                        \
              for (; j + 4 <= dv; j += 4)                                                          \
                  st4_##TAG(op + j, vmulq_n_f32(vld1q_f32(ap + j), inv));                          \
              for (; j < dv; j++) st1_##TAG(op + j, ap[j] * inv);                                  \
          }                                                                                        \
        });                                                                                        \
    }

ATTN_FLASH_HALF_IMPL(f16, __fp16)
ATTN_FLASH_HALF_IMPL(bf16, uint16_t)

// Single-query attention over an f16 KV cache: one query row's heads against
// `len` cached keys, grouped-query aware (query head h reads KV head
// h / (n_heads / n_kv_heads)). q is [n_heads][hd] f32, k and v are [len] rows
// of f16 with stride ld (KV head kh at kh*hd), out is [n_heads][hd] f32;
// scores is scratch for `len` floats. hd must be a multiple of 8.
//
// Scores take FMLAL (f16 x f16 -> f32): q is rounded to f16 once and every
// product and sum is exact-to-f32, four key rows at a time with low and high
// halves, and alternate 8-dim chunks, in separate accumulators so sixteen
// chains overlap; reading K from L2 (~110-125 GB/s a core on M5) is then the
// limit, not FMLAL, and the same holds for PV. The values widen to
// f32 and accumulate against f32 probabilities. Row-major keys keep an append a
// plain row copy; the horizontal sums cost less than the strided writes a
// transposed cache would need on every append.
static void kv_scores(float *s, const __fp16 *k, size_t len, size_t ld, const float *q,
                          size_t hd, float scale) {
    float16x8_t q16[32]; // hd <= 256
    for (size_t d = 0; d < hd; d += 8) {
        float32x4_t lo = vmulq_n_f32(vld1q_f32(q + d), scale);
        float32x4_t hi = vmulq_n_f32(vld1q_f32(q + d + 4), scale);
        q16[d / 8] = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
    }
    size_t t = 0;
    for (; (hd & 15) == 0 && t + 4 <= len; t += 4) {
        const __fp16 *r0 = k + t * ld, *r1 = r0 + ld, *r2 = r1 + ld, *r3 = r2 + ld;
        float32x4_t a[8], b[8];
        for (int i = 0; i < 8; i++)
            a[i] = vdupq_n_f32(0.0f), b[i] = a[i];
        for (size_t d = 0; d < hd; d += 16) {
            float16x8_t q0 = q16[d / 8], q1 = q16[d / 8 + 1];
#define SC_ROW(i, r)                                                                               \
    do {                                                                                           \
        float16x8_t x0 = vld1q_f16((r) + d), x1 = vld1q_f16((r) + d + 8);                          \
        a[i] = vfmlalq_low_f16(a[i], x0, q0), b[i] = vfmlalq_high_f16(b[i], x0, q0);               \
        a[i + 4] = vfmlalq_low_f16(a[i + 4], x1, q1),                                              \
              b[i + 4] = vfmlalq_high_f16(b[i + 4], x1, q1);                                       \
    } while (0)
            SC_ROW(0, r0);
            SC_ROW(1, r1);
            SC_ROW(2, r2);
            SC_ROW(3, r3);
#undef SC_ROW
        }
        for (int i = 0; i < 4; i++)
            a[i] = vaddq_f32(vaddq_f32(a[i], b[i]), vaddq_f32(a[i + 4], b[i + 4]));
        vst1q_f32(s + t, vpaddq_f32(vpaddq_f32(a[0], a[1]), vpaddq_f32(a[2], a[3])));
    }
    for (; t + 4 <= len; t += 4) {
        const __fp16 *r0 = k + t * ld, *r1 = r0 + ld, *r2 = r1 + ld, *r3 = r2 + ld;
        float32x4_t a0 = vdupq_n_f32(0.0f), b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0,
                    b3 = a0;
        for (size_t d = 0; d < hd; d += 8) {
            float16x8_t qv = q16[d / 8];
            float16x8_t k0 = vld1q_f16(r0 + d), k1 = vld1q_f16(r1 + d);
            float16x8_t k2 = vld1q_f16(r2 + d), k3 = vld1q_f16(r3 + d);
            a0 = vfmlalq_low_f16(a0, k0, qv), b0 = vfmlalq_high_f16(b0, k0, qv);
            a1 = vfmlalq_low_f16(a1, k1, qv), b1 = vfmlalq_high_f16(b1, k1, qv);
            a2 = vfmlalq_low_f16(a2, k2, qv), b2 = vfmlalq_high_f16(b2, k2, qv);
            a3 = vfmlalq_low_f16(a3, k3, qv), b3 = vfmlalq_high_f16(b3, k3, qv);
        }
        float32x4_t s01 = vpaddq_f32(vaddq_f32(a0, b0), vaddq_f32(a1, b1));
        float32x4_t s23 = vpaddq_f32(vaddq_f32(a2, b2), vaddq_f32(a3, b3));
        vst1q_f32(s + t, vpaddq_f32(s01, s23));
    }
    for (; t < len; t++) {
        const __fp16 *r = k + t * ld;
        float32x4_t a = vdupq_n_f32(0.0f), b = a;
        for (size_t d = 0; d < hd; d += 8) {
            float16x8_t kv = vld1q_f16(r + d);
            a = vfmlalq_low_f16(a, kv, q16[d / 8]), b = vfmlalq_high_f16(b, kv, q16[d / 8]);
        }
        s[t] = vaddvq_f32(vaddq_f32(a, b));
    }
}

// out[0..w) = inv * sum_t p[t] * v[t*ld + 0..w) for a constant w (8 to 64):
// eight probabilities round to f16 into one register and FMLAL takes each by
// lane, so a row costs one load and two FMLALs per 8 dims (no per-row widen or
// broadcast). Inlined per w so the dims loop unrolls and the accumulators stay
// in registers; 64 dims a pass is 16 chains, enough to cover FMLAL's latency
// (32 dims, 8 chains, ran at about half the FMLAL rate on M5).
__attribute__((always_inline)) static inline void kv_pv_pass(float *out, const __fp16 *v,
                                                             size_t len, size_t ld, const float *p,
                                                             float inv, const size_t w) {
    {
        float32x4_t acc[16];
        for (size_t i = 0; i < w / 4; i++)
            acc[i] = vdupq_n_f32(0.0f);
#define PV_ROW(L)                                                                                  \
    do {                                                                                           \
        const __fp16 *vr = v + (t + (L)) * ld;                                                     \
        for (size_t i = 0; i < w / 8; i++) {                                                       \
            float16x8_t x = vld1q_f16(vr + 8 * i);                                                 \
            acc[2 * i] = vfmlalq_laneq_low_f16(acc[2 * i], x, p8, L);                              \
            acc[2 * i + 1] = vfmlalq_laneq_high_f16(acc[2 * i + 1], x, p8, L);                     \
        }                                                                                          \
    } while (0)
        size_t t = 0;
        for (; t + 8 <= len; t += 8) {
            float16x8_t p8 =
                vcombine_f16(vcvt_f16_f32(vld1q_f32(p + t)), vcvt_f16_f32(vld1q_f32(p + t + 4)));
            PV_ROW(0);
            PV_ROW(1);
            PV_ROW(2);
            PV_ROW(3);
            PV_ROW(4);
            PV_ROW(5);
            PV_ROW(6);
            PV_ROW(7);
        }
        if (t < len) {
            float tail[8] = {0};
            for (size_t i = 0; t + i < len; i++)
                tail[i] = p[t + i];
            float16x8_t p8 =
                vcombine_f16(vcvt_f16_f32(vld1q_f32(tail)), vcvt_f16_f32(vld1q_f32(tail + 4)));
            switch (len - t) {
                case 7:
                    PV_ROW(6); // fallthrough
                case 6:
                    PV_ROW(5); // fallthrough
                case 5:
                    PV_ROW(4); // fallthrough
                case 4:
                    PV_ROW(3); // fallthrough
                case 3:
                    PV_ROW(2); // fallthrough
                case 2:
                    PV_ROW(1); // fallthrough
                default:
                    PV_ROW(0);
            }
        }
#undef PV_ROW
        for (size_t i = 0; i < w / 4; i++)
            vst1q_f32(out + 4 * i, vmulq_n_f32(acc[i], inv));
    }
}

// out[0..hd) = inv * sum_t p[t] * v[t*ld + 0..hd), hd a multiple of 8: 64
// dims a pass, then what is left in 32- and 8-dim passes.
static void kv_pv(float *out, const __fp16 *v, size_t len, size_t ld, const float *p, size_t hd,
                  float inv) {
    size_t d0 = 0;
    for (; d0 + 64 <= hd; d0 += 64)
        kv_pv_pass(out + d0, v + d0, len, ld, p, inv, 64);
    for (; d0 + 32 <= hd; d0 += 32)
        kv_pv_pass(out + d0, v + d0, len, ld, p, inv, 32);
    for (; d0 < hd; d0 += 8)
        kv_pv_pass(out + d0, v + d0, len, ld, p, inv, 8);
}

void attn_kv_f16(float *out, const float *q, const __fp16 *k, const __fp16 *v, size_t len,
                     size_t ld, size_t n_heads, size_t n_kv_heads, size_t hd, float scale,
                     float *scores) {
    size_t group = n_heads / n_kv_heads;
    for (size_t h = 0; h < n_heads; h++) {
        size_t kh = h / group;
        kv_scores(scores, k + kh * hd, len, ld, q + h * hd, hd, scale);
        float mx = row_maxv(scores, len);
        float sum = row_exp_sum(scores, len, mx);
        kv_pv(out + h * hd, v + kh * hd, len, ld, scores, hd, 1.0f / sum);
    }
}

// Causal attention for a block of query rows: row i (position start + i)
// attends to cached keys 0..start+i, so row i is attn_kv_f16 over the first
// start+i+1 keys. Rows are independent, so large blocks split across cores
// (dispatch_apply); each chunk mallocs its own score row. q and out are
// [rows][n_heads*hd]. Returns -1 if a scratch allocation fails.
int attn_kv_causal_f16(float *out, const float *q, const __fp16 *k, const __fp16 *v, size_t start,
                    size_t rows, size_t ld, size_t n_heads, size_t n_kv_heads, size_t hd,
                    float scale) {
    size_t qd = n_heads * hd;
    // Work ~ head-keys at ~3.5 ns each; below ~16K (~60 us) one thread does it.
    size_t work = n_heads * (rows * start + rows * (rows + 1) / 2);
    size_t chunks = work < (1u << 14) ? 1 : rows < ATTN_MAX_CHUNKS ? rows : ATTN_MAX_CHUNKS;
    __block atomic_int failed = 0;
    void (^body)(size_t) = ^(size_t ci) {
      // Later rows hold more keys: interleave rows across chunks to balance.
      float *scores = malloc((start + rows) * sizeof(float));
      if (!scores) {
          atomic_store_explicit(&failed, 1, memory_order_relaxed);
          return;
      }
      for (size_t i = ci; i < rows; i += chunks)
          attn_kv_f16(out + i * qd, q + i * qd, k, v, start + i + 1, ld, n_heads, n_kv_heads,
                          hd, scale, scores);
      free(scores);
    };
    if (chunks == 1)
        body(0);
    else
        dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), body);
    return atomic_load_explicit(&failed, memory_order_relaxed) ? -1 : 0;
}
