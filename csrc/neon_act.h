// Trailing transcendental activations for the m <= 7 GEMV paths, applied in
// NEON after the streaming kernel returns.
//
// Inside a streaming region gelu/silu/sigmoid/tanh cost ~2.5 ns an output on M5
// (streaming ALU throughput is tuned for MOPA, not polynomial chains): a
// 1x1536x384 Q4 GEMV takes 3.0 us plain and 6.9 us with a fused gelu. A GEMV
// writes only m*n outputs, so a NEON pass over them afterwards is far cheaper
// and computes the activation in f32 on the stored value -- which is the
// accumulator itself, since these paths accumulate in the output precision.
// Cheap ops (bias, scale, clamp, relu) stay fused in the kernel.
#ifndef SME_GEMM_NEON_ACT_H
#define SME_GEMM_NEON_ACT_H

#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

#include "epilogue.h"

// sme_warm.c: calls in a NEON phase. A call's busy mark (src/warm.rs) spans
// the whole C call, so without this the keep-awake helper would stand down for
// the post-pass and the unit would idle into the next call.
extern _Atomic uint32_t sme_warm_neon;
#define NA_NEON_PHASE(stmt)                                                                        \
    do {                                                                                           \
        atomic_fetch_add_explicit(&sme_warm_neon, 1, memory_order_release);                        \
        stmt;                                                                                      \
        atomic_fetch_sub_explicit(&sme_warm_neon, 1, memory_order_release);                        \
    } while (0)

// If ep ends in an activation worth moving out of the streaming region, return
// its kind and set *rest to the nodes before it; else EP_ACT_NONE.
static inline uint32_t na_split(const ep_desc16 *ep, ep_desc16 *rest) {
    if (!ep || ep->n_nodes == 0) return EP_ACT_NONE;
    const EpNode *last = &ep->nodes[ep->n_nodes - 1];
    if (last->op != EP_OP_ACT) return EP_ACT_NONE;
    if (last->aux != EP_ACT_GELU && last->aux != EP_ACT_SILU && last->aux != EP_ACT_SIGMOID &&
        last->aux != EP_ACT_TANH)
        return EP_ACT_NONE;
    rest->n_nodes = ep->n_nodes - 1;
    rest->nodes = ep->nodes;
    return last->aux;
}

// The epilogue the kernel still runs: ep itself, the nodes before a split-off
// activation, or NULL when that leaves none.
static inline const ep_desc16 *na_kernel_ep(const ep_desc16 *ep, uint32_t act,
                                            const ep_desc16 *rest) {
    if (act == EP_ACT_NONE) return ep;
    return rest->n_nodes ? rest : NULL;
}

// exp(x) to ~1e-4 relative -- the outputs here are f16/bf16 (2^-11 and 2^-8
// steps), so a degree-3 polynomial for 2^r on [-0.5, 0.5] is enough.
static inline float32x4_t na_exp(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.33654f)), vdupq_n_f32(88.72283f));
    float32x4_t t = vmulq_n_f32(x, 1.4426950408889634f);
    float32x4_t n = vrndnq_f32(t);
    float32x4_t r = vsubq_f32(t, n);
    float32x4_t p = vdupq_n_f32(5.5504109e-2f);
    p = vfmaq_f32(vdupq_n_f32(2.4022651e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(6.9314718e-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtnq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

// 1/(1 + exp(-x)): reciprocal estimate plus one Newton step (~2e-5), no divide.
static inline float32x4_t na_sigmoid(float32x4_t x) {
    float32x4_t d = vaddq_f32(vdupq_n_f32(1.0f), na_exp(vnegq_f32(x)));
    float32x4_t r = vrecpeq_f32(d);
    return vmulq_f32(r, vrecpsq_f32(d, r));
}

// act(x) for the kinds na_split accepts; NaN lanes pass through as NaN.
static inline float32x4_t na_apply(float32x4_t x, uint32_t act) {
    float32x4_t y;
    if (act == EP_ACT_GELU) {
        // tanh-form gelu: 0.5x(1 + tanh(c(x + 0.044715x^3))) = x * sigmoid(2c(x + 0.044715x^3))
        float32x4_t u =
            vmulq_f32(vmulq_n_f32(x, 1.5957691216f),
                      vfmaq_f32(vdupq_n_f32(1.0f), vmulq_f32(x, x), vdupq_n_f32(0.044715f)));
        y = vmulq_f32(x, na_sigmoid(u));
    } else if (act == EP_ACT_SILU) {
        y = vmulq_f32(x, na_sigmoid(x));
    } else if (act == EP_ACT_SIGMOID) {
        y = na_sigmoid(x);
    } else { // EP_ACT_TANH = 2 sigmoid(2x) - 1
        y = vsubq_f32(vmulq_n_f32(na_sigmoid(vmulq_n_f32(x, 2.0f)), 2.0f), vdupq_n_f32(1.0f));
    }
    return vbslq_f32(vceqq_f32(x, x), y, x);
}

// c[i*rs + j] = act(c[i*rs + j]) over an m x n f16 block.
static inline void na_act_f16(__fp16 *c, size_t m, size_t n, long rs, uint32_t act) {
    for (size_t i = 0; i < m; i++) {
        __fp16 *row = c + (long)i * rs;
        size_t j = 0;
        for (; j + 8 <= n; j += 8) {
            float16x8_t h = vld1q_f16(row + j);
            float32x4_t lo = na_apply(vcvt_f32_f16(vget_low_f16(h)), act);
            float32x4_t hi = na_apply(vcvt_high_f32_f16(h), act);
            vst1q_f16(row + j, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
        }
        for (; j < n; j += 4) {
            float buf[4] = {0};
            size_t w = n - j < 4 ? n - j : 4;
            for (size_t t = 0; t < w; t++)
                buf[t] = (float)row[j + t];
            vst1q_f32(buf, na_apply(vld1q_f32(buf), act));
            for (size_t t = 0; t < w; t++)
                row[j + t] = (__fp16)buf[t];
        }
    }
}

// f32 -> bf16 bits, round to nearest even; NaN stays a quiet NaN.
static inline uint16x4_t na_to_bf16(float32x4_t v) {
    uint32x4_t b = vreinterpretq_u32_f32(v);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(b, 16), vdupq_n_u32(1));
    uint32x4_t r = vshrq_n_u32(vaddq_u32(b, vaddq_u32(vdupq_n_u32(0x7fff), lsb)), 16);
    uint32x4_t nan = vmvnq_u32(vceqq_f32(v, v));
    r = vbslq_u32(nan, vdupq_n_u32(0x7fc0), r);
    return vmovn_u32(r);
}

// c[i*rs + j] = act(c[i*rs + j]) over an m x n bf16 block (bf16 as raw bits).
static inline void na_act_bf16(uint16_t *c, size_t m, size_t n, long rs, uint32_t act) {
    for (size_t i = 0; i < m; i++) {
        uint16_t *row = c + (long)i * rs;
        size_t j = 0;
        for (; j + 4 <= n; j += 4) {
            float32x4_t x = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(row + j), 16));
            vst1_u16(row + j, na_to_bf16(na_apply(x, act)));
        }
        if (j < n) {
            uint16_t buf[4] = {0};
            size_t w = n - j;
            for (size_t t = 0; t < w; t++)
                buf[t] = row[j + t];
            float32x4_t x = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(buf), 16));
            vst1_u16(buf, na_to_bf16(na_apply(x, act)));
            for (size_t t = 0; t < w; t++)
                row[j + t] = buf[t];
        }
    }
}

// Below this many outputs one thread does the pass (~16 us); above it row
// chunks spread over the cores.
#define NA_PAR_MIN (1u << 16)
#define NA_MAX_CHUNKS 32

// act over a whole output: unit column stride walks rows, unit row stride
// walks columns (an elementwise pass does not care which), in parallel row
// chunks when the output is large. bf selects bf16 (else f16).
static inline void na_post(uint16_t *c, size_t m, size_t n, long rs, long cs, uint32_t act,
                           int bf) {
    if (cs != 1) { // column-major: n runs of m contiguous outputs
        size_t t = m;
        m = n, n = t, rs = cs;
    }
    size_t cells = m * n;
    size_t chunks = cells < NA_PAR_MIN ? 1 : cells / (NA_PAR_MIN / 2);
    if (chunks > m) chunks = m;
    if (chunks > NA_MAX_CHUNKS) chunks = NA_MAX_CHUNKS;
    if (chunks <= 1) {
        if (bf)
            na_act_bf16(c, m, n, rs, act);
        else
            na_act_f16((__fp16 *)c, m, n, rs, act);
        return;
    }
    dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
      size_t r0 = m * ci / chunks, r1 = m * (ci + 1) / chunks;
      uint16_t *base = c + (long)r0 * rs;
      if (bf)
          na_act_bf16(base, r1 - r0, n, rs, act);
      else
          na_act_f16((__fp16 *)base, r1 - r0, n, rs, act);
    });
}
#define na_post_f16(c, m, n, rs, cs, act) na_post((uint16_t *)(c), m, n, rs, cs, act, 0)
#define na_post_bf16(c, m, n, rs, cs, act) na_post((uint16_t *)(c), m, n, rs, cs, act, 1)

#endif // SME_GEMM_NEON_ACT_H
