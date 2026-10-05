// Elementwise passes the Rust layer calls between kernels: plain NEON, on
// the calling thread or spread over the cores for large outputs.
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>

#include "neon_act.h"

void neon_glu_f16(__fp16 *out, const __fp16 *in, size_t m, size_t n, uint32_t act);

// One row of the gated pass: `in` holds 2n values, gate and up interleaved 32
// columns at a time (the last chunk w < 32 wide: w gate then w up).
static void glu_row(__fp16 *out, const __fp16 *in, size_t n, uint32_t act) {
    for (size_t c0 = 0; c0 < n; c0 += 32) {
        size_t w = n - c0 < 32 ? n - c0 : 32;
        const __fp16 *g = in + 2 * c0, *u = g + w;
        size_t j = 0;
        for (; j + 8 <= w; j += 8) {
            float16x8_t gv = vld1q_f16(g + j), uv = vld1q_f16(u + j);
            float32x4_t lo = vmulq_f32(na_apply(vcvt_f32_f16(vget_low_f16(gv)), act),
                                       vcvt_f32_f16(vget_low_f16(uv)));
            float32x4_t hi =
                vmulq_f32(na_apply(vcvt_high_f32_f16(gv), act), vcvt_high_f32_f16(uv));
            vst1q_f16(out + c0 + j, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
        }
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
