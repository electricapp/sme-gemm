// Row-softmax / flash-attention elementwise passes (attention.c).
#ifndef SME_GEMM_ATTENTION_H
#define SME_GEMM_ATTENTION_H

#include <stddef.h>
#include <stdint.h>

void attn_softmax_rows_f32(float *c, size_t m, size_t n, long rs, const float *row_max);

void attn_flash_block_f32(float *s, size_t m, size_t bj, long s_rs, float *o, size_t dv, long o_rs,
                         float *row_max, float *row_sum);

void attn_flash_finish_f32(float *o, size_t m, size_t dv, long o_rs, const float *row_sum);

// Half-precision flash steps. The score block `s` is f16/bf16 (as the GEMM
// wrote it) but every softmax statistic is f32, and the output accumulates in
// f32 across key blocks -- so a block is three calls:
//   _block  : s <- exp(scale*s - m_new), update row_max/row_sum, emit `corr`
//             (the scale is applied here, in f32, not folded into the GEMM's
//             half-precision beta)
//   _accum  : acc <- corr*acc + t     (t is the f16/bf16 P @ V_block product)
//   _finish : out <- acc / row_sum    (back down to f16/bf16)
#define ATTN_FLASH_HALF_DECLS(TAG, T)                                                               \
    void attn_flash_block_##TAG(T *s, size_t m, size_t bj, long s_rs, float scale,                  \
                               float *row_max, float *row_sum, float *corr);                       \
    void attn_flash_accum_##TAG(float *acc, const T *t, size_t m, size_t dv, long t_rs,             \
                               const float *corr);                                                 \
    void attn_flash_finish_##TAG(T *out, const float *acc, size_t m, size_t dv, long out_rs,        \
                                const float *row_sum);

ATTN_FLASH_HALF_DECLS(f16, __fp16)
ATTN_FLASH_HALF_DECLS(bf16, uint16_t)

// One query row against an f16 KV cache (see attention.c). hd % 8 == 0,
// hd <= 256, len >= 1, n_heads a multiple of n_kv_heads; scores holds len floats.
void attn_kv_f16(float *out, const float *q, const __fp16 *k, const __fp16 *v, size_t len,
                     size_t ld, size_t n_heads, size_t n_kv_heads, size_t hd, float scale,
                     float *scores);

// A block of query rows at positions start..start+rows against the cache
// (row i sees keys 0..start+i); rows spread across cores. 0, or -1 on OOM.
int attn_kv_causal_f16(float *out, const float *q, const __fp16 *k, const __fp16 *v, size_t start,
                    size_t rows, size_t ld, size_t n_heads, size_t n_kv_heads, size_t hd,
                    float scale);

#endif // SME_GEMM_ATTENTION_H
