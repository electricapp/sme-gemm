// Every function the kernels export, with C linkage: src/ffi.rs calls them by
// these names. Each .cc includes this header first, so the declaration here is
// the first one each definition meets and gives it C linkage; a definition
// whose signature drifts from its line here becomes a separate C++ overload and
// fails to link. Grouped by defining file.
#ifndef SME_GEMM_API_H
#define SME_GEMM_API_H

#include <atomic>
#include <stddef.h>
#include <stdint.h>

#include "epilogue.h"

extern "C" {

// attention.cc: row softmax and the flash-attention elementwise passes.
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
void attn_flash_block_f16(__fp16 *s, size_t m, size_t bj, long s_rs, float scale, float *row_max,
                          float *row_sum, float *corr);
void attn_flash_accum_f16(float *acc, const __fp16 *t, size_t m, size_t dv, long t_rs,
                          const float *corr);
void attn_flash_finish_f16(__fp16 *out, const float *acc, size_t m, size_t dv, long out_rs,
                           const float *row_sum);
void attn_flash_block_bf16(uint16_t *s, size_t m, size_t bj, long s_rs, float scale, float *row_max,
                           float *row_sum, float *corr);
void attn_flash_accum_bf16(float *acc, const uint16_t *t, size_t m, size_t dv, long t_rs,
                           const float *corr);
void attn_flash_finish_bf16(uint16_t *out, const float *acc, size_t m, size_t dv, long out_rs,
                            const float *row_sum);
// One query row against an f16 KV cache. hd % 8 == 0, hd <= 256, len >= 1,
// n_heads a multiple of n_kv_heads; scores holds len floats.
void attn_kv_f16(float *out, const float *q, const __fp16 *k, const __fp16 *v, size_t len,
                 size_t ld, size_t n_heads, size_t n_kv_heads, size_t hd, float scale,
                 float *scores);
// A block of query rows at positions start..start+rows against the cache
// (row i sees keys 0..start+i); rows spread across cores. 0, or -1 on OOM.
int attn_kv_causal_f16(float *out, const float *q, const __fp16 *k, const __fp16 *v, size_t start,
                       size_t rows, size_t ld, size_t n_heads, size_t n_kv_heads, size_t hd,
                       float scale);

// gemm_b16b16.cc: native bf16 GEMM (FEAT_SME_B16B16) and its Q4 path.
int gemm_sme_b16b16_batched(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                            const uint16_t *lhs, const uint16_t *rhs);
int gemm_sme_b16b16_batched_ep(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                               const uint16_t *lhs, const uint16_t *rhs, const ep_desc16 *ep);
void gemm_sme_b16b16_packb(uint16_t *b_pack, const uint16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs);
size_t gemm_sme_b16b16_packed_b_elems(size_t n, size_t k);
int gemm_sme_b16b16_q4(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                       const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                       const uint16_t *scales_bf16, const uint16_t *mins_bf16, size_t block,
                       const ep_desc16 *ep);
int gemm_sme_b16b16_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                        int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                        const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                        uint16_t beta_bits);
int gemm_sme_b16b16_run_packed(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs,
                               long dst_rs, int read_dst, const uint16_t *lhs, long lhs_cs,
                               long lhs_rs, const uint16_t *b_pack, uint16_t alpha_bits,
                               uint16_t beta_bits, const ep_desc16 *ep);

// gemm_widen.cc: widening 16-bit GEMMs into f32.
int gemm_sme_bf16f32_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                         int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                         const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                         uint16_t beta_bits);
int gemm_sme_f16f32_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                        int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                        const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                        uint16_t beta_bits);

// gemm_f16f16.cc: native f16 GEMM (FEAT_SME_F16F16) and its Q4 path.
int gemm_sme_f16f16_batched(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                            const uint16_t *lhs, const uint16_t *rhs);
int gemm_sme_f16f16_batched_ep(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                               const uint16_t *lhs, const uint16_t *rhs, const ep_desc16 *ep);
void gemm_sme_f16f16_packb(uint16_t *b_pack, const uint16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs);
size_t gemm_sme_f16f16_packed_b_elems(size_t n, size_t k);
int gemm_sme_f16f16_q4(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                       const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                       size_t block, const ep_desc16 *ep);
// The Q4 GEMM as one link of a chain (src/mlp.rs): raises *done as output
// columns are stored and waits on *ready for A's depths (either NULL).
int gemm_sme_f16f16_q4_chained(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                               const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                               size_t block, const ep_desc16 *ep, std::atomic<size_t> *done,
                               const std::atomic<size_t> *ready);
// Whether an m-row Q4 call runs as one GEMV on the calling thread.
int gemm_sme_f16f16_q4_single(size_t m, size_t n, size_t k);
int gemm_sme_f16f16_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                        int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                        const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                        uint16_t beta_bits);
int gemm_sme_f16f16_run_packed(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs,
                               long dst_rs, int read_dst, const uint16_t *lhs, long lhs_cs,
                               long lhs_rs, const uint16_t *b_pack, uint16_t alpha_bits,
                               uint16_t beta_bits, const ep_desc16 *ep);

// gemm_f32.cc
int gemm_sme_f32_batched(size_t count, size_t m, size_t n, size_t k, float *dst, const float *lhs,
                         const float *rhs);
int gemm_sme_f32_batched_ep(size_t count, size_t m, size_t n, size_t k, float *dst,
                            const float *lhs, const float *rhs, const ep_desc_f32 *ep);
void gemm_sme_f32_packb(float *b_pack, const float *rhs, size_t n, size_t k, long rhs_rs,
                        long rhs_cs);
size_t gemm_sme_f32_packed_b_elems(size_t n, size_t k);
int gemm_sme_f32_run(size_t m, size_t n, size_t k, float *dst, long dst_cs, long dst_rs,
                     int read_dst, const float *lhs, long lhs_cs, long lhs_rs, const float *rhs,
                     long rhs_cs, long rhs_rs, float alpha, float beta, const ep_desc_f32 *ep);
int gemm_sme_f32_run_packed(size_t m, size_t n, size_t k, float *dst, long dst_cs, long dst_rs,
                            int read_dst, const float *lhs, long lhs_cs, long lhs_rs,
                            const float *b_pack, float alpha, float beta, const ep_desc_f32 *ep);
// Dense row-major operands only (A m x k, B k x n, C m x n): C = nodes(beta*A@B)
// plus each finished row's sum and max (length m, either NULL). A reduction
// needs whole contiguous rows, so these entries have no strides to get wrong.
int gemm_sme_f32_rows(size_t m, size_t n, size_t k, float *dst, const float *lhs, const float *rhs,
                      float beta, const ep_desc_f32 *ep, float *row_sum, float *row_max);
int gemm_sme_f32_rows_packed(size_t m, size_t n, size_t k, float *dst, const float *lhs,
                             const float *b_pack, float beta, const ep_desc_f32 *ep, float *row_sum,
                             float *row_max);
int gemm_sme_f32_softmax(size_t m, size_t n, size_t k, float *dst, const float *lhs,
                         const float *b_pack, float *row_max);

// gemm_f64.cc (FEAT_SME_F64F64)
int gemm_sme_f64_batched(size_t count, size_t m, size_t n, size_t k, double *dst, const double *lhs,
                         const double *rhs);
int gemm_sme_f64_batched_ep(size_t count, size_t m, size_t n, size_t k, double *dst,
                            const double *lhs, const double *rhs, const ep_desc_f64 *ep);
void gemm_sme_f64_packb(double *b_pack, const double *rhs, size_t n, size_t k, long rhs_rs,
                        long rhs_cs);
size_t gemm_sme_f64_packed_b_elems(size_t n, size_t k);
int gemm_sme_f64_run(size_t m, size_t n, size_t k, double *dst, long dst_cs, long dst_rs,
                     int read_dst, const double *lhs, long lhs_cs, long lhs_rs, const double *rhs,
                     long rhs_cs, long rhs_rs, double alpha, double beta, const ep_desc_f64 *ep);
int gemm_sme_f64_run_packed(size_t m, size_t n, size_t k, double *dst, long dst_cs, long dst_rs,
                            int read_dst, const double *lhs, long lhs_cs, long lhs_rs,
                            const double *b_pack, double alpha, double beta, const ep_desc_f64 *ep);

// gemm_i16i64.cc (FEAT_SME_I16I64)
int gemm_sme_i16i64_batched(size_t count, size_t m, size_t n, size_t k, int64_t *dst,
                            const int16_t *lhs, const int16_t *rhs);
int gemm_sme_i16i64_batched_dequant(size_t count, size_t m, size_t n, size_t k, float *fdst,
                                    const int16_t *lhs, const int16_t *rhs, float scale,
                                    const float *scale_n, uint32_t n_nodes, const EpNode *nodes);
void gemm_sme_i16i64_packb(int16_t *b_pack, const int16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs);
size_t gemm_sme_i16i64_packed_b_elems(size_t n, size_t k);
int gemm_sme_i16i64_run(size_t m, size_t n, size_t k, int64_t *dst, long dst_cs, long dst_rs,
                        const int16_t *lhs, long lhs_cs, long lhs_rs, const int16_t *rhs,
                        long rhs_cs, long rhs_rs);
int gemm_sme_i16i64_run_dequant(size_t m, size_t n, size_t k, float *dst, long dst_cs, long dst_rs,
                                const int16_t *lhs, long lhs_cs, long lhs_rs, const int16_t *rhs,
                                long rhs_cs, long rhs_rs, float scale, const float *scale_n,
                                uint32_t n_nodes, const EpNode *nodes);
int gemm_sme_i16i64_run_packed(size_t m, size_t n, size_t k, int64_t *dst, long dst_cs, long dst_rs,
                               const int16_t *lhs, long lhs_cs, long lhs_rs, const int16_t *b_pack);
int gemm_sme_i16i64_run_packed_dequant(size_t m, size_t n, size_t k, float *dst, long dst_cs,
                                       long dst_rs, const int16_t *lhs, long lhs_cs, long lhs_rs,
                                       const int16_t *b_pack, float scale, const float *scale_n,
                                       uint32_t n_nodes, const EpNode *nodes);

// gemm_i8i32.cc
int gemm_sme_i8i32_batched(size_t count, size_t m, size_t n, size_t k, int32_t *dst,
                           const int8_t *lhs, const int8_t *rhs);
int gemm_sme_i8i32_batched_dequant(size_t count, size_t m, size_t n, size_t k, float *dst,
                                   const int8_t *lhs, const int8_t *rhs, float scale,
                                   const float *scale_n, uint32_t n_nodes, const EpNode *nodes);
void gemm_sme_i8i32_packb(int8_t *b_pack, const int8_t *rhs, size_t n, size_t k, long rhs_rs,
                          long rhs_cs);
size_t gemm_sme_i8i32_packed_b_elems(size_t n, size_t k);
int gemm_sme_i8i32_run(size_t m, size_t n, size_t k, int32_t *dst, long dst_cs, long dst_rs,
                       const int8_t *lhs, long lhs_cs, long lhs_rs, const int8_t *rhs, long rhs_cs,
                       long rhs_rs);
int gemm_sme_i8i32_run_packed(size_t m, size_t n, size_t k, int32_t *dst, long dst_cs, long dst_rs,
                              const int8_t *lhs, long lhs_cs, long lhs_rs, const int8_t *b_pack);
int gemm_sme_i8i32_run_packed_dequant(size_t m, size_t n, size_t k, float *dst, long dst_cs,
                                      long dst_rs, const int8_t *lhs, long lhs_cs, long lhs_rs,
                                      const int8_t *b_pack, float scale, const float *scale_n,
                                      uint32_t n_nodes, const EpNode *nodes);

// gemm_q2.cc: the 2-bit and 1-bit GEMMs (FEAT_SME2p1).
int gemm_sme_f16_q1(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                    const uint8_t *signs, const uint16_t *scales, size_t block,
                    const ep_desc16 *ep);
int gemm_sme_f16_q2(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                    const uint8_t *codes, const uint16_t *scales, const int8_t *codebook,
                    size_t block, const ep_desc16 *ep);

// neon_ops.cc: NEON passes over model activations.
void neon_act_f16(__fp16 *dst, const __fp16 *src, size_t n, uint32_t act);
void neon_add_f16_to_f32(float *dst, const __fp16 *src, size_t n);
void neon_f16_to_f32(float *dst, const __fp16 *src, size_t n);
void neon_f32_to_f16(__fp16 *dst, const float *src, size_t n);
void neon_glu_f16(__fp16 *out, const __fp16 *in, size_t m, size_t n, uint32_t act);
void neon_norm_row_f16(__fp16 *y, const float *x, const float *w, const float *b, size_t n,
                       float eps, int rms);
void neon_norm_row_f32(float *y, const float *x, const float *w, const float *b, size_t n,
                       float eps, int rms);

// sme_probe.cc: sysctl capability and core-count probes.
int sme_sysctl_flag(const char *name);
int64_t sme_sysctl_int(const char *name);

// sme_warm.cc: one burst of the SmeWarm helper's streaming work.
float sme_warm_tick(void);

} // extern "C"

#endif // SME_GEMM_API_H
