// Native bf16 GEMM (Apple M5+, FEAT_SME_B16B16): gemm_half.h instantiated for
// __bf16, plus the 4-bit-resident path (gemm_b16b16_q4.h). The alpha/beta
// combine uses FEAT_SVE_B16B16 arithmetic (svmul/svmla on bf16).

#include "sme_gemm_api.h"

#include "gemm_half.h"

typedef __bf16 bf16;

int gemm_sme_b16b16_run_packed(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs,
                               long dst_rs, int read_dst, const uint16_t *lhs, long lhs_cs,
                               long lhs_rs, const uint16_t *b_pack, uint16_t alpha_bits,
                               uint16_t beta_bits, const ep_desc16 *ep) {
    return run_packed<bf16>(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, b_pack,
                            alpha_bits, beta_bits, ep);
}

// Pack B once into a caller-allocated, zeroed buffer of
// gemm_sme_b16b16_packed_b_elems(n,k) bf16. Reuse across many GEMMs (weights).
void gemm_sme_b16b16_packb(uint16_t *b_pack, const uint16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs) {
    half_packb<bf16>(b_pack, rhs, n, k, rhs_rs, rhs_cs);
}

size_t gemm_sme_b16b16_packed_b_elems(size_t n, size_t k) {
    return half_packed_b_elems(n, k);
}

int gemm_sme_b16b16_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                        int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                        const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                        uint16_t beta_bits) {
    return run<bf16>(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs, rhs, rhs_cs,
                     rhs_rs, alpha_bits, beta_bits);
}

int gemm_sme_b16b16_batched_ep(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                               const uint16_t *lhs, const uint16_t *rhs, const ep_desc16 *ep) {
    return run_batched_ep<bf16>(count, m, n, k, dst, lhs, rhs, ep);
}

int gemm_sme_b16b16_batched(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                            const uint16_t *lhs, const uint16_t *rhs) {
    return run_batched_ep<bf16>(count, m, n, k, dst, lhs, rhs, NULL);
}

// The 4-bit-resident path, kept in its own file (it builds on the dense
// kernels above, so it is included rather than compiled separately).
#include "gemm_b16b16_q4.h"
