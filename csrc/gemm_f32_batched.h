// Batched f32: `count` independent same-shape GEMMs in ONE streaming session, so
// the streaming entry/exit is paid once for the whole batch rather than per item.
// Split out of gemm_f32.c (which owns the packing helpers and scratch this
// closes over) purely to keep both files navigable; it is included in place.
#ifndef SME_GEMM_F32_BATCHED_H
#define SME_GEMM_F32_BATCHED_H

// One row-major GEMM (C = A@B overwrite), already in streaming mode. B is read
// straight from the row-major input and each 16-row A band is transposed
// through ZA tile 0 into `a_scr` ([2][k][16]); see gemm_f16f16_batched.h.
static void batch_one(float *dst, const float *a, const float *b, float *a_scr, size_t m,
                      size_t n, size_t k, size_t m_tiles, size_t n_tiles, const EpNode *nodes,
                      uint32_t n_nodes) __arm_streaming __arm_inout("za") {
    svbool_t p32 = svptrue_b32();
    const svfloat32_t z32 = svdup_n_f32(0.0f);
    const svfloat32_t vb = svdup_n_f32(1.0f);
    const svfloat32_t va = svdup_n_f32(0.0f);
    int has_ep = nodes && n_nodes > 0;
    size_t per_band = ep_cmul(k, 16);
    const float *a_lo = a_scr;
    const float *a_hi = a_scr + per_band;
    for (size_t mt = 0; mt < m_tiles; mt++) {
        size_t m0 = mt * 32;
        size_t mr = m - m0 < 32 ? m - m0 : 32;
        size_t mr_lo = mr < 16 ? mr : 16;
        for (size_t d0 = 0; d0 < k; d0 += 16) {
            size_t dc = k - d0 < 16 ? k - d0 : 16;
            svbool_t pk = svwhilelt_b32((uint64_t)0, (uint64_t)dc);
            for (size_t band = 0; band < 2; band++) {
                svzero_za();
                for (size_t r = band * 16; r < mr && r < band * 16 + 16; r++)
                    svld1_hor_za32(0, (uint32_t)(r - band * 16), pk, a + (m0 + r) * k + d0);
                for (size_t c = 0; c < dc; c++)
                    svst1_ver_za32(0, (uint32_t)c, p32, a_scr + band * per_band + (d0 + c) * 16);
            }
        }
        for (size_t nt = 0; nt < n_tiles; nt++) {
            size_t n0 = nt * 32;
            size_t nc = n - n0 < 32 ? n - n0 : 32;
            svbool_t pbl = svwhilelt_b32((uint64_t)n0, (uint64_t)n);
            svbool_t pbh = nc > 16 ? svwhilelt_b32((uint64_t)(n0 + 16), (uint64_t)n) : svpfalse_b();
            const float *b_lo = b + n0;
            const float *b_hi = nc > 16 ? b + n0 + 16 : b;

            svzero_za();
            size_t d = 0;
            for (; d + 2 <= k; d += 2) {
                svfloat32_t al0 = svld1_f32(p32, a_lo + d * 16);
                svfloat32_t ah0 = svld1_f32(p32, a_hi + d * 16);
                svfloat32_t bl0 = svld1_f32(pbl, b_lo + d * n);
                svfloat32_t bh0 = svld1_f32(pbh, b_hi + d * n);
                svfloat32_t al1 = svld1_f32(p32, a_lo + (d + 1) * 16);
                svfloat32_t ah1 = svld1_f32(p32, a_hi + (d + 1) * 16);
                svfloat32_t bl1 = svld1_f32(pbl, b_lo + (d + 1) * n);
                svfloat32_t bh1 = svld1_f32(pbh, b_hi + (d + 1) * n);
                F32_STEP_FULL(al0, ah0, bl0, bh0);
                F32_STEP_FULL(al1, ah1, bl1, bh1);
            }
            for (; d < k; d++) {
                svfloat32_t al = svld1_f32(p32, a_lo + d * 16);
                svfloat32_t ah = svld1_f32(p32, a_hi + d * 16);
                svfloat32_t bl = svld1_f32(pbl, b_lo + d * n);
                svfloat32_t bh = svld1_f32(pbh, b_hi + d * n);
                F32_STEP_FULL(al, ah, bl, bh);
            }

            svbool_t plo = svwhilelt_b32((uint32_t)0, (uint32_t)(nc < 16 ? nc : 16));
            svbool_t phi = svwhilelt_b32((uint32_t)0, (uint32_t)(nc > 16 ? nc - 16 : 0));
            float *rbase = dst + m0 * n + n0;
            if (has_ep) {
#define EP_RDL0(s) svread_hor_za32_m(z32, p32, 0, (s))
#define EP_RDH0(s) svread_hor_za32_m(z32, p32, 1, (s))
                EP_STORE_TILE_ROWMAJOR_F32(p32, plo, phi, rbase, (long)n, EP_RDL0, EP_RDH0, 0, mr_lo,
                                           vb, va, /*read_dst=*/0, nodes, n_nodes, m0, n0);
#undef EP_RDL0
#undef EP_RDH0
#define EP_RDL2(s) svread_hor_za32_m(z32, p32, 2, (s))
#define EP_RDH2(s) svread_hor_za32_m(z32, p32, 3, (s))
                EP_STORE_TILE_ROWMAJOR_F32(p32, plo, phi, rbase, (long)n, EP_RDL2, EP_RDH2, 16, mr,
                                           vb, va, /*read_dst=*/0, nodes, n_nodes, m0, n0);
#undef EP_RDL2
#undef EP_RDH2
            } else {
                // clamp so no one-past-end pointer is formed; the hi-N band is
                // empty (phi all-false) when narrow-N (nc<=16).
                int wide = nc > 16;
                for (size_t r = 0; r < mr_lo; r++) {
                    float *rp = dst + (m0 + r) * n + n0;
                    svst1_hor_za32(0, (uint32_t)r, plo, rp);
                    svst1_hor_za32(1, (uint32_t)r, phi, wide ? rp + 16 : rp);
                }
                for (size_t r = 16; r < mr; r++) {
                    uint32_t rr = (uint32_t)(r - 16);
                    float *rp = dst + (m0 + r) * n + n0;
                    svst1_hor_za32(2, rr, plo, rp);
                    svst1_hor_za32(3, rr, phi, wide ? rp + 16 : rp);
                }
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void run_batched(
    size_t count, float *dst, const float *lhs, const float *rhs, float *a_scr, size_t m, size_t n,
    size_t k, size_t m_tiles, size_t n_tiles, const EpNode *nodes, uint32_t n_nodes) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(dst + bi * m * n, lhs + bi * m * k, rhs + bi * k * n, a_scr, m, n, k, m_tiles,
                  n_tiles, nodes, n_nodes);
}

// Batched f32 GEMM with an optional fused op-graph epilogue applied to EACH item
// (same nodes/operands for all items). `ep` NULL or n_nodes==0 is the raw-store
// path. `count` independent row-major C_i = A_i @ B_i (same shape), one streaming
// session. lhs/rhs/dst are contiguous batches (item strides m*k / k*n / m*n).
int gemm_sme_f32_batched_ep(size_t count, size_t m, size_t n, size_t k, float *dst,
                            const float *lhs, const float *rhs, const ep_desc_f32 *ep) {
    if (count == 0 || m == 0 || n == 0) return 0;
    int has_ep = ep && ep->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    // Two 16-row A bands for one M-tile, rounded up to whole transpose blocks.
    float *a_scr = apack_scratch(ep_cmul(ep_cmul((k + 15) & ~(size_t)15, 32), sizeof(float)));
    if (!a_scr) return -1;
    run_batched(count, dst, lhs, rhs, a_scr, m, n, k, m_tiles, n_tiles, nodes, n_nodes);
    return 0;
}

// Raw batched f32 (no epilogue) -- thin wrapper.
int gemm_sme_f32_batched(size_t count, size_t m, size_t n, size_t k, float *dst, const float *lhs,
                         const float *rhs) {
    return gemm_sme_f32_batched_ep(count, m, n, k, dst, lhs, rhs, NULL);
}

#endif // SME_GEMM_F32_BATCHED_H
