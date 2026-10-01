// Batched f64: `count` independent same-shape GEMMs in ONE streaming session, so
// the streaming entry/exit is paid once for the whole batch rather than per item.
// Split out of gemm_f64.c (which owns the packing helpers and scratch this
// closes over) purely to keep both files navigable; it is included in place.
#ifndef SME_GEMM_F64_BATCHED_H
#define SME_GEMM_F64_BATCHED_H

// One row-major GEMM (C = A@B overwrite), already in streaming mode. B is read
// straight from the row-major input and each 8-row A band is transposed through
// ZA64 tile 0 into `a_scr` ([2][k][8]); see gemm_f16f16_batched.h.
static void batch_one(double *dst, const double *a, const double *b, double *a_scr, size_t m,
                      size_t n, size_t k, size_t m_tiles, size_t n_tiles, const EpNode *nodes,
                      uint32_t n_nodes) __arm_streaming __arm_inout("za") {
    svbool_t p64 = svptrue_b64();
    const svfloat64_t z64 = svdup_n_f64(0.0);
    const svfloat64_t vb = svdup_n_f64(1.0);
    const svfloat64_t va = svdup_n_f64(0.0);
    int has_ep = nodes && n_nodes > 0;
    size_t per_band = ep_cmul(k, 8);
    const double *a_lo = a_scr;
    const double *a_hi = a_scr + per_band;
    for (size_t mt = 0; mt < m_tiles; mt++) {
        size_t m0 = mt * 16;
        size_t mr = m - m0 < 16 ? m - m0 : 16;
        size_t mr_lo = mr < 8 ? mr : 8;
        for (size_t d0 = 0; d0 < k; d0 += 8) {
            size_t dc = k - d0 < 8 ? k - d0 : 8;
            svbool_t pk = svwhilelt_b64((uint64_t)0, (uint64_t)dc);
            for (size_t band = 0; band < 2; band++) {
                svzero_za();
                for (size_t r = band * 8; r < mr && r < band * 8 + 8; r++)
                    svld1_hor_za64(0, (uint32_t)(r - band * 8), pk, a + (m0 + r) * k + d0);
                for (size_t c = 0; c < dc; c++)
                    svst1_ver_za64(0, (uint32_t)c, p64, a_scr + band * per_band + (d0 + c) * 8);
            }
        }
        for (size_t nt = 0; nt < n_tiles; nt++) {
            size_t n0 = nt * 16;
            size_t nc = n - n0 < 16 ? n - n0 : 16;
            svbool_t pbl = svwhilelt_b64((uint64_t)n0, (uint64_t)n);
            svbool_t pbh = nc > 8 ? svwhilelt_b64((uint64_t)(n0 + 8), (uint64_t)n) : svpfalse_b();
            const double *b_lo = b + n0;
            const double *b_hi = nc > 8 ? b + n0 + 8 : b;

            svzero_za();
            for (size_t d = 0; d < k; d++) {
                svfloat64_t al = svld1_f64(p64, a_lo + d * 8);
                svfloat64_t ah = svld1_f64(p64, a_hi + d * 8);
                svfloat64_t bl = svld1_f64(pbl, b_lo + d * n);
                svfloat64_t bh = svld1_f64(pbh, b_hi + d * n);
                svmopa_za64_f64_m(0, p64, p64, al, bl);
                svmopa_za64_f64_m(1, p64, p64, al, bh);
                svmopa_za64_f64_m(2, p64, p64, ah, bl);
                svmopa_za64_f64_m(3, p64, p64, ah, bh);
            }

            svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)(nc < 8 ? nc : 8));
            svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)(nc > 8 ? nc - 8 : 0));
            double *rbase = dst + m0 * n + n0;
            if (has_ep) {
#define EP_RDL0(s) svread_hor_za64_f64_m(z64, p64, 0, (s))
#define EP_RDH0(s) svread_hor_za64_f64_m(z64, p64, 1, (s))
                EP_STORE_TILE_ROWMAJOR_F64(p64, plo, phi, rbase, (long)n, EP_RDL0, EP_RDH0, 0, mr_lo,
                                           vb, va, /*read_dst=*/0, nodes, n_nodes, m0, n0);
#undef EP_RDL0
#undef EP_RDH0
#define EP_RDL2(s) svread_hor_za64_f64_m(z64, p64, 2, (s))
#define EP_RDH2(s) svread_hor_za64_f64_m(z64, p64, 3, (s))
                EP_STORE_TILE_ROWMAJOR_F64(p64, plo, phi, rbase, (long)n, EP_RDL2, EP_RDH2, 8, mr,
                                           vb, va, /*read_dst=*/0, nodes, n_nodes, m0, n0);
#undef EP_RDL2
#undef EP_RDH2
            } else {
                // clamp so no one-past-end pointer is formed; the hi-N band is
                // empty (phi all-false) when narrow-N (nc<=8).
                int wide = nc > 8;
                for (size_t r = 0; r < mr_lo; r++) {
                    double *rp = dst + (m0 + r) * n + n0;
                    svst1_hor_za64(0, (uint32_t)r, plo, rp);
                    svst1_hor_za64(1, (uint32_t)r, phi, wide ? rp + 8 : rp);
                }
                for (size_t r = 8; r < mr; r++) {
                    uint32_t rr = (uint32_t)(r - 8);
                    double *rp = dst + (m0 + r) * n + n0;
                    svst1_hor_za64(2, rr, plo, rp);
                    svst1_hor_za64(3, rr, phi, wide ? rp + 8 : rp);
                }
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void run_batched(
    size_t count, double *dst, const double *lhs, const double *rhs, double *a_scr, size_t m,
    size_t n, size_t k, size_t m_tiles, size_t n_tiles, const EpNode *nodes, uint32_t n_nodes) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(dst + bi * m * n, lhs + bi * m * k, rhs + bi * k * n, a_scr, m, n, k, m_tiles,
                  n_tiles, nodes, n_nodes);
}

// Batched f64 GEMM with an optional fused op-graph epilogue applied to EACH item
// (same nodes/operands for all items). `ep` NULL or n_nodes==0 is the raw-store
// path. `count` independent row-major C_i = A_i @ B_i (same shape), one streaming
// session. lhs/rhs/dst are contiguous batches (item strides m*k / k*n / m*n).
// M5-only (FEAT_SME_F64F64); the caller gates on the f64 cap.
int gemm_sme_f64_batched_ep(size_t count, size_t m, size_t n, size_t k, double *dst,
                            const double *lhs, const double *rhs, const ep_desc_f64 *ep) {
    if (count == 0 || m == 0 || n == 0) return 0;
    int has_ep = ep && ep->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t m_tiles = (m + 15) / 16;
    size_t n_tiles = (n + 15) / 16;
    // Two 8-row A bands for one M-tile, rounded up to whole transpose blocks.
    double *a_scr = apack_scratch(ep_cmul(ep_cmul((k + 7) & ~(size_t)7, 16), sizeof(double)));
    if (!a_scr) return -1;
    run_batched(count, dst, lhs, rhs, a_scr, m, n, k, m_tiles, n_tiles, nodes, n_nodes);
    return 0;
}

// Raw batched f64 (no epilogue) -- thin wrapper.
int gemm_sme_f64_batched(size_t count, size_t m, size_t n, size_t k, double *dst, const double *lhs,
                         const double *rhs) {
    return gemm_sme_f64_batched_ep(count, m, n, k, dst, lhs, rhs, NULL);
}

#endif // SME_GEMM_F64_BATCHED_H
