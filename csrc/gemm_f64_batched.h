// Batched f64: `count` independent same-shape GEMMs in ONE streaming session, so
// the streaming entry/exit is paid once for the whole batch rather than per item.
// Split out of gemm_f64.c (which owns the packing helpers and scratch this
// closes over) purely to keep both files navigable; it is included in place.
#ifndef SME_GEMM_F64_BATCHED_H
#define SME_GEMM_F64_BATCHED_H

// One row-major GEMM (C = A@B overwrite, packed A/B), already in streaming mode
// -- no SMSTART/SMSTOP of its own, so a batch pays the streaming entry once.
// `ep` (NULL / n_nodes==0 = raw store) applies the fused op-graph at the per-row
// store, mirroring run_streaming's row-major path: the output is row-major so
// COL/TENSOR are N-vector loads and ROW/SCALAR splats (span_n=1). A is packed
// [2*m_tiles][k,8] bands, B is packed [2*n_tiles][k,8] bands (caller zeroes B
// pad lanes). f64 16x16 super-tile = four 8x8 ZA64 quadrants.
static void batch_one(double *dst, const double *a_pack, const double *b_pack, size_t m, size_t n,
                      size_t k, size_t per_tile, size_t m_tiles, size_t n_tiles,
                      const EpNode *nodes, uint32_t n_nodes) __arm_streaming __arm_inout("za") {
    svbool_t p64 = svptrue_b64();
    const svfloat64_t z64 = svdup_n_f64(0.0);
    const svfloat64_t vb = svdup_n_f64(1.0);
    const svfloat64_t va = svdup_n_f64(0.0);
    int has_ep = nodes && n_nodes > 0;
    for (size_t mt = 0; mt < m_tiles; mt++) {
        const double *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
        const double *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
        size_t m0 = mt * 16;
        size_t mr = m - m0 < 16 ? m - m0 : 16;
        size_t mr_lo = mr < 8 ? mr : 8;
        for (size_t nt = 0; nt < n_tiles; nt++) {
            const double *b_lo = b_pack + (size_t)(2 * nt) * per_tile;
            const double *b_hi = b_pack + (size_t)(2 * nt + 1) * per_tile;
            size_t n0 = nt * 16;
            size_t nc = n - n0 < 16 ? n - n0 : 16;

            svzero_za();
            for (size_t d = 0; d < k; d++) {
                svfloat64_t al = svld1_f64(p64, a_lo + d * 8);
                svfloat64_t ah = svld1_f64(p64, a_hi + d * 8);
                svfloat64_t bl = svld1_f64(p64, b_lo + d * 8);
                svfloat64_t bh = svld1_f64(p64, b_hi + d * 8);
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
    size_t count, double *dst, const double *a_packs, const double *b_packs, size_t m, size_t n,
    size_t k, size_t per_tile, size_t m_tiles, size_t n_tiles, size_t a_per, size_t b_per,
    size_t c_per, const EpNode *nodes, uint32_t n_nodes) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(dst + bi * c_per, a_packs + bi * a_per, b_packs + bi * b_per, m, n, k, per_tile,
                  m_tiles, n_tiles, nodes, n_nodes);
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
    size_t per_tile = ep_cmul(k, 8);
    size_t a_per = (size_t)2 * m_tiles * per_tile;
    size_t b_per = (size_t)2 * n_tiles * per_tile;
    size_t c_per = m * n;

    // Pack every item's A and B (NEON, outside streaming mode). calloc on B so
    // the partial-band lanes past n are zero. `count` is the one factor not bounded
    // by a C-side size invariant, so guard the products against overflow (the calloc
    // nmemb `count*b_per` is multiplied before calloc can check it). -1 -> Rust falls back.
    size_t a_bytes, b_elems;
    if (__builtin_mul_overflow(count, a_per, &a_bytes) ||
        __builtin_mul_overflow(a_bytes, sizeof(double), &a_bytes) ||
        __builtin_mul_overflow(count, b_per, &b_elems)) {
        return -1;
    }
    double *a_packs = (double *)malloc(a_bytes);
    double *b_packs = (double *)calloc(b_elems, sizeof(double));
    if (!a_packs || !b_packs) {
        free(a_packs);
        free(b_packs);
        return -1;
    }
    for (size_t bi = 0; bi < count; bi++) {
        const double *a = lhs + bi * (m * k);
        const double *b = rhs + bi * (k * n);
        double *ap = a_packs + bi * a_per;
        double *bp = b_packs + bi * b_per;
        for (size_t st = 0; st < 2 * m_tiles; st++) {
            size_t r0 = st * 8;
            size_t vr = (r0 < m) ? ((m - r0 < 8) ? (m - r0) : 8) : 0;
            // Clamp the base for empty (vr/vc==0) bands so no out-of-bounds
            // pointer is formed (zero-filled without a deref either way).
            pack_band(ap + st * per_tile, vr ? a + (long)r0 * (long)k : a, (long)k, 1, k, vr);
        }
        for (size_t st = 0; st < 2 * n_tiles; st++) {
            size_t c0 = st * 8;
            size_t vc = (c0 < n) ? ((n - c0 < 8) ? (n - c0) : 8) : 0;
            pack_band(bp + st * per_tile, vc ? b + (long)c0 : b, 1, (long)n, k, vc);
        }
    }
    run_batched(count, dst, a_packs, b_packs, m, n, k, per_tile, m_tiles, n_tiles, a_per, b_per,
                c_per, nodes, n_nodes);
    free(a_packs);
    free(b_packs);
    return 0;
}

// Raw batched f64 (no epilogue) -- thin wrapper.
int gemm_sme_f64_batched(size_t count, size_t m, size_t n, size_t k, double *dst, const double *lhs,
                         const double *rhs) {
    return gemm_sme_f64_batched_ep(count, m, n, k, dst, lhs, rhs, NULL);
}

#endif // SME_GEMM_F64_BATCHED_H
