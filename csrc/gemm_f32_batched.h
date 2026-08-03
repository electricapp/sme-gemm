// Batched f32: `count` independent same-shape GEMMs in ONE streaming session, so
// the streaming entry/exit is paid once for the whole batch rather than per item.
// Split out of gemm_f32.c (which owns the packing helpers and scratch this
// closes over) purely to keep both files navigable; it is included in place.
#ifndef SME_GEMM_F32_BATCHED_H
#define SME_GEMM_F32_BATCHED_H

// One row-major GEMM (C = A@B overwrite, packed A/B), already in streaming mode
// -- no SMSTART/SMSTOP of its own, so a batch pays the streaming entry once.
// `ep` (NULL / n_nodes==0 = raw store) applies the fused op-graph at the per-row
// store, mirroring run_streaming's row-major path: the output is row-major so
// COL/TENSOR are N-vector loads and ROW/SCALAR splats (span_n=1). A is packed
// [2*m_tiles][k,16] bands, B is packed [2*n_tiles][k,16] bands (caller zeroes B
// pad lanes).
static void batch_one(float *dst, const float *a_pack, const float *b_pack, size_t m, size_t n,
                      size_t k, size_t per_tile, size_t m_tiles, size_t n_tiles,
                      const EpNode *nodes, uint32_t n_nodes) __arm_streaming __arm_inout("za") {
    svbool_t p32 = svptrue_b32();
    const svfloat32_t z32 = svdup_n_f32(0.0f);
    const svfloat32_t vb = svdup_n_f32(1.0f);
    const svfloat32_t va = svdup_n_f32(0.0f);
    int has_ep = nodes && n_nodes > 0;
    for (size_t mt = 0; mt < m_tiles; mt++) {
        const float *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
        const float *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
        size_t m0 = mt * 32;
        size_t mr = m - m0 < 32 ? m - m0 : 32;
        size_t mr_lo = mr < 16 ? mr : 16;
        for (size_t nt = 0; nt < n_tiles; nt++) {
            const float *b_lo = b_pack + (size_t)(2 * nt) * per_tile;
            const float *b_hi = b_pack + (size_t)(2 * nt + 1) * per_tile;
            size_t n0 = nt * 32;
            size_t nc = n - n0 < 32 ? n - n0 : 32;

            svzero_za();
            // Unrolled by 2, as run_streaming: runs the second step's loads
            // ahead of the first step's MOPAs instead of one load per MOPA group.
            size_t d = 0;
            for (; d + 2 <= k; d += 2) {
                svfloat32_t al0 = svld1_f32(p32, a_lo + d * 16);
                svfloat32_t ah0 = svld1_f32(p32, a_hi + d * 16);
                svfloat32_t bl0 = svld1_f32(p32, b_lo + d * 16);
                svfloat32_t bh0 = svld1_f32(p32, b_hi + d * 16);
                svfloat32_t al1 = svld1_f32(p32, a_lo + (d + 1) * 16);
                svfloat32_t ah1 = svld1_f32(p32, a_hi + (d + 1) * 16);
                svfloat32_t bl1 = svld1_f32(p32, b_lo + (d + 1) * 16);
                svfloat32_t bh1 = svld1_f32(p32, b_hi + (d + 1) * 16);
                F32_STEP_FULL(al0, ah0, bl0, bh0);
                F32_STEP_FULL(al1, ah1, bl1, bh1);
            }
            for (; d < k; d++) {
                svfloat32_t al = svld1_f32(p32, a_lo + d * 16);
                svfloat32_t ah = svld1_f32(p32, a_hi + d * 16);
                svfloat32_t bl = svld1_f32(p32, b_lo + d * 16);
                svfloat32_t bh = svld1_f32(p32, b_hi + d * 16);
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
    size_t count, float *dst, const float *a_packs, const float *b_packs, size_t m, size_t n,
    size_t k, size_t per_tile, size_t m_tiles, size_t n_tiles, size_t a_per, size_t b_per,
    size_t c_per, const EpNode *nodes, uint32_t n_nodes) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(dst + bi * c_per, a_packs + bi * a_per, b_packs + bi * b_per, m, n, k, per_tile,
                  m_tiles, n_tiles, nodes, n_nodes);
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
    size_t per_tile = ep_cmul(k, 16);
    size_t a_per = (size_t)2 * m_tiles * per_tile;
    size_t b_per = (size_t)2 * n_tiles * per_tile;
    size_t c_per = m * n;

    // Pack every item's A and B (NEON, outside streaming mode). calloc on B so
    // the partial-band lanes past n are zero. `count` is the one factor not bounded
    // by a C-side size invariant, so guard the products against overflow (the calloc
    // nmemb `count*b_per` is multiplied before calloc can check it). -1 -> Rust falls back.
    size_t a_bytes, b_elems;
    if (__builtin_mul_overflow(count, a_per, &a_bytes) ||
        __builtin_mul_overflow(a_bytes, sizeof(float), &a_bytes) ||
        __builtin_mul_overflow(count, b_per, &b_elems)) {
        return -1;
    }
    float *a_packs = (float *)malloc(a_bytes);
    float *b_packs = (float *)calloc(b_elems, sizeof(float));
    if (!a_packs || !b_packs) {
        free(a_packs);
        free(b_packs);
        return -1;
    }
    for (size_t bi = 0; bi < count; bi++) {
        const float *a = lhs + bi * (m * k);
        const float *b = rhs + bi * (k * n);
        float *ap = a_packs + bi * a_per;
        float *bp = b_packs + bi * b_per;
        for (size_t st = 0; st < 2 * m_tiles; st++) {
            size_t r0 = st * 16;
            size_t vr = (r0 < m) ? ((m - r0 < 16) ? (m - r0) : 16) : 0;
            pack_band(ap + st * per_tile, vr ? a + (long)r0 * (long)k : a, (long)k, 1, k, vr);
        }
        for (size_t st = 0; st < 2 * n_tiles; st++) {
            size_t c0 = st * 16;
            size_t vc = (c0 < n) ? ((n - c0 < 16) ? (n - c0) : 16) : 0;
            pack_band(bp + st * per_tile, vc ? b + (long)c0 : b, 1, (long)n, k, vc);
        }
    }
    run_batched(count, dst, a_packs, b_packs, m, n, k, per_tile, m_tiles, n_tiles, a_per, b_per,
                c_per, nodes, n_nodes);
    free(a_packs);
    free(b_packs);
    return 0;
}

// Raw batched f32 (no epilogue) -- thin wrapper.
int gemm_sme_f32_batched(size_t count, size_t m, size_t n, size_t k, float *dst, const float *lhs,
                         const float *rhs) {
    return gemm_sme_f32_batched_ep(count, m, n, k, dst, lhs, rhs, NULL);
}

#endif // SME_GEMM_F32_BATCHED_H
