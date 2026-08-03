// Batched f16: `count` independent same-shape GEMMs in ONE streaming session, so
// the streaming entry/exit is paid once for the whole batch rather than per item.
// Split out of gemm_f16f16.c (which owns the packing helpers and scratch this
// closes over) purely to keep both files navigable; it is included in place.
#ifndef SME_GEMM_F16F16_BATCHED_H
#define SME_GEMM_F16F16_BATCHED_H

// One row-major GEMM (C = A@B overwrite, packed A/B), already in streaming mode
// -- no SMSTART/SMSTOP of its own, so a batch pays the streaming entry once.
// `ep` (NULL / n_nodes==0 = raw store) applies the fused op-graph at the
// per-row store, mirroring run_streaming's row-major non-pure path: the output
// is row-major so COL/TENSOR are N-vector loads and ROW/SCALAR splats (span_n=1).
static void batch_one(f16 *dst, const f16 *a_pack, const f16 *b_pack, size_t m, size_t n, size_t k,
                      size_t per_tile, size_t n_tiles, const EpNode *nodes,
                      uint32_t n_nodes) __arm_streaming __arm_inout("za") {
    svbool_t p16 = svptrue_b16();
    const svfloat16_t z16 = svdup_n_f16((f16)0.0f);
    const svfloat16_t vb = svdup_n_f16((f16)1.0f);
    const svfloat16_t va = svdup_n_f16((f16)0.0f);
    int has_ep = nodes && n_nodes > 0;
    size_t m_tiles = (m + 31) / 32;
    for (size_t mt = 0; mt < m_tiles; mt++) {
        const f16 *at = a_pack + mt * per_tile;
        size_t m0 = mt * 32;
        size_t mrows = (m - m0 < 32) ? (m - m0) : 32;
        for (size_t nt = 0; nt < n_tiles; nt += 2) {
            const f16 *b0 = b_pack + nt * per_tile;
            const f16 *b1 = b_pack + (nt + 1) * per_tile;
            svzero_za();
            for (size_t d = 0; d < k; d++) {
                svfloat16_t av = svld1_f16(p16, at + d * 32);
                svmopa_za16_m(0, p16, p16, av, svld1_f16(p16, b0 + d * 32));
                svmopa_za16_m(1, p16, p16, av, svld1_f16(p16, b1 + d * 32));
            }
            for (size_t t = 0; t < 2; t++) {
                size_t ntt = nt + t;
                if (ntt >= n_tiles) break;
                size_t n0 = ntt * 32;
                size_t ncols = (n - n0 < 32) ? (n - n0) : 32;
                svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                if (has_ep) {
                    f16 *base = dst + m0 * n + n0;
                    if (t == 0) {
#define EP_RD0(s) svread_hor_za16_m(z16, p16, 0, (s))
                        EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, (long)n, EP_RD0, mrows, vb, va,
                                                   /*read_dst=*/0, nodes, n_nodes, m0, n0);
#undef EP_RD0
                    } else {
#define EP_RD1(s) svread_hor_za16_m(z16, p16, 1, (s))
                        EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, (long)n, EP_RD1, mrows, vb, va,
                                                   /*read_dst=*/0, nodes, n_nodes, m0, n0);
#undef EP_RD1
                    }
                } else if (t == 0) {
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(0, (uint32_t)r, pst, dst + (m0 + r) * n + n0);
                } else {
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(1, (uint32_t)r, pst, dst + (m0 + r) * n + n0);
                }
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void run_batched(
    size_t count, f16 *dst, const f16 *a_packs, const f16 *b_packs, size_t m, size_t n, size_t k,
    size_t per_tile, size_t n_tiles, size_t a_per, size_t b_per, size_t c_per, const EpNode *nodes,
    uint32_t n_nodes) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(dst + bi * c_per, a_packs + bi * a_per, b_packs + bi * b_per, m, n, k, per_tile,
                  n_tiles, nodes, n_nodes);
}

// Batched f16f16 GEMM with an optional fused op-graph epilogue applied to EACH
// item (same nodes/operands for all items -- e.g. shared bias+activation). `ep`
// NULL or n_nodes==0 is the raw-store path (bit-identical to the no-ep ABI).
// `count` independent row-major C_i = A_i @ B_i (same shape), one streaming-mode
// session. lhs/rhs/dst are contiguous batches (item strides m*k / k*n / m*n).
int gemm_sme_f16f16_batched_ep(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                               const uint16_t *lhs, const uint16_t *rhs, const ep_desc16 *ep) {
    if (count == 0 || m == 0 || n == 0) return 0;
    int has_ep = ep && ep->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    size_t n_tiles_pad = (n_tiles + 1) & ~(size_t)1;
    size_t per_tile = ep_cmul(k, 32);
    size_t a_per = m_tiles * per_tile, b_per = n_tiles_pad * per_tile, c_per = m * n;

    // Pack every item's A and B (NEON, outside streaming mode). `count` is the one
    // factor not bounded by a C-side size invariant, so guard the pack-buffer
    // products against overflow (the calloc nmemb `count*b_per` is multiplied
    // before calloc can check it). Return -1 -> the Rust side falls back.
    size_t a_bytes, b_elems;
    if (__builtin_mul_overflow(count, a_per, &a_bytes) ||
        __builtin_mul_overflow(a_bytes, sizeof(f16), &a_bytes) ||
        __builtin_mul_overflow(count, b_per, &b_elems)) {
        return -1;
    }
    f16 *a_packs = (f16 *)malloc(a_bytes);
    f16 *b_packs = (f16 *)calloc(b_elems, sizeof(f16));
    if (!a_packs || !b_packs) {
        free(a_packs);
        free(b_packs);
        return -1;
    }
    for (size_t bi = 0; bi < count; bi++) {
        packa(a_packs + bi * a_per, (const f16 *)lhs + bi * (m * k), m, k, k, 1, 0, m_tiles,
              per_tile);
        packb(b_packs + bi * b_per, (const f16 *)rhs + bi * (k * n), n, k, n, 1, n_tiles, per_tile);
    }
    run_batched(count, (f16 *)dst, a_packs, b_packs, m, n, k, per_tile, n_tiles, a_per, b_per,
                c_per, nodes, n_nodes);
    free(a_packs);
    free(b_packs);
    return 0;
}

// Raw batched f16f16 (no epilogue) -- unchanged ABI, thin wrapper.
int gemm_sme_f16f16_batched(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                            const uint16_t *lhs, const uint16_t *rhs) {
    return gemm_sme_f16f16_batched_ep(count, m, n, k, dst, lhs, rhs, NULL);
}

#endif // SME_GEMM_F16F16_BATCHED_H
