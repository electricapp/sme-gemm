// Batched f16: `count` independent same-shape GEMMs in ONE streaming session, so
// the streaming entry/exit is paid once for the whole batch rather than per item.
// Split out of gemm_f16f16.c (which owns the packing helpers and scratch this
// closes over) purely to keep both files navigable; it is included in place.
#ifndef SME_GEMM_F16F16_BATCHED_H
#define SME_GEMM_F16F16_BATCHED_H

// One row-major GEMM (C = A@B overwrite), already in streaming mode. B is read
// straight from the row-major input (a 32-column tile at depth d is contiguous)
// and each A tile is transposed through ZA tile 0 into `a_scr` ([k][32]), so
// nothing is packed outside the streaming session. `ep` applies the op-graph
// at the per-row store, as run_streaming's row-major path does.
static void batch_one(f16 *dst, const f16 *a, const f16 *b, f16 *a_scr, size_t m, size_t n,
                      size_t k, size_t n_tiles, const EpNode *nodes,
                      uint32_t n_nodes) __arm_streaming __arm_inout("za") {
    svbool_t p16 = svptrue_b16();
    const svfloat16_t z16 = svdup_n_f16((f16)0.0f);
    const svfloat16_t vb = svdup_n_f16((f16)1.0f);
    const svfloat16_t va = svdup_n_f16((f16)0.0f);
    int has_ep = nodes && n_nodes > 0;
    size_t m_tiles = (m + 31) / 32;
    for (size_t mt = 0; mt < m_tiles; mt++) {
        size_t m0 = mt * 32;
        size_t mrows = (m - m0 < 32) ? (m - m0) : 32;
        for (size_t d0 = 0; d0 < k; d0 += 32) {
            size_t dc = k - d0 < 32 ? k - d0 : 32;
            svbool_t pk = svwhilelt_b16((uint64_t)0, (uint64_t)dc);
            svzero_za();
            for (size_t r = 0; r < mrows; r++)
                svld1_hor_za16(0, (uint32_t)r, pk, a + (m0 + r) * k + d0);
            for (size_t c = 0; c < dc; c++)
                svst1_ver_za16(0, (uint32_t)c, p16, a_scr + (d0 + c) * 32);
        }
        for (size_t nt = 0; nt < n_tiles; nt += 2) {
            size_t n0 = nt * 32;
            size_t n1 = n0 + 32;
            svbool_t pb0 = svwhilelt_b16((uint64_t)n0, (uint64_t)n);
            svbool_t pb1 = n1 < n ? svwhilelt_b16((uint64_t)n1, (uint64_t)n) : svpfalse_b();
            const f16 *b1 = n1 < n ? b + n1 : b;
            svzero_za();
            for (size_t d = 0; d < k; d++) {
                svfloat16_t av = svld1_f16(p16, a_scr + d * 32);
                svmopa_za16_m(0, p16, p16, av, svld1_f16(pb0, b + d * n + n0));
                svmopa_za16_m(1, p16, p16, av, svld1_f16(pb1, b1 + d * n));
            }
            for (size_t t = 0; t < 2; t++) {
                size_t ntt = nt + t;
                if (ntt >= n_tiles) break;
                size_t nc0 = ntt * 32;
                size_t ncols = (n - nc0 < 32) ? (n - nc0) : 32;
                svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                if (has_ep) {
                    f16 *base = dst + m0 * n + nc0;
                    if (t == 0) {
#define EP_RD0(s) svread_hor_za16_m(z16, p16, 0, (s))
                        EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, (long)n, EP_RD0, mrows, vb, va,
                                                   /*read_dst=*/0, nodes, n_nodes, m0, nc0);
#undef EP_RD0
                    } else {
#define EP_RD1(s) svread_hor_za16_m(z16, p16, 1, (s))
                        EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, (long)n, EP_RD1, mrows, vb, va,
                                                   /*read_dst=*/0, nodes, n_nodes, m0, nc0);
#undef EP_RD1
                    }
                } else if (t == 0) {
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(0, (uint32_t)r, pst, dst + (m0 + r) * n + nc0);
                } else {
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(1, (uint32_t)r, pst, dst + (m0 + r) * n + nc0);
                }
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void run_batched(
    size_t count, f16 *dst, const f16 *lhs, const f16 *rhs, f16 *a_scr, size_t m, size_t n,
    size_t k, size_t n_tiles, const EpNode *nodes, uint32_t n_nodes) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(dst + bi * m * n, lhs + bi * m * k, rhs + bi * k * n, a_scr, m, n, k, n_tiles,
                  nodes, n_nodes);
}

// Batched f16f16 GEMM with an optional fused op-graph epilogue applied to EACH
// item (same nodes/operands for all items -- e.g. shared bias+activation). `ep`
// NULL or n_nodes==0 is the raw-store path (bit-identical to the no-ep ABI).
// `count` independent row-major C_i = A_i @ B_i (same shape), one streaming-mode
// session. lhs/rhs/dst are contiguous batches (item strides m*k / k*n / m*n).
static int batched_ep_core(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                           const uint16_t *lhs, const uint16_t *rhs, const ep_desc16 *ep) {
    if (count == 0 || m == 0 || n == 0) return 0;
    int has_ep = ep && ep->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    size_t n_tiles = (n + 31) / 32;
    // A scratch for one M-tile, rounded up to whole 32-deep transpose blocks.
    f16 *a_scr = apack_scratch(ep_cmul(ep_cmul((k + 31) & ~(size_t)31, 32), sizeof(f16)));
    if (!a_scr) return -1;
    run_batched(count, (f16 *)dst, (const f16 *)lhs, (const f16 *)rhs, a_scr, m, n, k, n_tiles,
                nodes, n_nodes);
    return 0;
}

// A trailing gelu/silu/sigmoid/tanh runs as a NEON pass over the output after
// the kernel (neon_act.h): in a streaming epilogue it costs ~2.5 ns an output.
int gemm_sme_f16f16_batched_ep(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                               const uint16_t *lhs, const uint16_t *rhs, const ep_desc16 *ep) {
    ep_desc16 rest;
    uint32_t act = na_split(ep, &rest);
    const ep_desc16 *kep = na_kernel_ep(ep, act, &rest);
    int rc = batched_ep_core(count, m, n, k, dst, lhs, rhs, kep);
    if (rc == 0 && act) NA_NEON_PHASE(na_post_f16(dst, count * m, n, (long)n, 1, act));
    return rc;
}

// Raw batched f16f16 (no epilogue) -- unchanged ABI, thin wrapper.
int gemm_sme_f16f16_batched(size_t count, size_t m, size_t n, size_t k, uint16_t *dst,
                            const uint16_t *lhs, const uint16_t *rhs) {
    return gemm_sme_f16f16_batched_ep(count, m, n, k, dst, lhs, rhs, NULL);
}

#endif // SME_GEMM_F16F16_BATCHED_H
