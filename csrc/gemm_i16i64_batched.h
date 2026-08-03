// Batched i16: `count` independent same-shape GEMMs in ONE streaming session, so
// the streaming entry/exit is paid once for the whole batch rather than per item.
// Split out of gemm_i16i64.c (which owns the packing helpers and scratch this
// closes over) purely to keep both files navigable; it is included in place.
#ifndef SME_GEMM_I16I64_BATCHED_H
#define SME_GEMM_I16I64_BATCHED_H

// One row-major GEMM (C(i64) = A(i16) @ B(i16)), already in streaming mode -- a
// batch pays the streaming entry once. See gemm_i8i32.c batch_one. When `dq` is
// NULL the raw i64 tile is stored; when set, the four i64 quadrants are read into
// a scratch and dequantized via the f32 op-graph (ep_dequant_cell_i64) into the
// f32 output `fdst` (row-major). This mirrors run_streaming's dq branch. A/B are
// packed [2*tiles][kp4,32] 4-way-interleaved bands; a 16x16 super-tile is four
// 8x8 i64 ZA quadrants.
static void batch_one(int64_t *dst, float *fdst, const int16_t *a_pack, const int16_t *b_pack,
                      size_t m, size_t n, size_t kp4, size_t per_tile, size_t n_tiles,
                      const ep_dq_f32 *dq) __arm_streaming __arm_inout("za") {
    svbool_t pb = svptrue_b16();
    svbool_t p64 = svptrue_b64();
    const svint64_t z64 = svdup_n_s64(0);
    size_t m_tiles = (m + 15) / 16;
    for (size_t mt = 0; mt < m_tiles; mt++) {
        const int16_t *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
        const int16_t *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
        size_t m0 = mt * 16;
        size_t mr = m - m0 < 16 ? m - m0 : 16;
        size_t mr_lo = mr < 8 ? mr : 8;
        for (size_t nt = 0; nt < n_tiles; nt++) {
            const int16_t *b_lo = b_pack + (size_t)(2 * nt) * per_tile;
            const int16_t *b_hi = b_pack + (size_t)(2 * nt + 1) * per_tile;
            size_t n0 = nt * 16;
            size_t nc = n - n0 < 16 ? n - n0 : 16;
            svzero_za();
            for (size_t p = 0; p < kp4; p++) {
                svint16_t al = svld1_s16(pb, a_lo + p * 32);
                svint16_t ah = svld1_s16(pb, a_hi + p * 32);
                svint16_t bl = svld1_s16(pb, b_lo + p * 32);
                svint16_t bh = svld1_s16(pb, b_hi + p * 32);
                svmopa_za64_s16_m(0, pb, pb, al, bl);
                svmopa_za64_s16_m(1, pb, pb, al, bh);
                svmopa_za64_s16_m(2, pb, pb, ah, bl);
                svmopa_za64_s16_m(3, pb, pb, ah, bh);
            }
            if (dq) {
                // Vectorized fused dequant store, as run_streaming's row-major
                // branch (dst is row-major m x n here, so row stride is n): read
                // the two i64 ZA quadrants per row, convert to f32 and uzp1-pack
                // them into one 16-lane row, scale, run the op-graph in-register,
                // store. ZA tile numbers are instruction immediates, hence the
                // split r<8 / r>=8 loops.
                svbool_t p32 = svptrue_b32();
                svbool_t pst = svwhilelt_b32((uint32_t)0, (uint32_t)nc);
                svfloat32_t sc =
                    dq->scale_n ? svld1_f32(pst, dq->scale_n + n0) : svdup_n_f32(dq->scale);
                const EpNode *nodes = dq->nodes;
                uint32_t n_nodes = dq->n_nodes;
                for (size_t r = 0; r < mr_lo; r++) {
                    svfloat32_t lo =
                        svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 0, (uint32_t)r));
                    svfloat32_t hi =
                        svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 1, (uint32_t)r));
                    svfloat32_t row = svmul_f32_x(p32, svuzp1_f32(lo, hi), sc);
                    row = ep_apply_nodes_f32(p32, pst, row, nodes, n_nodes, 1, m0 + r, n0, m0);
                    svst1_f32(pst, fdst + (m0 + r) * n + n0, row);
                }
                for (size_t r = 8; r < mr; r++) {
                    uint32_t rr = (uint32_t)(r - 8);
                    svfloat32_t lo = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 2, rr));
                    svfloat32_t hi = svcvt_f32_s64_x(p64, svread_hor_za64_s64_m(z64, p64, 3, rr));
                    svfloat32_t row = svmul_f32_x(p32, svuzp1_f32(lo, hi), sc);
                    row = ep_apply_nodes_f32(p32, pst, row, nodes, n_nodes, 1, m0 + r, n0, m0);
                    svst1_f32(pst, fdst + (m0 + r) * n + n0, row);
                }
                continue;
            }
            svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)(nc < 8 ? nc : 8));
            svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)(nc > 8 ? nc - 8 : 0));
            // clamp so no one-past-end pointer is formed; the hi-N band is
            // empty (phi all-false) when narrow-N (nc<=8).
            int wide = nc > 8;
            for (size_t r = 0; r < mr_lo; r++) {
                int64_t *rp = dst + (m0 + r) * n + n0;
                svst1_hor_za64(0, (uint32_t)r, plo, rp);
                svst1_hor_za64(1, (uint32_t)r, phi, wide ? rp + 8 : rp);
            }
            for (size_t r = 8; r < mr; r++) {
                uint32_t rr = (uint32_t)(r - 8);
                int64_t *rp = dst + (m0 + r) * n + n0;
                svst1_hor_za64(2, rr, plo, rp);
                svst1_hor_za64(3, rr, phi, wide ? rp + 8 : rp);
            }
        }
    }
}

__arm_locally_streaming __arm_new("za") static void run_batched(
    size_t count, int64_t *dst, float *fdst, const int16_t *a_packs, const int16_t *b_packs,
    size_t m, size_t n, size_t kp4, size_t per_tile, size_t n_tiles, size_t a_per, size_t b_per,
    size_t c_per, const ep_dq_f32 *dq) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(dst ? dst + bi * c_per : NULL, fdst ? fdst + bi * c_per : NULL,
                  a_packs + bi * a_per, b_packs + bi * b_per, m, n, kp4, per_tile, n_tiles, dq);
}

// Shared batched i16 driver. `dq` NULL: raw i64 output to `dst`. `dq` set:
// dequant + f32 op-graph applied to EACH item (same scale/nodes for all items),
// f32 output to `fdst`. One streaming session for the whole batch. Per-item A/B
// pack done outside streaming. M5-only (FEAT_SME_I16I64).
static int i16i64_batched_impl(size_t count, size_t m, size_t n, size_t k, int64_t *dst,
                               float *fdst, const int16_t *lhs, const int16_t *rhs,
                               const ep_dq_f32 *dq) {
    if (count == 0 || m == 0 || n == 0) return 0;
    size_t kp4 = (k + 3) / 4;
    size_t m_tiles = (m + 15) / 16;
    size_t n_tiles = (n + 15) / 16;
    size_t per_tile = ep_cmul(kp4, 32);
    size_t a_per = 2 * m_tiles * per_tile, b_per = 2 * n_tiles * per_tile, c_per = m * n;

    // pack_band fully writes every band (including zero-filling empty/partial
    // bands and partial p-groups), so the pack buffers need no calloc zeroing.
    // `count` is the one factor not bounded by a C-side size invariant, so guard
    // the byte products against overflow (return -1 -> the Rust side falls back).
    size_t a_elems, b_elems, a_bytes, b_bytes;
    if (__builtin_mul_overflow(count, a_per, &a_elems) ||
        __builtin_mul_overflow(count, b_per, &b_elems) ||
        __builtin_mul_overflow(a_elems, sizeof(int16_t), &a_bytes) ||
        __builtin_mul_overflow(b_elems, sizeof(int16_t), &b_bytes)) {
        return -1;
    }
    int16_t *a_packs = (int16_t *)malloc(a_bytes);
    int16_t *b_packs = (int16_t *)malloc(b_bytes);
    if (!a_packs || !b_packs) {
        free(a_packs);
        free(b_packs);
        return -1;
    }
    for (size_t bi = 0; bi < count; bi++) {
        const int16_t *li = lhs + bi * (m * k);
        const int16_t *ri = rhs + bi * (k * n);
        for (size_t st = 0; st < 2 * m_tiles; st++) {
            size_t r0 = st * 8;
            size_t vr = (r0 < m) ? ((m - r0 < 8) ? (m - r0) : 8) : 0;
            pack_band(a_packs + bi * a_per + st * per_tile, vr ? li + (long)r0 * (long)k : li,
                      (long)k, 1, k, vr);
        }
        for (size_t st = 0; st < 2 * n_tiles; st++) {
            size_t c0 = st * 8;
            size_t vc = (c0 < n) ? ((n - c0 < 8) ? (n - c0) : 8) : 0;
            pack_band(b_packs + bi * b_per + st * per_tile, vc ? ri + (long)c0 : ri, 1, (long)n, k,
                      vc);
        }
    }
    run_batched(count, dst, fdst, a_packs, b_packs, m, n, kp4, per_tile, n_tiles, a_per, b_per,
                c_per, dq);
    free(a_packs);
    free(b_packs);
    return 0;
}

// Batched i16 GEMM: `count` independent row-major C_i(i64) = A_i @ B_i (same
// shape), one streaming-mode session. See gemm_sme_i8i32_batched. M5-only.
int gemm_sme_i16i64_batched(size_t count, size_t m, size_t n, size_t k, int64_t *dst,
                            const int16_t *lhs, const int16_t *rhs) {
    return i16i64_batched_impl(count, m, n, k, dst, NULL, lhs, rhs, NULL);
}

// Batched i16 GEMM with fused dequant + op-graph applied to EACH item: the i64
// accumulator is scaled (per-tensor `scale`, or per-N `scale_n`) into f32 and the
// op-graph (`nodes`) is applied. f32 output to `dst`. Same scale/nodes/operands
// for every item. NULL nodes / n_nodes==0 = pure dequant. M5-only.
int gemm_sme_i16i64_batched_dequant(size_t count, size_t m, size_t n, size_t k, float *fdst,
                                    const int16_t *lhs, const int16_t *rhs, float scale,
                                    const float *scale_n, uint32_t n_nodes, const EpNode *nodes) {
    ep_dq_f32 dq = {scale, scale_n, n_nodes, nodes};
    return i16i64_batched_impl(count, m, n, k, NULL, fdst, lhs, rhs, &dq);
}

#endif // SME_GEMM_I16I64_BATCHED_H
