// f16 Q4 path: C(f16) = A(f16) @ dequant(B), B kept 4-bit resident.
// Included at the end of gemm_f16f16.c for its `f16` typedef and `packa`; the
// bf16 twin is gemm_b16b16_q4.h.
//
// Dequantize one packed Q4 B-tile ([k,16] nibbles, column-order 2/byte) into a
// [k,32] f16 scratch using the per-(column,K-block) scales, keeping the resident
// weights 4-bit while only one tile pair of f16 is live at a time. `bshift` is
// log2 of the K-block size (5 => 32, the Q4_0/Q4_1 default). `mins` is NULL for
// scale-only (Q4_0) or the matching per-(column,K-block) offsets for the affine
// form (Q4_1 / llama.cpp *_1): w = scale*code + min.
#ifndef SME_GEMM_F16F16_Q4_H
#define SME_GEMM_F16F16_Q4_H

static void dequant_q4_tile(f16 *scratch, const uint8_t *nib, const f16 *scales, const f16 *mins,
                            size_t k, unsigned bshift) {
    const uint8x8_t nib_mask = vdup_n_u8(0x0f);
    const uint8x8_t eight = vdup_n_u8(8);
    for (size_t d = 0; d < k; d++) {
        const f16 *sc = scales + (d >> bshift) * 32;
        const f16 *mn = mins ? mins + (d >> bshift) * 32 : NULL;
        const uint8_t *row = nib + d * 16;
        f16 *out = scratch + d * 32;
        // 16 bytes -> 32 codes, in two halves of 8 bytes. Low nibble is the even
        // column and high the odd, so vzip restores column order.
        for (size_t h = 0; h < 2; h++) {
            uint8x8_t b = vld1_u8(row + h * 8);
            uint8x8x2_t z = vzip_u8(vand_u8(b, nib_mask), vshr_n_u8(b, 4));
            for (size_t q = 0; q < 2; q++) {
                // sign-extend the 4-bit code to -8..7
                int8x8_t c8 = vreinterpret_s8_u8(vsub_u8(veor_u8(z.val[q], eight), eight));
                int16x8_t c16 = vmovl_s8(c8);
                size_t j = h * 16 + q * 8;
                float32x4_t w0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(c16))),
                                           vcvt_f32_f16(vld1_f16(sc + j)));
                float32x4_t w1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(c16))),
                                           vcvt_f32_f16(vld1_f16(sc + j + 4)));
                if (mn) {
                    w0 = vaddq_f32(w0, vcvt_f32_f16(vld1_f16(mn + j)));
                    w1 = vaddq_f32(w1, vcvt_f32_f16(vld1_f16(mn + j + 4)));
                }
                vst1_f16(out + j, vcvt_f16_f32(w0));
                vst1_f16(out + j + 4, vcvt_f16_f32(w1));
            }
        }
    }
}

// MOPAs for one already-dequantized N-tile pair against the M-tiles [mt_lo,
// mt_hi). Nothing but MOPA and store runs in streaming mode; the dequant that
// feeds `s0`/`s1` happens outside it, in the driver below.
// `ep` NULL (or empty) keeps the single-instruction ZA-slice store; otherwise the
// op-graph runs in-register via the shared node-major block store, exactly as the
// dense f16 path does. Output is always row-major, beta = 1, no dst read-back.
__arm_locally_streaming __arm_new("za") static void run_q4_pair(
    f16 *dst, const f16 *a_pack, const f16 *s0, const f16 *s1, int has1, size_t n, size_t k,
    size_t nt, size_t n_tiles, size_t per_tile, size_t m, size_t mt_lo, size_t mt_hi,
    const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    const svfloat16_t z16 = svdup_n_f16((ep_f16)0.0f);
    const EpNode *nodes = ep ? ep->nodes : NULL;
    uint32_t n_nodes = ep ? ep->n_nodes : 0;
    int has_ep = n_nodes != 0;
    svfloat16_t vb = svdup_n_f16((ep_f16)1.0f); // beta = 1
    svfloat16_t va = svdup_n_f16((ep_f16)0.0f); // alpha unused (read_dst = 0)
    for (size_t mt = mt_lo; mt < mt_hi; mt++) {
        const f16 *at = a_pack + mt * per_tile;
        size_t m0 = mt * 32;
        size_t mrows = (m - m0 < 32) ? (m - m0) : 32;
        svzero_za();
        for (size_t d = 0; d < k; d++) {
            svfloat16_t av = svld1_f16(p16, at + d * 32);
            svmopa_za16_m(0, p16, p16, av, svld1_f16(p16, s0 + d * 32));
            if (has1) svmopa_za16_m(1, p16, p16, av, svld1_f16(p16, s1 + d * 32));
        }
        for (size_t t = 0; t < 2; t++) {
            size_t ntt = nt + t;
            if (ntt >= n_tiles) break;
            size_t n0 = ntt * 32;
            size_t ncols = (n - n0 < 32) ? (n - n0) : 32;
            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
            f16 *base = dst + (long)m0 * (long)n + (long)n0;
            if (!has_ep) {
                if (t == 0)
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(0, (uint32_t)r, pst, dst + (m0 + r) * n + n0);
                else
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(1, (uint32_t)r, pst, dst + (m0 + r) * n + n0);
            } else if (t == 0) {
#define EP_Q4_RD0(s) svread_hor_za16_m(z16, p16, 0, (s))
                EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, (long)n, EP_Q4_RD0, mrows, vb, va, 0,
                                           nodes, n_nodes, m0, n0);
#undef EP_Q4_RD0
            } else {
#define EP_Q4_RD1(s) svread_hor_za16_m(z16, p16, 1, (s))
                EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, (long)n, EP_Q4_RD1, mrows, vb, va, 0,
                                           nodes, n_nodes, m0, n0);
#undef EP_Q4_RD1
            }
        }
    }
}

// Dequantize N-tile pair `nt` into `scratch` and run it against the M-tiles
// [ic, ic_end). One unit of Q4 work: the dequant half runs here (non-streaming),
// the MOPA half inside run_q4_pair.
static void q4_pair(f16 *dst, const f16 *a_pack, const uint8_t *nibbles, const f16 *scales,
                    const f16 *mins, size_t m, size_t n, size_t k, size_t nt, size_t n_tiles,
                    size_t per_tile, size_t nib_per_tile, size_t sc_per_tile, unsigned bshift,
                    f16 *scratch, size_t ic, size_t ic_end, const ep_desc16 *ep) {
    f16 *s0 = scratch;
    f16 *s1 = scratch + per_tile;
    dequant_q4_tile(s0, nibbles + nt * nib_per_tile, scales + nt * sc_per_tile,
                    mins ? mins + nt * sc_per_tile : NULL, k, bshift);
    int has1 = (nt + 1 < n_tiles);
    if (has1)
        dequant_q4_tile(s1, nibbles + (nt + 1) * nib_per_tile, scales + (nt + 1) * sc_per_tile,
                        mins ? mins + (nt + 1) * sc_per_tile : NULL, k, bshift);
    run_q4_pair(dst, a_pack, s0, s1, has1, n, k, nt, n_tiles, per_tile, m, ic, ic_end, ep);
}

// Per-thread reusable chunk pool, same pattern as gemm_i8i32.c's apack_scratch:
// the dequant scratch is up to tens of MB and every Q4 call needs it, so a
// malloc/free per call shows up directly in decode latency. Grows monotonically
// and lives for the thread's lifetime. The calling thread is the only one that
// resizes it; the dispatch_apply workers write disjoint slices of it and finish
// before the call returns.
static _Thread_local f16 *g_q4_pool = NULL;
static _Thread_local size_t g_q4_pool_cap = 0;
static f16 *q4_pool_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1; // never a history-dependent NULL on fresh threads
    if (g_q4_pool_cap < bytes) {
        free(g_q4_pool);
        g_q4_pool = (f16 *)malloc(bytes);
        g_q4_pool_cap = g_q4_pool ? bytes : 0;
    }
    return g_q4_pool;
}

// Tile-outer so each B-tile is dequantized once and reused across the M-tiles.
//
// N-tile pairs are independent -- each writes its own 64 output columns and only
// reads A, the nibbles and the scales -- so they parallelize directly. Each
// worker needs its own 2-tile dequant scratch, so the chunk count is bounded by
// the pool budget below as well as by the pair count.
static int run_q4(f16 *dst, const f16 *a_pack, const uint8_t *nibbles, const f16 *scales,
                  const f16 *mins, size_t m, size_t n, size_t k, size_t n_tiles, size_t per_tile,
                  size_t nib_per_tile, size_t sc_per_tile, unsigned bshift, const ep_desc16 *ep) {
    size_t m_tiles = (m + 31) / 32;

    // BLIS-style L2 blocking over the M-tiles. The A-panel for one M-tile is
    // k*32*2 bytes; without blocking every M-tile re-streams from DRAM once per
    // N-pair. Pick an M-block (mc tiles) whose A-block plus the two dequant
    // scratch tiles fit L2, so the A-block stays resident across the inner
    // N-pair loop. B is still dequantized on the fly a pair at a time into the
    // 2-tile scratch, so a second M-block costs a second full dequant pass. Tile
    // order within each output tile is unchanged, so the numeric result matches
    // the flat path.
    //
    // 16 MB rather than the 8 MB the dense kernels use, because here the budget
    // also decides how many full dequant passes a shape pays, and that dominates
    // the extra A traffic a larger block streams. Measured on M5 at n=k=4096:
    // 16 MB beats 8 MB by 2-6% for m >= 992 (where 8 MB forces a second block)
    // and ties below it, for both f16 and bf16; 4 MB is ~4% worse throughout and
    // 32 MB buys nothing further. It exceeds the 6 MB E-cluster L2 and wins
    // anyway -- the saved dequant outweighs the E workers' extra A misses.
    size_t tile_bytes = per_tile * sizeof(f16);
    size_t budget_tiles = (16u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    size_t mc = budget_tiles > 2 ? budget_tiles - 2 : 1;
    if (mc < 1) mc = 1;
    if (mc > m_tiles) mc = m_tiles;

    // Per-chunk dequant scratch, sized and obtained on the calling thread before
    // any worker runs: a worker-side allocation failure after other chunks have
    // already stored rows would break the documented "rc != 0 => C untouched"
    // contract. The serial path takes the first chunk's slice, so one buffer
    // serves both and nothing here allocates per call.
    //
    // Each chunk needs 2 tiles = 128*k bytes, so a flat byte cap sets the chunk
    // count and therefore the parallelism: at 8 MB that is 4 chunks at k=16384,
    // well under the core count. 32 MB keeps >= 16 chunks out to k=16384 and
    // measured 1.11x there (m=512). The cap has to stay bounded because this
    // buffer is per-thread and never shrinks.
    size_t n_pairs = (n_tiles + 1) / 2;
    size_t pair_elems = ep_cmul(2, per_tile);
    size_t pair_bytes = ep_cmul(pair_elems, sizeof(f16));
    size_t pool_cap = (32u * 1024 * 1024) / (pair_bytes ? pair_bytes : 1);
    size_t chunks = n_pairs < 16 ? n_pairs : 16;
    if (chunks > pool_cap) chunks = pool_cap;
    if (chunks < 1) chunks = 1;
    f16 *pool = q4_pool_scratch(ep_cmul(chunks, pair_bytes));
    if (!pool && chunks > 1) { // retry at the serial size before giving up
        chunks = 1;
        pool = q4_pool_scratch(pair_bytes);
    }
    if (!pool) return -1;
    size_t per_chunk = (n_pairs + chunks - 1) / chunks;

    for (size_t ic = 0; ic < m_tiles; ic += mc) {
        size_t ic_end = ic + mc < m_tiles ? ic + mc : m_tiles;
        if (chunks <= 1) {
            for (size_t nt = 0; nt < n_tiles; nt += 2)
                q4_pair(dst, a_pack, nibbles, scales, mins, m, n, k, nt, n_tiles, per_tile,
                        nib_per_tile, sc_per_tile, bshift, pool, ic, ic_end, ep);
            continue;
        }
        dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t p0 = ci * per_chunk;
          size_t p1 = p0 + per_chunk < n_pairs ? p0 + per_chunk : n_pairs;
          for (size_t pi = p0; pi < p1; pi++)
              q4_pair(dst, a_pack, nibbles, scales, mins, m, n, k, pi * 2, n_tiles, per_tile,
                      nib_per_tile, sc_per_tile, bshift, pool + ci * pair_elems, ic, ic_end, ep);
        });
    }
    return 0;
}

// Q4 GEMM with on-the-fly dequant: C(f16) = A(f16) @ dequant(B). B stays 4-bit
// resident (nibbles tile-major [n_tiles][k][16 bytes], column-order 2/byte) plus
// scales ([n_tiles][ceil(k/block)][32] f16); only one tile of f16 is live at a
// time. `block` is the K-block size (power of two; 32 = Q4_0/Q4_1). `mins` is
// NULL for the scale-only form, else the matching offsets: w = scale*code + min.
// `ep` is the fused f16 op-graph applied in-register at the store, or NULL.
int gemm_sme_f16f16_q4(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
                       const uint8_t *nibbles, const uint16_t *scales, const uint16_t *mins,
                       size_t block, const ep_desc16 *ep) {
    if (m == 0 || n == 0) return 0;
    if (block == 0 || (block & (block - 1)) != 0) return -1; // power of two only
    unsigned bshift = 0;
    while (((size_t)1 << bshift) < block) bshift++;
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    size_t per_tile = ep_cmul(k, 32);
    size_t nib_per_tile = ep_cmul(k, 16);
    size_t nbk = (k + block - 1) / block;
    size_t sc_per_tile = ep_cmul(nbk, 32);

    // Shares the dense path's per-thread A-pack buffer (this header is included
    // into gemm_f16f16.c): only one gemm call per thread is ever in flight, so
    // the two uses cannot overlap. run_q4 takes its dequant scratch the same way,
    // which leaves this entry point allocating nothing per call.
    f16 *a_pack = apack_scratch(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(f16)));
    if (!a_pack) return -1;
    packa(a_pack, (const f16 *)lhs, m, k, k, 1, 0, m_tiles, per_tile);
    return run_q4((f16 *)dst, a_pack, nibbles, (const f16 *)scales, (const f16 *)mins, m, n, k,
                  n_tiles, per_tile, nib_per_tile, sc_per_tile, bshift, ep);
}

#endif // SME_GEMM_F16F16_Q4_H
