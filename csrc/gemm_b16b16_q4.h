// bf16 Q4 path: C(bf16) = A(bf16) @ dequant(B), B kept 4-bit resident.
// Included at the end of gemm_b16b16.c for its `bf16` typedef and `packa`; the
// f16 twin lives inline in gemm_f16f16.c. Scales and mins stay f16 -- that is
// the GGUF Q4_0/Q4_1 on-disk type, and it carries more mantissa than bf16, so
// one Q4Weights serves both compute paths.
#ifndef SME_GEMM_B16B16_Q4_H
#define SME_GEMM_B16B16_Q4_H

// Dequantize one packed Q4 B-tile ([k,16] nibbles, column-order 2/byte) into a
// [k,32] bf16 scratch using the per-(column,K-block) scales. Runs outside the
// streaming region, as gemm_f16f16.c's dequant_q4_tile -- see there for why.
// `bshift` is log2 of the K-block size (5 => 32). `mins` is NULL for scale-only
// (Q4_0) or the matching offsets for the affine form (Q4_1): w = scale*code+min.
// f32 -> bf16, round to nearest even -- the integer form of what `(bf16)w`
// compiles to, matching it bit for bit on the finite values this path produces.
static inline void q4_st4_bf16(bf16 *p, float32x4_t v) {
    uint32x4_t u = vreinterpretq_u32_f32(v);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
    vst1_u16((uint16_t *)p, vshrn_n_u32(vaddq_u32(u, vaddq_u32(vdupq_n_u32(0x7fff), lsb)), 16));
}

static void dequant_q4_tile_bf16(bf16 *scratch, const uint8_t *nib, const ep_f16 *scales,
                                 const ep_f16 *mins, size_t k, unsigned bshift) {
    const uint8x8_t nib_mask = vdup_n_u8(0x0f);
    const uint8x8_t eight = vdup_n_u8(8);
    for (size_t d = 0; d < k; d++) {
        const ep_f16 *sc = scales + (d >> bshift) * 32;
        const ep_f16 *mn = mins ? mins + (d >> bshift) * 32 : NULL;
        const uint8_t *row = nib + d * 16;
        bf16 *out = scratch + d * 32;
        for (size_t h = 0; h < 2; h++) {
            uint8x8_t b = vld1_u8(row + h * 8);
            uint8x8x2_t z = vzip_u8(vand_u8(b, nib_mask), vshr_n_u8(b, 4));
            for (size_t q = 0; q < 2; q++) {
                // sign-extend the 4-bit code to -8..7
                int8x8_t c8 = vreinterpret_s8_u8(vsub_u8(veor_u8(z.val[q], eight), eight));
                int16x8_t c16 = vmovl_s8(c8);
                size_t j = h * 16 + q * 8;
                float32x4_t w0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(c16))),
                                           vcvt_f32_f16(vld1_f16((const __fp16 *)sc + j)));
                float32x4_t w1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(c16))),
                                           vcvt_f32_f16(vld1_f16((const __fp16 *)sc + j + 4)));
                if (mn) {
                    w0 = vaddq_f32(w0, vcvt_f32_f16(vld1_f16((const __fp16 *)mn + j)));
                    w1 = vaddq_f32(w1, vcvt_f32_f16(vld1_f16((const __fp16 *)mn + j + 4)));
                }
                q4_st4_bf16(out + j, w0);
                q4_st4_bf16(out + j + 4, w1);
            }
        }
    }
}

// MOPAs for one already-dequantized N-tile pair; the bf16 twin of run_q4_pair,
// including the optional in-register op-graph at the store.
__arm_locally_streaming __arm_new("za") static void run_q4_pair_bf16(
    bf16 *dst, const bf16 *a_pack, const bf16 *s0, const bf16 *s1, int has1, size_t n, size_t k,
    size_t nt, size_t n_tiles, size_t per_tile, size_t m, size_t mt_lo, size_t mt_hi,
    const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    const svbfloat16_t z16 = svdup_n_bf16((bf16)0.0f);
    const EpNode *nodes = ep ? ep->nodes : NULL;
    uint32_t n_nodes = ep ? ep->n_nodes : 0;
    int has_ep = n_nodes != 0;
    svbfloat16_t vb = svdup_n_bf16((bf16)1.0f); // beta = 1
    svbfloat16_t va = svdup_n_bf16((bf16)0.0f); // alpha unused (read_dst = 0)
    for (size_t mt = mt_lo; mt < mt_hi; mt++) {
        const bf16 *at = a_pack + mt * per_tile;
        size_t m0 = mt * 32;
        size_t mrows = (m - m0 < 32) ? (m - m0) : 32;
        svzero_za();
        for (size_t d = 0; d < k; d++) {
            svbfloat16_t av = svld1_bf16(p16, at + d * 32);
            svmopa_za16_bf16_m(0, p16, p16, av, svld1_bf16(p16, s0 + d * 32));
            if (has1) svmopa_za16_bf16_m(1, p16, p16, av, svld1_bf16(p16, s1 + d * 32));
        }
        for (size_t t = 0; t < 2; t++) {
            size_t ntt = nt + t;
            if (ntt >= n_tiles) break;
            size_t n0 = ntt * 32;
            size_t ncols = (n - n0 < 32) ? (n - n0) : 32;
            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
            bf16 *base = dst + (long)m0 * (long)n + (long)n0;
            if (!has_ep) {
                if (t == 0)
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(0, (uint32_t)r, pst, dst + (m0 + r) * n + n0);
                else
                    for (size_t r = 0; r < mrows; r++)
                        svst1_hor_za16(1, (uint32_t)r, pst, dst + (m0 + r) * n + n0);
            } else if (t == 0) {
#define EP_Q4_RD0(s) svread_hor_za16_bf16_m(z16, p16, 0, (s))
                EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, base, (long)n, EP_Q4_RD0, mrows, vb, va, 0,
                                            nodes, n_nodes, m0, n0);
#undef EP_Q4_RD0
            } else {
#define EP_Q4_RD1(s) svread_hor_za16_bf16_m(z16, p16, 1, (s))
                EP_STORE_TILE_ROWMAJOR_BF16(p16, pst, base, (long)n, EP_Q4_RD1, mrows, vb, va, 0,
                                            nodes, n_nodes, m0, n0);
#undef EP_Q4_RD1
            }
        }
    }
}

// One unit of Q4 work: dequant (non-streaming) then MOPA. See gemm_f16f16.c.
static void q4_pair_bf16(bf16 *dst, const bf16 *a_pack, const uint8_t *nibbles,
                         const ep_f16 *scales, const ep_f16 *mins, size_t m, size_t n, size_t k,
                         size_t nt, size_t n_tiles, size_t per_tile, size_t nib_per_tile,
                         size_t sc_per_tile, unsigned bshift, bf16 *scratch, size_t ic,
                         size_t ic_end, const ep_desc16 *ep) {
    bf16 *s0 = scratch;
    bf16 *s1 = scratch + per_tile;
    dequant_q4_tile_bf16(s0, nibbles + nt * nib_per_tile, scales + nt * sc_per_tile,
                         mins ? mins + nt * sc_per_tile : NULL, k, bshift);
    int has1 = (nt + 1 < n_tiles);
    if (has1)
        dequant_q4_tile_bf16(s1, nibbles + (nt + 1) * nib_per_tile,
                             scales + (nt + 1) * sc_per_tile,
                             mins ? mins + (nt + 1) * sc_per_tile : NULL, k, bshift);
    run_q4_pair_bf16(dst, a_pack, s0, s1, has1, n, k, nt, n_tiles, per_tile, m, ic, ic_end, ep);
}

// Per-thread reusable chunk pool; see gemm_f16f16_q4.h's q4_pool_scratch.
static _Thread_local bf16 *g_q4_pool_bf16 = NULL;
static _Thread_local size_t g_q4_pool_bf16_cap = 0;
static bf16 *q4_pool_scratch_bf16(size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (g_q4_pool_bf16_cap < bytes) {
        free(g_q4_pool_bf16);
        g_q4_pool_bf16 = (bf16 *)malloc(bytes);
        g_q4_pool_bf16_cap = g_q4_pool_bf16 ? bytes : 0;
    }
    return g_q4_pool_bf16;
}

// Tile-outer so each B-tile is dequantized once and reused across the M-tiles,
// with BLIS-style L2 blocking over the M-tiles so the A-block stays resident
// across the inner N-pair loop, and the independent N-tile pairs spread over the
// cluster. Mirrors gemm_f16f16.c's run_q4, including the pool sizing and the
// 16-chunk cap, both of which are explained there.
static int run_q4_bf16(bf16 *dst, const bf16 *a_pack, const uint8_t *nibbles, const ep_f16 *scales,
                       const ep_f16 *mins, size_t m, size_t n, size_t k, size_t n_tiles,
                       size_t per_tile, size_t nib_per_tile, size_t sc_per_tile, unsigned bshift,
                       const ep_desc16 *ep) {
    size_t m_tiles = (m + 31) / 32;

    size_t tile_bytes = per_tile * sizeof(bf16);
    size_t budget_tiles = (16u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    size_t mc = budget_tiles > 2 ? budget_tiles - 2 : 1;
    if (mc < 1) mc = 1;
    if (mc > m_tiles) mc = m_tiles;

    size_t n_pairs = (n_tiles + 1) / 2;
    size_t pair_elems = ep_cmul(2, per_tile);
    size_t pair_bytes = ep_cmul(pair_elems, sizeof(bf16));
    size_t pool_cap = (32u * 1024 * 1024) / (pair_bytes ? pair_bytes : 1);
    size_t chunks = n_pairs < 16 ? n_pairs : 16;
    if (chunks > pool_cap) chunks = pool_cap;
    if (chunks < 1) chunks = 1;
    bf16 *pool = q4_pool_scratch_bf16(ep_cmul(chunks, pair_bytes));
    if (!pool && chunks > 1) {
        chunks = 1;
        pool = q4_pool_scratch_bf16(pair_bytes);
    }
    if (!pool) return -1;
    size_t per_chunk = (n_pairs + chunks - 1) / chunks;

    for (size_t ic = 0; ic < m_tiles; ic += mc) {
        size_t ic_end = ic + mc < m_tiles ? ic + mc : m_tiles;
        if (chunks <= 1) {
            for (size_t nt = 0; nt < n_tiles; nt += 2)
                q4_pair_bf16(dst, a_pack, nibbles, scales, mins, m, n, k, nt, n_tiles, per_tile,
                             nib_per_tile, sc_per_tile, bshift, pool, ic, ic_end, ep);
            continue;
        }
        dispatch_apply(chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
          size_t p0 = ci * per_chunk;
          size_t p1 = p0 + per_chunk < n_pairs ? p0 + per_chunk : n_pairs;
          for (size_t pi = p0; pi < p1; pi++)
              q4_pair_bf16(dst, a_pack, nibbles, scales, mins, m, n, k, pi * 2, n_tiles, per_tile,
                           nib_per_tile, sc_per_tile, bshift, pool + ci * pair_elems, ic, ic_end, ep);
        });
    }
    return 0;
}

// Q4 GEMM with on-the-fly dequant, bf16 accumulation (M5+ FEAT_SME_B16B16). B
// stays 4-bit resident (nibbles tile-major [n_tiles][k][16 bytes], column-order
// 2/byte) plus f16 scales ([n_tiles][ceil(k/block)][32]). `block` is the K-block
// size (power of two; 32 = Q4_0/Q4_1). `mins` is NULL for the scale-only form,
// else the matching offsets: w = scale*code + min.
int gemm_sme_b16b16_q4(size_t m, size_t n, size_t k, uint16_t *dst, const uint16_t *lhs,
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

    // Shares the dense path's per-thread A-pack buffer; see gemm_f16f16_q4.h.
    bf16 *a_pack = apack_scratch(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(bf16)));
    if (!a_pack) return -1;
    packa(a_pack, (const bf16 *)lhs, m, k, k, 1, 0, m_tiles, per_tile);
    return run_q4_bf16((bf16 *)dst, a_pack, nibbles, (const ep_f16 *)scales, (const ep_f16 *)mins, m,
                       n, k, n_tiles, per_tile, nib_per_tile, sc_per_tile, bshift, ep);
}

#endif // SME_GEMM_B16B16_Q4_H
