// SME f16f16 non-widening GEMM driver (Apple M5+, FEAT_SME_F16F16): each MOPA
// is a 32x32 fp16 outer product, and two independent ZA16 tiles run in lockstep
// (za0 for N-tile nt, za1 for nt+1) so consecutive MOPAs never hit the same
// tile. ZA16 consumes one K-value per MOPA, so the packed panels are plain
// [K, 32] tile-major (32 contiguous lanes per depth).
//
// Computes dst = alpha*dst + beta*(A @ B); fp16 accumulate, fp16 alpha/beta
// combine. Single pass over K, single-threaded per cluster.

#include "epilogue.h"
#include "transpose16.h"
#include <arm_neon.h>
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef __fp16 f16;

static inline f16 f16_from_bits(uint16_t b) {
    f16 h;
    __builtin_memcpy(&h, &b, sizeof h);
    return h;
}

// Pack one 32-row M-tile of A into [K, 32]: dst[d*32 + i] = A[i, d].
static void pack_a_tile(f16 *dst, const f16 *a, long rs, long cs, size_t k, size_t vr) {
    if (rs == 1 && vr == 32) {
        // Fast path: rows contiguous, full tile -> 64-byte copy per depth.
        for (size_t d = 0; d < k; d++)
            memcpy(dst + d * 32, a + (long)d * cs, 32 * sizeof(f16));
        return;
    }
    for (size_t d = 0; d < k; d++)
        for (size_t i = 0; i < 32; i++)
            dst[d * 32 + i] = (i < vr) ? a[(long)i * rs + (long)d * cs] : (f16)0.0f;
}

// Transpose 32 depth-contiguous lanes into a [K, 32] tile: dst[d*32 + i] is lane
// i at depth d, where lane i starts at src + i*lane_stride and runs contiguously
// in depth. Lanes >= `valid` are zero-filled.
//
// This is the same shape for both operands -- row-major A (each ROW contiguous in
// depth) and column-major B (each COLUMN contiguous in depth) -- so both packs
// call it. NEON 8x8 transpose over the aligned bulk; the lane and depth edges go
// scalar, depth-outer so their stores stay within a cache line.
static void pack_lanes_depth_major(f16 *dt, const f16 *src, long lane_stride, size_t k,
                                   size_t valid) {
    size_t kfull = k & ~(size_t)7;
    size_t lfull = valid & ~(size_t)7;
    // Interleave two independent 8x8 transposes per step: the vtrn ladder is a
    // long dependency chain over strided loads, so a single block runs
    // latency-bound (~14 GB/s). Two blocks in flight overlap their chains and the
    // strided-load latency -> the pack stops being the small/medium-tile
    // bottleneck.
    size_t kfull2 = kfull & ~(size_t)15;
    for (size_t lg = 0; lg < lfull; lg += 8) {
        size_t kg = 0;
        for (; kg < kfull2; kg += 16) {
            uint16x8_t r[8], s[8];
            for (int i = 0; i < 8; i++) {
                const uint16_t *lane = (const uint16_t *)(src + (long)(lg + i) * lane_stride + kg);
                r[i] = vld1q_u16(lane);
                s[i] = vld1q_u16(lane + 8);
            }
            transpose_8x8_u16(r);
            transpose_8x8_u16(s);
            for (int i = 0; i < 8; i++) {
                vst1q_u16((uint16_t *)(dt + (kg + i) * 32 + lg), r[i]);
                vst1q_u16((uint16_t *)(dt + (kg + 8 + i) * 32 + lg), s[i]);
            }
        }
        for (; kg < kfull; kg += 8) {
            uint16x8_t r[8];
            for (int i = 0; i < 8; i++)
                r[i] = vld1q_u16((const uint16_t *)(src + (long)(lg + i) * lane_stride + kg));
            transpose_8x8_u16(r);
            for (int i = 0; i < 8; i++)
                vst1q_u16((uint16_t *)(dt + (kg + i) * 32 + lg), r[i]);
        }
    }
    // Lane tail (i in [lfull,32), all depths): real lanes up to `valid`, zero
    // past them. DEPTH-OUTER on purpose. The lane-outer form strides 32 elements
    // (one cache line) per store, so it re-walks the whole k*32 panel once per
    // lane -- at valid==1, lfull is 0 and that is 32 passes over the panel to
    // write a single lane, with 31 of them pure zero fill.
    for (size_t d = 0; d < k && lfull < 32; d++) {
        f16 *dp = dt + d * 32;
        for (size_t i = lfull; i < valid; i++)
            dp[i] = src[(long)i * lane_stride + d];
        for (size_t i = valid; i < 32; i++)
            dp[i] = (f16)0.0f;
    }
    // Depth tail (whole lanes, depths [kfull,k)).
    for (size_t i = 0; i < lfull; i++)
        for (size_t d = kfull; d < k; d++)
            dt[d * 32 + i] = src[(long)i * lane_stride + d];
}

// Pack one 32-col N-tile of B into [K, 32]: dst[d*32 + j] = B[d, j].
static void pack_b_tile(f16 *dst, const f16 *b, long rs, long cs, size_t k, size_t vc) {
    if (cs == 1 && vc == 32) {
        // Row-major B (e.g. [N,K] weights): contiguous 32 N-values per depth.
        for (size_t d = 0; d < k; d++)
            memcpy(dst + d * 32, b + (long)d * rs, 32 * sizeof(f16));
        return;
    }
    if (rs == 1) {
        // Col-major B (the transposed-B / Q@K^T case): each N-column is
        // contiguous in depth, which is the A-pack's layout with columns for
        // rows -- so it takes the same NEON transpose.
        pack_lanes_depth_major(dst, b, cs, k, vc);
        return;
    }
    for (size_t d = 0; d < k; d++)
        for (size_t j = 0; j < 32; j++)
            dst[d * 32 + j] = (j < vc) ? b[(long)d * rs + (long)j * cs] : (f16)0.0f;
}

// Narrow/combine/store one extracted ZA16 column or row (already 32 contiguous
// f16 for the stored direction). `acc` is the raw fp16 outer-product slice.
static inline void store_slice(svbool_t p16, svbool_t pst, f16 *ptr, svfloat16_t acc,
                               svfloat16_t vb, int read_dst, svfloat16_t va) __arm_streaming {
    acc = svmul_x(p16, acc, vb);
    if (read_dst) {
        acc = svmla_x(p16, acc, svld1_f16(pst, ptr), va);
    }
    svst1_f16(pst, ptr, acc);
}

// Zero ZA tiles 0/1 then initialize ZA[r][c] = bias[c] via a rank-1 FMOPA
// (ones[r] * bias[c]). Kept out of line so the bias-fold path does not perturb
// the (layout-sensitive) inlined store code of run_streaming. nt is the first of
// the two N-tiles; partial column tiles predicate the bias load.
__attribute__((noinline)) static void bias_init_za(svbool_t p16, const f16 *bias, size_t nt,
                                                   size_t n) __arm_streaming __arm_inout("za") {
    svzero_za();
    svfloat16_t v_ones = svdup_n_f16((f16)1.0f);
    size_t bn0 = (nt + 0) * 32;
    size_t bn1 = (nt + 1) * 32;
    svbool_t pb0 = (bn0 < n) ? svwhilelt_b16((uint64_t)bn0, (uint64_t)n) : svpfalse_b();
    svbool_t pb1 = (bn1 < n) ? svwhilelt_b16((uint64_t)bn1, (uint64_t)n) : svpfalse_b();
    // Clamp the pad-tile address: pb1 may be all-false (no lane is read), but
    // forming bias + bn1 past one-past-end would still be UB.
    svmopa_za16_m(0, p16, pb0, v_ones, svld1_f16(pb0, bias + bn0));
    svmopa_za16_m(1, p16, pb1, v_ones, svld1_f16(pb1, (bn1 < n) ? bias + bn1 : bias));
}

// Narrow-N (n_tiles == 1, n <= 32) row-major fast path. The dual-tile lockstep in
// run_streaming pairs N-tiles (za0 = nt, za1 = nt+1); when there is only ONE
// N-tile, za1 accumulates the zeroed pad tile -- ~half the MOPAs are wasted (n=32
// measured ~2x slow per flop vs n=64). Here both ZA tiles stay productive by
// pairing two M-tiles against the single B-tile: za0 = A[mt] @ B, za1 = A[mt+1] @ B.
// Row-major only (the decode/GEMV output layout); col-major / strided n_tiles==1
// stays on the general path. Bias is NOT folded (applied as a normal node at the
// store), so the store reuses run_streaming's pure / node-major row-major
// primitives verbatim -- no new store codegen, and run_streaming is untouched.
__arm_locally_streaming __arm_new("za") static void run_narrow_rowmajor(
    f16 *dst, long dst_cs, long dst_rs, const f16 *a_pack, const f16 *b_pack, size_t m, size_t n,
    size_t k, size_t mt_lo, size_t mt_hi, size_t nt, float alpha, float beta, int read_dst,
    const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    svcount_t pn = svptrue_c16();
    const svfloat16_t z16 = svdup_n_f16((f16)0.0f);
    const svfloat16_t va = svdup_n_f16((f16)alpha);
    const svfloat16_t vb = svdup_n_f16((f16)beta);
    size_t per_tile = ep_cmul(k, 32);
    int has_ep = ep && ep->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep->n_nodes : 0;
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    // `nt` is the single N-tile this pass covers: tile 0 for the n<=32 case, or
    // the odd trailing tile when the dual-N loop has run out of pairs.
    size_t n0 = nt * 32;
    size_t ncols = (n - n0 < 32) ? (n - n0) : 32;
    svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
    const f16 *b0 = b_pack + nt * per_tile;

    for (size_t mt = mt_lo; mt < mt_hi; mt += 2) {
        const f16 *a0t = a_pack + mt * per_tile;
        int have1 = (mt + 1 < mt_hi);
        const f16 *a1t = have1 ? a_pack + (mt + 1) * per_tile : a0t;
        size_t m0_0 = mt * 32;
        size_t mrows0 = (m - m0_0 < 32) ? (m - m0_0) : 32;
        size_t m0_1 = (mt + 1) * 32;
        size_t mrows1 = have1 ? ((m - m0_1 < 32) ? (m - m0_1) : 32) : 0;

        // za0 = A[mt] @ B, za1 = A[mt+1] @ B. Both share the single B-vector per
        // depth, so the two MOPAs issue back-to-back with no RAW stall (distinct
        // tiles), exactly like the dual-N scheme -- but both outputs are real.
        svzero_za();
        size_t d = 0;
        for (; d + 4 <= k; d += 4) {
            svfloat16x4_t A0 = svld1_f16_x4(pn, a0t + d * 32);
            svfloat16x4_t A1 = svld1_f16_x4(pn, a1t + d * 32);
            svfloat16x4_t U = svld1_f16_x4(pn, b0 + d * 32);
            svfloat16_t u0 = svget4_f16(U, 0), u1 = svget4_f16(U, 1), u2 = svget4_f16(U, 2),
                        u3 = svget4_f16(U, 3);
            svmopa_za16_m(0, p16, p16, svget4_f16(A0, 0), u0);
            svmopa_za16_m(1, p16, p16, svget4_f16(A1, 0), u0);
            svmopa_za16_m(0, p16, p16, svget4_f16(A0, 1), u1);
            svmopa_za16_m(1, p16, p16, svget4_f16(A1, 1), u1);
            svmopa_za16_m(0, p16, p16, svget4_f16(A0, 2), u2);
            svmopa_za16_m(1, p16, p16, svget4_f16(A1, 2), u2);
            svmopa_za16_m(0, p16, p16, svget4_f16(A0, 3), u3);
            svmopa_za16_m(1, p16, p16, svget4_f16(A1, 3), u3);
        }
        for (; d < k; d++) {
            svfloat16_t u = svld1_f16(p16, b0 + d * 32);
            svmopa_za16_m(0, p16, p16, svld1_f16(p16, a0t + d * 32), u);
            svmopa_za16_m(1, p16, p16, svld1_f16(p16, a1t + d * 32), u);
        }

        if (pure) {
            for (size_t r = 0; r < mrows0; r++)
                svst1_hor_za16(0, (uint32_t)r, pst,
                               dst + (long)(m0_0 + r) * dst_rs + (long)n0 * dst_cs);
            if (have1)
                for (size_t r = 0; r < mrows1; r++)
                    svst1_hor_za16(1, (uint32_t)r, pst,
                                   dst + (long)(m0_1 + r) * dst_rs + (long)n0 * dst_cs);
        } else if (has_ep) {
            f16 *base0 = dst + (long)m0_0 * dst_rs + (long)n0 * dst_cs;
#define EP_RD0(s) svread_hor_za16_m(z16, p16, 0, (s))
            EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base0, dst_rs, EP_RD0, mrows0, vb, va, read_dst,
                                       nodes, n_nodes, m0_0, n0);
#undef EP_RD0
            if (have1) {
                f16 *base1 = dst + (long)m0_1 * dst_rs + (long)n0 * dst_cs;
#define EP_RD1(s) svread_hor_za16_m(z16, p16, 1, (s))
                EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base1, dst_rs, EP_RD1, mrows1, vb, va, read_dst,
                                           nodes, n_nodes, m0_1, n0);
#undef EP_RD1
            }
        } else {
            // beta != 1 or read_dst (no ep): scale / accumulate per row.
            for (size_t r = 0; r < mrows0; r++)
                store_slice(p16, pst, dst + (long)(m0_0 + r) * dst_rs + (long)n0 * dst_cs,
                            svread_hor_za16_m(z16, p16, 0, (uint32_t)r), vb, read_dst, va);
            if (have1)
                for (size_t r = 0; r < mrows1; r++)
                    store_slice(p16, pst, dst + (long)(m0_1 + r) * dst_rs + (long)n0 * dst_cs,
                                svread_hor_za16_m(z16, p16, 1, (uint32_t)r), vb, read_dst, va);
        }
    }
}

// =============================================================================
// run_streaming: the SME hot loop. Computes the tile-blocked GEMM + epilogue for
// the M-tile range [mt_lo, mt_hi) of this (cluster's) chunk.
//
// INVARIANTS / WHY IT IS SHAPED THIS WAY -- read before touching this function.
//
// STREAMING-MODE REGION. The whole body runs in SME streaming mode: the
// `__arm_locally_streaming` attribute makes the compiler emit SMSTART on entry
// and SMSTOP on return, and `__arm_new("za")` gives it a fresh ZA state. So the
// SMSTART/SMSTOP boundaries are the function boundaries -- everything between is
// streaming-SVE (SVL=512: 32 f16 lanes per vector) with ZA live. Consequences:
// the SVE register width is the streaming width, ZA must be zeroed/initialized
// before use (svzero_za / bias_init_za), and libm / non-streaming code is NOT
// callable here (hence the in-register transcendental approximations in
// epilogue.h). The caller pays this streaming entry once per chunk; run_batched
// elsewhere reuses one region across many items for the same reason.
//
// ZA TILE LAYOUT. The kernel keeps TWO independent 32x32 ZA16 (fp16) tiles in
// flight per N-pair: za0 accumulates N-tile `nt`, za1 accumulates `nt+1`. They
// run in lockstep so the two MOPAs per K-step issue back-to-back with no
// read-after-write stall on a single tile (the K-loop is unrolled by 4 so the
// za0->za0 RAW distance is fully hidden by the interleaved za1 work). Each MOPA
// is a rank-1 fp16 outer product consuming one K-value; the packed panels are
// [K,32] tile-major, so one depth step is 32 contiguous lanes.
//
// HARD-CODED ZA TILE INDICES. The tile number in every svmopa_za16_m(0/1, ...),
// svread_{hor,ver}_za16_m(.., 0/1, ..) and svst1_{hor,ver}_za16(0/1, ..) is an
// INSTRUCTION IMMEDIATE -- it is encoded in the opcode and CANNOT be a runtime
// variable. That is why tiles 0 and 1 are spelled out literally in two parallel
// code arms (the `if (t == 0) ... else ...` and the EP_RD0/EP_RD1 macro pair)
// instead of being indexed by a loop variable. Parameterizing "which ZA tile"
// costs the dual-tile pipelining that the immediate enables.
//
// STRUCTURE (one large function on purpose):
//   1. Setup: predicates, alpha/beta splats, the bias-fold decision (a leading
//      ADD_COL bias, with beta==1 && !read_dst, is folded into ZA via a rank-1
//      FMOPA so the store reverts to the pure fast path), and `pure` (overwrite,
//      no epilogue) detection.
//   2. Cache blocking: jc->ic->mt->nt loops sized so the A- and B-blocks fit the
//      shared L2 (when the whole problem fits, the outer loops run once).
//   3. K-loop: per (mt, nt-pair), zero/bias-init ZA, then the unrolled-by-4
//      multi-vector LD1H + 8-MOPA inner loop over K, plus a scalar K-tail.
//   4. Store / epilogue hook, per N-tile, branching on output layout:
//        - `pure`: direct single-instruction ZA->C store (st1h vertical for
//          col-major, horizontal for row-major) -- no read/scale/store.
//        - col-major with epilogue: read each ZA vertical slice, run the op-graph
//          (TENSOR ops route to the scalar path, being a strided gather).
//        - row-major with epilogue: the NODE-MAJOR register-resident store
//          (EP_STORE_TILE_ROWMAJOR_F16) -- read a 4-row block, dispatch each node
//          once per block. See epilogue.h.
//        - fully strided: materialize a 32x32 scratch tile and store scalar-wise.
//
// REGISTER-FRAGILE / DO NOT SPLIT. This function is intentionally monolithic. The
// store arms inline so the live ZA slice never bounces through memory, and the
// codegen here is sensitive to register pressure -- the bias-fold is restricted to
// the bias-ONLY case precisely because folding through the relu store perturbs
// this layout and costs more than it saves. Splitting run_streaming into smaller
// functions costs ~8% of shared GEMM throughput: the streaming/ZA state and the
// inlined store stop composing. The monolith is load-bearing; benchmark ANY
// restructuring against examples/bench.rs and examples/epilogue_bench.rs (and run
// the correctness suite) before committing.
// =============================================================================
// [nt_lo, nt_hi) is the N-tile (pad) sub-range this invocation owns -- pass
// (0, n_tiles_pad) for the whole problem, or an even-aligned slice for the
// N-parallel flat-M path (see gemm_sme_f16f16_run_packed). Must be pair-aligned
// (even) so the dual-tile lockstep never splits a pair across workers.
__arm_locally_streaming __arm_new("za") static void run_streaming(
    f16 *dst, long dst_cs, long dst_rs, const f16 *a_pack, const f16 *b_pack, size_t m, size_t n,
    size_t k, size_t mt_lo, size_t mt_hi, size_t n_tiles, size_t nt_lo, size_t nt_hi, float alpha,
    float beta, int read_dst, const ep_desc16 *ep) {
    svbool_t p16 = svptrue_b16();
    // SME2 multi-vector LD1H: one load pulls 4 contiguous depth-vectors of a
    // packed [k,32] panel into a z-group, cutting the inner loop from 12 single
    // loads to 3. Throughput-neutral (the MOPA unit is the bottleneck), but
    // fewer uops and less load-port pressure under multi-cluster contention.
    svcount_t pn = svptrue_c16();
    const svfloat16_t z16 = svdup_n_f16((f16)0.0f);
    const svfloat16_t va = svdup_n_f16((f16)alpha);
    const svfloat16_t vb = svdup_n_f16((f16)beta);
    size_t per_tile = ep_cmul(k, 32);
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);
    // Pure overwrite (C = A@B): the epilogue is just "ZA -> C", which the
    // SME `st1h {zaNv}` family does in ONE instruction per slice -- no
    // read/scale/store. This removes ~2/3 of the epilogue ops, the dominant
    // cost at small K (tall-skinny shapes).
    // A fused epilogue (bias/activation) needs the accumulator in registers,
    // so it cannot use the single-instruction direct-store path.
    // Fold a leading per-column bias (ADD_COL) into the ZA accumulator before the
    // K-loop MOPAs via a rank-1 FMOPA (ones[r] * bias[c] = bias[c]), then drop it
    // from the store-time op-graph. Requires beta==1 (the store applies beta*acc,
    // which would also scale a pre-loaded bias) and !read_dst. The fold adds bias
    // in the f16 ZA element domain (bias-first, then accumulate) instead of after
    // accumulation -- a rounding-order change that stays within the sqrt(k) test
    // tolerance for the f16f16-Fast path. Gated to a bias-ONLY graph: the bias
    // then vanishes (the store reverts to the direct single-instruction path) for
    // free. (bias+activation keeps the cheaper in-store add: folding it perturbs
    // the layout-sensitive store codegen and costs more on the relu store than
    // the fold saves, so the fold is restricted to the bias-only case.)
    int fold_bias = ep && ep->n_nodes == 1 && beta == 1.0f && !read_dst &&
                    ep->nodes[0].op == EP_OP_ADD_COL;
    const f16 *fold_bias_ptr = fold_bias ? (const f16 *)ep->nodes[0].ptr : NULL;
    const ep_desc16 *ep_eff = fold_bias ? NULL : ep;
    int has_ep = ep_eff && ep_eff->n_nodes > 0;
    int pure = (beta == 1.0f && !read_dst && !has_ep);
    const EpNode *nodes = has_ep ? ep_eff->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep_eff->n_nodes : 0;
    int has_tensor = has_ep && ep_has_tensor(nodes, n_nodes);

    // Cache blocking (BLIS jc->ic->mt->nt). Each packed tile-panel is
    // k*32*2 bytes. Pick N- and M-block tile counts so the B-block and the
    // A-block both fit L2: the B-block stays resident across the inner M-tiles,
    // the A-block across the inner N-tiles, so each panel streams from DRAM once
    // instead of once per opposing tile. When the whole problem already fits,
    // the blocks span everything and the outer loops run once (no overhead vs
    // the flat path). This is what keeps 4096-and-up compute-bound instead of
    // collapsing to DRAM bandwidth.
    //
    // The 8 MB is sized to the SMALLER cluster, not the P-cluster: chunks go to
    // both, and on M5 the E-cluster L2 is 6 MB against the P-cluster's 16
    // (hw.perflevel1/0.l2cachesize), so a P-sized block thrashes the E workers
    // that share the loop. Sweeping 8 / 16 / 24 MB over 2048^3 through
    // 1024x8192x8192, f16 and bf16, moves nothing outside +-3% run-to-run noise
    // and not monotonically -- these shapes are MOPA-bound, so the budget is not
    // a live knob here. (Q4 is the exception: there it also sets how many full
    // dequant passes run, which is why 16 MB wins in gemm_f16f16_q4.h.)
    size_t tile_bytes = per_tile * sizeof(f16);
    size_t budget_tiles = (8u * 1024 * 1024) / (tile_bytes ? tile_bytes : 1);
    if (budget_tiles < 4) budget_tiles = 4;
    size_t nt_span = nt_hi - nt_lo;
    size_t nc = budget_tiles / 2;
    nc &= ~(size_t)1;
    if (nc < 2) nc = 2;
    if (nc > nt_span) nc = nt_span;
    size_t mc = budget_tiles / 2;
    if (mc < 1) mc = 1;
    if (mc > mt_hi - mt_lo) mc = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc) {
        size_t jc_end = jc + nc < nt_hi ? jc + nc : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc) {
            size_t ic_end = ic + mc < mt_hi ? ic + mc : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const f16 *at = a_pack + mt * per_tile;
                size_t m0 = mt * 32;
                size_t mrows = (m - m0 < 32) ? (m - m0) : 32;

                for (size_t nt = jc; nt < jc_end; nt += 2) {
                    const f16 *b0 = b_pack + (nt + 0) * per_tile;
                    const f16 *b1 = b_pack + (nt + 1) * per_tile;

                    // Dual-tile accumulate: za0 and za1 are independent, so the two
                    // MOPAs per K-step issue back-to-back with no RAW stall. Unrolled
                    // by 4 so the A-loads run ahead of the MOPAs and the za0/za0 RAW
                    // distance is fully covered by the interleaved za1 work.
                    if (fold_bias) {
                        bias_init_za(p16, fold_bias_ptr, nt, n);
                    } else {
                        svzero_za();
                    }
                    size_t d = 0;
                    for (; d + 4 <= k; d += 4) {
                        // One multi-vector LD1H per panel loads 4 contiguous depth-
                        // vectors (z-group) -- 3 load instructions instead of 12, so
                        // the load port does not gate the 8 MOPAs.
                        svfloat16x4_t A = svld1_f16_x4(pn, at + d * 32);
                        svfloat16x4_t U = svld1_f16_x4(pn, b0 + d * 32);
                        svfloat16x4_t V = svld1_f16_x4(pn, b1 + d * 32);
                        svfloat16_t a0 = svget4_f16(A, 0), a1 = svget4_f16(A, 1),
                                    a2 = svget4_f16(A, 2), a3 = svget4_f16(A, 3);
                        svmopa_za16_m(0, p16, p16, a0, svget4_f16(U, 0));
                        svmopa_za16_m(1, p16, p16, a0, svget4_f16(V, 0));
                        svmopa_za16_m(0, p16, p16, a1, svget4_f16(U, 1));
                        svmopa_za16_m(1, p16, p16, a1, svget4_f16(V, 1));
                        svmopa_za16_m(0, p16, p16, a2, svget4_f16(U, 2));
                        svmopa_za16_m(1, p16, p16, a2, svget4_f16(V, 2));
                        svmopa_za16_m(0, p16, p16, a3, svget4_f16(U, 3));
                        svmopa_za16_m(1, p16, p16, a3, svget4_f16(V, 3));
                    }
                    for (; d < k; d++) {
                        svfloat16_t av = svld1_f16(p16, at + d * 32);
                        svmopa_za16_m(0, p16, p16, av, svld1_f16(p16, b0 + d * 32));
                        svmopa_za16_m(1, p16, p16, av, svld1_f16(p16, b1 + d * 32));
                    }

                    // Epilogue for each of the two N-tiles (skip the odd padding tile).
                    for (size_t t = 0; t < 2; t++) {
                        size_t ntt = nt + t;
                        if (ntt >= n_tiles) break;
                        size_t n0 = ntt * 32;
                        size_t ncols = (n - n0 < 32) ? (n - n0) : 32;
                        if (col_major) {
                            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)mrows);
                            if (pure) {
                                // Direct ZA vertical-slice -> contiguous M-column.
                                if (t == 0)
                                    for (size_t c = 0; c < ncols; c++)
                                        svst1_ver_za16(0, (uint32_t)c, pst,
                                                       dst + (long)(n0 + c) * dst_cs +
                                                           (long)m0 * dst_rs);
                                else
                                    for (size_t c = 0; c < ncols; c++)
                                        svst1_ver_za16(1, (uint32_t)c, pst,
                                                       dst + (long)(n0 + c) * dst_cs +
                                                           (long)m0 * dst_rs);
                            } else {
                                // Col slice (M-oriented, vector spans M): ROW is the
                                // contiguous M-vector load (ptr+m0); COL/SCALAR splat;
                                // TENSOR (col-strided) routes to the scalar path below.
                                for (size_t c = 0; c < ncols; c++) {
                                    f16 *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                                    svfloat16_t acc =
                                        (t == 0) ? svread_ver_za16_m(z16, p16, 0, (uint32_t)c)
                                                 : svread_ver_za16_m(z16, p16, 1, (uint32_t)c);
                                    if (has_ep && !has_tensor) {
                                        ep_store_f16(p16, pst, col, acc, vb, va, read_dst, nodes,
                                                     n_nodes, /*span_n=*/0, /*i=*/n0 + c, n0, m0);
                                    } else if (has_ep) {
                                        // TENSOR present: col-major tensor is a strided
                                        // gather, so materialize the M-column and apply
                                        // the whole op-graph scalar-wise.
                                        f16 tmp[32];
                                        svst1_f16(pst, tmp, acc);
                                        for (size_t r = 0; r < mrows; r++) {
                                            float v = (float)tmp[r] * beta;
                                            v = ep_apply_nodes_scalar_f16(nodes, n_nodes, v, m0 + r,
                                                                          n0 + c);
                                            f16 *cell = col + (long)r * dst_rs;
                                            *cell = (f16)(read_dst ? alpha * (float)(*cell) + v : v);
                                        }
                                    } else {
                                        store_slice(p16, pst, col, acc, vb, read_dst, va);
                                    }
                                }
                            }
                        } else if (row_major) {
                            svbool_t pst = svwhilelt_b16((uint64_t)0, (uint64_t)ncols);
                            if (pure) {
                                // Direct ZA horizontal-slice -> contiguous N-row.
                                if (t == 0)
                                    for (size_t r = 0; r < mrows; r++)
                                        svst1_hor_za16(0, (uint32_t)r, pst,
                                                       dst + (long)(m0 + r) * dst_rs +
                                                           (long)n0 * dst_cs);
                                else
                                    for (size_t r = 0; r < mrows; r++)
                                        svst1_hor_za16(1, (uint32_t)r, pst,
                                                       dst + (long)(m0 + r) * dst_rs +
                                                           (long)n0 * dst_cs);
                            } else if (has_ep) {
                                // Row slice (N-oriented, vector spans N), NODE-MAJOR:
                                // read the live ZA tile directly into Z registers in
                                // 4-row blocks and dispatch each op-graph node ONCE
                                // per block across all its rows (invariant COL vector
                                // / scalars / act constants loaded once per node, not
                                // once per row); no memory bounce.
                                f16 *base = dst + (long)m0 * dst_rs + (long)n0 * dst_cs;
                                if (t == 0) {
#define EP_RD0(s) svread_hor_za16_m(z16, p16, 0, (s))
                                    EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, dst_rs, EP_RD0, mrows,
                                                               vb, va, read_dst, nodes, n_nodes, m0,
                                                               n0);
#undef EP_RD0
                                } else {
#define EP_RD1(s) svread_hor_za16_m(z16, p16, 1, (s))
                                    EP_STORE_TILE_ROWMAJOR_F16(p16, pst, base, dst_rs, EP_RD1, mrows,
                                                               vb, va, read_dst, nodes, n_nodes, m0,
                                                               n0);
#undef EP_RD1
                                }
                            } else {
                                for (size_t r = 0; r < mrows; r++) {
                                    f16 *rowp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                                    svfloat16_t acc =
                                        (t == 0) ? svread_hor_za16_m(z16, p16, 0, (uint32_t)r)
                                                 : svread_hor_za16_m(z16, p16, 1, (uint32_t)r);
                                    store_slice(p16, pst, rowp, acc, vb, read_dst, va);
                                }
                            }
                        } else {
                            f16 scratch[32 * 32];
                            svbool_t pg = svwhilelt_b16((uint64_t)0, (uint64_t)32);
                            for (uint32_t r = 0; r < 32; r++) {
                                svfloat16_t row = (t == 0) ? svread_hor_za16_m(z16, p16, 0, r)
                                                           : svread_hor_za16_m(z16, p16, 1, r);
                                svst1_f16(pg, scratch + (size_t)r * 32, row);
                            }
                            for (size_t r = 0; r < mrows; r++)
                                for (size_t c = 0; c < ncols; c++) {
                                    float ab = (float)scratch[r * 32 + c] * beta;
                                    if (has_ep)
                                        ab = ep_apply_nodes_scalar_f16(nodes, n_nodes, ab, m0 + r,
                                                                       n0 + c);
                                    f16 *cell =
                                        dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                                    *cell = (f16)(read_dst ? alpha * (float)(*cell) + ab : ab);
                                }
                        }
                    }
                }
            } // mt
        } // ic
    } // jc
}

// Pack A into [m_tiles, k, 32]. Fast paths for col-major (lhs_rs==1, memcpy)
// and row-major (lhs_cs==1, cache-blocked transpose); strided gather otherwise.
static void packa(f16 *a_pack, const f16 *a, size_t m, size_t k, long lhs_rs, long lhs_cs,
                  size_t mt_lo, size_t mt_hi, size_t per_tile) {
    if (lhs_rs == 1) {
        const size_t TB = 8; // tiles per block: bounds the write streams
        for (size_t tb = mt_lo; tb < mt_hi; tb += TB) {
            size_t tb_end = tb + TB < mt_hi ? tb + TB : mt_hi;
            for (size_t d = 0; d < k; d++) {
                const f16 *col = a + (long)d * lhs_cs;
                for (size_t mt = tb; mt < tb_end; mt++) {
                    size_t r0 = mt * 32;
                    size_t vr = (m - r0 < 32) ? (m - r0) : 32;
                    f16 *dp = a_pack + mt * per_tile + d * 32;
                    if (vr == 32) {
                        memcpy(dp, col + r0, 32 * sizeof(f16));
                    } else {
                        memcpy(dp, col + r0, vr * sizeof(f16));
                        for (size_t i = vr; i < 32; i++)
                            dp[i] = (f16)0.0f;
                    }
                }
            }
        }
    } else if (lhs_cs == 1) {
        // Row-major A: each row is contiguous in depth, which is exactly the
        // layout pack_lanes_depth_major transposes (rows as lanes).
        for (size_t mt = mt_lo; mt < mt_hi; mt++) {
            size_t r0 = mt * 32;
            size_t mrows = (m - r0 < 32) ? (m - r0) : 32;
            pack_lanes_depth_major(a_pack + mt * per_tile, a + (long)r0 * lhs_rs, lhs_rs, k, mrows);
        }
    } else {
        for (size_t mt = mt_lo; mt < mt_hi; mt++) {
            size_t r0 = mt * 32;
            size_t vr = (m - r0 < 32) ? (m - r0) : 32;
            pack_a_tile(a_pack + mt * per_tile, a + (long)r0 * lhs_rs, lhs_rs, lhs_cs, k, vr);
        }
    }
}

// Pack B into [n_tiles_pad, k, 32]; caller zeroes pad tiles. Fast path for
// row-major weights (rhs_cs==1); strided gather otherwise.
static void packb(f16 *b_pack, const f16 *b, size_t n, size_t k, long rhs_rs, long rhs_cs,
                  size_t n_tiles, size_t per_tile) {
    if (rhs_cs == 1) {
        const size_t TB = 8;
        for (size_t tb = 0; tb < n_tiles; tb += TB) {
            size_t tb_end = tb + TB < n_tiles ? tb + TB : n_tiles;
            for (size_t d = 0; d < k; d++) {
                const f16 *row = b + (long)d * rhs_rs;
                for (size_t nt = tb; nt < tb_end; nt++) {
                    size_t c0 = nt * 32;
                    size_t vc = (n - c0 < 32) ? (n - c0) : 32;
                    memcpy(b_pack + nt * per_tile + d * 32, row + c0, vc * sizeof(f16));
                }
            }
        }
    } else {
        for (size_t nt = 0; nt < n_tiles; nt++) {
            size_t c0 = nt * 32;
            size_t vc = (n - c0 < 32) ? (n - c0) : 32;
            pack_b_tile(b_pack + nt * per_tile, b + (long)c0 * rhs_cs, rhs_rs, rhs_cs, k, vc);
        }
    }
}

// Per-thread reusable A-pack scratch (no per-call malloc/free on the serial
// small/medium path). Grows monotonically, lives for the thread's lifetime.
static _Thread_local f16 *g_apack = NULL;
static _Thread_local size_t g_apack_cap = 0;
static f16 *apack_scratch(size_t bytes) {
    if (bytes == 0) bytes = 1; // k==0: return a real (tiny) buffer, not a
                               // history-dependent NULL on fresh threads
    if (g_apack_cap < bytes) {
        free(g_apack);
        g_apack = (f16 *)malloc(bytes);
        g_apack_cap = g_apack ? bytes : 0;
    }
    return g_apack;
}

// Packed-B GEMM: caller supplies B pre-packed via gemm_sme_f16f16_packb. Only
// A is packed here -- the inference pattern (weights packed once, reused).
int gemm_sme_f16f16_run_packed(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs,
                               long dst_rs, int read_dst, const uint16_t *lhs, long lhs_cs,
                               long lhs_rs, const uint16_t *b_pack, uint16_t alpha_bits,
                               uint16_t beta_bits, const ep_desc16 *ep) {
    if (m == 0 || n == 0) return 0;
    float alpha = read_dst ? (float)f16_from_bits(alpha_bits) : 0.0f;
    float beta = (float)f16_from_bits(beta_bits);
    size_t m_tiles = (m + 31) / 32;
    size_t n_tiles = (n + 31) / 32;
    size_t n_tiles_pad = (n_tiles + 1) & ~(size_t)1;
    size_t per_tile = ep_cmul(k, 32);
    const f16 *a = (const f16 *)lhs;
    const f16 *bp = (const f16 *)b_pack;
    f16 *d = (f16 *)dst;

    // n_tiles==1 (n<=32) row-major: the dual-tile scheme wastes za1 on the pad
    // N-tile (~2x slow per flop). run_narrow_rowmajor pairs two M-tiles against the
    // single B-tile instead. Same M-chunk parallelism; just a different inner loop.
    int narrow = (n_tiles == 1 && dst_cs == 1);
    // ODD N-TILE TAIL. run_streaming pairs N-tiles into za0/za1, so an odd tile
    // count leaves the final pair's za1 accumulating the zero pad tile -- it
    // wastes 1/(2*ceil(n_tiles/2)) of all MOPA issue, measured ~10% at
    // n_tiles=9 (3.86 vs 4.30/4.32 TF/s at the even neighbours). Handing that
    // last tile to run_narrow_rowmajor instead pairs two M-tiles against it, so
    // both ZA tiles stay productive. A-traffic is unchanged: the tile was
    // already being read once either way.
    //
    // Not when the bias folds. run_streaming folds a leading ADD_COL into ZA via
    // a rank-1 MOPA while run_narrow_rowmajor applies it at the store, so
    // splitting a folded call would round the two halves of one output
    // differently. Same predicate as `fold_bias` inside run_streaming.
    int fold_bias_here =
        ep && ep->n_nodes == 1 && beta == 1.0f && !read_dst && ep->nodes[0].op == EP_OP_ADD_COL;
    int odd_tail = (n_tiles > 1) && (n_tiles & 1) && dst_cs == 1 && !fold_bias_here;
    // Two tiles per chunk. The clusters are asymmetric, so fine chunks let
    // dispatch_apply work-steal instead of stalling on the slow half: 2 beats 4
    // by ~5% geomean and 9-13% at 256^3/512^3/4096^3. 1 beats 2 on 1024^3/2048^3
    // but loses 8% on 16384x512x512. Must stay EVEN -- run_narrow_rowmajor pairs
    // M-tiles, and an odd chunk would idle za1.
    size_t M_CHUNK = 2;
    size_t n_chunks = (m_tiles + M_CHUNK - 1) / M_CHUNK;
    int big = ep_flops(m, n, k) >= (1u << 21);
    // Flat-M (few M-tiles) but large and wide: M is the only parallel axis, so the
    // M-chunk scheme would run this on ONE cluster. Parallelize over N instead --
    // pack all of A once (cheap, m_tiles is small) into a shared buffer, then hand
    // even-aligned N-tile-pair chunks to both clusters. C columns per chunk are
    // disjoint; A and B are read-only and shared.
    // Tiles per chunk; must stay even so a dual-tile pair is never split. Every
    // chunk re-streams the whole A-panel for its N-range, so halving this to 2
    // buys parallelism more slowly than it adds traffic: 0.87-1.03x across
    // m<=64, n=64..512, k=4096..8192, worse on 5 of 7 shapes.
    size_t N_CHUNK = 4;
    size_t nn_chunks = (n_tiles_pad + N_CHUNK - 1) / N_CHUNK;
    // THREE chunks, not two. A 2-way split is slower than not splitting at all:
    // 1.03-1.15x serial across n=160..256, k=512..16384, m=1..64, both dtypes,
    // fading to parity by k=32768 where B-streaming dominates. The clusters are
    // asymmetric, so a 2-way split's makespan is the E-cluster's half, which
    // costs more than running the whole thing on the caller's cluster; from 3
    // chunks up dispatch_apply can work-steal and the split pays again.
    //
    // Not the same as the M-chunk path below, where a 2-way split *wins*
    // (forcing it serial measures 0.78-0.97x): those chunks pack their own A
    // slice, so a second cluster overlaps NEON packing with MOPA issue. Here A
    // is packed once up front and the chunks are pure MOPA, with nothing to
    // overlap. Nor is it the allocator -- swapping this malloc for the
    // thread-local scratch measures 1.00-1.01x.
    if (n_chunks <= 1 && big && nn_chunks >= 3) {
        f16 *a_pack = (f16 *)malloc(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(f16)));
        if (a_pack) {
            packa(a_pack, a, m, k, lhs_rs, lhs_cs, 0, m_tiles, per_tile);
            dispatch_apply(nn_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
              size_t nt0 = ci * N_CHUNK;
              size_t nt1 = nt0 + N_CHUNK < n_tiles_pad ? nt0 + N_CHUNK : n_tiles_pad;
              // The chunk holding the unpaired last tile stops one short and runs
              // that tile M-paired instead, as the serial path does. nt0 is a
              // multiple of N_CHUNK and n_tiles-1 is even here, so the truncated
              // range stays pair-aligned (and may be empty, which is fine).
              if (odd_tail && nt0 <= n_tiles - 1 && n_tiles - 1 < nt1) {
                  run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, nt0,
                                n_tiles - 1, alpha, beta, read_dst, ep);
                  run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles,
                                      n_tiles - 1, alpha, beta, read_dst, ep);
              } else {
                  run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, nt0,
                                nt1, alpha, beta, read_dst, ep);
              }
            });
            free(a_pack);
            return 0;
        }
        // malloc failed: fall through to the serial path.
    }
    if (n_chunks <= 1 || !big) {
        // Serial: per-thread reusable scratch, no per-call malloc/free.
        f16 *a_pack = apack_scratch(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(f16)));
        if (!a_pack) return -1;
        packa(a_pack, a, m, k, lhs_rs, lhs_cs, 0, m_tiles, per_tile);
        if (narrow) {
            run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, 0, alpha, beta,
                                read_dst, ep);
        } else if (odd_tail) {
            run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, 0,
                          n_tiles - 1, alpha, beta, read_dst, ep);
            run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles - 1,
                                alpha, beta, read_dst, ep);
        } else {
            run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, 0, m_tiles, n_tiles, 0,
                          n_tiles_pad, alpha, beta, read_dst, ep);
        }
        return 0;
    }
    f16 *a_pack = (f16 *)malloc(ep_cmul(ep_cmul(m_tiles, per_tile), sizeof(f16)));
    if (!a_pack) return -1;
    dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t ci) {
      size_t mt0 = ci * M_CHUNK;
      size_t mt1 = mt0 + M_CHUNK < m_tiles ? mt0 + M_CHUNK : m_tiles;
      packa(a_pack, a, m, k, lhs_rs, lhs_cs, mt0, mt1, per_tile);
      if (narrow) {
          run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, mt0, mt1, 0, alpha, beta,
                              read_dst, ep);
      } else if (odd_tail) {
          run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, mt0, mt1, n_tiles, 0, n_tiles - 1,
                        alpha, beta, read_dst, ep);
          run_narrow_rowmajor(d, dst_cs, dst_rs, a_pack, bp, m, n, k, mt0, mt1, n_tiles - 1, alpha,
                              beta, read_dst, ep);
      } else {
          run_streaming(d, dst_cs, dst_rs, a_pack, bp, m, n, k, mt0, mt1, n_tiles, 0, n_tiles_pad,
                        alpha, beta, read_dst, ep);
      }
    });
    free(a_pack);
    return 0;
}

// Pack B once into a caller-allocated, zeroed buffer of
// gemm_sme_f16f16_packed_b_elems(n,k) f16. Reuse across many GEMMs (weights).
void gemm_sme_f16f16_packb(uint16_t *b_pack, const uint16_t *rhs, size_t n, size_t k, long rhs_rs,
                           long rhs_cs) {
    packb((f16 *)b_pack, (const f16 *)rhs, n, k, rhs_rs, rhs_cs, (n + 31) / 32, ep_cmul(k, 32));
}

// Saturating: SIZE_MAX on overflow (see gemm_sme_i8i32_packed_b_elems).
size_t gemm_sme_f16f16_packed_b_elems(size_t n, size_t k) {
    size_t n_tiles_pad = (((n + 31) / 32) + 1) & ~(size_t)1;
    return ep_cmul(ep_cmul(n_tiles_pad, k), 32);
}

int gemm_sme_f16f16_run(size_t m, size_t n, size_t k, uint16_t *dst, long dst_cs, long dst_rs,
                        int read_dst, const uint16_t *lhs, long lhs_cs, long lhs_rs,
                        const uint16_t *rhs, long rhs_cs, long rhs_rs, uint16_t alpha_bits,
                        uint16_t beta_bits) {
    if (m == 0 || n == 0) return 0;
    size_t per_tile = ep_cmul(k, 32);
    size_t n_tiles_pad = (((n + 31) / 32) + 1) & ~(size_t)1;
    f16 *b_pack = (f16 *)calloc(ep_cmul(n_tiles_pad, per_tile), sizeof(f16));
    if (!b_pack) return -1;
    packb(b_pack, (const f16 *)rhs, n, k, rhs_rs, rhs_cs, (n + 31) / 32, per_tile);
    int rc = gemm_sme_f16f16_run_packed(m, n, k, dst, dst_cs, dst_rs, read_dst, lhs, lhs_cs, lhs_rs,
                                        (const uint16_t *)b_pack, alpha_bits, beta_bits, NULL);
    free(b_pack);
    return rc;
}

// Batched f16: `count` independent same-shape GEMMs in ONE streaming session, so
#include "gemm_f16f16_batched.h"

// The 4-bit-resident path, kept in its own file (it needs `packa` and the f16
// typedef, so it is included rather than compiled separately).
#include "gemm_f16f16_q4.h"
