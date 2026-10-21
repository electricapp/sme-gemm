// The f32 and f64 MOPA kernels as one template over the element type T: each
// super-tile is 2x2 ZA tiles of W = 64/sizeof(T) lanes (16 for f32, 8 for f64),
// one K value per MOPA, fp in and out, so the store is either a direct ZA->C
// write (pure C = A@B) or read/scale/epilogue/store. Included by gemm_f32.cc and
// gemm_f64.cc, which instantiate it under their own target features and keep
// the dtype-specific drivers, packing and GEMV paths.
#ifndef SME_GEMM_FLOAT_H
#define SME_GEMM_FLOAT_H

#include <arm_sme.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <type_traits>

#include "epilogue.h"
#include "sme_tile.h"

// Resident-bytes budget (A-block plus B-block) for run_streaming's cache
// blocking. Larger than the 16-bit drivers' because a float panel holds fewer K
// values per byte.
#define FLOAT_CACHE_BUDGET_BYTES ((size_t)24 << 20)

// N-block width for the drivers' outer jc loop; run_streaming re-derives the
// same split for its own range, so the two agree on block boundaries.
static inline size_t float_nc_blk(size_t k, size_t n_tiles) {
    return quad_nc_blk(quad_budget(k, FLOAT_CACHE_BUDGET_BYTES), n_tiles);
}

// K-loop load widths: f32 full super-tiles take four depths per load (one LD1W
// x4 per band) and narrow ones two; f64 takes two.
template <class T> using FloatK = std::conditional_t<sizeof(T) == 4, KSteps<4, 2>, KSteps<2>>;

// The epilogue entry points by element type. The scalar walker is not streaming
// (it calls libm), so the kernels call it out of line, as the rare paths it serves
// always have.
static inline float ep_nodes_scalar(const EpNode *nodes, uint32_t n_nodes, float x, size_t i,
                                    size_t j) {
    return ep_apply_nodes_scalar_f32(nodes, n_nodes, x, i, j);
}
static inline double ep_nodes_scalar(const EpNode *nodes, uint32_t n_nodes, double x, size_t i,
                                     size_t j) {
    return ep_apply_nodes_scalar_f64(nodes, n_nodes, x, i, j);
}
// What one call's store does, decided once. A leading per-column bias
// (ADD_COL) alone folds into ZA through rank-1 MOPAs before the K-loop and is
// dropped from the graph, so a bias-only graph reverts to the direct store (the
// accumulator domain matches the bias domain, so this is exact). It requires
// beta == 1 and no read of C, and stays gated to bias-only: folding
// bias+activation perturbs the store codegen.
template <class T> struct FloatEpi {
    const T *bias;       // folded ADD_COL operand, or NULL
    const EpNode *nodes; // the graph left after the fold, NULL if none
    uint32_t n_nodes;
    int has_tensor, pure, read_dst;
    T alpha, beta;
};
template <class T, class Ep>
SME_INLINE FloatEpi<T> float_epi(const Ep *ep, T alpha, T beta,
                                 int read_dst) __arm_streaming_compatible {
    FloatEpi<T> e = {};
    int fold =
        ep && ep->n_nodes == 1 && beta == (T)1 && !read_dst && ep->nodes[0].op == EP_OP_ADD_COL;
    e.bias = fold ? (const T *)ep->nodes[0].ptr : NULL;
    if (!fold && ep && ep->n_nodes > 0) {
        e.nodes = ep->nodes;
        e.n_nodes = ep->n_nodes;
        e.has_tensor = ep_has_tensor(e.nodes, e.n_nodes);
    }
    e.pure = beta == (T)1 && !read_dst && !e.n_nodes;
    e.read_dst = read_dst;
    e.alpha = alpha;
    e.beta = beta;
    return e;
}

// Zero ZA, then seed the live quadrants with the bias row through rank-1 MOPAs
// (ones x bias): the lo bias half inits tiles 0 and 2, the hi half 1 and 3, and
// the tiles the K-loop leaves dead stay zero. Out of line so the fold does not
// perturb the store codegen.
template <class T>
__attribute__((noinline)) static void bias_init_za(const T *bias, size_t n0, size_t n, size_t nc,
                                                   size_t mr) __arm_streaming __arm_inout("za") {
    using A = Mopa<T, T>;
    constexpr size_t W = A::kTile;
    svbool_t p = ptrue<sizeof(T)>();
    svzero_za();
    Vec<T> ones = Sve<T>::dup(1);
    size_t nlo = n0, nhi = n0 + W;
    svbool_t plo = nlo < n ? whilelt<sizeof(T)>(nlo, n) : svpfalse_b();
    svbool_t phi = nhi < n ? whilelt<sizeof(T)>(nhi, n) : svpfalse_b();
    // Clamp the bias addresses for the all-false (pad) halves: no lane is read,
    // but forming bias + nlo/nhi past one-past-end would still be UB.
    Vec<T> blo = svld1(plo, nlo < n ? bias + nlo : bias);
    Vec<T> bhi = svld1(phi, nhi < n ? bias + nhi : bias);
    A::template mopa<0>(p, plo, ones, blo);
    if (nc > W) A::template mopa<1>(p, phi, ones, bhi);
    if (mr > W) A::template mopa<2>(p, plo, ones, blo);
    if (nc > W && mr > W) A::template mopa<3>(p, phi, ones, bhi);
}

// Zero ZA, or seed it with the folded bias.
template <class T>
SME_INLINE void quad_init(const FloatEpi<T> &e, size_t n0, size_t n, size_t nc,
                          size_t mr) __arm_streaming __arm_inout("za") {
    if (e.bias)
        bias_init_za(e.bias, n0, n, nc, mr);
    else
        svzero_za();
}

// Columns [c0, c1) of a col-major super-tile from the ZA tile pair (ZL, ZH), each
// through the vector op-graph. The values come in as parameters, not lambda
// captures, and the epilogue call is the plain one with its constant span_n:
// around an out-of-line call per column, either difference costs registers.
template <int ZL, int ZH, class T>
SME_INLINE void store_cols_ep(T *dst, long dst_cs, size_t m0, size_t n0, size_t c0, size_t c1,
                              size_t mr_hi, svbool_t plo, svbool_t phi, Vec<T> va, Vec<T> vb,
                              int read_dst, const EpNode *nodes,
                              uint32_t n_nodes) __arm_streaming __arm_in("za") {
    constexpr size_t W = 64 / sizeof(T);
    svbool_t p = ptrue<sizeof(T)>();
    for (size_t c = c0; c < c1; c++) {
        uint32_t s = (uint32_t)(c - c0);
        T *col = dst + (long)(n0 + c) * dst_cs + (long)m0;
        T *colh = mr_hi ? col + W : col;
        Vec<T> accl = za_col<ZL, T>(s), acch = za_col<ZH, T>(s);
        if constexpr (sizeof(T) == 4) {
            ep_store_f32(p, plo, col, accl, vb, va, read_dst, nodes, n_nodes, 0, n0 + c, n0, m0);
            ep_store_f32(p, phi, colh, acch, vb, va, read_dst, nodes, n_nodes, 0, n0 + c, n0,
                         m0 + W);
        } else {
            ep_store_f64(p, plo, col, accl, vb, va, read_dst, nodes, n_nodes, 0, n0 + c, n0, m0);
            ep_store_f64(p, phi, colh, acch, vb, va, read_dst, nodes, n_nodes, 0, n0 + c, n0,
                         m0 + W);
        }
    }
}

// Store the live mr x nc corner of super-tile (m0, n0) to C:
// C = alpha*C + nodes(beta*acc). RowMajor is set when C is known row-major.
template <bool RowMajor = false, class T>
SME_INLINE void quad_store(T *dst, long dst_cs, long dst_rs, size_t m0, size_t n0, size_t mr,
                           size_t nc, const FloatEpi<T> &e, Vec<T> va,
                           Vec<T> vb) __arm_streaming __arm_in("za") {
    constexpr size_t W = 64 / sizeof(T);
    svbool_t p = ptrue<sizeof(T)>();
    size_t mr_lo = mr < W ? mr : W, mr_hi = mr > W ? mr - W : 0;
    size_t nc_lo = nc < W ? nc : W, nc_hi = nc > W ? nc - W : 0;
    const EpNode *nodes = e.nodes;
    uint32_t n_nodes = e.n_nodes;
    int read_dst = e.read_dst;
    if (e.pure) {
        quad_store_raw<RowMajor>(dst, dst_cs, dst_rs, m0, n0, mr, nc);
        return;
    }
    if (!RowMajor && dst_rs == 1) {
        // Col-major C: one column of the super-tile per ZA column, lo-M rows from
        // tile zl and hi-M from zh.
        svbool_t plo = whilelt<sizeof(T)>(0, mr_lo), phi = whilelt<sizeof(T)>(0, mr_hi);
        // Clamp so no one-past-end pointer is formed; the hi-M band is empty
        // (phi all-false) when narrow-M.
        size_t hw = mr_hi ? W : 0;
        if (n_nodes && !e.has_tensor) {
            store_cols_ep<0, 2>(dst, dst_cs, m0, n0, 0, nc_lo, mr_hi, plo, phi, va, vb, read_dst,
                                nodes, n_nodes);
            store_cols_ep<1, 3>(dst, dst_cs, m0, n0, W, nc, mr_hi, plo, phi, va, vb, read_dst,
                                nodes, n_nodes);
            return;
        }
        quad_cols<W>(
            nc, [&] [[gnu::always_inline]] (auto zl, auto zh, uint32_t s,
                                            size_t c) __arm_streaming __arm_in("za") {
                T *col = dst + (long)(n0 + c) * dst_cs + (long)m0, *colh = col + hw;
                Vec<T> accl = za_col<zl, T>(s), acch = za_col<zh, T>(s);
                if (n_nodes) {
                    // A TENSOR operand read down a column is a gather: go scalar.
                    T tmp[2 * W];
                    svst1(plo, tmp, accl);
                    svst1(phi, tmp + W, acch);
                    for (size_t r = 0; r < mr; r++) {
                        T v = ep_nodes_scalar(nodes, n_nodes, tmp[r] * e.beta, m0 + r, n0 + c);
                        T *cell = col + (long)r * dst_rs;
                        *cell = read_dst ? e.alpha * *cell + v : v;
                    }
                } else {
                    Vec<T> lo = svmul_x(p, accl, vb), hi = svmul_x(p, acch, vb);
                    if (read_dst) {
                        lo = svmla_x(p, lo, svld1(plo, col), va);
                        hi = svmla_x(p, hi, svld1(phi, colh), va);
                    }
                    svst1(plo, col, lo);
                    svst1(phi, colh, hi);
                }
            });
    } else if (RowMajor || dst_cs == 1) {
        svbool_t plo = whilelt<sizeof(T)>(0, nc_lo), phi = whilelt<sizeof(T)>(0, nc_hi);
        if (n_nodes) {
            // Node-major: read the live ZA half-rows in 4-row blocks and dispatch
            // each node once per block across all halves.
            T *rbase = dst + (long)m0 * dst_rs + (long)n0;
            ep_store_rowmajor(p, plo, phi, rbase, dst_rs, ZaRowPair<0, 1, T>{}, 0, mr_lo, vb, va,
                              read_dst, nodes, n_nodes, m0, n0);
            ep_store_rowmajor(p, plo, phi, rbase, dst_rs, ZaRowPair<2, 3, T>{}, W, mr, vb, va,
                              read_dst, nodes, n_nodes, m0, n0);
            return;
        }
        // Clamp so no one-past-end pointer is formed; the hi-N band is empty
        // (phi all-false) when narrow-N.
        size_t hw = nc_hi ? W : 0;
        quad_rows<W>(
            mr, [&] [[gnu::always_inline]] (auto zl, auto zh, uint32_t s,
                                            size_t r) __arm_streaming __arm_in("za") {
                T *rp = dst + (long)(m0 + r) * dst_rs + (long)n0, *rph = rp + hw;
                Vec<T> lo = svmul_x(p, za_row<zl, T>(s), vb), hi = svmul_x(p, za_row<zh, T>(s), vb);
                if (read_dst) {
                    lo = svmla_x(p, lo, svld1(plo, rp), va);
                    hi = svmla_x(p, hi, svld1(phi, rph), va);
                }
                svst1(plo, rp, lo);
                svst1(phi, rph, hi);
            });
    } else {
        // Fully general strides: through a scratch, scalar combine. Rare.
        T scratch[4 * W * W];
        quad_to_scratch(scratch);
        for (size_t r = 0; r < mr; r++)
            for (size_t c = 0; c < nc; c++) {
                T ab = scratch[r * 2 * W + c] * e.beta;
                if (n_nodes) ab = ep_nodes_scalar(nodes, n_nodes, ab, m0 + r, n0 + c);
                T *cell = dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                *cell = read_dst ? e.alpha * *cell + ab : ab;
            }
    }
}

// The per-row outputs of a row-major GEMM: each finished row's sum and max
// (either NULL to skip), length m. Only run_rows takes them, and it has no
// column stride or N-range to give, so a reduction can only ever run over
// whole, contiguous, finished rows.
template <class T> struct RowStats {
    T *sum, *max;
};
struct NoRowStats {};

// One horizontal sum and max per finished row, over the whole row, right after
// its M-tile's last N-tile wrote it -- so the row is still L1-resident. Doing it
// per (row, N-tile) instead measured 2.2x slower than the GEMM itself:
// svaddv/svmaxv are serializing cross-lane ops, and n_tiles of them per row
// dwarf the sweep they save.
template <class T>
SME_INLINE void reduce_rows(const T *dst, size_t m0, size_t mr, size_t n,
                            RowStats<T> out) __arm_streaming {
    svbool_t p = ptrue<sizeof(T)>();
    constexpr size_t W = 64 / sizeof(T);
    for (size_t r = 0; r < mr; r++) {
        const T *rp = dst + (m0 + r) * n;
        // Accumulate into VECTORS and reduce across lanes exactly once per row.
        // Calling svaddv/svmaxv per chunk instead builds an n/W-long serial
        // dependency chain of cross-lane ops -- measured 3.5x the whole GEMM.
        Vec<T> vs = Sve<T>::zero(), vm = Sve<T>::dup((T)-INFINITY);
        size_t j = 0;
        for (; j + W <= n; j += W) {
            Vec<T> v = svld1(p, rp + j);
            vs = svadd_x(p, vs, v);
            vm = svmaxnm_x(p, vm, v);
        }
        if (j < n) {
            svbool_t pt = whilelt<sizeof(T)>(j, n);
            Vec<T> v = svld1(pt, rp + j);
            // Identity in the inactive lanes so the tail cannot perturb either.
            vs = svadd_m(pt, vs, v);
            vm = svmaxnm_m(pt, vm, v);
        }
        if (out.sum) out.sum[m0 + r] = svaddv(p, vs);
        if (out.max) out.max[m0 + r] = svmaxv(p, vm);
    }
}

// The packed loop nest behind run_streaming and run_rows. With RowStats, C is
// row-major by construction: the store never reads dst_cs, and all of N is one
// block so each M-tile's rows are finished before they are reduced.
template <class T, class Ep, class Stats>
SME_INLINE void packed_tiles(T *dst, long dst_cs, long dst_rs, const T *a_pack, const T *b_pack,
                             size_t m, size_t n, size_t mt_lo, size_t mt_hi, size_t nt_lo,
                             size_t nt_hi, size_t k, T alpha, T beta, int read_dst, const Ep *ep,
                             Stats stats) __arm_streaming __arm_inout("za") {
    using A = Mopa<T, T>;
    constexpr size_t W = A::kTile;
    constexpr bool kRows = std::is_same_v<Stats, RowStats<T>>;
    const Vec<T> va = Sve<T>::dup(alpha), vb = Sve<T>::dup(beta);
    size_t per_tile = ep_cmul(k, W);
    FloatEpi<T> e = float_epi(ep, alpha, beta, read_dst);

    // Cache blocking (BLIS jc->ic) in super-tiles. mc_blk is capped at this
    // call's M range, so in the parallel paths only nc_blk grows; cross-chunk B
    // reuse comes from the driver running this once per N-block.
    size_t budget = quad_budget(k, FLOAT_CACHE_BUDGET_BYTES);
    size_t nt_span = nt_hi - nt_lo;
    size_t nc_blk = kRows ? nt_span : quad_nc_blk(budget, nt_span);
    size_t mc_blk = budget / 2;
    if (mc_blk < 1) mc_blk = 1;
    if (mc_blk > mt_hi - mt_lo) mc_blk = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc_blk) {
        size_t jc_end = jc + nc_blk < nt_hi ? jc + nc_blk : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc_blk) {
            size_t ic_end = ic + mc_blk < mt_hi ? ic + mc_blk : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const T *a_lo = a_pack + 2 * mt * per_tile, *a_hi = a_lo + per_tile;
                size_t m0 = mt * 2 * W, mr = m - m0 < 2 * W ? m - m0 : 2 * W;
                for (size_t nt = jc; nt < jc_end; nt++) {
                    const T *b_lo = b_pack + 2 * (nt - nt_lo) * per_tile, *b_hi = b_lo + per_tile;
                    size_t n0 = nt * 2 * W, nc = n - n0 < 2 * W ? n - n0 : 2 * W;
                    quad_init(e, n0, n, nc, mr);
                    quad_accumulate<A, FloatK<T>>(k, a_lo, a_hi, b_lo, b_hi, mr > W, nc > W);
                    quad_store<kRows>(dst, dst_cs, dst_rs, m0, n0, mr, nc, e, va, vb);
                }
                if constexpr (kRows) reduce_rows(dst, m0, mr, n, stats);
            }
        }
    }
}

// The packed kernel: [nt_lo, nt_hi) is the N-tile sub-range this invocation
// owns -- (0, n_tiles) for the whole problem, or a slice for the N-parallel
// paths -- and b_pack points at super-tile nt_lo's bands. Super-tiles are
// independent (no cross-tile state), so any chunk granularity is safe.
template <class T, class Ep>
__arm_locally_streaming __arm_new("za") static void run_streaming(
    T *dst, long dst_cs, long dst_rs, const T *a_pack, const T *b_pack, size_t m, size_t n,
    size_t mt_lo, size_t mt_hi, size_t nt_lo, size_t nt_hi, size_t k, T alpha, T beta, int read_dst,
    const Ep *ep) {
    packed_tiles(dst, dst_cs, dst_rs, a_pack, b_pack, m, n, mt_lo, mt_hi, nt_lo, nt_hi, k, alpha,
                 beta, read_dst, ep, NoRowStats{});
}

// M-tiles [mt_lo, mt_hi) of C = nodes(beta*A@B) into dense row-major C (row
// stride n) across all of N, writing each finished row's sum and max.
template <class T, class Ep>
__arm_locally_streaming __arm_new("za") static void run_rows(T *dst, const T *a_pack,
                                                             const T *b_pack, size_t m, size_t n,
                                                             size_t mt_lo, size_t mt_hi, size_t k,
                                                             T beta, const Ep *ep,
                                                             RowStats<T> stats) {
    size_t n_tiles = (n + 128 / sizeof(T) - 1) / (128 / sizeof(T));
    packed_tiles(dst, 1, (long)n, a_pack, b_pack, m, n, mt_lo, mt_hi, 0, n_tiles, k, (T)0, beta, 0,
                 ep, stats);
}

// Small-GEMM path: serial (one streaming region, no GCD), A pre-packed, B read
// straight from row-major rhs -- B[d, n0..] is already contiguous, so no B pack.
// Targets the small regime where the parallel path's dispatch and full A/B
// packing dwarf the compute.
template <class T, class Ep>
__arm_locally_streaming __arm_new("za") static void run_small(
    T *dst, long dst_cs, long dst_rs, const T *a_pack, const T *rhs, long rhs_rs, size_t m,
    size_t n, size_t k, size_t mt_lo, size_t mt_hi, T alpha, T beta, int read_dst, const Ep *ep) {
    using A = Mopa<T, T>;
    constexpr size_t W = A::kTile;
    const Vec<T> va = Sve<T>::dup(alpha), vb = Sve<T>::dup(beta);
    size_t per_tile = ep_cmul(k, W);
    size_t n_tiles = (n + 2 * W - 1) / (2 * W);
    FloatEpi<T> e = float_epi(ep, alpha, beta, read_dst);
    for (size_t mt = mt_lo; mt < mt_hi; mt++) {
        const T *a_lo = a_pack + 2 * mt * per_tile, *a_hi = a_lo + per_tile;
        size_t m0 = mt * 2 * W, mr = m - m0 < 2 * W ? m - m0 : 2 * W;
        for (size_t nt = 0; nt < n_tiles; nt++) {
            size_t n0 = nt * 2 * W, nc = n - n0 < 2 * W ? n - n0 : 2 * W;
            size_t vc1 = nc > W ? nc - W : 0;
            // Predicated loads zero the partial N-band tail (and stop the last
            // band reading past the rhs buffer). The hi-N band is only read when
            // it has columns; otherwise its base is clamped to the lo band's.
            svbool_t pb0 = whilelt<sizeof(T)>(0, nc < W ? nc : W), pb1 = whilelt<sizeof(T)>(0, vc1);
            const T *b0 = rhs + (long)n0, *b1 = vc1 ? b0 + W : b0;
            quad_init(e, n0, n, nc, mr);
            quad_accumulate_rowb<A, 2>(k, a_lo, a_hi, b0, b1, rhs_rs, pb0, pb1, mr > W, nc > W);
            quad_store(dst, dst_cs, dst_rs, m0, n0, mr, nc, e, va, vb);
        }
    }
}

// One row-major batch item (C = nodes(A@B), overwrite), already in streaming
// mode. B is read straight from the row-major input, and each W-row A band is
// transposed through ZA tile 0 into `a_scr` ([2][k][W]) -- a load into a tile
// row comes back out as a column.
template <class T>
static void batch_one(T *dst, const T *a, const T *b, T *a_scr, size_t m, size_t n, size_t k,
                      size_t m_tiles, size_t n_tiles, const EpNode *nodes,
                      uint32_t n_nodes) __arm_streaming __arm_inout("za") {
    using A = Mopa<T, T>;
    constexpr size_t W = A::kTile;
    svbool_t p = ptrue<sizeof(T)>();
    const Vec<T> va = Sve<T>::zero(), vb = Sve<T>::dup(1);
    FloatEpi<T> e = {};
    e.nodes = n_nodes ? nodes : NULL;
    e.n_nodes = n_nodes;
    e.pure = !n_nodes;
    e.beta = 1;
    size_t per_band = ep_cmul(k, W);
    const T *a_lo = a_scr, *a_hi = a_scr + per_band;
    for (size_t mt = 0; mt < m_tiles; mt++) {
        size_t m0 = mt * 2 * W, mr = m - m0 < 2 * W ? m - m0 : 2 * W;
        for (size_t d0 = 0; d0 < k; d0 += W) {
            size_t dc = k - d0 < W ? k - d0 : W;
            svbool_t pk = whilelt<sizeof(T)>(0, dc);
            for (size_t band = 0; band < 2; band++) {
                svzero_za();
                for (size_t r = band * W; r < mr && r < band * W + W; r++)
                    za_ld_row<0>((uint32_t)(r - band * W), pk, a + (m0 + r) * k + d0);
                for (size_t c = 0; c < dc; c++)
                    za_st_col<0>((uint32_t)c, p, a_scr + band * per_band + (d0 + c) * W);
            }
        }
        for (size_t nt = 0; nt < n_tiles; nt++) {
            size_t n0 = nt * 2 * W, nc = n - n0 < 2 * W ? n - n0 : 2 * W;
            svbool_t pbl = whilelt<sizeof(T)>(n0, n);
            svbool_t pbh = nc > W ? whilelt<sizeof(T)>(n0 + W, n) : svpfalse_b();
            const T *b_lo = b + n0, *b_hi = nc > W ? b + n0 + W : b;
            svzero_za();
            quad_accumulate_rowb<A, 2>(k, a_lo, a_hi, b_lo, b_hi, (long)n, pbl, pbh, mr > W,
                                       nc > W);
            quad_store<true>(dst, 1, (long)n, m0, n0, mr, nc, e, va, vb);
        }
    }
}

template <class T>
__arm_locally_streaming __arm_new("za") static void run_batched(size_t count, T *dst, const T *lhs,
                                                                const T *rhs, T *a_scr, size_t m,
                                                                size_t n, size_t k, size_t m_tiles,
                                                                size_t n_tiles, const EpNode *nodes,
                                                                uint32_t n_nodes) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(dst + bi * m * n, lhs + bi * m * k, rhs + bi * k * n, a_scr, m, n, k, m_tiles,
                  n_tiles, nodes, n_nodes);
}

// `count` independent row-major C_i = nodes(A_i @ B_i) of one shape in ONE
// streaming session, so the entry and exit are paid once for the whole batch.
// lhs/rhs/dst are contiguous batches (item strides m*k / k*n / m*n); the same
// epilogue (NULL or no nodes for none) applies to every item.
template <class T, class Ep>
static int run_batched_ep(size_t count, size_t m, size_t n, size_t k, T *dst, const T *lhs,
                          const T *rhs, const Ep *ep) {
    constexpr size_t W = 64 / sizeof(T);
    if (count == 0 || m == 0 || n == 0) return 0;
    uint32_t n_nodes = ep ? ep->n_nodes : 0;
    size_t m_tiles = (m + 2 * W - 1) / (2 * W), n_tiles = (n + 2 * W - 1) / (2 * W);
    // Two A bands for one M-tile, rounded up to whole transpose blocks.
    T *a_scr = apack_scratch<T>(ep_cmul(ep_cmul((k + W - 1) & ~(W - 1), 2 * W), sizeof(T)));
    if (!a_scr) return -1;
    run_batched(count, dst, lhs, rhs, a_scr, m, n, k, m_tiles, n_tiles, n_nodes ? ep->nodes : NULL,
                n_nodes);
    return 0;
}

#endif // SME_GEMM_FLOAT_H
