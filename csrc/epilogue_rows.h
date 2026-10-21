// The NODE-MAJOR row-major tile epilogue (vector spans N), one template over the
// stored element type. Included by epilogue.h after the per-dtype evaluators,
// whose activations and bf16 widen/narrow it builds on.
//
// The per-slice interpreters (ep_apply_nodes_*) pay a full op-graph dispatch on
// EVERY output row -- for a wide output that is n_nodes*n_rows interpreted
// dispatches plus an out-of-line call per row (the activation rationals bloat
// the body past the inliner). The op-graph is built once per GEMM and is tiny,
// so that dispatch is loop-invariant across the rows of a tile.
//
// ep_block_rowmajor applies the whole op-graph to a BLOCK of `nr` (<= 4) rows
// held in Z registers, node-major: each node dispatches ONCE for the block.
// Invariant operands (COL N-vector, scalars, activation constants) load once per
// node; only the per-row operands (ROW splat, TENSOR row-vector) index by row.
// The rows never touch memory between the ZA read and the final store, so the
// only cost beyond the bare store is the op math. `r0` is the block's first row
// within the tile, `i_base` the tile's first output row (m0).
//
// A row computes as one vector (f16) or as two halves: f32 and f64, whose rows
// span two ZA tiles, and bf16, which widens its 32 lanes into two f32 halves,
// runs the graph in f32 and rounds once at the store. pm is the element-size
// predicate the kernel holds; pl / ph are the N predicates of the lo / hi half
// (for a one-vector row, both are its N predicate).
#ifndef SME_GEMM_EPILOGUE_ROWS_H
#define SME_GEMM_EPILOGUE_ROWS_H

#include <type_traits>

#include "sme_tile.h"

// x / d for a splat divisor: a multiply by the reciprocal when it is finite,
// else the true divide (d is 0, or so small that 1/d overflows C). W is the
// scalar type the reciprocal is formed in.
template <class C, class W>
SME_INLINE Vec<C> ep_div_splat(svbool_t p, Vec<C> x, W d) __arm_streaming {
    C r = (C)((W)1 / d);
    return __builtin_isfinite((W)r) ? svmul_x(p, x, r) : svdiv_x(p, x, Sve<C>::dup((C)d));
}

// What differs per stored element type. C computes, In is one row as the
// kernel reads it, and W carries the node scalars.
template <class E> struct EpRow;
template <> struct EpRow<ep_f16> {
    using C = ep_f16;
    using W = float;
    using In = svfloat16_t;
    static constexpr int kHalves = 1, kHi = 0;
    static SME_INLINE svbool_t pred(svbool_t pm) __arm_streaming { return pm; }
    static SME_INLINE svfloat16_t half(In x, int) __arm_streaming { return x; }
    static SME_INLINE svfloat16_t ld(svbool_t pl, svbool_t, const ep_f16 *q, int) __arm_streaming {
        return svld1(pl, q);
    }
    static SME_INLINE svfloat16_t act(svbool_t p, svfloat16_t x, int kind,
                                      float alpha) __arm_streaming {
        return ep_act_apply(p, x, kind, alpha);
    }
};
template <> struct EpRow<ep_bf16> {
    using C = float;
    using W = float;
    using In = svbfloat16_t;
    static constexpr int kHalves = 2, kHi = 0;
    // f32 math is unpredicated: inactive lanes hold garbage but are never stored.
    static SME_INLINE svbool_t pred(svbool_t) __arm_streaming { return svptrue_b32(); }
    static SME_INLINE svfloat32_t half(In x, int h) __arm_streaming {
        return h ? ep_bf16_hi(x) : ep_bf16_lo(x);
    }
    static SME_INLINE svfloat32_t ld(svbool_t pl, svbool_t, const ep_bf16 *q,
                                     int h) __arm_streaming {
        return half(svld1(pl, q), h);
    }
    static SME_INLINE svfloat32_t act(svbool_t p, svfloat32_t x, int kind,
                                      float alpha) __arm_streaming {
        return ep_act_apply_f32(p, x, kind, alpha);
    }
};
template <class T, int Hi> struct EpRowWide {
    using C = T;
    using W = T;
    using In = std::conditional_t<sizeof(T) == 4, svfloat32x2_t, svfloat64x2_t>;
    static constexpr int kHalves = 2, kHi = Hi;
    static SME_INLINE svbool_t pred(svbool_t pm) __arm_streaming { return pm; }
    static SME_INLINE Vec<T> half(In x, int h) __arm_streaming {
        return h ? svget2(x, 1) : svget2(x, 0);
    }
    static SME_INLINE Vec<T> ld(svbool_t pl, svbool_t ph, const T *q, int h) __arm_streaming {
        return h ? svld1(ph, q + Hi) : svld1(pl, q);
    }
    static SME_INLINE Vec<T> act(svbool_t p, Vec<T> x, int kind, T alpha) __arm_streaming {
        if constexpr (sizeof(T) == 4)
            return ep_act_apply_f32(p, x, kind, alpha);
        else
            return ep_act_apply_f64(p, x, kind, alpha);
    }
};
template <> struct EpRow<float> : EpRowWide<float, 16> {};
template <> struct EpRow<double> : EpRowWide<double, 8> {};

template <class E>
SME_INLINE void ep_block_rowmajor(svbool_t pm, svbool_t pl, svbool_t ph, E *dst, long row_stride,
                                  typename EpRow<E>::In x0, typename EpRow<E>::In x1,
                                  typename EpRow<E>::In x2, typename EpRow<E>::In x3, size_t nr,
                                  Vec<E> vb, Vec<E> va, int read_dst, const EpNode *nodes,
                                  uint32_t n_nodes, size_t i_base, size_t r0,
                                  size_t n0) __arm_streaming {
    using G = EpRow<E>;
    using C = typename G::C;
    using W = typename G::W;
    using V = Vec<C>;
    constexpr bool two = G::kHalves == 2, widen = !std::is_same_v<C, E>;
    svbool_t p = G::pred(pm);
    V l0 = G::half(x0, 0), l1 = G::half(x1, 0), l2 = G::half(x2, 0), l3 = G::half(x3, 0);
    V h0 = G::half(x0, 1), h1 = G::half(x1, 1), h2 = G::half(x2, 1), h3 = G::half(x3, 1);
    // f(x, r, h) on half h of every live row r.
    auto each = [&] [[gnu::always_inline]] (auto f) __arm_streaming {
        if (nr > 0) {
            l0 = f(l0, 0, 0);
            if constexpr (two) h0 = f(h0, 0, 1);
        }
        if (nr > 1) {
            l1 = f(l1, 1, 0);
            if constexpr (two) h1 = f(h1, 1, 1);
        }
        if (nr > 2) {
            l2 = f(l2, 2, 0);
            if constexpr (two) h2 = f(h2, 2, 1);
        }
        if (nr > 3) {
            l3 = f(l3, 3, 0);
            if constexpr (two) h3 = f(h3, 3, 1);
        }
    };
#define EP_EACH(expr)                                                                              \
    each([&] [[gnu::always_inline]] (V x, size_t r, int h) __arm_streaming {                       \
        (void)r, (void)h;                                                                          \
        return (expr);                                                                             \
    })
    if constexpr (widen) {
        V bl = G::half(vb, 0), bh = G::half(vb, 1);
        EP_EACH(svmul_x(p, x, h ? bh : bl));
    } else {
        EP_EACH(svmul_x(p, x, vb));
    }
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        const E *q = (const E *)nd->ptr;
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: {
                C s = (C)nd->scalar;
                EP_EACH(svadd_x(p, x, s));
                break;
            }
            case EP_OP_MUL_SCALAR: {
                C s = (C)nd->scalar;
                EP_EACH(svmul_x(p, x, s));
                break;
            }
            case EP_OP_MAX_SCALAR: {
                C s = (C)nd->scalar;
                EP_EACH(svmaxnm_x(p, x, s));
                break;
            }
            case EP_OP_MIN_SCALAR: {
                C s = (C)nd->scalar;
                EP_EACH(svminnm_x(p, x, s));
                break;
            }
            case EP_OP_ADD_ROW:
                EP_EACH(svadd_x(p, x, (C)q[i_base + r0 + r]));
                break;
            case EP_OP_MUL_ROW:
                EP_EACH(svmul_x(p, x, (C)q[i_base + r0 + r]));
                break;
            case EP_OP_ADD_COL: {
                V cl = G::ld(pl, ph, q + n0, 0), ch = two ? G::ld(pl, ph, q + n0, 1) : cl;
                EP_EACH(svadd_x(p, x, h ? ch : cl));
                break;
            }
            case EP_OP_MUL_COL: {
                V cl = G::ld(pl, ph, q + n0, 0), ch = two ? G::ld(pl, ph, q + n0, 1) : cl;
                EP_EACH(svmul_x(p, x, h ? ch : cl));
                break;
            }
            case EP_OP_ADD_TENSOR:
                EP_EACH(svadd_x(p, x, G::ld(pl, ph, q + (i_base + r0 + r) * nd->ld + n0, h)));
                break;
            case EP_OP_MUL_TENSOR:
                EP_EACH(svmul_x(p, x, G::ld(pl, ph, q + (i_base + r0 + r) * nd->ld + n0, h)));
                break;
            case EP_OP_ACT: {
                int act = (int)nd->aux;
                W alpha = (W)nd->scalar;
                EP_EACH(G::act(p, x, act, alpha));
                break;
            }
            case EP_OP_SUB_SCALAR: {
                C s = (C)nd->scalar;
                EP_EACH(svsub_x(p, x, s));
                break;
            }
            case EP_OP_DIV_SCALAR: {
                W d = (W)nd->scalar;
                EP_EACH(ep_div_splat<C>(p, x, d));
                break;
            }
            case EP_OP_SUB_ROW:
                EP_EACH(svsub_x(p, x, (C)q[i_base + r0 + r]));
                break;
            case EP_OP_DIV_ROW:
                EP_EACH(ep_div_splat<C>(p, x, (W)q[i_base + r0 + r]));
                break;
            case EP_OP_MAX_ROW:
                EP_EACH(svmaxnm_x(p, x, (C)q[i_base + r0 + r]));
                break;
            case EP_OP_MIN_ROW:
                EP_EACH(svminnm_x(p, x, (C)q[i_base + r0 + r]));
                break;
            case EP_OP_SUB_COL: {
                V cl = G::ld(pl, ph, q + n0, 0), ch = two ? G::ld(pl, ph, q + n0, 1) : cl;
                EP_EACH(svsub_x(p, x, h ? ch : cl));
                break;
            }
            case EP_OP_DIV_COL: {
                V cl = G::ld(pl, ph, q + n0, 0), ch = two ? G::ld(pl, ph, q + n0, 1) : cl;
                EP_EACH(svdiv_x(p, x, h ? ch : cl));
                break;
            }
            case EP_OP_MAX_COL: {
                V cl = G::ld(pl, ph, q + n0, 0), ch = two ? G::ld(pl, ph, q + n0, 1) : cl;
                EP_EACH(svmaxnm_x(p, x, h ? ch : cl));
                break;
            }
            case EP_OP_MIN_COL: {
                V cl = G::ld(pl, ph, q + n0, 0), ch = two ? G::ld(pl, ph, q + n0, 1) : cl;
                EP_EACH(svminnm_x(p, x, h ? ch : cl));
                break;
            }
            case EP_OP_SUB_TENSOR:
                EP_EACH(svsub_x(p, x, G::ld(pl, ph, q + (i_base + r0 + r) * nd->ld + n0, h)));
                break;
            case EP_OP_DIV_TENSOR:
                EP_EACH(svdiv_x(p, x, G::ld(pl, ph, q + (i_base + r0 + r) * nd->ld + n0, h)));
                break;
            case EP_OP_MAX_TENSOR:
                EP_EACH(svmaxnm_x(p, x, G::ld(pl, ph, q + (i_base + r0 + r) * nd->ld + n0, h)));
                break;
            case EP_OP_MIN_TENSOR:
                EP_EACH(svminnm_x(p, x, G::ld(pl, ph, q + (i_base + r0 + r) * nd->ld + n0, h)));
                break;
            default:
                break;
        }
    }
    if constexpr (widen) {
        // Round each row back to E once, then the optional alpha*C and the store,
        // both in E.
        Vec<E> y0 = ep_bf16_pack(p, l0, h0), y1 = ep_bf16_pack(p, l1, h1);
        Vec<E> y2 = ep_bf16_pack(p, l2, h2), y3 = ep_bf16_pack(p, l3, h3);
        auto d = [&] [[gnu::always_inline]] (size_t r) __arm_streaming {
            return dst + (long)(r0 + r) * row_stride;
        };
        if (read_dst) {
            if (nr > 0) y0 = svmla_x(pm, y0, svld1(pl, d(0)), va);
            if (nr > 1) y1 = svmla_x(pm, y1, svld1(pl, d(1)), va);
            if (nr > 2) y2 = svmla_x(pm, y2, svld1(pl, d(2)), va);
            if (nr > 3) y3 = svmla_x(pm, y3, svld1(pl, d(3)), va);
        }
        if (nr > 0) svst1(pl, d(0), y0);
        if (nr > 1) svst1(pl, d(1), y1);
        if (nr > 2) svst1(pl, d(2), y2);
        if (nr > 3) svst1(pl, d(3), y3);
    } else {
        if (read_dst)
            EP_EACH(svmla_x(
                p, x, svld1(h ? ph : pl, dst + (long)(r0 + r) * row_stride + h * G::kHi), va));
        auto st = [&] [[gnu::always_inline]] (size_t r, V l, V hh) __arm_streaming {
            svst1(pl, dst + (long)(r0 + r) * row_stride, l);
            if constexpr (two) svst1(ph, dst + (long)(r0 + r) * row_stride + G::kHi, hh);
        };
        if (nr > 0) st(0, l0, h0);
        if (nr > 1) st(1, l1, h1);
        if (nr > 2) st(2, l2, h2);
        if (nr > 3) st(3, l3, h3);
    }
#undef EP_EACH
}

// Node-major store of tile rows [lo, hi) of row-major C at dst (tile row r at
// dst + r*row_stride), row r read by rd(r - lo) -- a ZA slice (the tile number
// is the reader's template argument), or a row already in registers.
template <class E, class RD>
SME_INLINE void ep_store_rowmajor(svbool_t pm, svbool_t pl, svbool_t ph, E *dst, long row_stride,
                                  RD &&rd, size_t lo, size_t hi, Vec<E> vb, Vec<E> va, int read_dst,
                                  const EpNode *nodes, uint32_t n_nodes, size_t i_base,
                                  size_t n0) __arm_streaming __arm_in("za") {
    size_t r = lo;
    for (; r + 4 <= hi; r += 4) {
        uint32_t s = (uint32_t)(r - lo);
        ep_block_rowmajor<E>(pm, pl, ph, dst, row_stride, rd(s), rd(s + 1), rd(s + 2), rd(s + 3), 4,
                             vb, va, read_dst, nodes, n_nodes, i_base, r, n0);
    }
    if (r < hi) {
        size_t nr = hi - r;
        uint32_t s = (uint32_t)(r - lo);
        typename EpRow<E>::In x0 = rd(s);
        typename EpRow<E>::In x1 = nr > 1 ? rd(s + 1) : x0;
        typename EpRow<E>::In x2 = nr > 2 ? rd(s + 2) : x0;
        ep_block_rowmajor<E>(pm, pl, ph, dst, row_stride, x0, x1, x2, x0, nr, vb, va, read_dst,
                             nodes, n_nodes, i_base, r, n0);
    }
}

// The same for one row already in registers (a GEMV output row).
template <class E>
SME_INLINE void ep_store_row(svbool_t pm, svbool_t pl, svbool_t ph, E *dst, typename EpRow<E>::In x,
                             Vec<E> vb, Vec<E> va, int read_dst, const EpNode *nodes,
                             uint32_t n_nodes, size_t i_base, size_t n0) __arm_streaming {
    ep_block_rowmajor<E>(pm, pl, ph, dst, 0, x, x, x, x, 1, vb, va, read_dst, nodes, n_nodes,
                         i_base, 0, n0);
}

#endif // SME_GEMM_EPILOGUE_ROWS_H
