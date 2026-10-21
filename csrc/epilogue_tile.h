// The NODE-MAJOR tile epilogue, one template over the stored element type and
// the slice orientation: rows of row-major C (vector spans N) or columns of
// col-major C (vector spans M). Included by epilogue.h after the per-dtype
// evaluators, whose activations it builds on.
//
// Interpreting the op-graph per output slice pays a full dispatch on EVERY row
// or column -- for a wide output that is n_nodes*n_slices interpreted dispatches
// plus an out-of-line call per slice (the activation rationals bloat the body
// past the inliner), and the call, made with ZA live, sets up a lazy save each
// time. The op-graph is built once per GEMM and is tiny, so that dispatch is
// loop-invariant across the slices of a tile.
//
// ep_block applies the whole op-graph to a BLOCK of `nr` (<= 4) slices held in
// Z registers, node-major: each node dispatches ONCE for the block. The operand
// that runs along a slice is one vector per node (COL for rows, ROW for
// columns), the one fixed per slice a splat (ROW for rows, COL for columns),
// and scalars and activation constants load once per node. The slices never
// touch memory between the ZA read and the final store, so the only cost beyond
// the bare store is the op math. Slice s of the block is tile slice s0 + s; the
// tile starts at output row m0, column n0.
//
// A slice computes as one vector (f16) or as two halves: f32 and f64, whose
// slices span two ZA tiles, and bf16, which widens its 32 lanes into two f32
// halves, runs the graph in f32 and rounds once at the store. pm is the
// element-size predicate the kernel holds; pl / ph are the predicates of the lo
// / hi half along the slice (for a one-vector slice, both are its predicate),
// and the hi half sits hoff elements past the lo one.
#ifndef SME_GEMM_EPILOGUE_TILE_H
#define SME_GEMM_EPILOGUE_TILE_H

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

// An op-graph with no TENSOR node: the only kind a column store can run (a
// TENSOR operand read down a column is a gather, which streaming SVE lacks
// here). Only `of` makes one, so a column store cannot be handed a TENSOR.
class EpColGraph {
  public:
    class Maybe;
    static SME_INLINE Maybe of(const EpNode *nodes, uint32_t n_nodes) __arm_streaming_compatible;
    SME_INLINE const EpNode *nodes() const __arm_streaming_compatible { return nodes_; }
    SME_INLINE uint32_t size() const __arm_streaming_compatible { return n_nodes_; }

  private:
    SME_INLINE EpColGraph(const EpNode *nodes, uint32_t n_nodes) __arm_streaming_compatible
        : nodes_(nodes),
          n_nodes_(n_nodes) {}
    const EpNode *nodes_;
    uint32_t n_nodes_;
};

// The graph, or none when it holds a TENSOR node. Not std::optional: its
// members are not streaming-compatible, so building one in a streaming body
// is a call that switches mode.
class EpColGraph::Maybe {
  public:
    SME_INLINE Maybe() __arm_streaming_compatible : g_(nullptr, 0), ok_(false) {}
    SME_INLINE explicit operator bool() const __arm_streaming_compatible { return ok_; }
    SME_INLINE EpColGraph operator*() const __arm_streaming_compatible { return g_; }

  private:
    friend class EpColGraph;
    SME_INLINE explicit Maybe(EpColGraph g) __arm_streaming_compatible : g_(g), ok_(true) {}
    EpColGraph g_;
    bool ok_;
};

SME_INLINE EpColGraph::Maybe EpColGraph::of(const EpNode *nodes,
                                            uint32_t n_nodes) __arm_streaming_compatible {
    for (uint32_t t = 0; t < n_nodes; t++)
        if (EP_OP_IS_TENSOR(nodes[t].op)) return Maybe();
    return Maybe(EpColGraph(nodes, n_nodes));
}

// RELU and NONE in line; the rational kinds call the dtype's out-of-line
// evaluator. Calls with ZA live pay a lazy save each, so the common kinds
// must not be calls.
template <class C, class W, class R>
SME_INLINE Vec<C> ep_act(svbool_t p, Vec<C> x, int kind, W alpha, R rational) __arm_streaming {
    if (kind == EP_ACT_RELU) return svmaxnm_x(p, x, (C)0);
    if (kind == EP_ACT_NONE) return x;
    return rational(p, x, kind, alpha);
}

// What differs per stored element type. C computes, In is one slice as the
// kernel reads it, and W carries the node scalars.
template <class E> struct EpRow;
template <> struct EpRow<ep_f16> {
    using C = ep_f16;
    using W = float;
    using In = svfloat16_t;
    static constexpr int kHalves = 1, kHi = 0;
    static SME_INLINE svbool_t pred(svbool_t pm) __arm_streaming { return pm; }
    static SME_INLINE svfloat16_t half(In x, int) __arm_streaming { return x; }
    static SME_INLINE svfloat16_t ld(svbool_t pl, svbool_t, const ep_f16 *q, int,
                                     size_t) __arm_streaming {
        return svld1(pl, q);
    }
    // *q in every lane: the operand fixed per slice.
    static SME_INLINE svfloat16_t splat(const ep_f16 *q) __arm_streaming { return svdup_n_f16(*q); }
    static SME_INLINE svfloat16_t act(svbool_t p, svfloat16_t x, int kind,
                                      float alpha) __arm_streaming {
        return ep_act<ep_f16>(p, x, kind, alpha, ep_act_rational_f16);
    }
};
// bf16 accumulates in bf16 but runs the graph in f32, operands widened too:
// B16B16 has no divide and streaming bf16 arithmetic is unreliable here
// (svsub/svmin operand order), so f32 gives it the same op set as the others.
template <> struct EpRow<ep_bf16> {
    using C = float;
    using W = float;
    using In = svbfloat16_t;
    static constexpr int kHalves = 2, kHi = 0;
    // f32 math is unpredicated: inactive lanes hold garbage but are never stored.
    static SME_INLINE svbool_t pred(svbool_t) __arm_streaming { return svptrue_b32(); }
    // bf16 is the top half of an f32: zip a zero below each lane, lanes
    // 0..15 for h = 0 and 16..31 for h = 1.
    static SME_INLINE svfloat32_t half(In x, int h) __arm_streaming {
        svuint16_t z = svdup_n_u16(0), u = svreinterpret_u16(x);
        return svreinterpret_f32(h ? svzip2(z, u) : svzip1(z, u));
    }
    static SME_INLINE svfloat32_t ld(svbool_t pl, svbool_t, const ep_bf16 *q, int h,
                                     size_t) __arm_streaming {
        return half(svld1(pl, q), h);
    }
    // Broadcast as bf16 and widen in vector registers: widening the scalar
    // first takes a core register and a move into the streaming unit per slice.
    static SME_INLINE svfloat32_t splat(const ep_bf16 *q) __arm_streaming {
        uint16_t b; // its bits: a 16-bit integer broadcast load is one LD1RH
        __builtin_memcpy(&b, q, sizeof b);
        return half(svreinterpret_bf16(svdup_n_u16(b)), 0);
    }
    static SME_INLINE svfloat32_t act(svbool_t p, svfloat32_t x, int kind,
                                      float alpha) __arm_streaming {
        return ep_act<float>(p, x, kind, alpha, ep_act_rational_f32);
    }
    // Round two halves back to one slice (nearest-even): the conversion
    // fills the even lanes, and uzp1 inverts the split.
    static SME_INLINE svbfloat16_t pack(svbool_t p, svfloat32_t lo,
                                        svfloat32_t hi) __arm_streaming {
        return svuzp1(svcvt_bf16_x(p, lo), svcvt_bf16_x(p, hi));
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
    static SME_INLINE Vec<T> ld(svbool_t pl, svbool_t ph, const T *q, int h,
                                size_t hoff) __arm_streaming {
        return h ? svld1(ph, q + hoff) : svld1(pl, q);
    }
    static SME_INLINE Vec<T> splat(const T *q) __arm_streaming { return Sve<T>::dup(*q); }
    static SME_INLINE Vec<T> act(svbool_t p, Vec<T> x, int kind, T alpha) __arm_streaming {
        if constexpr (sizeof(T) == 4)
            return ep_act<T>(p, x, kind, alpha, ep_act_rational_f32);
        else
            return ep_act<T>(p, x, kind, alpha, ep_act_rational_f64);
    }
};
template <> struct EpRow<float> : EpRowWide<float, 16> {};
template <> struct EpRow<double> : EpRowWide<double, 8> {};

template <class E, bool Cols, bool Scale>
SME_INLINE void ep_block(svbool_t pm, svbool_t pl, svbool_t ph, E *dst, long stride, size_t hoff,
                         typename EpRow<E>::In x0, typename EpRow<E>::In x1,
                         typename EpRow<E>::In x2, typename EpRow<E>::In x3, size_t nr, Vec<E> vb,
                         Vec<E> va, int read_dst, const EpNode *nodes, uint32_t n_nodes, size_t m0,
                         size_t s0, size_t n0) __arm_streaming {
    using G = EpRow<E>;
    using C = typename G::C;
    using W = typename G::W;
    using V = Vec<C>;
    constexpr bool two = G::kHalves == 2, widen = !std::is_same_v<C, E>;
    svbool_t p = G::pred(pm);
    V l0 = G::half(x0, 0), l1 = G::half(x1, 0), l2 = G::half(x2, 0), l3 = G::half(x3, 0);
    V h0 = G::half(x0, 1), h1 = G::half(x1, 1), h2 = G::half(x2, 1), h3 = G::half(x3, 1);
    // f(x, r, h) on half h of every live slice r.
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
// The operand along the slices: one vector per node, from q + at.
#define EP_ALONG(f, at)                                                                            \
    {                                                                                              \
        V cl = G::ld(pl, ph, q + (at), 0, hoff), ch = two ? G::ld(pl, ph, q + (at), 1, hoff) : cl; \
        EP_EACH(f(p, x, h ? ch : cl));                                                             \
    }
    if constexpr (Scale && widen) {
        V bl = G::half(vb, 0), bh = G::half(vb, 1);
        EP_EACH(svmul_x(p, x, h ? bh : bl));
    } else if constexpr (Scale) {
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
                if constexpr (Cols)
                    EP_ALONG(svadd_x, m0)
                else
                    EP_EACH(svadd_x(p, x, G::splat(q + (m0 + s0 + r))));
                break;
            case EP_OP_MUL_ROW:
                if constexpr (Cols)
                    EP_ALONG(svmul_x, m0)
                else
                    EP_EACH(svmul_x(p, x, G::splat(q + (m0 + s0 + r))));
                break;
            case EP_OP_ADD_COL:
                if constexpr (Cols)
                    EP_EACH(svadd_x(p, x, G::splat(q + (n0 + s0 + r))));
                else
                    EP_ALONG(svadd_x, n0)
                break;
            case EP_OP_MUL_COL:
                if constexpr (Cols)
                    EP_EACH(svmul_x(p, x, G::splat(q + (n0 + s0 + r))));
                else
                    EP_ALONG(svmul_x, n0)
                break;
            case EP_OP_ADD_TENSOR:
                if constexpr (Cols)
                    __builtin_unreachable(); // EpColGraph has none
                else
                    EP_EACH(svadd_x(p, x, G::ld(pl, ph, q + (m0 + s0 + r) * nd->ld + n0, h, hoff)));
                break;
            case EP_OP_MUL_TENSOR:
                if constexpr (Cols)
                    __builtin_unreachable();
                else
                    EP_EACH(svmul_x(p, x, G::ld(pl, ph, q + (m0 + s0 + r) * nd->ld + n0, h, hoff)));
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
                if constexpr (Cols)
                    EP_ALONG(svsub_x, m0)
                else
                    EP_EACH(svsub_x(p, x, G::splat(q + (m0 + s0 + r))));
                break;
            case EP_OP_DIV_ROW:
                if constexpr (Cols)
                    EP_ALONG(svdiv_x, m0)
                else
                    EP_EACH(ep_div_splat<C>(p, x, (W)q[m0 + s0 + r]));
                break;
            case EP_OP_MAX_ROW:
                if constexpr (Cols)
                    EP_ALONG(svmaxnm_x, m0)
                else
                    EP_EACH(svmaxnm_x(p, x, G::splat(q + (m0 + s0 + r))));
                break;
            case EP_OP_MIN_ROW:
                if constexpr (Cols)
                    EP_ALONG(svminnm_x, m0)
                else
                    EP_EACH(svminnm_x(p, x, G::splat(q + (m0 + s0 + r))));
                break;
            case EP_OP_SUB_COL:
                if constexpr (Cols)
                    EP_EACH(svsub_x(p, x, G::splat(q + (n0 + s0 + r))));
                else
                    EP_ALONG(svsub_x, n0)
                break;
            case EP_OP_DIV_COL:
                if constexpr (Cols)
                    EP_EACH(ep_div_splat<C>(p, x, (W)q[n0 + s0 + r]));
                else
                    EP_ALONG(svdiv_x, n0)
                break;
            case EP_OP_MAX_COL:
                if constexpr (Cols)
                    EP_EACH(svmaxnm_x(p, x, G::splat(q + (n0 + s0 + r))));
                else
                    EP_ALONG(svmaxnm_x, n0)
                break;
            case EP_OP_MIN_COL:
                if constexpr (Cols)
                    EP_EACH(svminnm_x(p, x, G::splat(q + (n0 + s0 + r))));
                else
                    EP_ALONG(svminnm_x, n0)
                break;
            case EP_OP_SUB_TENSOR:
                if constexpr (Cols)
                    __builtin_unreachable();
                else
                    EP_EACH(svsub_x(p, x, G::ld(pl, ph, q + (m0 + s0 + r) * nd->ld + n0, h, hoff)));
                break;
            case EP_OP_DIV_TENSOR:
                if constexpr (Cols)
                    __builtin_unreachable();
                else
                    EP_EACH(svdiv_x(p, x, G::ld(pl, ph, q + (m0 + s0 + r) * nd->ld + n0, h, hoff)));
                break;
            case EP_OP_MAX_TENSOR:
                if constexpr (Cols)
                    __builtin_unreachable();
                else
                    EP_EACH(
                        svmaxnm_x(p, x, G::ld(pl, ph, q + (m0 + s0 + r) * nd->ld + n0, h, hoff)));
                break;
            case EP_OP_MIN_TENSOR:
                if constexpr (Cols)
                    __builtin_unreachable();
                else
                    EP_EACH(
                        svminnm_x(p, x, G::ld(pl, ph, q + (m0 + s0 + r) * nd->ld + n0, h, hoff)));
                break;
            default:
                break;
        }
    }
    if constexpr (widen) {
        // Round each slice back to E once, then the optional alpha*C and the
        // store, both in E.
        Vec<E> y0 = G::pack(p, l0, h0), y1 = G::pack(p, l1, h1);
        Vec<E> y2 = G::pack(p, l2, h2), y3 = G::pack(p, l3, h3);
        auto d = [&] [[gnu::always_inline]] (size_t r) __arm_streaming {
            return dst + (long)(s0 + r) * stride;
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
            EP_EACH(
                svmla_x(p, x, svld1(h ? ph : pl, dst + (long)(s0 + r) * stride + h * hoff), va));
        auto st = [&] [[gnu::always_inline]] (size_t r, V l, V hh) __arm_streaming {
            svst1(pl, dst + (long)(s0 + r) * stride, l);
            if constexpr (two) svst1(ph, dst + (long)(s0 + r) * stride + hoff, hh);
        };
        if (nr > 0) st(0, l0, h0);
        if (nr > 1) st(1, l1, h1);
        if (nr > 2) st(2, l2, h2);
        if (nr > 3) st(3, l3, h3);
    }
#undef EP_ALONG
#undef EP_EACH
}

// Slices [lo, hi) of a tile at dst (slice s at dst + s*stride), slice s read by
// rd(s - lo): a ZA slice (the tile number is the reader's template argument),
// or one already in registers. Scale applies beta (vb) first; a caller that
// knows beta is 1 drops it, since every vector op on a slice costs and a runtime
// test of it perturbs the store's code.
template <class E, bool Cols, bool Scale, class RD>
SME_INLINE void ep_store_slices(svbool_t pm, svbool_t pl, svbool_t ph, E *dst, long stride,
                                size_t hoff, RD &&rd, size_t lo, size_t hi, Vec<E> vb, Vec<E> va,
                                int read_dst, const EpNode *nodes, uint32_t n_nodes, size_t m0,
                                size_t n0) __arm_streaming __arm_in("za") {
    size_t n = hi > lo ? hi - lo : 0, s = 0;
    for (; s + 4 <= n; s += 4)
        ep_block<E, Cols, Scale>(pm, pl, ph, dst, stride, hoff, rd((uint32_t)(s + 0)),
                                 rd((uint32_t)(s + 1)), rd((uint32_t)(s + 2)),
                                 rd((uint32_t)(s + 3)), 4, vb, va, read_dst, nodes, n_nodes, m0,
                                 lo + s, n0);
    if (s < n) {
        size_t nr = n - s;
        // Implied by the loop exit, but stated, the compiler drops the tail
        // block's fourth slice (about 500 instructions of run_streaming).
        __builtin_assume(nr < 4);
        typename EpRow<E>::In x0 = rd((uint32_t)(s + 0));
        typename EpRow<E>::In x1 = nr > 1 ? rd((uint32_t)(s + 1)) : x0;
        typename EpRow<E>::In x2 = nr > 2 ? rd((uint32_t)(s + 2)) : x0;
        ep_block<E, Cols, Scale>(pm, pl, ph, dst, stride, hoff, x0, x1, x2, x0, nr, vb, va,
                                 read_dst, nodes, n_nodes, m0, lo + s, n0);
    }
}

// Tile rows [lo, hi) of row-major C at dst (tile row r at dst + r*row_stride);
// pl / ph are the N predicates of the row's halves. i_base is the tile's first
// output row.
template <bool Scale = true, class E, class RD>
SME_INLINE void ep_store_rowmajor(svbool_t pm, svbool_t pl, svbool_t ph, E *dst, long row_stride,
                                  RD &&rd, size_t lo, size_t hi, Vec<E> vb, Vec<E> va, int read_dst,
                                  const EpNode *nodes, uint32_t n_nodes, size_t i_base,
                                  size_t n0) __arm_streaming __arm_in("za") {
    ep_store_slices<E, false, Scale>(pm, pl, ph, dst, row_stride, EpRow<E>::kHi, rd, lo, hi, vb, va,
                                     read_dst, nodes, n_nodes, i_base, n0);
}

// Tile columns [lo, hi) of col-major C at dst (tile column c at dst +
// c*col_stride); pl / ph are the M predicates of the column's halves and hoff
// the hi half's offset (0 when that half is empty, so no past-the-end pointer
// is formed). The tile starts at output row m0, column n0.
template <bool Scale = true, class E, class RD>
SME_INLINE void ep_store_colmajor(svbool_t pm, svbool_t pl, svbool_t ph, E *dst, long col_stride,
                                  size_t hoff, RD &&rd, size_t lo, size_t hi, Vec<E> vb, Vec<E> va,
                                  int read_dst, EpColGraph g, size_t m0,
                                  size_t n0) __arm_streaming __arm_in("za") {
    ep_store_slices<E, true, Scale>(pm, pl, ph, dst, col_stride, hoff, rd, lo, hi, vb, va, read_dst,
                                    g.nodes(), g.size(), m0, n0);
}

// One row already in registers (a GEMV output row).
template <bool Scale = true, class E>
SME_INLINE void ep_store_row(svbool_t pm, svbool_t pl, svbool_t ph, E *dst, typename EpRow<E>::In x,
                             Vec<E> vb, Vec<E> va, int read_dst, const EpNode *nodes,
                             uint32_t n_nodes, size_t i_base, size_t n0) __arm_streaming {
    ep_block<E, false, Scale>(pm, pl, ph, dst, 0, EpRow<E>::kHi, x, x, x, x, 1, vb, va, read_dst,
                              nodes, n_nodes, i_base, 0, n0);
}

#endif // SME_GEMM_EPILOGUE_TILE_H
