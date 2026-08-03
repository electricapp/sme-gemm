// f16 in-register epilogue: vector op-graph, store, and node-major tile store.
// Included by epilogue.h, which defines the shared enums, EpNode/ep_desc
// structs, and helper macros these evaluators use.
#ifndef SME_GEMM_EPILOGUE_F16_H
#define SME_GEMM_EPILOGUE_F16_H

// --- f16 vectorized op-graph ------------------------------------------------

// Rational tanh on a live f16 slice: t*(27+t^2)/(27+9t^2), clamped. This same
// rational backs GELU, SiLU, and the bare tanh/sigmoid activations: end-to-end
// error ~4e-2 worst-case vs libm (measured) -- masked by the envelope for
// gelu/silu, but directly exposed for a bare .tanh()/.sigmoid() epilogue.
// Acceptable for fused f16 inference. The input clamp keeps t^2 from
// overflowing f16 and saturates |tanh|->1.

// ===========================================================================
// F16 SECTION: in-register f16 epilogue. ep_tanh_f16 / ep_act_apply /
// ep_apply_nodes_f16 / ep_store_f16, the node-major ep_block_rowmajor_f16 and
// its EP_STORE_TILE_ROWMAJOR_F16 macro. Exotic activations upcast to f32 and
// reuse the f32 transcendentals forward-declared above.
// ===========================================================================
static inline svfloat16_t ep_tanh_f16(svbool_t p16, svfloat16_t t) __arm_streaming {
    t = svmin_n_f16_x(p16, svmax_n_f16_x(p16, t, (ep_f16)-8.0f), (ep_f16)8.0f);
    svfloat16_t t2 = svmul_f16_x(p16, t, t);
    svfloat16_t num = svadd_n_f16_x(p16, t2, (ep_f16)27.0f);
    svfloat16_t den = svadd_n_f16_x(p16, svmul_n_f16_x(p16, t2, (ep_f16)9.0f), (ep_f16)27.0f);
    svfloat16_t th = svmul_f16_x(p16, t, svdiv_f16_x(p16, num, den));
    return svmin_n_f16_x(p16, svmax_n_f16_x(p16, th, (ep_f16)-1.0f), (ep_f16)1.0f);
}

// The rational GELU/SiLU/TANH/SIGMOID math (the bulk of the activation menu) is
// pulled OUT of line: it is what bloats the epilogue past the inliner, and
// keeping it here lets the common add/mul/relu node path inline cleanly into the
// node-major tile loops. RELU is handled at the (inlined) call site -- it is a
// single svmax and the dominant fused activation. Per node-major tile, the act
// dispatch is paid once per row-block, not once per row.
// Apply an f32 unary activation to an f16 vector by upcasting each half to f32,
// applying, and downcasting (svcvt). The f16 vector holds VL/16 lanes; the lo
// half is the even-indexed (bottom) f32 lanes, the hi half the top. Correctness
// over speed for the exotic activations (as specified). `ACTF32` is an
// expression in `lo`/`hi` (svfloat32_t) yielding the transformed half.
#define EP_F16_VIA_F32(ACTF32)                                                          \
    do {                                                                               \
        svbool_t p32 = svptrue_b32();                                                  \
        svfloat32_t lo = svcvt_f32_f16_x(p16, svzip1_f16(x, x));                       \
        svfloat32_t hi = svcvt_f32_f16_x(p16, svzip2_f16(x, x));                       \
        svfloat16_t rlo = svcvt_f16_f32_x(p32, (ACTF32(lo)));                          \
        svfloat16_t rhi = svcvt_f16_f32_x(p32, (ACTF32(hi)));                          \
        return svuzp1_f16(rlo, rhi);                                                   \
    } while (0)

__attribute__((noinline)) static svfloat16_t ep_act_rational_f16(svbool_t p16, svfloat16_t x,
                                                                 int act,
                                                                 float alpha) __arm_streaming {
    // Group A exact ops, computed directly in f16 (no transcendental, no upcast).
    if (act == EP_ACT_LEAKY_RELU) {
        svfloat16_t pos = svmax_n_f16_x(p16, x, (ep_f16)0.0f);
        svfloat16_t neg = svmin_n_f16_x(p16, x, (ep_f16)0.0f);
        return svmla_n_f16_x(p16, pos, neg, (ep_f16)alpha);
    }
    if (act == EP_ACT_RELU6) {
        return svmin_n_f16_x(p16, svmax_n_f16_x(p16, x, (ep_f16)0.0f), (ep_f16)6.0f);
    }
    if (act == EP_ACT_HARDSIGMOID) {
        svfloat16_t h = svadd_n_f16_x(p16, svmul_n_f16_x(p16, x, (ep_f16)(1.0f / 6.0f)), (ep_f16)0.5f);
        return svmin_n_f16_x(p16, svmax_n_f16_x(p16, h, (ep_f16)0.0f), (ep_f16)1.0f);
    }
    if (act == EP_ACT_HARDSWISH) {
        svfloat16_t h = svadd_n_f16_x(p16, svmul_n_f16_x(p16, x, (ep_f16)(1.0f / 6.0f)), (ep_f16)0.5f);
        h = svmin_n_f16_x(p16, svmax_n_f16_x(p16, h, (ep_f16)0.0f), (ep_f16)1.0f);
        return svmul_f16_x(p16, x, h);
    }
    if (act == EP_ACT_ABS) return svabs_f16_x(p16, x);
    if (act == EP_ACT_NEG) return svneg_f16_x(p16, x);
    if (act == EP_ACT_SQUARE) return svmul_f16_x(p16, x, x);
    if (act == EP_ACT_SIGN) {
        svfloat16_t pos =
            svsel_f16(svcmpgt_n_f16(p16, x, (ep_f16)0.0f), svdup_n_f16((ep_f16)1.0f),
                      svdup_n_f16((ep_f16)0.0f));
        return svsel_f16(svcmplt_n_f16(p16, x, (ep_f16)0.0f), svdup_n_f16((ep_f16)-1.0f), pos);
    }
    if (act == EP_ACT_SQRT) return svsqrt_f16_x(p16, x);
    if (act == EP_ACT_SOFTSIGN) {
        return svdiv_f16_x(p16, x, svadd_n_f16_x(p16, svabs_f16_x(p16, x), (ep_f16)1.0f));
    }
    if (act == EP_ACT_RECIP) {
        return svdiv_f16_x(p16, svdup_n_f16((ep_f16)1.0f), x);
    }
    if (act == EP_ACT_RSQRT) {
        return svdiv_f16_x(p16, svdup_n_f16((ep_f16)1.0f), svsqrt_f16_x(p16, x));
    }
    // Group B transcendentals: upcast each half to f32, apply, downcast.
    if (act == EP_ACT_EXP) {
#define EP_AF(v) ep_exp_f32(p32, v)
        EP_F16_VIA_F32(EP_AF);
#undef EP_AF
    }
    if (act == EP_ACT_LOG) {
#define EP_AF(v) ep_log_f32(p32, v)
        EP_F16_VIA_F32(EP_AF);
#undef EP_AF
    }
    if (act == EP_ACT_ELU) {
#define EP_AF(v) ep_act_rational_f32(p32, v, EP_ACT_ELU, alpha)
        EP_F16_VIA_F32(EP_AF);
#undef EP_AF
    }
    if (act == EP_ACT_SELU) {
#define EP_AF(v) ep_act_rational_f32(p32, v, EP_ACT_SELU, alpha)
        EP_F16_VIA_F32(EP_AF);
#undef EP_AF
    }
    if (act == EP_ACT_SOFTPLUS) {
#define EP_AF(v) ep_softplus_f32(p32, v)
        EP_F16_VIA_F32(EP_AF);
#undef EP_AF
    }
    if (act == EP_ACT_MISH) {
#define EP_AF(v) ep_act_rational_f32(p32, v, EP_ACT_MISH, alpha)
        EP_F16_VIA_F32(EP_AF);
#undef EP_AF
    }
    if (act == EP_ACT_GELU_EXACT) {
#define EP_AF(v) ep_act_rational_f32(p32, v, EP_ACT_GELU_EXACT, alpha)
        EP_F16_VIA_F32(EP_AF);
#undef EP_AF
    }
    if (act == EP_ACT_GELU) {
        // t = c0*(x + c1*x^3) = c0*x*(1 + c1*x^2)
        svfloat16_t x2 = svmul_f16_x(p16, x, x);
        svfloat16_t inner = svmul_f16_x(
            p16, x, svadd_n_f16_x(p16, svmul_n_f16_x(p16, x2, (ep_f16)0.044715f), (ep_f16)1.0f));
        svfloat16_t t = svmul_n_f16_x(p16, inner, (ep_f16)0.7978845608f);
        svfloat16_t th = ep_tanh_f16(p16, t);
        return svmul_n_f16_x(p16, svmul_f16_x(p16, x, svadd_n_f16_x(p16, th, (ep_f16)1.0f)),
                             (ep_f16)0.5f);
    }
    if (act == EP_ACT_SILU) {
        // silu(x) = x*sigmoid(x) = 0.5*x*(1 + tanh(x/2))
        svfloat16_t th = ep_tanh_f16(p16, svmul_n_f16_x(p16, x, (ep_f16)0.5f));
        return svmul_n_f16_x(p16, svmul_f16_x(p16, x, svadd_n_f16_x(p16, th, (ep_f16)1.0f)),
                             (ep_f16)0.5f);
    }
    if (act == EP_ACT_TANH) {
        return ep_tanh_f16(p16, x);
    }
    if (act == EP_ACT_SIGMOID) {
        // sigmoid(x) = 0.5*(1 + tanh(x/2))
        svfloat16_t th = ep_tanh_f16(p16, svmul_n_f16_x(p16, x, (ep_f16)0.5f));
        return svmul_n_f16_x(p16, svadd_n_f16_x(p16, th, (ep_f16)1.0f), (ep_f16)0.5f);
    }
    return x; // EP_ACT_NONE
}

// Apply the activation to a live f16 slice (streaming-SVE). RELU stays inline (one
// svmax); the rationals go out of line via ep_act_rational_f16.
static inline svfloat16_t ep_act_apply(svbool_t p16, svfloat16_t x, int act,
                                       float alpha) __arm_streaming {
    if (act == EP_ACT_RELU) {
        return svmaxnm_n_f16_x(p16, x, (ep_f16)0.0f);
    }
    if (act == EP_ACT_NONE) {
        return x;
    }
    return ep_act_rational_f16(p16, x, act, alpha);
}

// Walk the op-graph over a live f16 SVE vector `x`. `span_n` selects orientation:
//   span_n != 0 (row-major dst, vector spans N over lanes [n0, n0+VL)):
//       COL/TENSOR are vector loads (ptr+n0 / ptr+i*ld+n0); ROW/SCALAR are splats.
//   span_n == 0 (col-major dst, vector spans M over lanes [m0, m0+VL)):
//       ROW is the vector load (ptr+m0); COL/SCALAR are splats; TENSOR is column-
//       strided -- not vectorizable here, so the caller uses the scalar path and
//       this routine is not invoked for graphs containing a TENSOR node.
// `i` is the fixed row (when span_n) / column (when !span_n) index for splats.
static inline svfloat16_t ep_apply_nodes_f16(svbool_t p16, svbool_t pst, svfloat16_t x,
                                             const EpNode *nodes, uint32_t n_nodes, int span_n,
                                             size_t i, size_t n0, size_t m0) __arm_streaming {
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        const ep_f16 *p = (const ep_f16 *)nd->ptr;
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: x = svadd_n_f16_x(p16, x, (ep_f16)nd->scalar); break;
            case EP_OP_MUL_SCALAR: x = svmul_n_f16_x(p16, x, (ep_f16)nd->scalar); break;
            case EP_OP_ADD_ROW:
                if (span_n) x = svadd_n_f16_x(p16, x, p[i]);
                else x = svadd_f16_x(p16, x, svld1_f16(pst, p + m0));
                break;
            case EP_OP_MUL_ROW:
                if (span_n) x = svmul_n_f16_x(p16, x, p[i]);
                else x = svmul_f16_x(p16, x, svld1_f16(pst, p + m0));
                break;
            case EP_OP_ADD_COL:
                if (span_n) x = svadd_f16_x(p16, x, svld1_f16(pst, p + n0));
                else x = svadd_n_f16_x(p16, x, p[i]);
                break;
            case EP_OP_MUL_COL:
                if (span_n) x = svmul_f16_x(p16, x, svld1_f16(pst, p + n0));
                else x = svmul_n_f16_x(p16, x, p[i]);
                break;
            case EP_OP_ADD_TENSOR:
                // Only reached for span_n (row-major); the col-major caller routes
                // TENSOR graphs to the scalar path.
                x = svadd_f16_x(p16, x, svld1_f16(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_MUL_TENSOR:
                x = svmul_f16_x(p16, x, svld1_f16(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_ACT: x = ep_act_apply(p16, x, (int)nd->aux, nd->scalar); break;
            case EP_OP_MAX_SCALAR: x = svmaxnm_n_f16_x(p16, x, (ep_f16)nd->scalar); break;
            case EP_OP_MIN_SCALAR: x = svminnm_n_f16_x(p16, x, (ep_f16)nd->scalar); break;
            case EP_OP_SUB_SCALAR: x = svsub_n_f16_x(p16, x, (ep_f16)nd->scalar); break;
            case EP_OP_DIV_SCALAR: x = EP_DIV_SPLAT_F16(p16, x, nd->scalar); break;
            case EP_OP_SUB_ROW:
                if (span_n) x = svsub_n_f16_x(p16, x, p[i]);
                else x = svsub_f16_x(p16, x, svld1_f16(pst, p + m0));
                break;
            case EP_OP_DIV_ROW:
                if (span_n) x = EP_DIV_SPLAT_F16(p16, x, (float)p[i]);
                else x = svdiv_f16_x(p16, x, svld1_f16(pst, p + m0));
                break;
            case EP_OP_MAX_ROW:
                if (span_n) x = svmaxnm_n_f16_x(p16, x, p[i]);
                else x = svmaxnm_f16_x(p16, x, svld1_f16(pst, p + m0));
                break;
            case EP_OP_MIN_ROW:
                if (span_n) x = svminnm_n_f16_x(p16, x, p[i]);
                else x = svminnm_f16_x(p16, x, svld1_f16(pst, p + m0));
                break;
            case EP_OP_SUB_COL:
                if (span_n) x = svsub_f16_x(p16, x, svld1_f16(pst, p + n0));
                else x = svsub_n_f16_x(p16, x, p[i]);
                break;
            case EP_OP_DIV_COL:
                if (span_n) x = svdiv_f16_x(p16, x, svld1_f16(pst, p + n0));
                else x = EP_DIV_SPLAT_F16(p16, x, (float)p[i]);
                break;
            case EP_OP_MAX_COL:
                if (span_n) x = svmaxnm_f16_x(p16, x, svld1_f16(pst, p + n0));
                else x = svmaxnm_n_f16_x(p16, x, p[i]);
                break;
            case EP_OP_MIN_COL:
                if (span_n) x = svminnm_f16_x(p16, x, svld1_f16(pst, p + n0));
                else x = svminnm_n_f16_x(p16, x, p[i]);
                break;
            case EP_OP_SUB_TENSOR:
                x = svsub_f16_x(p16, x, svld1_f16(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_DIV_TENSOR:
                x = svdiv_f16_x(p16, x, svld1_f16(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_MAX_TENSOR:
                x = svmaxnm_f16_x(p16, x, svld1_f16(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_MIN_TENSOR:
                x = svminnm_f16_x(p16, x, svld1_f16(pst, p + i * nd->ld + n0));
                break;
            default: break;
        }
    }
    return x;
}

// True if the op-graph contains a TENSOR node (which the col-major store cannot
// vectorize and must route to the scalar path).
static inline int ep_has_tensor(const EpNode *nodes, uint32_t n_nodes) {
    for (uint32_t t = 0; t < n_nodes; t++)
        if (EP_OP_IS_TENSOR(nodes[t].op)) return 1;
    return 0;
}

// Combine + store one f16 slice with the fused op-graph epilogue:
//   x = beta*acc; x = nodes(x); [x = alpha*C + x]; store
static inline void ep_store_f16(svbool_t p16, svbool_t pst, ep_f16 *ptr, svfloat16_t acc,
                                svfloat16_t vb, svfloat16_t va, int read_dst, const EpNode *nodes,
                                uint32_t n_nodes, int span_n, size_t i, size_t n0,
                                size_t m0) __arm_streaming {
    svfloat16_t x = svmul_f16_x(p16, acc, vb);
    x = ep_apply_nodes_f16(p16, pst, x, nodes, n_nodes, span_n, i, n0, m0);
    if (read_dst) x = svmla_f16_x(p16, x, svld1_f16(pst, ptr), va);
    svst1_f16(pst, ptr, x);
}

// --- f16 NODE-MAJOR tile epilogue (row-major dst, vector spans N) -----------
//
// The per-row interpreter (ep_apply_nodes_f16) pays a full op-graph dispatch on
// EVERY output row -- for a wide output that is n_nodes*n_rows interpreted
// dispatches plus an out-of-line call per row (the activation rationals bloat
// the body past the inliner). The op-graph is built once per GEMM and is tiny,
// so that dispatch is loop-invariant across the rows of a tile.
//
// Apply the whole op-graph to a BLOCK of `nr` (<=4) live f16 rows held in Z
// registers, node-major: each node dispatches ONCE for the whole block. Invariant
// operands (COL N-vector, scalars, activation constants) are loaded once per node;
// only the genuinely per-row operands (ROW splat, TENSOR row-vector) index by row.
// The rows never touch memory between the ZA read and the final store, so the only
// cost beyond the bare store is the actual op math: no per-row out-of-line call,
// no interpreted jump-table dispatch. Dispatch is n_nodes per block of 4 rows
// (~n_nodes*mrows/4), against n_nodes*mrows for a per-row scheme. `r0` is the
// block's first row within the tile; `i_base` is the tile's first output row (m0).
#define EP_BLK_F16_OP(EXPR)                                              \
    do {                                                                 \
        if (nr > 0) x0 = (EXPR(x0, 0));                                  \
        if (nr > 1) x1 = (EXPR(x1, 1));                                  \
        if (nr > 2) x2 = (EXPR(x2, 2));                                  \
        if (nr > 3) x3 = (EXPR(x3, 3));                                  \
    } while (0)

__attribute__((always_inline)) static inline void
ep_block_rowmajor_f16(svbool_t p16, svbool_t pst, ep_f16 *dst, long row_stride, svfloat16_t x0,
                      svfloat16_t x1, svfloat16_t x2, svfloat16_t x3, size_t nr, svfloat16_t vb,
                      svfloat16_t va, int read_dst, const EpNode *nodes, uint32_t n_nodes,
                      size_t i_base, size_t r0, size_t n0) __arm_streaming {
#define EP_BETA(x, r) svmul_f16_x(p16, x, vb)
    EP_BLK_F16_OP(EP_BETA);
#undef EP_BETA
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        const ep_f16 *p = (const ep_f16 *)nd->ptr;
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: {
                ep_f16 s = (ep_f16)nd->scalar;
#define EP_OPX(x, r) svadd_n_f16_x(p16, x, s)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_MUL_SCALAR: {
                ep_f16 s = (ep_f16)nd->scalar;
#define EP_OPX(x, r) svmul_n_f16_x(p16, x, s)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_MAX_SCALAR: {
                ep_f16 s = (ep_f16)nd->scalar;
#define EP_OPX(x, r) svmaxnm_n_f16_x(p16, x, s)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_MIN_SCALAR: {
                ep_f16 s = (ep_f16)nd->scalar;
#define EP_OPX(x, r) svminnm_n_f16_x(p16, x, s)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_ADD_ROW:
#define EP_OPX(x, r) svadd_n_f16_x(p16, x, p[i_base + r0 + (r)])
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_MUL_ROW:
#define EP_OPX(x, r) svmul_n_f16_x(p16, x, p[i_base + r0 + (r)])
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_ADD_COL: { // invariant N-vector: load ONCE per block
                svfloat16_t col = svld1_f16(pst, p + n0);
#define EP_OPX(x, r) svadd_f16_x(p16, x, col)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_MUL_COL: {
                svfloat16_t col = svld1_f16(pst, p + n0);
#define EP_OPX(x, r) svmul_f16_x(p16, x, col)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_ADD_TENSOR:
#define EP_OPX(x, r) svadd_f16_x(p16, x, svld1_f16(pst, p + (i_base + r0 + (r)) * nd->ld + n0))
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_MUL_TENSOR:
#define EP_OPX(x, r) svmul_f16_x(p16, x, svld1_f16(pst, p + (i_base + r0 + (r)) * nd->ld + n0))
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_ACT: {
                int act = (int)nd->aux;
                float alpha = nd->scalar;
#define EP_OPX(x, r) ep_act_apply(p16, x, act, alpha)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_SUB_SCALAR: {
                ep_f16 s = (ep_f16)nd->scalar;
#define EP_OPX(x, r) svsub_n_f16_x(p16, x, s)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_DIV_SCALAR: {
                float s = nd->scalar;
#define EP_OPX(x, r) EP_DIV_SPLAT_F16(p16, x, s)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_SUB_ROW:
#define EP_OPX(x, r) svsub_n_f16_x(p16, x, p[i_base + r0 + (r)])
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_DIV_ROW:
#define EP_OPX(x, r) EP_DIV_SPLAT_F16(p16, x, p[i_base + r0 + (r)])
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_MAX_ROW:
#define EP_OPX(x, r) svmaxnm_n_f16_x(p16, x, p[i_base + r0 + (r)])
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_MIN_ROW:
#define EP_OPX(x, r) svminnm_n_f16_x(p16, x, p[i_base + r0 + (r)])
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_SUB_COL: {
                svfloat16_t col = svld1_f16(pst, p + n0);
#define EP_OPX(x, r) svsub_f16_x(p16, x, col)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_DIV_COL: {
                svfloat16_t col = svld1_f16(pst, p + n0);
#define EP_OPX(x, r) svdiv_f16_x(p16, x, col)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_MAX_COL: {
                svfloat16_t col = svld1_f16(pst, p + n0);
#define EP_OPX(x, r) svmaxnm_f16_x(p16, x, col)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_MIN_COL: {
                svfloat16_t col = svld1_f16(pst, p + n0);
#define EP_OPX(x, r) svminnm_f16_x(p16, x, col)
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_SUB_TENSOR:
#define EP_OPX(x, r) svsub_f16_x(p16, x, svld1_f16(pst, p + (i_base + r0 + (r)) * nd->ld + n0))
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_DIV_TENSOR:
#define EP_OPX(x, r) svdiv_f16_x(p16, x, svld1_f16(pst, p + (i_base + r0 + (r)) * nd->ld + n0))
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_MAX_TENSOR:
#define EP_OPX(x, r) svmaxnm_f16_x(p16, x, svld1_f16(pst, p + (i_base + r0 + (r)) * nd->ld + n0))
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_MIN_TENSOR:
#define EP_OPX(x, r) svminnm_f16_x(p16, x, svld1_f16(pst, p + (i_base + r0 + (r)) * nd->ld + n0))
                EP_BLK_F16_OP(EP_OPX);
#undef EP_OPX
                break;
            default: break;
        }
    }
    if (read_dst) {
#define EP_DST(x, r) svmla_f16_x(p16, x, svld1_f16(pst, dst + (long)(r0 + (r)) * row_stride), va)
        EP_BLK_F16_OP(EP_DST);
#undef EP_DST
    }
    if (nr > 0) svst1_f16(pst, dst + (long)(r0 + 0) * row_stride, x0);
    if (nr > 1) svst1_f16(pst, dst + (long)(r0 + 1) * row_stride, x1);
    if (nr > 2) svst1_f16(pst, dst + (long)(r0 + 2) * row_stride, x2);
    if (nr > 3) svst1_f16(pst, dst + (long)(r0 + 3) * row_stride, x3);
}

// Node-major store of a 32-row row-major tile (vector spans N), reading the live
// ZA horizontal slices directly into Z registers in blocks of 4 rows that stay
// resident across the whole op-graph. No memory bounce: each node dispatches once
// per row-block (n_nodes per 4 rows) and the running rows never touch memory
// between the ZA read and the final store, so the only cost over the bare store is
// the actual op math. Invariant operands load once per node.
//
// The ZA tile index must be a compile-time immediate to the svread intrinsic, so
// the read is supplied by the caller as a macro `RD(slice)` (with the literal tile
// baked in) rather than a runtime argument. `dst`/`row_stride` locate output row r
// at dst + r*row_stride; `pst` is the N predicate; `i_base` is m0 (the tile's
// first output row).
#define EP_STORE_TILE_ROWMAJOR_F16(p16, pst, dst, row_stride, RD, mrows, vb, va, read_dst, nodes,  \
                                   n_nodes, i_base, n0)                                             \
    do {                                                                                           \
        size_t ep__mrows = (mrows);                                                                \
        size_t ep__r0 = 0;                                                                         \
        for (; ep__r0 + 4 <= ep__mrows; ep__r0 += 4) {                                             \
            ep_block_rowmajor_f16((p16), (pst), (dst), (row_stride), RD((uint32_t)(ep__r0 + 0)),   \
                                  RD((uint32_t)(ep__r0 + 1)), RD((uint32_t)(ep__r0 + 2)),          \
                                  RD((uint32_t)(ep__r0 + 3)), 4, (vb), (va), (read_dst), (nodes),  \
                                  (n_nodes), (i_base), ep__r0, (n0));                              \
        }                                                                                          \
        if (ep__r0 < ep__mrows) {                                                                  \
            size_t ep__nr = ep__mrows - ep__r0;                                                    \
            svfloat16_t ep__x0 = RD((uint32_t)(ep__r0 + 0));                                       \
            svfloat16_t ep__x1 = ep__nr > 1 ? RD((uint32_t)(ep__r0 + 1)) : ep__x0;                 \
            svfloat16_t ep__x2 = ep__nr > 2 ? RD((uint32_t)(ep__r0 + 2)) : ep__x0;                 \
            ep_block_rowmajor_f16((p16), (pst), (dst), (row_stride), ep__x0, ep__x1, ep__x2,       \
                                  ep__x0, ep__nr, (vb), (va), (read_dst), (nodes), (n_nodes),      \
                                  (i_base), ep__r0, (n0));                                         \
        }                                                                                          \
    } while (0)


#endif // SME_GEMM_EPILOGUE_F16_H
