// Scalar op-graph evaluators (one cell at a time) and the dequant cells.
// Included by epilogue.h, which defines the shared enums, EpNode/ep_desc
// structs, and helper macros these evaluators use.
#ifndef SME_GEMM_EPILOGUE_SCALAR_H
#define SME_GEMM_EPILOGUE_SCALAR_H

// ===========================================================================
// SCALAR EVALUATORS: walk the op-graph one cell at a time, in scalar f32/f64.
// Used by the non-vectorizable store paths (strided/general/col-major-TENSOR)
// and the dequant cells.
//
// NOTE ON libm + STREAMING: these evaluators call libm (tanhf/expf/logf/erff).
// They are invoked from inside the __arm_locally_streaming kernels on the
// strided / col-major-TENSOR fallback arms, so the compiler brackets each libm
// call with SMSTOP/SMSTART + a TPIDR2 lazy-save (ABI-correct, compiler-rt
// linked). That is fine for these deliberately-rare "correctness over speed"
// paths, but it is NOT free -- do not route a hot store through here.
//
// NaN SEMANTICS (unified across SIMD / C scalar / Rust; pinned by
// tests/correctness/nan.rs):
//   - MAX/MIN nodes and the standalone ReLU are maxNum/minNum. Lowering them to
//     FMAX/FMIN instead makes the paths disagree -- those PROPAGATE NaN, so the
//     answer would depend on the output layout. FMAXNM costs the same as FMAX
//     (epilogue_bench: no change).
//   - The activation clamps (relu6/hardsigmoid/hardswish, tanh/exp range
//     clamps) deliberately still PROPAGATE NaN, matching Rust's `clamp`. Do not
//     move those onto svmaxnm; it would break parity they already have.
//
// REMAINING CROSS-PATH DIVERGENCE (bounded, within the sqrt(K) tolerance):
//   - DIV by a scalar/splat operand is a multiply-by-reciprocal, so ~1 ULP off
//     true division (see EP_DIV_SPLAT_*, which guards the overflow case). DIV by
//     a full vector operand (ROW/COL-vector, TENSOR) is true svdiv and exact.
//   - The five NAMED activations relu/gelu/silu/tanh/sigmoid use the SAME
//     `ep_act_rational_*` minimax approximation, tuned for the gelu/silu
//     envelope. Used as a BARE `.tanh()`/`.sigmoid()` epilogue (no 0.5*x* mask)
//     this is ~4e-2 abs vs the exact libm `ep_act_scalar` used here -- e.g.
//     tanh(2): rational 0.984 vs exact 0.964. gelu/silu have the same ~4e-2 but
//     it is masked by the envelope. All other activations match closely.
// log/exp ARE specified at the boundaries (log(0)=-inf, log(x<0)=NaN, exp
// overflow=+inf) and the SIMD approximations honor them. Callers asserting
// tight cross-path bit-identity on a bare tanh/sigmoid or a splat-DIV should
// not rely on it -- use the documented tolerance.
// ===========================================================================

// Accurate scalar activation in f32 (libm). Used by the strided/general store
// path and the i8 dequant (which is already f32) -- not perf-critical, so it
// uses the exact tanhf/expf rather than the vector approximations.
static inline float ep_act_scalar(float x, int act) {
    switch (act) {
        case EP_ACT_RELU:
            return x > 0.0f ? x : 0.0f;
        case EP_ACT_GELU:
            return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        case EP_ACT_SILU:
            return x / (1.0f + expf(-x));
        case EP_ACT_TANH:
            return tanhf(x);
        case EP_ACT_SIGMOID:
            return 1.0f / (1.0f + expf(-x));
        // --- Group A (exact) ---
        case EP_ACT_LEAKY_RELU:
            return x; // alpha folded in by caller (needs node->scalar); see _aux variant
        case EP_ACT_RELU6:
            return x < 0.0f ? 0.0f : (x > 6.0f ? 6.0f : x);
        case EP_ACT_HARDSIGMOID: {
            float h = x / 6.0f + 0.5f;
            return h < 0.0f ? 0.0f : (h > 1.0f ? 1.0f : h);
        }
        case EP_ACT_HARDSWISH: {
            float h = x / 6.0f + 0.5f;
            h = h < 0.0f ? 0.0f : (h > 1.0f ? 1.0f : h);
            return x * h;
        }
        case EP_ACT_ABS:
            return fabsf(x);
        case EP_ACT_NEG:
            return -x;
        case EP_ACT_SQUARE:
            return x * x;
        case EP_ACT_SIGN:
            return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
        case EP_ACT_SQRT:
            return sqrtf(x);
        case EP_ACT_SOFTSIGN:
            return x / (1.0f + fabsf(x));
        case EP_ACT_RECIP:
            return 1.0f / x;
        case EP_ACT_RSQRT:
            return 1.0f / sqrtf(x);
        // --- Group B (transcendental) ---
        case EP_ACT_EXP:
            return expf(x);
        case EP_ACT_LOG:
            return logf(x);
        case EP_ACT_ELU:
            return x; // alpha folded in by caller; see _aux variant
        case EP_ACT_SELU:
            return x >= 0.0f ? EP_SELU_LAMBDA * x
                             : EP_SELU_LAMBDA * EP_SELU_ALPHA * (expf(x) - 1.0f);
        case EP_ACT_SOFTPLUS:
            return (x > 0.0f ? x : 0.0f) + log1pf(expf(-fabsf(x)));
        case EP_ACT_MISH: {
            float sp = (x > 0.0f ? x : 0.0f) + log1pf(expf(-fabsf(x)));
            return x * tanhf(sp);
        }
        case EP_ACT_GELU_EXACT:
            return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
        default:
            return x;
    }
}

// Scalar activation taking the node `aux` (kind) AND `scalar` (the alpha param
// for LEAKY_RELU/ELU). All other kinds ignore alpha.
static inline float ep_act_scalar_aux(float x, int act, float alpha) {
    if (act == EP_ACT_LEAKY_RELU) return x >= 0.0f ? x : alpha * x;
    if (act == EP_ACT_ELU) return x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
    return ep_act_scalar(x, act);
}

// maxNum/minNum (NaN operand discarded, number wins) -- matches Rust's
// f32::max and the svmaxnm/svminnm the vector paths use. `x > v ? x : v` is NOT
// equivalent: it returns NaN when the OPERAND is NaN. fmaxf lowers to a single
// FMAXNM here, so no libm call enters the streaming region. Not for the
// activation clamps (relu6/hardsigmoid/hardswish/tanh/exp), which propagate NaN
// to match Rust's `clamp`.
#define EP_MAXNUM(a, b) _Generic((a), float: fmaxf, double: fmax)((a), (b))
#define EP_MINNUM(a, b) _Generic((a), float: fminf, double: fmin)((a), (b))

// x / d for a SPLAT divisor (a *_SCALAR node, or ROW/COL in its broadcast
// orientation). SVE has no divide-by-scalar, so multiply by 1/d -- except when
// 1/d OVERFLOWS, where x*(1/d) gives +-inf but x/d is finite. Reachable with
// ordinary f16 numbers: 1/d overflows f16 below |d| ~= 1.53e-5. One predictable
// scalar test picks the true divide there; every normal input keeps the multiply
// (and its documented ~1 ULP).
#define EP_DIV_SPLAT_F16(P, X, D)                                                                   \
    ({                                                                                              \
        float ep_d_ = (float)(D);                                                                   \
        ep_f16 ep_r_ = (ep_f16)(1.0f / ep_d_);                                                      \
        __builtin_isfinite((float)ep_r_)                                                            \
            ? svmul_n_f16_x((P), (X), ep_r_)                                                        \
            : svdiv_f16_x((P), (X), svdup_n_f16((ep_f16)ep_d_));                                    \
    })
#define EP_DIV_SPLAT_F32(P, X, D)                                                                   \
    ({                                                                                              \
        float ep_d_ = (float)(D), ep_r_ = 1.0f / ep_d_;                                             \
        __builtin_isfinite(ep_r_) ? svmul_n_f32_x((P), (X), ep_r_)                                  \
                                  : svdiv_f32_x((P), (X), svdup_n_f32(ep_d_));                      \
    })
#define EP_DIV_SPLAT_F64(P, X, D)                                                                   \
    ({                                                                                              \
        double ep_d_ = (double)(D), ep_r_ = 1.0 / ep_d_;                                            \
        __builtin_isfinite(ep_r_) ? svmul_n_f64_x((P), (X), ep_r_)                                  \
                                  : svdiv_f64_x((P), (X), svdup_n_f64(ep_d_));                      \
    })

// Shared scalar-interpreter cases for the elementwise binary ops (SUB/DIV/MAX/
// MIN against scalar/row/col/tensor). RD(idx) reads operand element idx in the
// node's native type and widens to the working type (float/double). LD resolves
// the tensor row stride. Used by every scalar interpreter below.
#define EP_SCALAR_BINARY_CASES(RD, LD)                                                              \
    case EP_OP_SUB_SCALAR: x -= nd->scalar; break;                                                  \
    case EP_OP_SUB_ROW: x -= RD(i); break;                                                          \
    case EP_OP_SUB_COL: x -= RD(j); break;                                                          \
    case EP_OP_SUB_TENSOR: x -= RD(i * (LD) + j); break;                                            \
    case EP_OP_DIV_SCALAR: x /= nd->scalar; break;                                                  \
    case EP_OP_DIV_ROW: x /= RD(i); break;                                                          \
    case EP_OP_DIV_COL: x /= RD(j); break;                                                          \
    case EP_OP_DIV_TENSOR: x /= RD(i * (LD) + j); break;                                            \
    case EP_OP_MAX_ROW: { __typeof__(x) v = RD(i); x = EP_MAXNUM(x, v); break; }                    \
    case EP_OP_MAX_COL: { __typeof__(x) v = RD(j); x = EP_MAXNUM(x, v); break; }                    \
    case EP_OP_MAX_TENSOR: { __typeof__(x) v = RD(i * (LD) + j); x = EP_MAXNUM(x, v); break; }      \
    case EP_OP_MIN_ROW: { __typeof__(x) v = RD(i); x = EP_MINNUM(x, v); break; }                    \
    case EP_OP_MIN_COL: { __typeof__(x) v = RD(j); x = EP_MINNUM(x, v); break; }                    \
    case EP_OP_MIN_TENSOR: { __typeof__(x) v = RD(i * (LD) + j); x = EP_MINNUM(x, v); break; }

// Apply the whole op-graph to one scalar cell (i,j) of running value x. Used by
// the strided/general store fallbacks and the i8 dequant path. All operands are
// read in the node's native pointer type (T for 16-bit nodes, f32 for dq).
static inline float ep_apply_nodes_scalar_f16(const EpNode *nodes, uint32_t n_nodes, float x,
                                              size_t i, size_t j) {
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: x += nd->scalar; break;
            case EP_OP_MUL_SCALAR: x *= nd->scalar; break;
            case EP_OP_ADD_ROW: x += (float)((const ep_f16 *)nd->ptr)[i]; break;
            case EP_OP_MUL_ROW: x *= (float)((const ep_f16 *)nd->ptr)[i]; break;
            case EP_OP_ADD_COL: x += (float)((const ep_f16 *)nd->ptr)[j]; break;
            case EP_OP_MUL_COL: x *= (float)((const ep_f16 *)nd->ptr)[j]; break;
            case EP_OP_ADD_TENSOR: x += (float)((const ep_f16 *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_MUL_TENSOR: x *= (float)((const ep_f16 *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_ACT: x = ep_act_scalar_aux(x, (int)nd->aux, nd->scalar); break;
            case EP_OP_MAX_SCALAR: x = EP_MAXNUM(x, nd->scalar); break;
            case EP_OP_MIN_SCALAR: x = EP_MINNUM(x, nd->scalar); break;
#define EP_RD16(idx) ((float)((const ep_f16 *)nd->ptr)[(idx)])
            EP_SCALAR_BINARY_CASES(EP_RD16, nd->ld)
#undef EP_RD16
            default: break;
        }
    }
    return x;
}

static inline float ep_apply_nodes_scalar_bf16(const EpNode *nodes, uint32_t n_nodes, float x,
                                               size_t i, size_t j) {
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: x += nd->scalar; break;
            case EP_OP_MUL_SCALAR: x *= nd->scalar; break;
            case EP_OP_ADD_ROW: x += (float)((const ep_bf16 *)nd->ptr)[i]; break;
            case EP_OP_MUL_ROW: x *= (float)((const ep_bf16 *)nd->ptr)[i]; break;
            case EP_OP_ADD_COL: x += (float)((const ep_bf16 *)nd->ptr)[j]; break;
            case EP_OP_MUL_COL: x *= (float)((const ep_bf16 *)nd->ptr)[j]; break;
            case EP_OP_ADD_TENSOR: x += (float)((const ep_bf16 *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_MUL_TENSOR: x *= (float)((const ep_bf16 *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_ACT: x = ep_act_scalar_aux(x, (int)nd->aux, nd->scalar); break;
            case EP_OP_MAX_SCALAR: x = EP_MAXNUM(x, nd->scalar); break;
            case EP_OP_MIN_SCALAR: x = EP_MINNUM(x, nd->scalar); break;
#define EP_RDB16(idx) ((float)((const ep_bf16 *)nd->ptr)[(idx)])
            EP_SCALAR_BINARY_CASES(EP_RDB16, nd->ld)
#undef EP_RDB16
            default: break;
        }
    }
    return x;
}

// f32-domain op-graph applied to one scalar cell (strided/general f32 store).
static inline float ep_apply_nodes_scalar_f32(const EpNode *nodes, uint32_t n_nodes, float x,
                                              size_t i, size_t j) {
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: x += nd->scalar; break;
            case EP_OP_MUL_SCALAR: x *= nd->scalar; break;
            case EP_OP_ADD_ROW: x += ((const float *)nd->ptr)[i]; break;
            case EP_OP_MUL_ROW: x *= ((const float *)nd->ptr)[i]; break;
            case EP_OP_ADD_COL: x += ((const float *)nd->ptr)[j]; break;
            case EP_OP_MUL_COL: x *= ((const float *)nd->ptr)[j]; break;
            case EP_OP_ADD_TENSOR: x += ((const float *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_MUL_TENSOR: x *= ((const float *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_ACT: x = ep_act_scalar_aux(x, (int)nd->aux, nd->scalar); break;
            case EP_OP_MAX_SCALAR: x = EP_MAXNUM(x, nd->scalar); break;
            case EP_OP_MIN_SCALAR: x = EP_MINNUM(x, nd->scalar); break;
#define EP_RDF32(idx) (((const float *)nd->ptr)[(idx)])
            EP_SCALAR_BINARY_CASES(EP_RDF32, nd->ld)
#undef EP_RDF32
            default: break;
        }
    }
    return x;
}

// Accurate scalar activation in f64 (libm). Used by the strided/general store
// path and the col-major TENSOR fallback. The EpNode scalar operands are f32 and
// widen to double at the call site; here x is already double.
static inline double ep_act_scalar_f64(double x, int act) {
    switch (act) {
        case EP_ACT_RELU:
            return x > 0.0 ? x : 0.0;
        case EP_ACT_GELU:
            return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
        case EP_ACT_SILU:
            return x / (1.0 + exp(-x));
        case EP_ACT_TANH:
            return tanh(x);
        case EP_ACT_SIGMOID:
            return 1.0 / (1.0 + exp(-x));
        case EP_ACT_LEAKY_RELU:
            return x;
        case EP_ACT_RELU6:
            return x < 0.0 ? 0.0 : (x > 6.0 ? 6.0 : x);
        case EP_ACT_HARDSIGMOID: {
            double h = x / 6.0 + 0.5;
            return h < 0.0 ? 0.0 : (h > 1.0 ? 1.0 : h);
        }
        case EP_ACT_HARDSWISH: {
            double h = x / 6.0 + 0.5;
            h = h < 0.0 ? 0.0 : (h > 1.0 ? 1.0 : h);
            return x * h;
        }
        case EP_ACT_ABS:
            return fabs(x);
        case EP_ACT_NEG:
            return -x;
        case EP_ACT_SQUARE:
            return x * x;
        case EP_ACT_SIGN:
            return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
        case EP_ACT_SQRT:
            return sqrt(x);
        case EP_ACT_SOFTSIGN:
            return x / (1.0 + fabs(x));
        case EP_ACT_RECIP:
            return 1.0 / x;
        case EP_ACT_RSQRT:
            return 1.0 / sqrt(x);
        case EP_ACT_EXP:
            return exp(x);
        case EP_ACT_LOG:
            return log(x);
        case EP_ACT_ELU:
            return x;
        case EP_ACT_SELU:
            return x >= 0.0 ? (double)EP_SELU_LAMBDA * x
                            : (double)EP_SELU_LAMBDA * (double)EP_SELU_ALPHA * (exp(x) - 1.0);
        case EP_ACT_SOFTPLUS:
            return (x > 0.0 ? x : 0.0) + log1p(exp(-fabs(x)));
        case EP_ACT_MISH: {
            double sp = (x > 0.0 ? x : 0.0) + log1p(exp(-fabs(x)));
            return x * tanh(sp);
        }
        case EP_ACT_GELU_EXACT:
            return 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
        default:
            return x;
    }
}

static inline double ep_act_scalar_f64_aux(double x, int act, double alpha) {
    if (act == EP_ACT_LEAKY_RELU) return x >= 0.0 ? x : alpha * x;
    if (act == EP_ACT_ELU) return x >= 0.0 ? x : alpha * (exp(x) - 1.0);
    return ep_act_scalar_f64(x, act);
}

// f64-domain op-graph applied to one scalar cell (strided/general f64 store and
// the col-major TENSOR fallback). Operands are read as f64; the node `scalar`
// field is f32 and widens to double.
static inline double ep_apply_nodes_scalar_f64(const EpNode *nodes, uint32_t n_nodes, double x,
                                               size_t i, size_t j) {
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: x += (double)nd->scalar; break;
            case EP_OP_MUL_SCALAR: x *= (double)nd->scalar; break;
            case EP_OP_ADD_ROW: x += ((const double *)nd->ptr)[i]; break;
            case EP_OP_MUL_ROW: x *= ((const double *)nd->ptr)[i]; break;
            case EP_OP_ADD_COL: x += ((const double *)nd->ptr)[j]; break;
            case EP_OP_MUL_COL: x *= ((const double *)nd->ptr)[j]; break;
            case EP_OP_ADD_TENSOR: x += ((const double *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_MUL_TENSOR: x *= ((const double *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_ACT: x = ep_act_scalar_f64_aux(x, (int)nd->aux, (double)nd->scalar); break;
            case EP_OP_MAX_SCALAR: x = EP_MAXNUM(x, (double)nd->scalar); break;
            case EP_OP_MIN_SCALAR: x = EP_MINNUM(x, (double)nd->scalar); break;
#define EP_RDF64(idx) (((const double *)nd->ptr)[(idx)])
#define EP_SCSUB (double)nd->scalar
            case EP_OP_SUB_SCALAR: x -= (double)nd->scalar; break;
            case EP_OP_SUB_ROW: x -= EP_RDF64(i); break;
            case EP_OP_SUB_COL: x -= EP_RDF64(j); break;
            case EP_OP_SUB_TENSOR: x -= EP_RDF64(i * nd->ld + j); break;
            case EP_OP_DIV_SCALAR: x /= (double)nd->scalar; break;
            case EP_OP_DIV_ROW: x /= EP_RDF64(i); break;
            case EP_OP_DIV_COL: x /= EP_RDF64(j); break;
            case EP_OP_DIV_TENSOR: x /= EP_RDF64(i * nd->ld + j); break;
            case EP_OP_MAX_ROW: { double v = EP_RDF64(i); x = EP_MAXNUM(x, v); break; }
            case EP_OP_MAX_COL: { double v = EP_RDF64(j); x = EP_MAXNUM(x, v); break; }
            case EP_OP_MAX_TENSOR: { double v = EP_RDF64(i * nd->ld + j); x = EP_MAXNUM(x, v); break; }
            case EP_OP_MIN_ROW: { double v = EP_RDF64(i); x = EP_MINNUM(x, v); break; }
            case EP_OP_MIN_COL: { double v = EP_RDF64(j); x = EP_MINNUM(x, v); break; }
            case EP_OP_MIN_TENSOR: { double v = EP_RDF64(i * nd->ld + j); x = EP_MINNUM(x, v); break; }
#undef EP_RDF64
#undef EP_SCSUB
            default: break;
        }
    }
    return x;
}

// Dequant-domain (f32) op-graph applied to one scalar cell.
static inline float ep_apply_nodes_scalar_dq(const EpNode *nodes, uint32_t n_nodes, float x,
                                             size_t i, size_t j) {
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: x += nd->scalar; break;
            case EP_OP_MUL_SCALAR: x *= nd->scalar; break;
            case EP_OP_ADD_ROW: x += ((const float *)nd->ptr)[i]; break;
            case EP_OP_MUL_ROW: x *= ((const float *)nd->ptr)[i]; break;
            case EP_OP_ADD_COL: x += ((const float *)nd->ptr)[j]; break;
            case EP_OP_MUL_COL: x *= ((const float *)nd->ptr)[j]; break;
            case EP_OP_ADD_TENSOR: x += ((const float *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_MUL_TENSOR: x *= ((const float *)nd->ptr)[i * nd->ld + j]; break;
            case EP_OP_ACT: x = ep_act_scalar_aux(x, (int)nd->aux, nd->scalar); break;
            case EP_OP_MAX_SCALAR: x = EP_MAXNUM(x, nd->scalar); break;
            case EP_OP_MIN_SCALAR: x = EP_MINNUM(x, nd->scalar); break;
#define EP_RDDQ(idx) (((const float *)nd->ptr)[(idx)])
            EP_SCALAR_BINARY_CASES(EP_RDDQ, nd->ld)
#undef EP_RDDQ
            default: break;
        }
    }
    return x;
}

// Dequant one i32 cell at output row i and column j into the real (f32) domain
// (scale, per-tensor or per-N), then run the op-graph.
static inline float ep_dequant_cell(const ep_dq_f32 *dq, int32_t acc, size_t i, size_t j) {
    float s = dq->scale_n ? dq->scale_n[j] : dq->scale;
    float x = s * (float)acc;
    return ep_apply_nodes_scalar_dq(dq->nodes, dq->n_nodes, x, i, j);
}

// Dequant one i64 cell (i16->i64 SMOPA accumulator) at output row i, column j
// into the real (f32) domain (scale, per-tensor or per-N), then run the
// op-graph. The i64 -> f32 conversion may lose precision for accumulators that
// exceed the 24-bit f32 mantissa -- expected for quantized inference.
static inline float ep_dequant_cell_i64(const ep_dq_f32 *dq, int64_t acc, size_t i, size_t j) {
    float s = dq->scale_n ? dq->scale_n[j] : dq->scale;
    float x = s * (float)acc;
    return ep_apply_nodes_scalar_dq(dq->nodes, dq->n_nodes, x, i, j);
}

// Forward declarations of the f32 vector transcendentals (defined in the f32
// section below). The f16 path upcasts to f32 to evaluate the exotic

#endif // SME_GEMM_EPILOGUE_SCALAR_H
