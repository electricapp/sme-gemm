// bf16 in-register epilogue (op-graph evaluated in f32, rounded to bf16).
// Included by epilogue.h, which defines the shared enums, EpNode/ep_desc
// structs, and helper macros these evaluators use.
#ifndef SME_GEMM_EPILOGUE_BF16_H
#define SME_GEMM_EPILOGUE_BF16_H

// --- bf16 vectorized op-graph (FEAT_SME_B16B16 non-widening path) -----------
//
// The bf16 GEMM ACCUMULATES in bf16 (B16B16 MOPA), but the fused EPILOGUE runs
// the ENTIRE op-graph in f32: the live bf16 accumulator is upcast to f32, every
// operand (bias/row/col/tensor, in bf16 storage) is upcast to f32, the whole
// node graph -- add/sub/mul/div/max/min, all activations, all unary math -- is
// evaluated in the f32 domain (reusing the f32 node ops + ep_act_apply_f32 +
// the f32 transcendentals), then the result is rounded back to bf16 for the
// store. This avoids B16B16's missing divide AND Apple's unreliable streaming
// bf16 arith (svsub/svmin operand order), so bf16 supports the SAME op set as
// f16/f32. A bf16 SVE vector holds 32 lanes at SVL=512; it splits into two f32
// halves (lo = lanes 0..15, hi = lanes 16..31) the same way the f16 path does.

// Widen the lo / hi half of a bf16 vector to f32. bf16 is the top 16 bits of an
// f32, so zip the bf16 lanes into the high halfwords of f32 lanes (low halfword
// zeroed). svzip1_u16(0, x) takes the bottom 16 bf16 lanes -> lo f32 half;
// svzip2_u16 -> the top 16 -> hi f32 half.
static inline svfloat32_t ep_bf16_lo(svbfloat16_t x) __arm_streaming {
    svuint16_t xu = svreinterpret_u16_bf16(x);
    return svreinterpret_f32_u16(svzip1_u16(svdup_n_u16(0), xu));
}
static inline svfloat32_t ep_bf16_hi(svbfloat16_t x) __arm_streaming {
    svuint16_t xu = svreinterpret_u16_bf16(x);
    return svreinterpret_f32_u16(svzip2_u16(svdup_n_u16(0), xu));
}
// Round two f32 halves back to one bf16 vector (round-to-nearest-even). svcvt
// writes the bf16 result into the even bf16 lanes; uzp1 collects lo's evens then
// hi's evens, inverting the lo/hi split above.
static inline svbfloat16_t ep_bf16_pack(svbool_t p32, svfloat32_t lo,
                                        svfloat32_t hi) __arm_streaming {
    return svuzp1_bf16(svcvt_bf16_f32_x(p32, lo), svcvt_bf16_f32_x(p32, hi));
}

// Apply the whole op-graph to one bf16 accumulator vector, computed in f32. lo /
// hi are the two upcast halves of a single 32-lane bf16 vector (lo = lanes 0..15,
// hi = lanes 16..31); both are mutated in lockstep, one f32 node op per half. All
// f32 math is unpredicated (svptrue_b32) -- inactive lanes hold garbage but are
// never stored (the caller's svst1_bf16(pst,...) masks them). bf16 operand vectors
// are loaded with the b16 predicate `pst` so the load is in-bounds and inactive
// lanes widen to 0. span_n selects orientation exactly as the f16/f32 interpreters
// do (COL/TENSOR are vector loads when span_n, ROW when !span_n; the others are
// per-cell splats read as (float)bf16). The col-major caller routes TENSOR graphs
// to the scalar path, so TENSOR here is reached only for span_n.
static inline void ep_apply_nodes_bf16_f32(svbool_t p32, svbool_t pst, svfloat32_t *lo_p,
                                           svfloat32_t *hi_p, const EpNode *nodes,
                                           uint32_t n_nodes, int span_n, size_t i, size_t n0,
                                           size_t m0) __arm_streaming {
    svfloat32_t lo = *lo_p, hi = *hi_p;
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        const ep_bf16 *p = (const ep_bf16 *)nd->ptr;
        switch (nd->op) {
            case EP_OP_ADD_SCALAR:
                lo = svadd_n_f32_x(p32, lo, nd->scalar); hi = svadd_n_f32_x(p32, hi, nd->scalar);
                break;
            case EP_OP_MUL_SCALAR:
                lo = svmul_n_f32_x(p32, lo, nd->scalar); hi = svmul_n_f32_x(p32, hi, nd->scalar);
                break;
            case EP_OP_SUB_SCALAR:
                lo = svsub_n_f32_x(p32, lo, nd->scalar); hi = svsub_n_f32_x(p32, hi, nd->scalar);
                break;
            case EP_OP_DIV_SCALAR: {
                float s = nd->scalar;
                lo = EP_DIV_SPLAT_F32(p32, lo, s); hi = EP_DIV_SPLAT_F32(p32, hi, s);
                break;
            }
            case EP_OP_MAX_SCALAR:
                lo = svmaxnm_n_f32_x(p32, lo, nd->scalar); hi = svmaxnm_n_f32_x(p32, hi, nd->scalar);
                break;
            case EP_OP_MIN_SCALAR:
                lo = svminnm_n_f32_x(p32, lo, nd->scalar); hi = svminnm_n_f32_x(p32, hi, nd->scalar);
                break;
            case EP_OP_ADD_ROW:
                if (span_n) {
                    float s = (float)p[i];
                    lo = svadd_n_f32_x(p32, lo, s); hi = svadd_n_f32_x(p32, hi, s);
                } else {
                    svbfloat16_t v = svld1_bf16(pst, p + m0);
                    lo = svadd_f32_x(p32, lo, ep_bf16_lo(v)); hi = svadd_f32_x(p32, hi, ep_bf16_hi(v));
                }
                break;
            case EP_OP_MUL_ROW:
                if (span_n) {
                    float s = (float)p[i];
                    lo = svmul_n_f32_x(p32, lo, s); hi = svmul_n_f32_x(p32, hi, s);
                } else {
                    svbfloat16_t v = svld1_bf16(pst, p + m0);
                    lo = svmul_f32_x(p32, lo, ep_bf16_lo(v)); hi = svmul_f32_x(p32, hi, ep_bf16_hi(v));
                }
                break;
            case EP_OP_SUB_ROW:
                if (span_n) {
                    float s = (float)p[i];
                    lo = svsub_n_f32_x(p32, lo, s); hi = svsub_n_f32_x(p32, hi, s);
                } else {
                    svbfloat16_t v = svld1_bf16(pst, p + m0);
                    lo = svsub_f32_x(p32, lo, ep_bf16_lo(v)); hi = svsub_f32_x(p32, hi, ep_bf16_hi(v));
                }
                break;
            case EP_OP_DIV_ROW:
                if (span_n) {
                    float s = (float)p[i];
                    lo = EP_DIV_SPLAT_F32(p32, lo, s); hi = EP_DIV_SPLAT_F32(p32, hi, s);
                } else {
                    svbfloat16_t v = svld1_bf16(pst, p + m0);
                    lo = svdiv_f32_x(p32, lo, ep_bf16_lo(v)); hi = svdiv_f32_x(p32, hi, ep_bf16_hi(v));
                }
                break;
            case EP_OP_MAX_ROW:
                if (span_n) {
                    float s = (float)p[i];
                    lo = svmaxnm_n_f32_x(p32, lo, s); hi = svmaxnm_n_f32_x(p32, hi, s);
                } else {
                    svbfloat16_t v = svld1_bf16(pst, p + m0);
                    lo = svmaxnm_f32_x(p32, lo, ep_bf16_lo(v)); hi = svmaxnm_f32_x(p32, hi, ep_bf16_hi(v));
                }
                break;
            case EP_OP_MIN_ROW:
                if (span_n) {
                    float s = (float)p[i];
                    lo = svminnm_n_f32_x(p32, lo, s); hi = svminnm_n_f32_x(p32, hi, s);
                } else {
                    svbfloat16_t v = svld1_bf16(pst, p + m0);
                    lo = svminnm_f32_x(p32, lo, ep_bf16_lo(v)); hi = svminnm_f32_x(p32, hi, ep_bf16_hi(v));
                }
                break;
            case EP_OP_ADD_COL:
                if (span_n) {
                    svbfloat16_t v = svld1_bf16(pst, p + n0);
                    lo = svadd_f32_x(p32, lo, ep_bf16_lo(v)); hi = svadd_f32_x(p32, hi, ep_bf16_hi(v));
                } else {
                    float s = (float)p[i];
                    lo = svadd_n_f32_x(p32, lo, s); hi = svadd_n_f32_x(p32, hi, s);
                }
                break;
            case EP_OP_MUL_COL:
                if (span_n) {
                    svbfloat16_t v = svld1_bf16(pst, p + n0);
                    lo = svmul_f32_x(p32, lo, ep_bf16_lo(v)); hi = svmul_f32_x(p32, hi, ep_bf16_hi(v));
                } else {
                    float s = (float)p[i];
                    lo = svmul_n_f32_x(p32, lo, s); hi = svmul_n_f32_x(p32, hi, s);
                }
                break;
            case EP_OP_SUB_COL:
                if (span_n) {
                    svbfloat16_t v = svld1_bf16(pst, p + n0);
                    lo = svsub_f32_x(p32, lo, ep_bf16_lo(v)); hi = svsub_f32_x(p32, hi, ep_bf16_hi(v));
                } else {
                    float s = (float)p[i];
                    lo = svsub_n_f32_x(p32, lo, s); hi = svsub_n_f32_x(p32, hi, s);
                }
                break;
            case EP_OP_DIV_COL:
                if (span_n) {
                    svbfloat16_t v = svld1_bf16(pst, p + n0);
                    lo = svdiv_f32_x(p32, lo, ep_bf16_lo(v)); hi = svdiv_f32_x(p32, hi, ep_bf16_hi(v));
                } else {
                    float s = (float)p[i];
                    lo = EP_DIV_SPLAT_F32(p32, lo, s); hi = EP_DIV_SPLAT_F32(p32, hi, s);
                }
                break;
            case EP_OP_MAX_COL:
                if (span_n) {
                    svbfloat16_t v = svld1_bf16(pst, p + n0);
                    lo = svmaxnm_f32_x(p32, lo, ep_bf16_lo(v)); hi = svmaxnm_f32_x(p32, hi, ep_bf16_hi(v));
                } else {
                    float s = (float)p[i];
                    lo = svmaxnm_n_f32_x(p32, lo, s); hi = svmaxnm_n_f32_x(p32, hi, s);
                }
                break;
            case EP_OP_MIN_COL:
                if (span_n) {
                    svbfloat16_t v = svld1_bf16(pst, p + n0);
                    lo = svminnm_f32_x(p32, lo, ep_bf16_lo(v)); hi = svminnm_f32_x(p32, hi, ep_bf16_hi(v));
                } else {
                    float s = (float)p[i];
                    lo = svminnm_n_f32_x(p32, lo, s); hi = svminnm_n_f32_x(p32, hi, s);
                }
                break;
            // TENSOR (row-major / span_n only; col-major routes to scalar path).
            case EP_OP_ADD_TENSOR: {
                svbfloat16_t v = svld1_bf16(pst, p + i * nd->ld + n0);
                lo = svadd_f32_x(p32, lo, ep_bf16_lo(v)); hi = svadd_f32_x(p32, hi, ep_bf16_hi(v));
                break;
            }
            case EP_OP_MUL_TENSOR: {
                svbfloat16_t v = svld1_bf16(pst, p + i * nd->ld + n0);
                lo = svmul_f32_x(p32, lo, ep_bf16_lo(v)); hi = svmul_f32_x(p32, hi, ep_bf16_hi(v));
                break;
            }
            case EP_OP_SUB_TENSOR: {
                svbfloat16_t v = svld1_bf16(pst, p + i * nd->ld + n0);
                lo = svsub_f32_x(p32, lo, ep_bf16_lo(v)); hi = svsub_f32_x(p32, hi, ep_bf16_hi(v));
                break;
            }
            case EP_OP_DIV_TENSOR: {
                svbfloat16_t v = svld1_bf16(pst, p + i * nd->ld + n0);
                lo = svdiv_f32_x(p32, lo, ep_bf16_lo(v)); hi = svdiv_f32_x(p32, hi, ep_bf16_hi(v));
                break;
            }
            case EP_OP_MAX_TENSOR: {
                svbfloat16_t v = svld1_bf16(pst, p + i * nd->ld + n0);
                lo = svmaxnm_f32_x(p32, lo, ep_bf16_lo(v)); hi = svmaxnm_f32_x(p32, hi, ep_bf16_hi(v));
                break;
            }
            case EP_OP_MIN_TENSOR: {
                svbfloat16_t v = svld1_bf16(pst, p + i * nd->ld + n0);
                lo = svminnm_f32_x(p32, lo, ep_bf16_lo(v)); hi = svminnm_f32_x(p32, hi, ep_bf16_hi(v));
                break;
            }
            case EP_OP_ACT:
                lo = ep_act_apply_f32(p32, lo, (int)nd->aux, nd->scalar);
                hi = ep_act_apply_f32(p32, hi, (int)nd->aux, nd->scalar);
                break;
            default: break;
        }
    }
    *lo_p = lo;
    *hi_p = hi;
}

static inline void ep_store_bf16(svbool_t p16, svbool_t pst, ep_bf16 *ptr, svbfloat16_t acc,
                                 svbfloat16_t vb, svbfloat16_t va, int read_dst,
                                 const EpNode *nodes, uint32_t n_nodes, int span_n, size_t i,
                                 size_t n0, size_t m0) __arm_streaming {
    // beta in bf16, then upcast to f32 for the whole op-graph.
    svbfloat16_t xb = svmul_bf16_x(p16, acc, vb);
    svbool_t p32 = svptrue_b32();
    svfloat32_t lo = ep_bf16_lo(xb), hi = ep_bf16_hi(xb);
    ep_apply_nodes_bf16_f32(p32, pst, &lo, &hi, nodes, n_nodes, span_n, i, n0, m0);
    svbfloat16_t x = ep_bf16_pack(p32, lo, hi);
    if (read_dst) x = svmla_bf16_x(p16, x, svld1_bf16(pst, ptr), va);
    svst1_bf16(pst, ptr, x);
}

#endif // SME_GEMM_EPILOGUE_BF16_H
