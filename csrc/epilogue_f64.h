// f64 in-register epilogue: vector op-graph, store, and node-major tile store.
// Included by epilogue.h, which defines the shared enums, EpNode/ep_desc
// structs, and helper macros these evaluators use.
#ifndef SME_GEMM_EPILOGUE_F64_H
#define SME_GEMM_EPILOGUE_F64_H

// --- f64 vectorized op-graph (FMOPA f64 accumulator domain) -----------------

// f64 vector exp: range reduction by ln2 + a degree-11 Taylor/minimax poly on
// the reduced argument, then 2^k by exponent assembly. ~1e-13 relative.
static inline svfloat64_t ep_exp_f64(svbool_t p64, svfloat64_t x) __arm_streaming {
    // exp is finite up to ln(DBL_MAX) = 709.78271; +inf above. See ep_exp_f32.
    svbool_t over = svcmpgt_n_f64(p64, x, 709.78271289338397);
    // x < -708 has underflowed to ~0; mask to 0.0 so the clamp does not pin it
    // to exp(-708) (see ep_exp_f32). Also yields exp(-inf) = 0.
    svbool_t under = svcmplt_n_f64(p64, x, -708.0);
    // NaN guard: NaN survives the clamp below (FMAX/FMIN propagate it) and the
    // range reduction would still yield a finite value via FCVTZS(NaN) -> 0;
    // force NaN at the end instead (see ep_exp_f32).
    svbool_t isnan = svcmpne_f64(p64, x, x); // NaN is the only value != itself
    x = svmin_n_f64_x(p64, svmax_n_f64_x(p64, x, -708.0), 709.78271289338397);
    const double LOG2E = 1.4426950408889634;
    const double LN2_HI = 0.6931471805599453;
    svfloat64_t kf = svrintn_f64_x(p64, svmul_n_f64_x(p64, x, LOG2E));
    svfloat64_t r = svmls_n_f64_x(p64, x, kf, LN2_HI);
    // exp(r) via Horner on 1/n! up to r^11.
    svfloat64_t poly = svdup_n_f64(2.5052108385441720e-8);   // 1/11!
    poly = svmad_n_f64_x(p64, poly, r, 2.7557319223985893e-7); // 1/10!
    poly = svmad_n_f64_x(p64, poly, r, 2.7557319223985888e-6); // 1/9!
    poly = svmad_n_f64_x(p64, poly, r, 2.4801587301587302e-5); // 1/8!
    poly = svmad_n_f64_x(p64, poly, r, 1.9841269841269841e-4); // 1/7!
    poly = svmad_n_f64_x(p64, poly, r, 1.3888888888888889e-3); // 1/6!
    poly = svmad_n_f64_x(p64, poly, r, 8.3333333333333332e-3); // 1/5!
    poly = svmad_n_f64_x(p64, poly, r, 4.1666666666666664e-2); // 1/4!
    poly = svmad_n_f64_x(p64, poly, r, 1.6666666666666666e-1); // 1/3!
    poly = svmad_n_f64_x(p64, poly, r, 5.0000000000000000e-1); // 1/2!
    poly = svmad_n_f64_x(p64, poly, r, 1.0);
    poly = svmad_n_f64_x(p64, poly, r, 1.0);
    // 2^k in two halves (see ep_exp_f32) so a finite result never trips the inf
    // exponent pattern at the k=1024 overflow boundary.
    svint64_t ki = svcvt_s64_f64_x(p64, kf);
    svint64_t k1 = svasr_n_s64_x(p64, ki, 1);
    svint64_t k2 = svsub_s64_x(p64, ki, k1);
    svfloat64_t p1 = svreinterpret_f64_s64(svlsl_n_s64_x(p64, svadd_n_s64_x(p64, k1, 1023), 52));
    svfloat64_t p2 = svreinterpret_f64_s64(svlsl_n_s64_x(p64, svadd_n_s64_x(p64, k2, 1023), 52));
    svfloat64_t res = svmul_f64_x(p64, svmul_f64_x(p64, poly, p1), p2);
    res = svsel_f64(over, svreinterpret_f64_u64(svdup_n_u64(0x7ff0000000000000ULL)), res); // +inf
    res = svsel_f64(under, svdup_n_f64(0.0), res);                                         // -> 0
    return svsel_f64(isnan, svreinterpret_f64_u64(svdup_n_u64(0x7ff8000000000000ULL)),
                     res); // NaN -> NaN
}

// f64 vector log: extract IEEE-754 exponent + degree-? minimax on the mantissa.
static inline svfloat64_t ep_log_f64(svbool_t p64, svfloat64_t x) __arm_streaming {
    // Domain guards (see ep_log_f32): log(x<0) = NaN, log(0) = -inf,
    // log(+inf) = +inf, log(NaN) = NaN, and scale subnormals (< DBL_MIN =
    // 2^-1022) up by 2^54 into the normal range.
    svbool_t neg = svcmplt_n_f64(p64, x, 0.0);
    svbool_t zero = svcmpeq_n_f64(p64, x, 0.0);
    svbool_t posinf =
        svcmpeq_f64(p64, x, svreinterpret_f64_u64(svdup_n_u64(0x7ff0000000000000ULL)));
    svbool_t isnan = svcmpne_f64(p64, x, x); // NaN is the only value != itself
    svbool_t sub = svcmplt_n_f64(p64, x, 2.2250738585072014e-308);
    x = svsel_f64(sub, svmul_n_f64_x(p64, x, 18014398509481984.0), x); // *2^54
    svuint64_t xu = svreinterpret_u64_f64(x);
    svint64_t ei = svsub_n_s64_x(p64, svreinterpret_s64_u64(svlsr_n_u64_x(p64, xu, 52)), 1023);
    svuint64_t mu = svorr_n_u64_x(p64, svand_n_u64_x(p64, xu, 0x000fffffffffffffULL),
                                  0x3ff0000000000000ULL);
    svfloat64_t m = svreinterpret_f64_u64(mu);
    svfloat64_t ef = svcvt_f64_s64_x(p64, ei);
    svbool_t big = svcmpge_n_f64(p64, m, 1.4142135623730951);
    m = svsel_f64(big, svmul_n_f64_x(p64, m, 0.5), m);
    ef = svadd_f64_x(p64, ef, svsel_f64(big, svdup_n_f64(1.0), svdup_n_f64(0.0)));
    // s = (m-1)/(m+1); log(m) = 2*(s + s^3/3 + s^5/5 + ...). Atanh series.
    svfloat64_t s = svdiv_f64_x(p64, svsub_n_f64_x(p64, m, 1.0), svadd_n_f64_x(p64, m, 1.0));
    svfloat64_t s2 = svmul_f64_x(p64, s, s);
    svfloat64_t poly = svdup_n_f64(1.0 / 15.0);
    poly = svmad_n_f64_x(p64, poly, s2, 1.0 / 13.0);
    poly = svmad_n_f64_x(p64, poly, s2, 1.0 / 11.0);
    poly = svmad_n_f64_x(p64, poly, s2, 1.0 / 9.0);
    poly = svmad_n_f64_x(p64, poly, s2, 1.0 / 7.0);
    poly = svmad_n_f64_x(p64, poly, s2, 1.0 / 5.0);
    poly = svmad_n_f64_x(p64, poly, s2, 1.0 / 3.0);
    poly = svmad_n_f64_x(p64, poly, s2, 1.0);
    svfloat64_t lm = svmul_n_f64_x(p64, svmul_f64_x(p64, s, poly), 2.0);
    const double LN2 = 0.6931471805599453;
    svfloat64_t res = svmla_n_f64_x(p64, lm, ef, LN2);
    res = svsub_f64_x(p64, res, svsel_f64(sub, svdup_n_f64(54.0 * LN2), svdup_n_f64(0.0)));
    res = svsel_f64(zero, svreinterpret_f64_u64(svdup_n_u64(0xfff0000000000000ULL)), res); // -inf
    res = svsel_f64(neg, svreinterpret_f64_u64(svdup_n_u64(0x7ff8000000000000ULL)), res);  // NaN
    res = svsel_f64(posinf, svreinterpret_f64_u64(svdup_n_u64(0x7ff0000000000000ULL)), res); // +inf
    res = svsel_f64(isnan, svreinterpret_f64_u64(svdup_n_u64(0x7ff8000000000000ULL)), res);  // NaN
    return res;
}

// f64 vector erf via A&S 7.1.26 (same rational form as f32; ~1.5e-7 abs error).
static inline svfloat64_t ep_erf_f64(svbool_t p64, svfloat64_t x) __arm_streaming {
    svfloat64_t ax = svabs_f64_x(p64, x);
    svfloat64_t t = svdiv_f64_x(p64, svdup_n_f64(1.0),
                                svadd_n_f64_x(p64, svmul_n_f64_x(p64, ax, 0.3275911), 1.0));
    svfloat64_t poly = svdup_n_f64(1.061405429);
    poly = svmad_n_f64_x(p64, poly, t, -1.453152027);
    poly = svmad_n_f64_x(p64, poly, t, 1.421413741);
    poly = svmad_n_f64_x(p64, poly, t, -0.284496736);
    poly = svmad_n_f64_x(p64, poly, t, 0.254829592);
    poly = svmul_f64_x(p64, poly, t);
    svfloat64_t ex2 = ep_exp_f64(p64, svneg_f64_x(p64, svmul_f64_x(p64, ax, ax)));
    svfloat64_t e = svsub_f64_x(p64, svdup_n_f64(1.0), svmul_f64_x(p64, poly, ex2));
    return svsel_f64(svcmplt_n_f64(p64, x, 0.0), svneg_f64_x(p64, e), e);
}

static inline svfloat64_t ep_softplus_f64(svbool_t p64, svfloat64_t x) __arm_streaming {
    svfloat64_t ax = svabs_f64_x(p64, x);
    svfloat64_t e = ep_exp_f64(p64, svneg_f64_x(p64, ax));
    svfloat64_t l = ep_log_f64(p64, svadd_n_f64_x(p64, e, 1.0));
    return svadd_f64_x(p64, svmax_n_f64_x(p64, x, 0.0), l);
}

// Rational tanh on a live f64 slice: same form as the f32 path at double
// precision. End-to-end GELU/SiLU/tanh error is dominated by the rational form
// (~4e-2 worst-case), not the double precision; acceptable for fused inference.
static inline svfloat64_t ep_tanh_f64(svbool_t p64, svfloat64_t t) __arm_streaming {
    t = svmin_n_f64_x(p64, svmax_n_f64_x(p64, t, -20.0), 20.0);
    svfloat64_t t2 = svmul_f64_x(p64, t, t);
    svfloat64_t num = svadd_n_f64_x(p64, t2, 27.0);
    svfloat64_t den = svadd_n_f64_x(p64, svmul_n_f64_x(p64, t2, 9.0), 27.0);
    svfloat64_t th = svmul_f64_x(p64, t, svdiv_f64_x(p64, num, den));
    return svmin_n_f64_x(p64, svmax_n_f64_x(p64, th, -1.0), 1.0);
}

// Rational GELU/SiLU/TANH/SIGMOID pulled out of line (mirrors the f16/f32 split)
// so the common add/mul/relu node path inlines cleanly into the node-major loops.
__attribute__((noinline)) static svfloat64_t ep_act_rational_f64(svbool_t p64, svfloat64_t x,
                                                                 int act,
                                                                 double alpha) __arm_streaming {
    if (act == EP_ACT_GELU) {
        svfloat64_t x2 = svmul_f64_x(p64, x, x);
        svfloat64_t inner = svmul_f64_x(
            p64, x, svadd_n_f64_x(p64, svmul_n_f64_x(p64, x2, 0.044715), 1.0));
        svfloat64_t t = svmul_n_f64_x(p64, inner, 0.7978845608028654);
        svfloat64_t th = ep_tanh_f64(p64, t);
        return svmul_n_f64_x(p64, svmul_f64_x(p64, x, svadd_n_f64_x(p64, th, 1.0)), 0.5);
    }
    if (act == EP_ACT_SILU) {
        svfloat64_t th = ep_tanh_f64(p64, svmul_n_f64_x(p64, x, 0.5));
        return svmul_n_f64_x(p64, svmul_f64_x(p64, x, svadd_n_f64_x(p64, th, 1.0)), 0.5);
    }
    if (act == EP_ACT_TANH) {
        return ep_tanh_f64(p64, x);
    }
    if (act == EP_ACT_SIGMOID) {
        svfloat64_t th = ep_tanh_f64(p64, svmul_n_f64_x(p64, x, 0.5));
        return svmul_n_f64_x(p64, svadd_n_f64_x(p64, th, 1.0), 0.5);
    }
    // --- Group A ---
    if (act == EP_ACT_LEAKY_RELU) {
        svfloat64_t pos = svmax_n_f64_x(p64, x, 0.0);
        svfloat64_t neg = svmin_n_f64_x(p64, x, 0.0);
        return svmla_n_f64_x(p64, pos, neg, alpha);
    }
    if (act == EP_ACT_RELU6) {
        return svmin_n_f64_x(p64, svmax_n_f64_x(p64, x, 0.0), 6.0);
    }
    if (act == EP_ACT_HARDSIGMOID) {
        svfloat64_t h = svadd_n_f64_x(p64, svmul_n_f64_x(p64, x, 1.0 / 6.0), 0.5);
        return svmin_n_f64_x(p64, svmax_n_f64_x(p64, h, 0.0), 1.0);
    }
    if (act == EP_ACT_HARDSWISH) {
        svfloat64_t h = svadd_n_f64_x(p64, svmul_n_f64_x(p64, x, 1.0 / 6.0), 0.5);
        h = svmin_n_f64_x(p64, svmax_n_f64_x(p64, h, 0.0), 1.0);
        return svmul_f64_x(p64, x, h);
    }
    if (act == EP_ACT_ABS) return svabs_f64_x(p64, x);
    if (act == EP_ACT_NEG) return svneg_f64_x(p64, x);
    if (act == EP_ACT_SQUARE) return svmul_f64_x(p64, x, x);
    if (act == EP_ACT_SIGN) {
        svfloat64_t pos =
            svsel_f64(svcmpgt_n_f64(p64, x, 0.0), svdup_n_f64(1.0), svdup_n_f64(0.0));
        return svsel_f64(svcmplt_n_f64(p64, x, 0.0), svdup_n_f64(-1.0), pos);
    }
    if (act == EP_ACT_SQRT) return svsqrt_f64_x(p64, x);
    if (act == EP_ACT_SOFTSIGN) {
        return svdiv_f64_x(p64, x, svadd_n_f64_x(p64, svabs_f64_x(p64, x), 1.0));
    }
    if (act == EP_ACT_RECIP) return svdiv_f64_x(p64, svdup_n_f64(1.0), x);
    if (act == EP_ACT_RSQRT) return svdiv_f64_x(p64, svdup_n_f64(1.0), svsqrt_f64_x(p64, x));
    // --- Group B ---
    if (act == EP_ACT_EXP) return ep_exp_f64(p64, x);
    if (act == EP_ACT_LOG) return ep_log_f64(p64, x);
    if (act == EP_ACT_ELU) {
        svfloat64_t em1 = svsub_n_f64_x(p64, ep_exp_f64(p64, x), 1.0);
        svfloat64_t neg = svmul_n_f64_x(p64, em1, alpha);
        return svsel_f64(svcmpge_n_f64(p64, x, 0.0), x, neg);
    }
    if (act == EP_ACT_SELU) {
        svfloat64_t em1 = svsub_n_f64_x(p64, ep_exp_f64(p64, x), 1.0);
        svfloat64_t neg = svmul_n_f64_x(p64, em1, (double)EP_SELU_ALPHA);
        svfloat64_t v = svsel_f64(svcmpge_n_f64(p64, x, 0.0), x, neg);
        return svmul_n_f64_x(p64, v, (double)EP_SELU_LAMBDA);
    }
    if (act == EP_ACT_SOFTPLUS) return ep_softplus_f64(p64, x);
    if (act == EP_ACT_MISH) {
        svfloat64_t sp = ep_softplus_f64(p64, x);
        return svmul_f64_x(p64, x, ep_tanh_f64(p64, sp));
    }
    if (act == EP_ACT_GELU_EXACT) {
        svfloat64_t e = ep_erf_f64(p64, svmul_n_f64_x(p64, x, 0.7071067811865476));
        return svmul_n_f64_x(p64, svmul_f64_x(p64, x, svadd_n_f64_x(p64, e, 1.0)), 0.5);
    }
    return x;
}

static inline svfloat64_t ep_act_apply_f64(svbool_t p64, svfloat64_t x, int act,
                                           double alpha) __arm_streaming {
    if (act == EP_ACT_RELU) {
        return svmaxnm_n_f64_x(p64, x, 0.0);
    }
    if (act == EP_ACT_NONE) {
        return x;
    }
    return ep_act_rational_f64(p64, x, act, alpha);
}

// Walk the op-graph over a live f64 SVE vector `x`. `span_n` mirrors the f32
// interpreter: span_n != 0 (row-major dst, vector spans N) -> COL/TENSOR are
// vector loads, ROW/SCALAR splats; span_n == 0 (col-major dst, vector spans M)
// -> ROW is the vector load, COL/SCALAR splats; TENSOR routes to the scalar path.
static inline svfloat64_t ep_apply_nodes_f64(svbool_t p64, svbool_t pst, svfloat64_t x,
                                             const EpNode *nodes, uint32_t n_nodes, int span_n,
                                             size_t i, size_t n0, size_t m0) __arm_streaming {
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        const double *p = (const double *)nd->ptr;
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: x = svadd_n_f64_x(p64, x, (double)nd->scalar); break;
            case EP_OP_MUL_SCALAR: x = svmul_n_f64_x(p64, x, (double)nd->scalar); break;
            case EP_OP_ADD_ROW:
                if (span_n) x = svadd_n_f64_x(p64, x, p[i]);
                else x = svadd_f64_x(p64, x, svld1_f64(pst, p + m0));
                break;
            case EP_OP_MUL_ROW:
                if (span_n) x = svmul_n_f64_x(p64, x, p[i]);
                else x = svmul_f64_x(p64, x, svld1_f64(pst, p + m0));
                break;
            case EP_OP_ADD_COL:
                if (span_n) x = svadd_f64_x(p64, x, svld1_f64(pst, p + n0));
                else x = svadd_n_f64_x(p64, x, p[i]);
                break;
            case EP_OP_MUL_COL:
                if (span_n) x = svmul_f64_x(p64, x, svld1_f64(pst, p + n0));
                else x = svmul_n_f64_x(p64, x, p[i]);
                break;
            case EP_OP_ADD_TENSOR:
                x = svadd_f64_x(p64, x, svld1_f64(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_MUL_TENSOR:
                x = svmul_f64_x(p64, x, svld1_f64(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_ACT: x = ep_act_apply_f64(p64, x, (int)nd->aux, (double)nd->scalar); break;
            case EP_OP_MAX_SCALAR: x = svmaxnm_n_f64_x(p64, x, (double)nd->scalar); break;
            case EP_OP_MIN_SCALAR: x = svminnm_n_f64_x(p64, x, (double)nd->scalar); break;
            case EP_OP_SUB_SCALAR: x = svsub_n_f64_x(p64, x, (double)nd->scalar); break;
            case EP_OP_DIV_SCALAR: x = EP_DIV_SPLAT_F64(p64, x, nd->scalar); break;
            case EP_OP_SUB_ROW:
                if (span_n) x = svsub_n_f64_x(p64, x, p[i]);
                else x = svsub_f64_x(p64, x, svld1_f64(pst, p + m0));
                break;
            case EP_OP_DIV_ROW:
                if (span_n) x = EP_DIV_SPLAT_F64(p64, x, p[i]);
                else x = svdiv_f64_x(p64, x, svld1_f64(pst, p + m0));
                break;
            case EP_OP_MAX_ROW:
                if (span_n) x = svmaxnm_n_f64_x(p64, x, p[i]);
                else x = svmaxnm_f64_x(p64, x, svld1_f64(pst, p + m0));
                break;
            case EP_OP_MIN_ROW:
                if (span_n) x = svminnm_n_f64_x(p64, x, p[i]);
                else x = svminnm_f64_x(p64, x, svld1_f64(pst, p + m0));
                break;
            case EP_OP_SUB_COL:
                if (span_n) x = svsub_f64_x(p64, x, svld1_f64(pst, p + n0));
                else x = svsub_n_f64_x(p64, x, p[i]);
                break;
            case EP_OP_DIV_COL:
                if (span_n) x = svdiv_f64_x(p64, x, svld1_f64(pst, p + n0));
                else x = EP_DIV_SPLAT_F64(p64, x, p[i]);
                break;
            case EP_OP_MAX_COL:
                if (span_n) x = svmaxnm_f64_x(p64, x, svld1_f64(pst, p + n0));
                else x = svmaxnm_n_f64_x(p64, x, p[i]);
                break;
            case EP_OP_MIN_COL:
                if (span_n) x = svminnm_f64_x(p64, x, svld1_f64(pst, p + n0));
                else x = svminnm_n_f64_x(p64, x, p[i]);
                break;
            case EP_OP_SUB_TENSOR:
                x = svsub_f64_x(p64, x, svld1_f64(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_DIV_TENSOR:
                x = svdiv_f64_x(p64, x, svld1_f64(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_MAX_TENSOR:
                x = svmaxnm_f64_x(p64, x, svld1_f64(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_MIN_TENSOR:
                x = svminnm_f64_x(p64, x, svld1_f64(pst, p + i * nd->ld + n0));
                break;
            default: break;
        }
    }
    return x;
}

// Combine + store one f64 slice with the fused op-graph epilogue:
//   x = beta*acc; x = nodes(x); [x = alpha*C + x]; store
static inline void ep_store_f64(svbool_t p64, svbool_t pst, double *ptr, svfloat64_t acc,
                                svfloat64_t vb, svfloat64_t va, int read_dst, const EpNode *nodes,
                                uint32_t n_nodes, int span_n, size_t i, size_t n0,
                                size_t m0) __arm_streaming {
    svfloat64_t x = svmul_f64_x(p64, acc, vb);
    x = ep_apply_nodes_f64(p64, pst, x, nodes, n_nodes, span_n, i, n0, m0);
    if (read_dst) x = svmla_f64_x(p64, x, svld1_f64(pst, ptr), va);
    svst1_f64(pst, ptr, x);
}

// --- f64 NODE-MAJOR tile epilogue (row-major dst, vector spans N) -----------
// f64 stores a 16-wide output row as two 8-lane halves (lanes 0..7 "lo", lanes
// 8..15 "hi"), held across four ZA tiles (0/1 for rows 0..7, 2/3 for rows 8..15).
// Same node-major block structure as f32 with an 8-lane hi offset.
#define EP_BLK_F64_OP(EXPRLO, EXPRHI)                                    \
    do {                                                                 \
        if (nr > 0) { l0 = (EXPRLO(l0, 0)); h0 = (EXPRHI(h0, 0)); }      \
        if (nr > 1) { l1 = (EXPRLO(l1, 1)); h1 = (EXPRHI(h1, 1)); }      \
        if (nr > 2) { l2 = (EXPRLO(l2, 2)); h2 = (EXPRHI(h2, 2)); }      \
        if (nr > 3) { l3 = (EXPRLO(l3, 3)); h3 = (EXPRHI(h3, 3)); }      \
    } while (0)

__attribute__((always_inline)) static inline void
ep_block_rowmajor_f64(svbool_t p64, svbool_t plo, svbool_t phi, double *dst, long row_stride,
                      svfloat64_t l0, svfloat64_t h0, svfloat64_t l1, svfloat64_t h1,
                      svfloat64_t l2, svfloat64_t h2, svfloat64_t l3, svfloat64_t h3, size_t nr,
                      svfloat64_t vb, svfloat64_t va, int read_dst, const EpNode *nodes,
                      uint32_t n_nodes, size_t i_base, size_t r0, size_t n0) __arm_streaming {
#define EP_BETA(x, r) svmul_f64_x(p64, x, vb)
    EP_BLK_F64_OP(EP_BETA, EP_BETA);
#undef EP_BETA
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        const double *p = (const double *)nd->ptr;
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: {
                double s = (double)nd->scalar;
#define EP_OPX(x, r) svadd_n_f64_x(p64, x, s)
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_MUL_SCALAR: {
                double s = (double)nd->scalar;
#define EP_OPX(x, r) svmul_n_f64_x(p64, x, s)
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_MAX_SCALAR: {
                double s = (double)nd->scalar;
#define EP_OPX(x, r) svmaxnm_n_f64_x(p64, x, s)
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_MIN_SCALAR: {
                double s = (double)nd->scalar;
#define EP_OPX(x, r) svminnm_n_f64_x(p64, x, s)
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_ADD_ROW:
#define EP_OPX(x, r) svadd_n_f64_x(p64, x, p[i_base + r0 + (r)])
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_MUL_ROW:
#define EP_OPX(x, r) svmul_n_f64_x(p64, x, p[i_base + r0 + (r)])
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_ADD_COL: {
                svfloat64_t cl = svld1_f64(plo, p + n0);
                svfloat64_t ch = svld1_f64(phi, p + n0 + 8);
#define EP_OPL(x, r) svadd_f64_x(p64, x, cl)
#define EP_OPH(x, r) svadd_f64_x(p64, x, ch)
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            }
            case EP_OP_MUL_COL: {
                svfloat64_t cl = svld1_f64(plo, p + n0);
                svfloat64_t ch = svld1_f64(phi, p + n0 + 8);
#define EP_OPL(x, r) svmul_f64_x(p64, x, cl)
#define EP_OPH(x, r) svmul_f64_x(p64, x, ch)
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            }
            case EP_OP_ADD_TENSOR:
#define EP_OPL(x, r) svadd_f64_x(p64, x, svld1_f64(plo, p + (i_base + r0 + (r)) * nd->ld + n0))
#define EP_OPH(x, r) svadd_f64_x(p64, x, svld1_f64(phi, p + (i_base + r0 + (r)) * nd->ld + n0 + 8))
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            case EP_OP_MUL_TENSOR:
#define EP_OPL(x, r) svmul_f64_x(p64, x, svld1_f64(plo, p + (i_base + r0 + (r)) * nd->ld + n0))
#define EP_OPH(x, r) svmul_f64_x(p64, x, svld1_f64(phi, p + (i_base + r0 + (r)) * nd->ld + n0 + 8))
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            case EP_OP_ACT: {
                int act = (int)nd->aux;
                double alpha = (double)nd->scalar;
#define EP_OPX(x, r) ep_act_apply_f64(p64, x, act, alpha)
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_SUB_SCALAR: {
                double s = (double)nd->scalar;
#define EP_OPX(x, r) svsub_n_f64_x(p64, x, s)
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_DIV_SCALAR: {
                double s = (double)nd->scalar;
#define EP_OPX(x, r) EP_DIV_SPLAT_F64(p64, x, s)
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            }
            case EP_OP_SUB_ROW:
#define EP_OPX(x, r) svsub_n_f64_x(p64, x, p[i_base + r0 + (r)])
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_DIV_ROW:
#define EP_OPX(x, r) EP_DIV_SPLAT_F64(p64, x, p[i_base + r0 + (r)])
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_MAX_ROW:
#define EP_OPX(x, r) svmaxnm_n_f64_x(p64, x, p[i_base + r0 + (r)])
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_MIN_ROW:
#define EP_OPX(x, r) svminnm_n_f64_x(p64, x, p[i_base + r0 + (r)])
                EP_BLK_F64_OP(EP_OPX, EP_OPX);
#undef EP_OPX
                break;
            case EP_OP_SUB_COL: {
                svfloat64_t cl = svld1_f64(plo, p + n0);
                svfloat64_t ch = svld1_f64(phi, p + n0 + 8);
#define EP_OPL(x, r) svsub_f64_x(p64, x, cl)
#define EP_OPH(x, r) svsub_f64_x(p64, x, ch)
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            }
            case EP_OP_DIV_COL: {
                svfloat64_t cl = svld1_f64(plo, p + n0);
                svfloat64_t ch = svld1_f64(phi, p + n0 + 8);
#define EP_OPL(x, r) svdiv_f64_x(p64, x, cl)
#define EP_OPH(x, r) svdiv_f64_x(p64, x, ch)
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            }
            case EP_OP_MAX_COL: {
                svfloat64_t cl = svld1_f64(plo, p + n0);
                svfloat64_t ch = svld1_f64(phi, p + n0 + 8);
#define EP_OPL(x, r) svmaxnm_f64_x(p64, x, cl)
#define EP_OPH(x, r) svmaxnm_f64_x(p64, x, ch)
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            }
            case EP_OP_MIN_COL: {
                svfloat64_t cl = svld1_f64(plo, p + n0);
                svfloat64_t ch = svld1_f64(phi, p + n0 + 8);
#define EP_OPL(x, r) svminnm_f64_x(p64, x, cl)
#define EP_OPH(x, r) svminnm_f64_x(p64, x, ch)
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            }
            case EP_OP_SUB_TENSOR:
#define EP_OPL(x, r) svsub_f64_x(p64, x, svld1_f64(plo, p + (i_base + r0 + (r)) * nd->ld + n0))
#define EP_OPH(x, r) svsub_f64_x(p64, x, svld1_f64(phi, p + (i_base + r0 + (r)) * nd->ld + n0 + 8))
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            case EP_OP_DIV_TENSOR:
#define EP_OPL(x, r) svdiv_f64_x(p64, x, svld1_f64(plo, p + (i_base + r0 + (r)) * nd->ld + n0))
#define EP_OPH(x, r) svdiv_f64_x(p64, x, svld1_f64(phi, p + (i_base + r0 + (r)) * nd->ld + n0 + 8))
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            case EP_OP_MAX_TENSOR:
#define EP_OPL(x, r) svmaxnm_f64_x(p64, x, svld1_f64(plo, p + (i_base + r0 + (r)) * nd->ld + n0))
#define EP_OPH(x, r) svmaxnm_f64_x(p64, x, svld1_f64(phi, p + (i_base + r0 + (r)) * nd->ld + n0 + 8))
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            case EP_OP_MIN_TENSOR:
#define EP_OPL(x, r) svminnm_f64_x(p64, x, svld1_f64(plo, p + (i_base + r0 + (r)) * nd->ld + n0))
#define EP_OPH(x, r) svminnm_f64_x(p64, x, svld1_f64(phi, p + (i_base + r0 + (r)) * nd->ld + n0 + 8))
                EP_BLK_F64_OP(EP_OPL, EP_OPH);
#undef EP_OPL
#undef EP_OPH
                break;
            default: break;
        }
    }
    if (read_dst) {
#define EP_DL(x, r) svmla_f64_x(p64, x, svld1_f64(plo, dst + (long)(r0 + (r)) * row_stride), va)
#define EP_DH(x, r) svmla_f64_x(p64, x, svld1_f64(phi, dst + (long)(r0 + (r)) * row_stride + 8), va)
        EP_BLK_F64_OP(EP_DL, EP_DH);
#undef EP_DL
#undef EP_DH
    }
    if (nr > 0) { svst1_f64(plo, dst + (long)(r0 + 0) * row_stride, l0);
                  svst1_f64(phi, dst + (long)(r0 + 0) * row_stride + 8, h0); }
    if (nr > 1) { svst1_f64(plo, dst + (long)(r0 + 1) * row_stride, l1);
                  svst1_f64(phi, dst + (long)(r0 + 1) * row_stride + 8, h1); }
    if (nr > 2) { svst1_f64(plo, dst + (long)(r0 + 2) * row_stride, l2);
                  svst1_f64(phi, dst + (long)(r0 + 2) * row_stride + 8, h2); }
    if (nr > 3) { svst1_f64(plo, dst + (long)(r0 + 3) * row_stride, l3);
                  svst1_f64(phi, dst + (long)(r0 + 3) * row_stride + 8, h3); }
}

#define EP_STORE_TILE_ROWMAJOR_F64(p64, plo, phi, dst, row_stride, RDL, RDH, base_r, rmax, vb, va,  \
                                   read_dst, nodes, n_nodes, i_base, n0)                            \
    do {                                                                                           \
        size_t ep__lo = (base_r), ep__hi = (rmax);                                                 \
        size_t ep__r = ep__lo;                                                                     \
        for (; ep__r + 4 <= ep__hi; ep__r += 4) {                                                  \
            ep_block_rowmajor_f64(                                                                 \
                (p64), (plo), (phi), (dst), (row_stride), RDL((uint32_t)(ep__r + 0 - ep__lo)),     \
                RDH((uint32_t)(ep__r + 0 - ep__lo)), RDL((uint32_t)(ep__r + 1 - ep__lo)),          \
                RDH((uint32_t)(ep__r + 1 - ep__lo)), RDL((uint32_t)(ep__r + 2 - ep__lo)),          \
                RDH((uint32_t)(ep__r + 2 - ep__lo)), RDL((uint32_t)(ep__r + 3 - ep__lo)),          \
                RDH((uint32_t)(ep__r + 3 - ep__lo)), 4, (vb), (va), (read_dst), (nodes),           \
                (n_nodes), (i_base), ep__r, (n0));                                                 \
        }                                                                                          \
        if (ep__r < ep__hi) {                                                                      \
            size_t ep__nr = ep__hi - ep__r;                                                        \
            svfloat64_t ep__l0 = RDL((uint32_t)(ep__r + 0 - ep__lo));                              \
            svfloat64_t ep__h0 = RDH((uint32_t)(ep__r + 0 - ep__lo));                              \
            svfloat64_t ep__l1 = ep__nr > 1 ? RDL((uint32_t)(ep__r + 1 - ep__lo)) : ep__l0;        \
            svfloat64_t ep__h1 = ep__nr > 1 ? RDH((uint32_t)(ep__r + 1 - ep__lo)) : ep__h0;        \
            svfloat64_t ep__l2 = ep__nr > 2 ? RDL((uint32_t)(ep__r + 2 - ep__lo)) : ep__l0;        \
            svfloat64_t ep__h2 = ep__nr > 2 ? RDH((uint32_t)(ep__r + 2 - ep__lo)) : ep__h0;        \
            ep_block_rowmajor_f64((p64), (plo), (phi), (dst), (row_stride), ep__l0, ep__h0, ep__l1,\
                                  ep__h1, ep__l2, ep__h2, ep__l0, ep__h0, ep__nr, (vb), (va),      \
                                  (read_dst), (nodes), (n_nodes), (i_base), ep__r, (n0));          \
        }                                                                                          \
    } while (0)


#endif // SME_GEMM_EPILOGUE_F64_H
