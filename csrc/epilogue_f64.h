// f64 in-register activations and transcendentals for the tile epilogue
// (epilogue_tile.h). Included by epilogue.h, which defines the shared enums,
// EpNode/ep_desc structs, and helper macros these evaluators use.
#ifndef SME_GEMM_EPILOGUE_F64_H
#define SME_GEMM_EPILOGUE_F64_H

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
__attribute__((noinline)) static inline svfloat64_t
ep_act_rational_f64(svbool_t p64, svfloat64_t x, int act, double alpha) __arm_streaming {
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

#endif // SME_GEMM_EPILOGUE_F64_H
