// f32 in-register epilogue, including the vector transcendentals every dtype reuses.
// Included by epilogue.h, which defines the shared enums, EpNode/ep_desc
// structs, and helper macros these evaluators use.
#ifndef SME_GEMM_EPILOGUE_F32_H
#define SME_GEMM_EPILOGUE_F32_H

// --- f32 vectorized op-graph (FMOPA f32 accumulator domain) -----------------

// --- f32 vector transcendentals (exp / log / erf) ---------------------------
//
// Streaming-SVE approximations in f32 working precision, pulled out of line
// (noinline, like ep_act_rational_*) so they never bloat the node loop. Accuracy
// target ~1e-4 relative for the inference regime, matching the spirit of the
// rational tanh. exp uses range reduction by ln2 + a degree-5 minimax poly; log
// extracts the f32 exponent via integer bit tricks + a degree-5 poly on the
// mantissa; erf is Abramowitz-Stegun 7.1.26 (rational, ~1.5e-7 max abs error).

// exp(x) ~= 2^k * P(r), x = k*ln2 + r, |r| <= ln2/2.
static inline svfloat32_t ep_exp_f32(svbool_t p32, svfloat32_t x) __arm_streaming {
    // exp is finite up to ln(FLT_MAX) = 88.72284; above that the true result is
    // +inf. Mask that region (on the original x) and force +inf afterwards,
    // rather than saturating to a large finite value early: clamping at 88.0
    // instead loses up to ~2x in (88, 88.72] and never reaches inf. Clamp the
    // poly argument to the finite range to keep the reduction well-conditioned.
    svbool_t over = svcmpgt_n_f32(p32, x, 88.72283905f);
    // Below the low clamp the true result has underflowed to ~0; the -88 clamp
    // alone would pin every such input to exp(-88) ~= 6e-39, so mask x < -88 to
    // 0.0 afterwards (matching scalar expf, which underflows large-negative x to
    // 0). Symmetric to the `over` -> +inf select. Also yields exp(-inf) = 0.
    svbool_t under = svcmplt_n_f32(p32, x, -88.0f);
    // NaN is neither over nor under, so it survives the clamp below into the
    // range reduction, where FCVTZS(NaN) -> 0 makes the 2^k assembly produce a
    // finite scale. Capture NaN now and force it back at the end, matching
    // scalar expf and the ep_log NaN guard. (The clamp uses svmax/svmin = Arm
    // FMAX/FMIN, which PROPAGATE NaN -- not maxNum -- so it cannot quiet it.)
    svbool_t isnan = svcmpne_f32(p32, x, x); // NaN is the only value != itself
    x = svmin_n_f32_x(p32, svmax_n_f32_x(p32, x, -88.0f), 88.72283905f);
    const float LOG2E = 1.4426950408889634f;
    const float LN2_HI = 0.6931471824645996f;   // high part of ln2 (f32-exact)
    const float LN2_LO = -1.904654323148236e-9f; // low correction
    svfloat32_t kf = svrintn_f32_x(p32, svmul_n_f32_x(p32, x, LOG2E)); // round to nearest int
    // r = x - k*ln2 (two-step for accuracy)
    svfloat32_t r = svmls_n_f32_x(p32, x, kf, LN2_HI);
    r = svmls_n_f32_x(p32, r, kf, LN2_LO);
    // P(r) = 1 + r + r^2/2 + ... degree-5 minimax (exp on [-ln2/2, ln2/2]).
    svfloat32_t poly = svdup_n_f32(1.9875691500e-4f);
    poly = svmad_n_f32_x(p32, poly, r, 1.3981999507e-3f);
    poly = svmad_n_f32_x(p32, poly, r, 8.3334519073e-3f);
    poly = svmad_n_f32_x(p32, poly, r, 4.1665795894e-2f);
    poly = svmad_n_f32_x(p32, poly, r, 1.6666665459e-1f);
    poly = svmad_n_f32_x(p32, poly, r, 5.0000001201e-1f);
    // e = 1 + r + r^2*poly
    svfloat32_t r2 = svmul_f32_x(p32, r, r);
    svfloat32_t e = svadd_n_f32_x(p32, svadd_f32_x(p32, svmul_f32_x(p32, r2, poly), r), 1.0f);
    // 2^k in TWO halves so the exponent field never overflows for a finite
    // result (a single (k+127)<<23 hits the inf bit pattern at k=128, which the
    // true overflow boundary k=128 reaches while the value is still finite).
    svint32_t ki = svcvt_s32_f32_x(p32, kf);
    svint32_t k1 = svasr_n_s32_x(p32, ki, 1);
    svint32_t k2 = svsub_s32_x(p32, ki, k1);
    svfloat32_t p1 = svreinterpret_f32_s32(svlsl_n_s32_x(p32, svadd_n_s32_x(p32, k1, 127), 23));
    svfloat32_t p2 = svreinterpret_f32_s32(svlsl_n_s32_x(p32, svadd_n_s32_x(p32, k2, 127), 23));
    svfloat32_t res = svmul_f32_x(p32, svmul_f32_x(p32, e, p1), p2);
    res = svsel_f32(over, svreinterpret_f32_u32(svdup_n_u32(0x7f800000u)), res);  // +inf above
    res = svsel_f32(under, svdup_n_f32(0.0f), res);                               // underflow -> 0
    return svsel_f32(isnan, svreinterpret_f32_u32(svdup_n_u32(0x7fc00000u)), res); // NaN -> NaN
}

// log(x) ~= e*ln2 + poly(m), x = 2^e * m, m in [sqrt(1/2), sqrt(2)).
static inline svfloat32_t ep_log_f32(svbool_t p32, svfloat32_t x) __arm_streaming {
    // Domain guards (computed on the original x): log(x<0) = NaN, log(0) = -inf,
    // log(+inf) = +inf, log(NaN) = NaN. Without these the bit-tricks below return
    // finite garbage (e.g. the sign bit lands in the exponent field, or +inf/NaN
    // decode to a finite ~88.7) -- a real bug for an all-zero / overflowed
    // accumulator row with a .log() epilogue, and one that diverged from libm.
    svbool_t neg = svcmplt_n_f32(p32, x, 0.0f);
    svbool_t zero = svcmpeq_n_f32(p32, x, 0.0f);
    svbool_t posinf = svcmpeq_f32(p32, x, svreinterpret_f32_u32(svdup_n_u32(0x7f800000u)));
    svbool_t isnan = svcmpne_f32(p32, x, x); // NaN is the only value != itself
    // Subnormals have a zero exponent field, so the plain extraction is wrong by
    // up to ~4. Scale them up by 2^24 (exact) into the normal range and subtract
    // 24*ln2 from the result. FLT_MIN = 2^-126 = 1.1754944e-38.
    svbool_t sub = svcmplt_n_f32(p32, x, 1.1754943508222875e-38f);
    x = svsel_f32(sub, svmul_n_f32_x(p32, x, 16777216.0f), x); // *2^24 for subnormals
    // Decompose x = 2^e * m via the IEEE-754 exponent field (use u32 for the
    // logical shift; svlsr is defined only for unsigned lanes).
    svuint32_t xu = svreinterpret_u32_f32(x);
    svint32_t ei = svsub_n_s32_x(p32, svreinterpret_s32_u32(svlsr_n_u32_x(p32, xu, 23)), 127);
    // mantissa in [1,2): clear exponent bits, set to bias 127.
    svuint32_t mu = svorr_n_u32_x(p32, svand_n_u32_x(p32, xu, 0x007fffffu), 0x3f800000u);
    svfloat32_t m = svreinterpret_f32_u32(mu);
    svfloat32_t ef = svcvt_f32_s32_x(p32, ei);
    // If m >= sqrt(2) (~1.4142), halve it and bump e (keeps the poly argument
    // centered around 0 for accuracy).
    svbool_t big = svcmpge_n_f32(p32, m, 1.4142135623730951f);
    m = svsel_f32(big, svmul_n_f32_x(p32, m, 0.5f), m);
    ef = svadd_f32_x(p32, ef, svsel_f32(big, svdup_n_f32(1.0f), svdup_n_f32(0.0f)));
    svfloat32_t f = svsub_n_f32_x(p32, m, 1.0f); // f in [-0.293, 0.414]
    // atanh-form range reduction: s = f/(2+f), so log(1+f) = 2*atanh(s) =
    // 2s + 2s^3*(1/3 + 1/5 z + 1/7 z^2 + ...), z = s^2 (|s| <= ~0.172, so the
    // odd series converges fast). Symmetric in s -> ~1e-9 with few terms.
    svfloat32_t s = svdiv_f32_x(p32, f, svadd_n_f32_x(p32, f, 2.0f));
    svfloat32_t z = svmul_f32_x(p32, s, s);
    // P(z) = 1/3 + 1/5 z + 1/7 z^2 + 1/9 z^3 (minimax-rounded reciprocals).
    svfloat32_t p = svdup_n_f32(0.11111110f);  // ~1/9
    p = svmad_n_f32_x(p32, p, z, 0.14285722f); // ~1/7
    p = svmad_n_f32_x(p32, p, z, 0.19999999f); // 1/5
    p = svmad_n_f32_x(p32, p, z, 0.33333333f); // 1/3
    // log(1+f) = 2s + 2s*z*P(z) = 2s*(1 + z*P(z))
    svfloat32_t lm = svmul_n_f32_x(p32, svmul_f32_x(p32, s, svmad_f32_x(p32, z, p, svdup_n_f32(1.0f))),
                                   2.0f);
    const float LN2 = 0.6931471805599453f;
    svfloat32_t res = svmla_n_f32_x(p32, lm, ef, LN2);
    // Undo the 2^24 subnormal scaling, then apply the domain selects.
    res = svsub_f32_x(p32, res,
                      svsel_f32(sub, svdup_n_f32(24.0f * LN2), svdup_n_f32(0.0f)));
    res = svsel_f32(zero, svreinterpret_f32_u32(svdup_n_u32(0xff800000u)), res);   // -inf
    res = svsel_f32(neg, svreinterpret_f32_u32(svdup_n_u32(0x7fc00000u)), res);    // NaN
    res = svsel_f32(posinf, svreinterpret_f32_u32(svdup_n_u32(0x7f800000u)), res); // +inf
    res = svsel_f32(isnan, svreinterpret_f32_u32(svdup_n_u32(0x7fc00000u)), res);  // NaN
    return res;
}

// erf(x) via Abramowitz-Stegun 7.1.26: for x>=0, erf = 1 - (a1*t+...+a5*t^5)*e^{-x^2},
// t = 1/(1+p*x). Odd extension for x<0. ~1.5e-7 max abs error.
static inline svfloat32_t ep_erf_f32(svbool_t p32, svfloat32_t x) __arm_streaming {
    svfloat32_t ax = svabs_f32_x(p32, x);
    const float P = 0.3275911f;
    svfloat32_t t = svdiv_f32_x(p32, svdup_n_f32(1.0f),
                                svadd_n_f32_x(p32, svmul_n_f32_x(p32, ax, P), 1.0f));
    // poly(t) = ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
    svfloat32_t poly = svdup_n_f32(1.061405429f);
    poly = svmad_n_f32_x(p32, poly, t, -1.453152027f);
    poly = svmad_n_f32_x(p32, poly, t, 1.421413741f);
    poly = svmad_n_f32_x(p32, poly, t, -0.284496736f);
    poly = svmad_n_f32_x(p32, poly, t, 0.254829592f);
    poly = svmul_f32_x(p32, poly, t);
    svfloat32_t ex2 = ep_exp_f32(p32, svneg_f32_x(p32, svmul_f32_x(p32, ax, ax)));
    svfloat32_t e = svsub_f32_x(p32, svdup_n_f32(1.0f), svmul_f32_x(p32, poly, ex2));
    // Apply the sign of x (erf is odd).
    return svsel_f32(svcmplt_n_f32(p32, x, 0.0f), svneg_f32_x(p32, e), e);
}

// softplus(x) = max(x,0) + log1p(exp(-|x|)), numerically stable.
static inline svfloat32_t ep_softplus_f32(svbool_t p32, svfloat32_t x) __arm_streaming {
    svfloat32_t ax = svabs_f32_x(p32, x);
    svfloat32_t e = ep_exp_f32(p32, svneg_f32_x(p32, ax));
    svfloat32_t l = ep_log_f32(p32, svadd_n_f32_x(p32, e, 1.0f)); // log1p(exp(-|x|))
    return svadd_f32_x(p32, svmax_n_f32_x(p32, x, 0.0f), l);
}

// Rational tanh on a live f32 slice: f32 has svdiv, so this uses the same form
// as the f16 path but at single precision with a wider clamp (no f16 overflow
// concern). End-to-end GELU/SiLU/tanh error ~4e-2 worst-case, dominated by the
// rational form; acceptable for fused f32 inference.
static inline svfloat32_t ep_tanh_f32(svbool_t p32, svfloat32_t t) __arm_streaming {
    t = svmin_n_f32_x(p32, svmax_n_f32_x(p32, t, -9.0f), 9.0f);
    svfloat32_t t2 = svmul_f32_x(p32, t, t);
    svfloat32_t num = svadd_n_f32_x(p32, t2, 27.0f);
    svfloat32_t den = svadd_n_f32_x(p32, svmul_n_f32_x(p32, t2, 9.0f), 27.0f);
    svfloat32_t th = svmul_f32_x(p32, t, svdiv_f32_x(p32, num, den));
    return svmin_n_f32_x(p32, svmax_n_f32_x(p32, th, -1.0f), 1.0f);
}

// Rational GELU/SiLU/TANH/SIGMOID + the transcendental Group-B activations +
// the divide/sqrt Group-A activations, all pulled out of line (mirrors the f16
// split): this is what bloats the epilogue past the inliner, so keeping it here
// lets the common add/mul/relu node path inline cleanly into the node-major tile
// loops. `alpha` is the LEAKY_RELU/ELU parameter (ignored by other kinds).
__attribute__((noinline)) static svfloat32_t ep_act_rational_f32(svbool_t p32, svfloat32_t x,
                                                                 int act,
                                                                 float alpha) __arm_streaming {
    if (act == EP_ACT_GELU) {
        svfloat32_t x2 = svmul_f32_x(p32, x, x);
        svfloat32_t inner = svmul_f32_x(
            p32, x, svadd_n_f32_x(p32, svmul_n_f32_x(p32, x2, 0.044715f), 1.0f));
        svfloat32_t t = svmul_n_f32_x(p32, inner, 0.7978845608f);
        svfloat32_t th = ep_tanh_f32(p32, t);
        return svmul_n_f32_x(p32, svmul_f32_x(p32, x, svadd_n_f32_x(p32, th, 1.0f)), 0.5f);
    }
    if (act == EP_ACT_SILU) {
        svfloat32_t th = ep_tanh_f32(p32, svmul_n_f32_x(p32, x, 0.5f));
        return svmul_n_f32_x(p32, svmul_f32_x(p32, x, svadd_n_f32_x(p32, th, 1.0f)), 0.5f);
    }
    if (act == EP_ACT_TANH) {
        return ep_tanh_f32(p32, x);
    }
    if (act == EP_ACT_SIGMOID) {
        svfloat32_t th = ep_tanh_f32(p32, svmul_n_f32_x(p32, x, 0.5f));
        return svmul_n_f32_x(p32, svadd_n_f32_x(p32, th, 1.0f), 0.5f);
    }
    // --- Group A (exact, but out-of-line to keep the node loop slim) ---
    if (act == EP_ACT_LEAKY_RELU) {
        svfloat32_t pos = svmax_n_f32_x(p32, x, 0.0f);
        svfloat32_t neg = svmin_n_f32_x(p32, x, 0.0f);
        return svmla_n_f32_x(p32, pos, neg, alpha);
    }
    if (act == EP_ACT_RELU6) {
        return svmin_n_f32_x(p32, svmax_n_f32_x(p32, x, 0.0f), 6.0f);
    }
    if (act == EP_ACT_HARDSIGMOID) {
        svfloat32_t h = svadd_n_f32_x(p32, svmul_n_f32_x(p32, x, 1.0f / 6.0f), 0.5f);
        return svmin_n_f32_x(p32, svmax_n_f32_x(p32, h, 0.0f), 1.0f);
    }
    if (act == EP_ACT_HARDSWISH) {
        svfloat32_t h = svadd_n_f32_x(p32, svmul_n_f32_x(p32, x, 1.0f / 6.0f), 0.5f);
        h = svmin_n_f32_x(p32, svmax_n_f32_x(p32, h, 0.0f), 1.0f);
        return svmul_f32_x(p32, x, h);
    }
    if (act == EP_ACT_ABS) {
        return svabs_f32_x(p32, x);
    }
    if (act == EP_ACT_NEG) {
        return svneg_f32_x(p32, x);
    }
    if (act == EP_ACT_SQUARE) {
        return svmul_f32_x(p32, x, x);
    }
    if (act == EP_ACT_SIGN) {
        svfloat32_t pos = svsel_f32(svcmpgt_n_f32(p32, x, 0.0f), svdup_n_f32(1.0f), svdup_n_f32(0.0f));
        return svsel_f32(svcmplt_n_f32(p32, x, 0.0f), svdup_n_f32(-1.0f), pos);
    }
    if (act == EP_ACT_SQRT) {
        return svsqrt_f32_x(p32, x);
    }
    if (act == EP_ACT_SOFTSIGN) {
        return svdiv_f32_x(p32, x, svadd_n_f32_x(p32, svabs_f32_x(p32, x), 1.0f));
    }
    if (act == EP_ACT_RECIP) {
        return svdiv_f32_x(p32, svdup_n_f32(1.0f), x);
    }
    if (act == EP_ACT_RSQRT) {
        return svdiv_f32_x(p32, svdup_n_f32(1.0f), svsqrt_f32_x(p32, x));
    }
    // --- Group B (transcendental) ---
    if (act == EP_ACT_EXP) {
        return ep_exp_f32(p32, x);
    }
    if (act == EP_ACT_LOG) {
        return ep_log_f32(p32, x);
    }
    if (act == EP_ACT_ELU) {
        // x>=0 ? x : alpha*(exp(x)-1)
        svfloat32_t em1 = svsub_n_f32_x(p32, ep_exp_f32(p32, x), 1.0f);
        svfloat32_t neg = svmul_n_f32_x(p32, em1, alpha);
        return svsel_f32(svcmpge_n_f32(p32, x, 0.0f), x, neg);
    }
    if (act == EP_ACT_SELU) {
        svfloat32_t em1 = svsub_n_f32_x(p32, ep_exp_f32(p32, x), 1.0f);
        svfloat32_t neg = svmul_n_f32_x(p32, em1, EP_SELU_ALPHA);
        svfloat32_t v = svsel_f32(svcmpge_n_f32(p32, x, 0.0f), x, neg);
        return svmul_n_f32_x(p32, v, EP_SELU_LAMBDA);
    }
    if (act == EP_ACT_SOFTPLUS) {
        return ep_softplus_f32(p32, x);
    }
    if (act == EP_ACT_MISH) {
        svfloat32_t sp = ep_softplus_f32(p32, x);
        return svmul_f32_x(p32, x, ep_tanh_f32(p32, sp));
    }
    if (act == EP_ACT_GELU_EXACT) {
        svfloat32_t e = ep_erf_f32(p32, svmul_n_f32_x(p32, x, 0.7071067811865476f));
        return svmul_n_f32_x(p32, svmul_f32_x(p32, x, svadd_n_f32_x(p32, e, 1.0f)), 0.5f);
    }
    return x;
}

static inline svfloat32_t ep_act_apply_f32(svbool_t p32, svfloat32_t x, int act,
                                           float alpha) __arm_streaming {
    if (act == EP_ACT_RELU) {
        return svmaxnm_n_f32_x(p32, x, 0.0f);
    }
    if (act == EP_ACT_NONE) {
        return x;
    }
    return ep_act_rational_f32(p32, x, act, alpha);
}

// Walk the op-graph over one live f32 vector `x`: the integer kernels'
// dequantized slices. span_n != 0 (row-major dst, vector spans N) -> COL/TENSOR
// are vector loads, ROW/SCALAR splats; span_n == 0 (col-major dst, vector spans
// M) -> ROW is the vector load, COL/SCALAR splats; TENSOR routes to the scalar
// path.
static inline svfloat32_t ep_apply_nodes_f32(svbool_t p32, svbool_t pst, svfloat32_t x,
                                             const EpNode *nodes, uint32_t n_nodes, int span_n,
                                             size_t i, size_t n0, size_t m0) __arm_streaming {
    for (uint32_t t = 0; t < n_nodes; t++) {
        const EpNode *nd = &nodes[t];
        const float *p = (const float *)nd->ptr;
        switch (nd->op) {
            case EP_OP_ADD_SCALAR: x = svadd_n_f32_x(p32, x, nd->scalar); break;
            case EP_OP_MUL_SCALAR: x = svmul_n_f32_x(p32, x, nd->scalar); break;
            case EP_OP_ADD_ROW:
                if (span_n) x = svadd_n_f32_x(p32, x, p[i]);
                else x = svadd_f32_x(p32, x, svld1_f32(pst, p + m0));
                break;
            case EP_OP_MUL_ROW:
                if (span_n) x = svmul_n_f32_x(p32, x, p[i]);
                else x = svmul_f32_x(p32, x, svld1_f32(pst, p + m0));
                break;
            case EP_OP_ADD_COL:
                if (span_n) x = svadd_f32_x(p32, x, svld1_f32(pst, p + n0));
                else x = svadd_n_f32_x(p32, x, p[i]);
                break;
            case EP_OP_MUL_COL:
                if (span_n) x = svmul_f32_x(p32, x, svld1_f32(pst, p + n0));
                else x = svmul_n_f32_x(p32, x, p[i]);
                break;
            case EP_OP_ADD_TENSOR:
                x = svadd_f32_x(p32, x, svld1_f32(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_MUL_TENSOR:
                x = svmul_f32_x(p32, x, svld1_f32(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_ACT: x = ep_act_apply_f32(p32, x, (int)nd->aux, nd->scalar); break;
            case EP_OP_MAX_SCALAR: x = svmaxnm_n_f32_x(p32, x, nd->scalar); break;
            case EP_OP_MIN_SCALAR: x = svminnm_n_f32_x(p32, x, nd->scalar); break;
            case EP_OP_SUB_SCALAR: x = svsub_n_f32_x(p32, x, nd->scalar); break;
            case EP_OP_DIV_SCALAR: x = EP_DIV_SPLAT_F32(p32, x, nd->scalar); break;
            case EP_OP_SUB_ROW:
                if (span_n) x = svsub_n_f32_x(p32, x, p[i]);
                else x = svsub_f32_x(p32, x, svld1_f32(pst, p + m0));
                break;
            case EP_OP_DIV_ROW:
                if (span_n) x = EP_DIV_SPLAT_F32(p32, x, p[i]);
                else x = svdiv_f32_x(p32, x, svld1_f32(pst, p + m0));
                break;
            case EP_OP_MAX_ROW:
                if (span_n) x = svmaxnm_n_f32_x(p32, x, p[i]);
                else x = svmaxnm_f32_x(p32, x, svld1_f32(pst, p + m0));
                break;
            case EP_OP_MIN_ROW:
                if (span_n) x = svminnm_n_f32_x(p32, x, p[i]);
                else x = svminnm_f32_x(p32, x, svld1_f32(pst, p + m0));
                break;
            case EP_OP_SUB_COL:
                if (span_n) x = svsub_f32_x(p32, x, svld1_f32(pst, p + n0));
                else x = svsub_n_f32_x(p32, x, p[i]);
                break;
            case EP_OP_DIV_COL:
                if (span_n) x = svdiv_f32_x(p32, x, svld1_f32(pst, p + n0));
                else x = EP_DIV_SPLAT_F32(p32, x, p[i]);
                break;
            case EP_OP_MAX_COL:
                if (span_n) x = svmaxnm_f32_x(p32, x, svld1_f32(pst, p + n0));
                else x = svmaxnm_n_f32_x(p32, x, p[i]);
                break;
            case EP_OP_MIN_COL:
                if (span_n) x = svminnm_f32_x(p32, x, svld1_f32(pst, p + n0));
                else x = svminnm_n_f32_x(p32, x, p[i]);
                break;
            case EP_OP_SUB_TENSOR:
                x = svsub_f32_x(p32, x, svld1_f32(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_DIV_TENSOR:
                x = svdiv_f32_x(p32, x, svld1_f32(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_MAX_TENSOR:
                x = svmaxnm_f32_x(p32, x, svld1_f32(pst, p + i * nd->ld + n0));
                break;
            case EP_OP_MIN_TENSOR:
                x = svminnm_f32_x(p32, x, svld1_f32(pst, p + i * nd->ld + n0));
                break;
            default: break;
        }
    }
    return x;
}

#endif // SME_GEMM_EPILOGUE_F32_H
