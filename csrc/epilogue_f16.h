// f16 in-register activations for the tile epilogue (epilogue_tile.h).
// Included by epilogue.h, which defines the shared enums, EpNode/ep_desc
// structs, and helper macros these evaluators use.
#ifndef SME_GEMM_EPILOGUE_F16_H
#define SME_GEMM_EPILOGUE_F16_H

// Rational tanh on a live f16 slice: t*(27+t^2)/(27+9t^2), clamped. This same
// rational backs GELU, SiLU, and the bare tanh/sigmoid activations: end-to-end
// error ~4e-2 worst-case vs libm (measured) -- masked by the envelope for
// gelu/silu, but directly exposed for a bare .tanh()/.sigmoid() epilogue.
// Acceptable for fused f16 inference. The input clamp keeps t^2 from
// overflowing f16 and saturates |tanh|->1.

// Exotic activations upcast to f32 and reuse the f32 transcendentals
// forward-declared in epilogue.h.
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

#endif // SME_GEMM_EPILOGUE_F16_H
