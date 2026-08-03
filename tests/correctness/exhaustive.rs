//! Exhaustive activation / elementwise op coverage, f64 oracle.

use half::{bf16, f16};
use sme_gemm::{Gemm, caps, prepack_f32};

fn erf64(x: f64) -> f64 {
    // A&S 7.1.26 (same as the lib + kernel approximation), for the oracle of
    // gelu_exact -- this matches the implementation's approximation, not
    // libm's exact erf, so the test checks consistency at a reasonable tol.
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    let e = 1.0 - poly * (-ax * ax).exp();
    if x < 0.0 { -e } else { e }
}

// Apply one unary op to a scalar in f64 (the oracle).
fn unary(name: &str, x: f64) -> f64 {
    match name {
        "leaky" => {
            if x >= 0.0 {
                x
            } else {
                0.1 * x
            }
        }
        "relu6" => x.clamp(0.0, 6.0),
        "hardsigmoid" => (x / 6.0 + 0.5).clamp(0.0, 1.0),
        "hardswish" => x * (x / 6.0 + 0.5).clamp(0.0, 1.0),
        "abs" => x.abs(),
        "neg" => -x,
        "square" => x * x,
        "sign" => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        "sqrt" => x.abs().sqrt(), // we feed |.| via a prior abs in the graph
        "softsign" => x / (1.0 + x.abs()),
        "recip" => 1.0 / x,
        "rsqrt" => 1.0 / x.abs().sqrt(),
        "exp" => x.exp(),
        "log" => x.abs().ln(),
        "elu" => {
            if x >= 0.0 {
                x
            } else {
                0.5 * x.exp_m1()
            }
        }
        "selu" => {
            1.050_700_987_355_480_5
                * if x >= 0.0 {
                    x
                } else {
                    1.673_263_242_354_377_2 * x.exp_m1()
                }
        }
        "softplus" => x.max(0.0) + (-x.abs()).exp().ln_1p(),
        "mish" => x * (x.max(0.0) + (-x.abs()).exp().ln_1p()).tanh(),
        "gelu_exact" => 0.5 * x * (1.0 + erf64(x * std::f64::consts::FRAC_1_SQRT_2)),
        _ => unreachable!(),
    }
}

fn rnd(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
}

fn oracle(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
            }
            c[i * n + j] = acc;
        }
    }
    c
}

const SIZES: &[(usize, usize, usize)] = &[(32, 32, 16), (33, 17, 9), (64, 48, 24), (7, 5, 3)];

// f32: every unary op vs the f64 oracle. Group A: tight tol; Group B: loose.
// `chain` builds the op onto a fresh Gemm; `abs_in` prepends .abs() so domain-
// restricted ops (sqrt/log/rsqrt) get a non-negative argument.
#[test]
fn f32_unary_ops() {
    if !caps().sme {
        return;
    }
    macro_rules! run {
        ($name:expr, $g:ident, $chain:expr, $abs_in:expr, $tol:expr) => {{
            for &(m, n, k) in SIZES {
                let mut s = 0x0fe1_a55e_0055_u64.wrapping_add((m * 131 + n * 17 + k) as u64);
                let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut s)).collect();
                let b: Vec<f32> = (0..k * n).map(|_| rnd(&mut s)).collect();
                let base = oracle(&a, &b, m, n, k);
                let packed = prepack_f32(&b, n, k);
                let mut c = vec![0.0f32; m * n];
                let $g = Gemm::new(&a, &packed, m);
                let $g = if $abs_in { $g.abs() } else { $g };
                ($chain).run(&mut c);
                let mut mr = 0.0f64;
                for i in 0..m {
                    for j in 0..n {
                        let mut xin = base[i * n + j];
                        if $abs_in {
                            xin = xin.abs();
                        }
                        let want = unary($name, xin);
                        if $name == "recip" && want.abs() > 1e3 {
                            continue; // 1/x blows up near 0; skip
                        }
                        let got = f64::from(c[i * n + j]);
                        mr = mr.max((got - want).abs() / (1.0 + want.abs()));
                    }
                }
                assert!(
                    mr < $tol,
                    "f32 {} {m}x{n}x{k}: max_rel={mr} >= {}",
                    $name,
                    $tol
                );
            }
        }};
    }
    // Group A (exact): tight tol.
    run!("leaky", g, g.leaky_relu(0.1), false, 2e-3);
    run!("relu6", g, g.relu6(), false, 2e-3);
    run!("hardsigmoid", g, g.hardsigmoid(), false, 2e-3);
    run!("hardswish", g, g.hardswish(), false, 2e-3);
    run!("abs", g, g.abs(), false, 2e-3);
    run!("neg", g, g.neg(), false, 2e-3);
    run!("square", g, g.square(), false, 2e-3);
    run!("sign", g, g.sign(), false, 2e-3);
    run!("softsign", g, g.softsign(), false, 2e-3);
    run!("recip", g, g.recip(), false, 2e-3);
    run!("sqrt", g, g.sqrt(), true, 2e-3);
    run!("rsqrt", g, g.rsqrt(), true, 2e-3);
    // Group B: the f32 minimax exp/log + A&S erf are near machine precision
    // (exp/elu/selu/softplus/gelu_exact ~1-2e-7 rel; log ~2e-5, limited by
    // accumulator conditioning for near-zero inputs, not the polynomial). Tol
    // is set just above the measured worst case to actually assert the accuracy.
    run!("exp", g, g.exp(), false, 1e-6);
    run!("log", g, g.log(), true, 1e-4);
    run!("elu", g, g.elu(0.5), false, 1e-6);
    run!("selu", g, g.selu(), false, 1e-6);
    run!("softplus", g, g.softplus(), false, 1e-6);
    // mish = x*tanh(softplus(x)), so it inherits the rational tanh's ~4e-2 abs
    // envelope (csrc/epilogue.h) -- unlike the rest of Group B, which is pure
    // exp/log/erf. Measured 1.3e-2 here. A 1e-6 bound would only hold if these
    // shapes fell below the floor and ran the exact SCALAR evaluator instead of
    // the kernel.
    run!("mish", g, g.mish(), false, 5e-2);
    run!("gelu_exact", g, g.gelu_exact(), false, 1e-6);
}

// Boundary behavior of the vectorized log/exp: log(0) = -inf, log(x<0) = NaN,
// and exp saturates to +inf only past ln(FLT_MAX), staying finite just below it.
// Drive the VECTORIZED SME epilogue (ep_log_f32 / ep_exp_f32), NOT the scalar
// libm reference. That requires the kernel actually run: sme_worth_it needs
// k >= 2 AND m*n*k >= 2^18, so a k=1 probe would route to libm and test nothing
// in-kernel. Build a real SME GEMM (128^3 = 2^21 flops) whose
// every output cell equals vals[j]: A row = [1, 0, ..., 0], B row 0 = vals
// (zero-padded), other B rows = 0, so acc[i][j] = 1*vals[j] = vals[j] exactly
// -- inf/NaN included (1*x == x; the zero terms are 0*0 = 0, never inf*0).
// Returns row 0, cols 0..len.
fn log_exp_probe(vals: &[f32], which: &str) -> Vec<f32> {
    let len = vals.len();
    let (m, n, k) = (128usize, 128usize, 128usize);
    assert!(len <= n, "probe values must fit in n columns");
    let mut a = vec![0.0f32; m * k];
    for row in a.chunks_exact_mut(k) {
        row[0] = 1.0; // column 0 = 1, the rest 0
    }
    let mut b = vec![0.0f32; k * n]; // k x n row-major
    b[..len].copy_from_slice(vals); // row 0 holds the probe values
    let packed = prepack_f32(&b, n, k);
    let mut c = vec![0.0f32; m * n];
    let g = Gemm::new(&a, &packed, m);
    match which {
        "log" => g.log().run(&mut c),
        "exp" => g.exp().run(&mut c),
        _ => unreachable!(),
    }
    c[..len].to_vec()
}

#[test]
fn f32_log_exp_boundaries() {
    if !caps().sme {
        return;
    }
    let probe = log_exp_probe;

    // log domain, including +inf/NaN, which a bit-trick decode turns into a
    // finite ~88.7.
    let c = probe(
        &[
            0.0,
            -1.0,
            -4.0,
            1.0,
            std::f32::consts::E,
            f32::INFINITY,
            f32::NAN,
        ],
        "log",
    );
    assert!(
        c[0].is_infinite() && c[0].is_sign_negative(),
        "log(0) must be -inf, got {}",
        c[0]
    );
    assert!(c[1].is_nan(), "log(-1) must be NaN, got {}", c[1]);
    assert!(c[2].is_nan(), "log(-4) must be NaN, got {}", c[2]);
    assert!((c[3] - 0.0).abs() < 1e-4, "log(1) ~ 0, got {}", c[3]);
    assert!((c[4] - 1.0).abs() < 1e-4, "log(e) ~ 1, got {}", c[4]);
    assert!(
        c[5].is_infinite() && c[5].is_sign_positive(),
        "log(+inf) must be +inf, got {}",
        c[5]
    );
    assert!(c[6].is_nan(), "log(NaN) must be NaN, got {}", c[6]);

    // exp overflow boundary: finite just below ln(FLT_MAX)=88.7228, +inf above;
    // NaN propagates; -inf and very-negative underflow to 0.
    let c = probe(
        &[
            88.0,
            88.5,
            89.0,
            100.0,
            f32::NAN,
            f32::NEG_INFINITY,
            -1000.0,
        ],
        "exp",
    );
    assert!(
        c[0].is_finite() && c[0] > 1.6e38,
        "exp(88) finite-large, got {}",
        c[0]
    );
    assert!(c[1].is_finite(), "exp(88.5) must stay finite, got {}", c[1]);
    // true exp(88.5) = 2.51e38; an 88.0 clamp returns ~1.65e38, ~1.5x low.
    assert!(
        c[1] > 2.0e38,
        "exp(88.5) must be ~2.5e38, not clamped low: {}",
        c[1]
    );
    assert!(
        c[2].is_infinite() && c[2].is_sign_positive(),
        "exp(89) must be +inf, got {}",
        c[2]
    );
    assert!(
        c[3].is_infinite() && c[3].is_sign_positive(),
        "exp(100) must be +inf, got {}",
        c[3]
    );
    // The clamp quiets NaN to -88 via min/maxNum; without the isnan guard this
    // returned a finite ~exp(-88) instead of NaN.
    assert!(c[4].is_nan(), "exp(NaN) must be NaN, got {}", c[4]);
    assert!(
        c[5] == 0.0 && c[5].is_sign_positive(),
        "exp(-inf) must be +0, got {}",
        c[5]
    );
    assert!(c[6] == 0.0, "exp(-1000) must underflow to 0, got {}", c[6]);

    // log of a subnormal accumulator: the scaled-extraction path stays accurate
    // (not garbage). Assumes the f32 FMOPA accumulator is not flushed to zero.
    let sub = f32::from_bits(0x0000_4000); // a subnormal ~ 2.3e-41
    let c = probe(&[sub], "log");
    let want = f64::from(sub).ln();
    assert!(
        (f64::from(c[0]) - want).abs() < 0.5,
        "log(subnormal) ~ {want}, got {}",
        c[0]
    );
}

// f16: the transcendental ops (upcast-to-f32 path) vs the f64 oracle, loose
// tol (f16 storage + approximation).
#[test]
fn f16_transcendental_ops() {
    use sme_gemm::{Gemm as G16, prepack_f16};
    if !caps().sme {
        return;
    }
    macro_rules! run16 {
        ($name:expr, $g:ident, $chain:expr, $abs_in:expr, $tol:expr) => {{
            for &(m, n, k) in SIZES {
                let mut s = 0x16e1_f00d_u64.wrapping_add((m * 131 + n * 17 + k) as u64);
                let a: Vec<f16> = (0..m * k).map(|_| f16::from_f32(rnd(&mut s))).collect();
                let b: Vec<f16> = (0..k * n).map(|_| f16::from_f32(rnd(&mut s))).collect();
                let af: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
                let bf: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
                let base = oracle(&af, &bf, m, n, k);
                let packed = prepack_f16(&b, n, k);
                let mut c = vec![f16::ZERO; m * n];
                let $g = G16::new(&a, &packed, m);
                let $g = if $abs_in { $g.abs() } else { $g };
                ($chain).run(&mut c);
                let mut mr = 0.0f64;
                for i in 0..m {
                    for j in 0..n {
                        let mut xin = base[i * n + j];
                        if $abs_in {
                            xin = xin.abs();
                        }
                        let want = unary($name, xin);
                        let got = f64::from(c[i * n + j].to_f32());
                        mr = mr.max((got - want).abs() / (1.0 + want.abs()));
                    }
                }
                assert!(
                    mr < $tol,
                    "f16 {} {m}x{n}x{k}: max_rel={mr} >= {}",
                    $name,
                    $tol
                );
            }
        }};
    }
    // The f32 polynomials are near machine precision, so f16 error is dominated
    // by f16 STORAGE rounding (~1e-3), not the approximation. Tolerances are set
    // to that storage-bound floor (just above the measured worst case): the
    // smooth ops sit at ~1e-3; sqrt/mish reach ~1.5e-2 at their knees; log of
    // near-zero accumulators is conditioning-limited (~0.11), not polynomial.
    run16!("exp", g, g.exp(), false, 2e-3);
    run16!("log", g, g.log(), true, 1.5e-1);
    run16!("elu", g, g.elu(0.5), false, 2e-3);
    run16!("selu", g, g.selu(), false, 2e-3);
    run16!("softplus", g, g.softplus(), false, 2e-3);
    run16!("mish", g, g.mish(), false, 2e-2);
    run16!("gelu_exact", g, g.gelu_exact(), false, 2e-3);
    run16!("leaky", g, g.leaky_relu(0.1), false, 2e-3);
    run16!("relu6", g, g.relu6(), false, 2e-3);
    run16!("hardswish", g, g.hardswish(), false, 2e-3);
    run16!("sqrt", g, g.sqrt(), true, 1.5e-2);
}

// f32 elementwise binary ops (sub/div/max/min vs scalar/col/tensor) vs oracle.
#[test]
fn f32_binary_ops() {
    if !caps().sme {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0xb16a_09e5_u64.wrapping_add((m * 131 + n * 17 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut s)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rnd(&mut s)).collect();
        let col: Vec<f32> = (0..n).map(|_| rnd(&mut s) + 1.5).collect(); // away from 0
        let tens: Vec<f32> = (0..m * n).map(|_| rnd(&mut s) + 1.5).collect();
        let base = oracle(&a, &b, m, n, k);
        let packed = prepack_f32(&b, n, k);

        // sub_scalar -> div_col -> max_tensor -> min_col
        let mut c = vec![0.0f32; m * n];
        Gemm::new(&a, &packed, m)
            .sub_scalar(0.25)
            .div_col(&col)
            .max_tensor(&tens)
            .min_col(&col)
            .run(&mut c);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut v = base[i * n + j] - 0.25;
                v /= f64::from(col[j]);
                v = v.max(f64::from(tens[i * n + j]));
                v = v.min(f64::from(col[j]));
                mr = mr.max((f64::from(c[i * n + j]) - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(mr < 2e-3, "f32 binary chain {m}x{n}x{k}: {mr}");
    }
}

// Composed graphs mixing new ops: mul_col -> elu -> clamp, and add_col -> mish.
#[test]
fn composed_graphs() {
    if !caps().sme {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0xc0_de_d0_0d_u64.wrapping_add((m * 131 + n * 17 + k) as u64);
        let a: Vec<f32> = (0..m * k).map(|_| rnd(&mut s)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rnd(&mut s)).collect();
        let col: Vec<f32> = (0..n).map(|_| rnd(&mut s)).collect();
        let base = oracle(&a, &b, m, n, k);
        let packed = prepack_f32(&b, n, k);

        // mul_col -> elu(0.7) -> clamp(-0.5, 2.0)
        let mut c = vec![0.0f32; m * n];
        Gemm::new(&a, &packed, m)
            .mul_col(&col)
            .elu(0.7)
            .clamp(-0.5, 2.0)
            .run(&mut c);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let x = base[i * n + j] * f64::from(col[j]);
                let e = if x >= 0.0 { x } else { 0.7 * x.exp_m1() };
                let v = e.clamp(-0.5, 2.0);
                mr = mr.max((f64::from(c[i * n + j]) - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(mr < 5e-2, "composed mul_col->elu->clamp {m}x{n}x{k}: {mr}");

        // add_col -> mish
        let mut c2 = vec![0.0f32; m * n];
        Gemm::new(&a, &packed, m).add_col(&col).mish().run(&mut c2);
        let mut mr2 = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let x = base[i * n + j] + f64::from(col[j]);
                let v = x * (x.max(0.0) + (-x.abs()).exp().ln_1p()).tanh();
                mr2 = mr2.max((f64::from(c2[i * n + j]) - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(mr2 < 5e-2, "composed add_col->mish {m}x{n}x{k}: {mr2}");
    }
}

// bf16 Group-A no-divide ops are allowed (leaky_relu, relu6, hardsigmoid,
// hardswish, abs, neg, square, and the sub/max/min elementwise ops).
// Validated against the f64 oracle; bf16's ~3-digit mantissa gives a loose
// tol scaled by sqrt(k) like the existing bf16 tests.
// unary bf16 op -> f64 oracle (leaky alpha fixed at 0.1).
fn bf16_uref(name: &str, x: f64) -> f64 {
    match name {
        "leaky" => {
            if x >= 0.0 {
                x
            } else {
                0.1 * x
            }
        }
        "relu6" => x.clamp(0.0, 6.0),
        "hardsigmoid" => (x / 6.0 + 0.5).clamp(0.0, 1.0),
        "hardswish" => x * (x / 6.0 + 0.5).clamp(0.0, 1.0),
        "abs" => x.abs(),
        "neg" => -x,
        "square" => x * x,
        // Newly enabled via the f32-computed bf16 epilogue (full op parity).
        "sign" => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        "sqrt" => x.sqrt(),
        "softsign" => x / (1.0 + x.abs()),
        "recip" => 1.0 / x,
        "rsqrt" => 1.0 / x.sqrt(),
        "exp" => x.exp(),
        "log" => x.ln(),
        "elu" => {
            if x >= 0.0 {
                x
            } else {
                0.1 * x.exp_m1()
            }
        }
        "selu" => {
            1.050_700_987_355_480_5
                * if x >= 0.0 {
                    x
                } else {
                    1.673_263_242_354_377_2 * x.exp_m1()
                }
        }
        "softplus" => x.max(0.0) + (-x.abs()).exp().ln_1p(),
        "mish" => x * (x.max(0.0) + (-x.abs()).exp().ln_1p()).tanh(),
        "gelu_exact" => 0.5 * x * (1.0 + libm_erf(x * core::f64::consts::FRAC_1_SQRT_2)),
        "gelu" => 0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh()),
        "silu" => x / (1.0 + (-x).exp()),
        "tanh" => x.tanh(),
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        _ => unreachable!(),
    }
}

// erf via Abramowitz-Stegun 7.1.26 (~1.5e-7 abs) for the gelu_exact oracle.
fn libm_erf(x: f64) -> f64 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = ((((1.061_405_429 * t - 1.453_152_027) * t + 1.421_413_741) * t - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    let e = 1.0 - poly * (-ax * ax).exp();
    if x < 0.0 { -e } else { e }
}

#[test]
fn bf16_nodivide_ops() {
    use sme_gemm::prepack_bf16;
    if !caps().sme_b16b16 {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0xb16a_0a0a_u64.wrapping_add((m * 131 + n * 17 + k) as u64);
        // scale up so accumulators span past the relu6/hardsigmoid clamp knees.
        let a: Vec<bf16> = (0..m * k)
            .map(|_| bf16::from_f32(rnd(&mut s) * 2.0))
            .collect();
        let b: Vec<bf16> = (0..k * n)
            .map(|_| bf16::from_f32(rnd(&mut s) * 2.0))
            .collect();
        let af: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let bf: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
        let base = oracle(&af, &bf, m, n, k);
        let packed = prepack_bf16(&b, n, k);
        let tol = (6e-2 * (k as f64).sqrt() / 4.0).max(6e-2);
        macro_rules! chk {
            ($name:expr, $g:ident, $chain:expr) => {{
                let mut c = vec![bf16::ZERO; m * n];
                let $g = Gemm::new(&a, &packed, m);
                ($chain).run(&mut c);
                let mut mr = 0.0f64;
                for i in 0..m {
                    for j in 0..n {
                        let v = bf16_uref($name, base[i * n + j]);
                        let got = f64::from(c[i * n + j].to_f32());
                        mr = mr.max((got - v).abs() / (1.0 + v.abs()));
                    }
                }
                assert!(mr < tol, "bf16 {} {m}x{n}x{k}: max_rel={mr}", $name);
            }};
        }
        chk!("leaky", g, g.leaky_relu(0.1));
        chk!("relu6", g, g.relu6());
        chk!("hardsigmoid", g, g.hardsigmoid());
        chk!("hardswish", g, g.hardswish());
        chk!("abs", g, g.abs());
        chk!("neg", g, g.neg());
        chk!("square", g, g.square());

        // elementwise sub_col / max_col against the f64 oracle.
        let col: Vec<bf16> = (0..n).map(|_| bf16::from_f32(rnd(&mut s))).collect();
        let mut c = vec![bf16::ZERO; m * n];
        Gemm::new(&a, &packed, m)
            .sub_col(&col)
            .max_col(&col)
            .run(&mut c);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let cj = f64::from(col[j].to_f32());
                let v = (base[i * n + j] - cj).max(cj);
                mr = mr.max((f64::from(c[i * n + j].to_f32()) - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(mr < tol, "bf16 sub_col->max_col {m}x{n}x{k}: {mr}");
    }
}

// bf16 full op-graph parity: div / sqrt / recip / rsqrt / softsign / sign and
// the Group-B transcendentals (exp/log/elu/selu/softplus/mish/gelu_exact) plus
// the rational tanh-family (gelu/silu/tanh/sigmoid) now run via the f32-
// computed bf16 epilogue (formerly the `should_panic` cases). Validated vs the
// f64 oracle. The bf16 epilogue rounds to bf16 (~3-digit mantissa) after the
// f32 transcendental, so the tol is bf16-storage dominated, scaled by sqrt(k).
#[test]
fn bf16_fullop_parity() {
    use sme_gemm::prepack_bf16;
    if !caps().sme_b16b16 {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0xb16f_0b07_u64.wrapping_add((m * 131 + n * 17 + k) as u64);
        // Keep accumulators in a moderate range so exp/log/sqrt stay well-posed.
        let a: Vec<bf16> = (0..m * k)
            .map(|_| bf16::from_f32(rnd(&mut s) * 0.5))
            .collect();
        let b: Vec<bf16> = (0..k * n)
            .map(|_| bf16::from_f32(rnd(&mut s) * 0.5))
            .collect();
        let af: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
        let bf: Vec<f32> = b.iter().map(|x| x.to_f32()).collect();
        let base = oracle(&af, &bf, m, n, k);
        let packed = prepack_bf16(&b, n, k);
        // bf16 storage floor (~8-bit mantissa) dominates; scale loosely by k.
        let tol = (8e-2 * (k as f64).sqrt() / 4.0).max(8e-2);
        // chk applies a leading affine to put x in the op's valid domain, runs
        // the op on the bf16 kernel, and compares to the same-shifted oracle.
        macro_rules! chk {
            ($name:expr, $shift:expr, $g:ident, $chain:expr) => {{
                let mut c = vec![bf16::ZERO; m * n];
                let $g = Gemm::new(&a, &packed, m);
                ($chain).run(&mut c);
                let mut mr = 0.0f64;
                for i in 0..m {
                    for j in 0..n {
                        let x = base[i * n + j] + $shift;
                        let v = bf16_uref($name, x);
                        let got = f64::from(c[i * n + j].to_f32());
                        mr = mr.max((got - v).abs() / (1.0 + v.abs()));
                    }
                }
                assert!(
                    mr < tol,
                    "bf16 {} {m}x{n}x{k}: max_rel={mr} tol={tol}",
                    $name
                );
            }};
        }
        // Group A divide/compare ops (formerly bf16-gated). sign is shifted
        // by +1.0 so accumulators stay clear of the zero-crossing knee (where
        // f64-oracle vs bf16-rounded sign can legitimately disagree).
        chk!("sign", 1.0, g, g.add_scalar(bf16::from_f32(1.0)).sign());
        chk!("sqrt", 0.5, g, g.add_scalar(bf16::from_f32(0.5)).sqrt());
        chk!("softsign", 0.0, g, g.softsign());
        chk!("recip", 2.0, g, g.add_scalar(bf16::from_f32(2.0)).recip());
        chk!("rsqrt", 0.5, g, g.add_scalar(bf16::from_f32(0.5)).rsqrt());
        // Group B transcendentals.
        chk!("exp", 0.0, g, g.exp());
        chk!("log", 0.5, g, g.add_scalar(bf16::from_f32(0.5)).log());
        chk!("elu", 0.0, g, g.elu(0.1));
        chk!("selu", 0.0, g, g.selu());
        chk!("softplus", 0.0, g, g.softplus());
        chk!("mish", 0.0, g, g.mish());
        chk!("gelu_exact", 0.0, g, g.gelu_exact());
        // Rational tanh-family.
        chk!("gelu", 0.0, g, g.gelu());
        chk!("silu", 0.0, g, g.silu());
        chk!("tanh", 0.0, g, g.tanh());
        chk!("sigmoid", 0.0, g, g.sigmoid());

        // Divide elementwise ops against a non-zero col operand, vs oracle.
        let col: Vec<bf16> = (0..n).map(|_| bf16::from_f32(rnd(&mut s) + 1.5)).collect();
        let mut c = vec![bf16::ZERO; m * n];
        Gemm::new(&a, &packed, m).div_col(&col).run(&mut c);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let v = base[i * n + j] / f64::from(col[j].to_f32());
                mr = mr.max((f64::from(c[i * n + j].to_f32()) - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(mr < tol, "bf16 div_col {m}x{n}x{k}: {mr}");
    }
}

// Dequant (i8 -> f32) new ops in the f32 post-dequant domain.
#[test]
fn dequant_new_ops() {
    use sme_gemm::{Dequant, matmul_i8_packed_dequant, prepack_i8};
    if !caps().sme {
        return;
    }
    let (m, n, k) = (32, 48, 24);
    let mut s = 0xd0de_8a11_u64;
    let mut ri8 = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 56) as i8) >> 2
    };
    let a: Vec<i8> = (0..m * k).map(|_| ri8()).collect();
    let b: Vec<i8> = (0..k * n).map(|_| ri8()).collect();
    let scale = 0.01f32;
    let packed = prepack_i8(&b, n, k);
    let mut c = vec![0.0f32; m * n];
    // dequant -> elu(0.3) -> clamp
    matmul_i8_packed_dequant(
        &a,
        &packed,
        &mut c,
        m,
        &Dequant::new(scale).elu(0.3).clamp(-1.0, 5.0),
    );
    let mut mr = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i64;
            for l in 0..k {
                acc += i64::from(a[i * k + l]) * i64::from(b[l * n + j]);
            }
            let x = f64::from(scale) * acc as f64;
            let e = if x >= 0.0 { x } else { 0.3 * x.exp_m1() };
            let v = e.clamp(-1.0, 5.0);
            mr = mr.max((f64::from(c[i * n + j]) - v).abs() / (1.0 + v.abs()));
        }
    }
    assert!(mr < 5e-2, "dequant elu->clamp: {mr}");
}
