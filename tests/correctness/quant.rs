//! Fused dequant (i8/i16 -> f32). The 4-bit paths live in `q4.rs`.

use crate::{Act, SIZES, TAIL_SIZES, gelu_f64};

#[test]
fn i8_dequant() {
    use sme_gemm::{Dequant, matmul_i8_packed_dequant, prepack_i8};
    for &(m, n, k) in SIZES {
        let mut s = 0x2a2a_5b5b_8c8c_d0d0 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 48) as i32 % 255 - 127) as i8
        };
        let a: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i8> = (0..k * n).map(|_| rnd()).collect();
        // exact i32 accumulation oracle
        let mut acc = vec![0i64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0i64;
                for l in 0..k {
                    s += i64::from(a[i * k + l]) * i64::from(b[l * n + j]);
                }
                acc[i * n + j] = s;
            }
        }
        let scale = 0.000_123_f32;
        let scale_n: Vec<f32> = (0..n).map(|j| 0.0001 * (1.0 + (j % 7) as f32)).collect();
        let bias: Vec<f32> = (0..n).map(|j| 0.05 * (j as f32 - n as f32 / 2.0)).collect();

        // (use_per_n_scale, use_bias, act)
        let cases: &[(bool, bool, Act)] = &[
            (false, false, Act::None),
            (false, true, Act::Relu),
            (true, true, Act::None),
            (true, true, Act::Relu),
        ];
        for &(pn, ub, act) in cases {
            let packed = prepack_i8(&b, n, k);
            let mut c = vec![0.0f32; m * n];
            let mut dq = Dequant::new(scale);
            if pn {
                dq = dq.scale_per_n(&scale_n);
            }
            if ub {
                dq = dq.add_col(&bias);
            }
            dq = apply_act!(dq, act);
            matmul_i8_packed_dequant(&a, &packed, &mut c, m, &dq);

            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let sc = if pn { scale_n[j] } else { scale };
                    let mut want = f64::from(sc) * acc[i * n + j] as f64;
                    if ub {
                        want += f64::from(bias[j]);
                    }
                    if act == Act::Relu && want < 0.0 {
                        want = 0.0;
                    }
                    mr = mr.max((f64::from(c[i * n + j]) - want).abs() / (1.0 + want.abs()));
                }
            }
            assert!(
                mr < 1e-4,
                "i8-dequant {m}x{n}x{k} pn={pn} bias={ub} {act:?}: max_rel={mr}"
            );
        }
    }
}

// Composable additive bias on the i8->f32 dequant path: each new component
// (bias_row, bias_scalar, residual) alone and combined, validated against the
// exact i32 oracle dequantized in f64.
#[test]
fn i8_dequant_composable() {
    use sme_gemm::{Dequant, matmul_i8_packed_dequant, prepack_i8};
    for &(m, n, k) in SIZES {
        let mut s = 0x9d9d_4c4c_a1a1_7070 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 48) as i32 % 255 - 127) as i8
        };
        let a: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i8> = (0..k * n).map(|_| rnd()).collect();
        let mut acc = vec![0i64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut t = 0i64;
                for l in 0..k {
                    t += i64::from(a[i * k + l]) * i64::from(b[l * n + j]);
                }
                acc[i * n + j] = t;
            }
        }
        let scale = 0.000_123_f32;
        let bias_col: Vec<f32> = (0..n).map(|j| 0.05 * (j as f32 - n as f32 / 2.0)).collect();
        let bias_row: Vec<f32> = (0..m).map(|i| 0.03 * (i as f32 - m as f32 / 2.0)).collect();
        let scalar = 0.017_f32;
        let residual: Vec<f32> = (0..m * n)
            .map(|t| 0.011 * ((t % 13) as f32 - 6.0))
            .collect();

        // (use_row, use_col, use_scalar, use_resid, act)
        let cases: &[(bool, bool, bool, bool, Act)] = &[
            (true, false, false, false, Act::None), // bias_row only
            (false, false, false, true, Act::None), // residual only
            (false, false, true, false, Act::None), // scalar only
            (true, true, false, false, Act::None),  // row+col
            (true, true, true, true, Act::None),    // all
            (true, true, true, true, Act::Relu),    // all + relu
        ];
        for &(ur, uc, us, ud, act) in cases {
            let packed = prepack_i8(&b, n, k);
            let mut c = vec![0.0f32; m * n];
            let mut dq = apply_act!(Dequant::new(scale), act);
            if ur {
                dq = dq.add_row(&bias_row);
            }
            if uc {
                dq = dq.add_col(&bias_col);
            }
            if us {
                dq = dq.add_scalar(scalar);
            }
            if ud {
                dq = dq.add_tensor(&residual);
            }
            matmul_i8_packed_dequant(&a, &packed, &mut c, m, &dq);

            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let mut want = f64::from(scale) * acc[i * n + j] as f64;
                    if us {
                        want += f64::from(scalar);
                    }
                    if ur {
                        want += f64::from(bias_row[i]);
                    }
                    if uc {
                        want += f64::from(bias_col[j]);
                    }
                    if ud {
                        want += f64::from(residual[i * n + j]);
                    }
                    if act == Act::Relu && want < 0.0 {
                        want = 0.0;
                    }
                    mr = mr.max((f64::from(c[i * n + j]) - want).abs() / (1.0 + want.abs()));
                }
            }
            assert!(
                mr < 1e-4,
                "i8-dequant-composable {m}x{n}x{k} r={ur} c={uc} s={us} d={ud} {act:?}: max_rel={mr}"
            );
        }
    }
}

#[test]
fn i16_dequant() {
    use sme_gemm::{Dequant, matmul_i16_dequant};
    // Sizes that clear the SME_MIN_FLOPS threshold so the real i16 SME kernel
    // runs (and exercises the dequant store), plus its M/N/K tails.
    for &(m, n, k) in TAIL_SIZES {
        let mut s = 0x16d6_16d6_a5a5_0001 ^ ((m * 131 + n * 17 + k) as u64);
        // Bounded i16 inputs (|a|,|b| <= 64) keep the i64 accumulator small enough
        // that the i64 -> f32 store conversion is effectively exact for the tol.
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 40) as i32 % 129 - 64) as i16
        };
        let a: Vec<i16> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i16> = (0..k * n).map(|_| rnd()).collect();
        // exact i64 accumulation oracle
        let mut acc = vec![0i64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut t = 0i64;
                for l in 0..k {
                    t += i64::from(a[i * k + l]) * i64::from(b[l * n + j]);
                }
                acc[i * n + j] = t;
            }
        }
        let scale = 0.000_037_f32;
        let scale_n: Vec<f32> = (0..n).map(|j| 0.00003 * (1.0 + (j % 7) as f32)).collect();
        let bias: Vec<f32> = (0..n).map(|j| 0.05 * (j as f32 - n as f32 / 2.0)).collect();

        // (use_per_n_scale, use_bias, act, clamp)
        let cases: &[(bool, bool, Act, bool)] = &[
            (false, false, Act::None, false),
            (false, true, Act::Relu, false),
            (true, true, Act::None, false),
            (true, true, Act::Relu, false),
            (false, false, Act::None, true),
        ];
        for &(pn, ub, act, clamp) in cases {
            let mut c = vec![0.0f32; m * n];
            let mut dq = Dequant::new(scale);
            if pn {
                dq = dq.scale_per_n(&scale_n);
            }
            if ub {
                dq = dq.add_col(&bias);
            }
            if clamp {
                dq = dq.clamp(-1.0, 1.0);
            }
            dq = apply_act!(dq, act);
            matmul_i16_dequant(&a, &b, &mut c, m, n, k, &dq);

            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let sc = if pn { scale_n[j] } else { scale };
                    let mut want = f64::from(sc) * acc[i * n + j] as f64;
                    if ub {
                        want += f64::from(bias[j]);
                    }
                    if clamp {
                        want = want.clamp(-1.0, 1.0);
                    }
                    if act == Act::Relu && want < 0.0 {
                        want = 0.0;
                    }
                    mr = mr.max((f64::from(c[i * n + j]) - want).abs() / (1.0 + want.abs()));
                }
            }
            assert!(
                mr < 1e-4,
                "i16-dequant {m}x{n}x{k} pn={pn} bias={ub} {act:?} clamp={clamp}: max_rel={mr}"
            );
        }
    }
}

// i8->f32 dequant op-graph: scale -> mul_col(gain) -> add_row(bias_row) ->
// add_tensor(residual) -> gelu(), validated against the exact i32 oracle
// dequantized + evaluated through the same node sequence in f64.
#[test]
fn opgraph_i8_dequant() {
    use sme_gemm::{Dequant, matmul_i8_packed_dequant, prepack_i8};
    for &(m, n, k) in SIZES {
        let mut s = 0xd1d1_8e8e_2c2c_4040 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 48) as i32 % 255 - 127) as i8
        };
        let a: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i8> = (0..k * n).map(|_| rnd()).collect();
        let mut acc = vec![0i64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut t = 0i64;
                for l in 0..k {
                    t += i64::from(a[i * k + l]) * i64::from(b[l * n + j]);
                }
                acc[i * n + j] = t;
            }
        }
        let scale = 0.000_211_f32;
        let gain: Vec<f32> = (0..n).map(|j| 0.5 + 0.1 * (j % 5) as f32).collect();
        let bias_row: Vec<f32> = (0..m).map(|i| 0.02 * (i as f32 - m as f32 / 2.0)).collect();
        let residual: Vec<f32> = (0..m * n)
            .map(|t| 0.013 * ((t % 11) as f32 - 5.0))
            .collect();

        let packed = prepack_i8(&b, n, k);
        let mut c = vec![0.0f32; m * n];
        let dq = Dequant::new(scale)
            .mul_col(&gain)
            .add_row(&bias_row)
            .add_tensor(&residual)
            .gelu();
        matmul_i8_packed_dequant(&a, &packed, &mut c, m, &dq);

        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut v = f64::from(scale) * acc[i * n + j] as f64;
                v *= f64::from(gain[j]);
                v += f64::from(bias_row[i]);
                v += f64::from(residual[i * n + j]);
                v = gelu_f64(v);
                mr = mr.max((f64::from(c[i * n + j]) - v).abs() / (1.0 + v.abs()));
            }
        }
        // 3e-2: the fused dequant store is vectorized (like every other epilogue
        // path), so the trailing gelu uses the in-register rational-tanh
        // approximation (~2e-2 here) rather than a per-cell scalar libm call.
        assert!(mr < 3e-2, "opgraph i8-dequant {m}x{n}x{k}: max_rel={mr}");

        // Tight check of the dequant-specific arithmetic (i32->f32 convert + scale)
        // and the exact ops, with relu (exact in both paths) so no approximation
        // masks an arithmetic bug in the vectorized store.
        let mut c2 = vec![0.0f32; m * n];
        let dq2 = Dequant::new(scale)
            .mul_col(&gain)
            .add_row(&bias_row)
            .add_tensor(&residual)
            .relu();
        matmul_i8_packed_dequant(&a, &packed, &mut c2, m, &dq2);
        let mut mr2 = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut v = f64::from(scale) * acc[i * n + j] as f64;
                v *= f64::from(gain[j]);
                v += f64::from(bias_row[i]);
                v += f64::from(residual[i * n + j]);
                v = v.max(0.0);
                mr2 = mr2.max((f64::from(c2[i * n + j]) - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(
            mr2 < 2e-3,
            "opgraph i8-dequant (relu) {m}x{n}x{k}: max_rel={mr2}"
        );
    }
}
