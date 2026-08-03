//! f16/bf16 fused-epilogue op-graphs and the `Gemm` builder.

use crate::{Act, SIZES, fill, max_rel, oracle, silu_f64};
use half::{bf16, f16};
use sme_gemm::{caps, prepack_f16};

#[test]
fn f16_packed_epilogue() {
    use sme_gemm::Gemm;
    for &(m, n, k) in SIZES {
        let mut s = 0xe9f0_1122_3344_5566 ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let bias = fill(&mut s, n);
        let base = oracle(&a, &b, m, n, k); // f64 A@B

        // Cases: bias only, relu only, bias+relu.
        let cases: &[(bool, Act)] = &[(true, Act::None), (false, Act::Relu), (true, Act::Relu)];
        for &(use_bias, act) in cases {
            let packed = prepack_f16(&b, n, k);
            let mut c = vec![f16::ZERO; m * n];
            let mut g = apply_act!(Gemm::new(&a, &packed, m), act);
            if use_bias {
                g = g.add_col(&bias);
            }
            g.run(&mut c);

            // Oracle epilogue in f64.
            let mut want = base.clone();
            for i in 0..m {
                for j in 0..n {
                    let mut v = want[i * n + j];
                    if use_bias {
                        v += f64::from(bias[j]);
                    }
                    if act == Act::Relu && v < 0.0 {
                        v = 0.0;
                    }
                    want[i * n + j] = v;
                }
            }
            let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
            let mr = max_rel(&c, &want);
            assert!(
                mr < tol,
                "epilogue {m}x{n}x{k} bias={use_bias} {act:?}: max_rel={mr}"
            );
        }
    }
}

// Composable additive bias: out = act(A@B + scalar + bias_row[i] + bias_col[j]
// + residual[i,j]), each component independently optional. Exercises the new
// f16 store paths (row-major and col-major) against the f64 oracle.
#[test]
fn f16_composable_bias() {
    use sme_gemm::Gemm;
    for &(m, n, k) in SIZES {
        let mut s = 0x5151_a7a7_3c3c_9090 ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let bias_col = fill(&mut s, n);
        let bias_row = fill(&mut s, m);
        let scalar = fill(&mut s, 1)[0];
        let residual = fill(&mut s, m * n);
        let base = oracle(&a, &b, m, n, k);

        // (use_row, use_col, use_scalar, use_resid, act, col_major_out)
        let cases: &[(bool, bool, bool, bool, Act, bool)] = &[
            (true, false, false, false, Act::None, false), // bias_row only
            (false, false, false, true, Act::None, false), // residual only
            (false, false, true, false, Act::None, false), // scalar only
            (true, true, false, false, Act::None, false),  // row+col (rank-1)
            (true, true, true, true, Act::Relu, false),    // all together
            (true, false, true, false, Act::None, true),   // col-major: row+scalar
            (false, true, false, true, Act::Relu, true),   // col-major: col+resid
            (true, true, true, true, Act::None, true),     // col-major: all
        ];
        for &(ur, uc, us, ud, act, cmaj) in cases {
            let packed = prepack_f16(&b, n, k);
            let mut c = vec![f16::ZERO; m * n];
            let mut g = apply_act!(Gemm::new(&a, &packed, m), act);
            if cmaj {
                g = g.col_major_output();
            }
            if ur {
                g = g.add_row(&bias_row);
            }
            if uc {
                g = g.add_col(&bias_col);
            }
            if us {
                g = g.add_scalar(scalar);
            }
            if ud {
                g = g.add_tensor(&residual);
            }
            g.run(&mut c);

            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let mut v = base[i * n + j];
                    if us {
                        v += f64::from(scalar);
                    }
                    if ur {
                        v += f64::from(bias_row[i]);
                    }
                    if uc {
                        v += f64::from(bias_col[j]);
                    }
                    if ud {
                        v += f64::from(residual[i * n + j]);
                    }
                    if act == Act::Relu && v < 0.0 {
                        v = 0.0;
                    }
                    let got = if cmaj { c[i + j * m] } else { c[i * n + j] };
                    mr = mr.max((f64::from(got) - v).abs() / (1.0 + v.abs()));
                }
            }
            let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
            assert!(
                mr < tol,
                "f16-composable {m}x{n}x{k} r={ur} c={uc} s={us} d={ud} {act:?} cmaj={cmaj}: max_rel={mr}"
            );
        }
    }
}

#[test]
fn f16_activations_gelu_silu() {
    use sme_gemm::{Gemm, prepack_f16};
    // f64 reference matching the implemented formulas (tanh-approx GELU, SiLU).
    fn gelu(x: f64) -> f64 {
        0.5 * x * (1.0 + (0.797_884_560_802_865_4 * (x + 0.044_715 * x * x * x)).tanh())
    }
    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }
    type ActCase = (Act, fn(f64) -> f64);
    let mut worst = 0.0f64;
    for &(m, n, k) in SIZES {
        let mut s = 0x70e1_5a5a_0f0f_3c3c ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let base = oracle(&a, &b, m, n, k);
        let cases: [ActCase; 2] = [(Act::Gelu, gelu), (Act::Silu, silu)];
        for &(act, f) in &cases {
            let packed = prepack_f16(&b, n, k);
            let mut c = vec![f16::ZERO; m * n];
            apply_act!(Gemm::new(&a, &packed, m), act).run(&mut c);
            let want: Vec<f64> = base.iter().map(|&v| f(v)).collect();
            let mr = max_rel(&c, &want);
            worst = worst.max(mr);
            // f16 rational-tanh approx: ~2-3% near the f16 floor.
            assert!(mr < 6e-2, "f16 {act:?} {m}x{n}x{k}: max_rel={mr}");
        }
    }
    eprintln!("f16 GELU/SiLU worst max_rel = {worst:.4}");
}

#[test]
fn bf16_packed_epilogue() {
    use sme_gemm::{Gemm, prepack_bf16};
    for &(m, n, k) in SIZES {
        let mut s = 0x1357_9bdf_2468_ace0 ^ ((m * 131 + n * 17 + k) as u64);
        let rb = |s: &mut u64| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            bf16::from_f32((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        };
        let a: Vec<bf16> = (0..m * k).map(|_| rb(&mut s)).collect();
        let b: Vec<bf16> = (0..k * n).map(|_| rb(&mut s)).collect();
        let bias: Vec<bf16> = (0..n).map(|_| rb(&mut s)).collect();

        let cases: &[(bool, Act)] = &[(true, Act::None), (false, Act::Relu), (true, Act::Relu)];
        for &(use_bias, act) in cases {
            let packed = prepack_bf16(&b, n, k);
            let mut c = vec![bf16::ZERO; m * n];
            let mut gb = apply_act!(Gemm::new(&a, &packed, m), act);
            if use_bias {
                gb = gb.add_col(&bias);
            }
            gb.run(&mut c);

            let mut want = vec![0.0f64; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for l in 0..k {
                        acc += f64::from(a[i * k + l].to_f32()) * f64::from(b[l * n + j].to_f32());
                    }
                    if use_bias {
                        acc += f64::from(bias[j].to_f32());
                    }
                    if act == Act::Relu && acc < 0.0 {
                        acc = 0.0;
                    }
                    want[i * n + j] = acc;
                }
            }
            let tol = (6e-2 * (k as f64).sqrt() / 4.0).max(6e-2);
            let mr = c
                .iter()
                .zip(&want)
                .map(|(g, w)| (f64::from(g.to_f32()) - w).abs() / (1.0 + w.abs()))
                .fold(0.0, f64::max);
            assert!(
                mr < tol,
                "bf16-epilogue {m}x{n}x{k} bias={use_bias} {act:?}: max_rel={mr}"
            );
        }
    }
}

// Composable additive bias on the bf16 path (None/Relu only; B16B16 has no
// divide for tanh-based Gelu/Silu).
#[test]
fn bf16_composable_bias() {
    use sme_gemm::{Gemm, prepack_bf16};
    if !caps().sme_b16b16 {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0x7b16_2c2c_d4d4_0e0e ^ ((m * 131 + n * 17 + k) as u64);
        let rb = |s: &mut u64| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            bf16::from_f32((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        };
        let a: Vec<bf16> = (0..m * k).map(|_| rb(&mut s)).collect();
        let b: Vec<bf16> = (0..k * n).map(|_| rb(&mut s)).collect();
        let bias_col: Vec<bf16> = (0..n).map(|_| rb(&mut s)).collect();
        let bias_row: Vec<bf16> = (0..m).map(|_| rb(&mut s)).collect();
        let scalar = rb(&mut s);
        let residual: Vec<bf16> = (0..m * n).map(|_| rb(&mut s)).collect();

        // (use_row, use_col, use_scalar, use_resid, act, col_major_out)
        let cases: &[(bool, bool, bool, bool, Act, bool)] = &[
            (true, false, false, false, Act::None, false), // bias_row only
            (false, false, false, true, Act::None, false), // residual only
            (false, false, true, false, Act::None, false), // scalar only
            (true, true, false, false, Act::None, false),  // row+col
            (true, true, true, true, Act::Relu, false),    // all together
            (true, false, true, false, Act::None, true),   // col-major: row+scalar
            (false, true, false, true, Act::None, true),   // col-major: col+resid
            (true, true, true, true, Act::Relu, true),     // col-major: all
        ];
        for &(ur, uc, us, ud, act, cmaj) in cases {
            let packed = prepack_bf16(&b, n, k);
            let mut c = vec![bf16::ZERO; m * n];
            let mut g = apply_act!(Gemm::new(&a, &packed, m), act);
            if cmaj {
                g = g.col_major_output();
            }
            if ur {
                g = g.add_row(&bias_row);
            }
            if uc {
                g = g.add_col(&bias_col);
            }
            if us {
                g = g.add_scalar(scalar);
            }
            if ud {
                g = g.add_tensor(&residual);
            }
            g.run(&mut c);

            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let mut v = 0.0f64;
                    for l in 0..k {
                        v += f64::from(a[i * k + l].to_f32()) * f64::from(b[l * n + j].to_f32());
                    }
                    if us {
                        v += f64::from(scalar.to_f32());
                    }
                    if ur {
                        v += f64::from(bias_row[i].to_f32());
                    }
                    if uc {
                        v += f64::from(bias_col[j].to_f32());
                    }
                    if ud {
                        v += f64::from(residual[i * n + j].to_f32());
                    }
                    if act == Act::Relu && v < 0.0 {
                        v = 0.0;
                    }
                    let got = if cmaj { c[i + j * m] } else { c[i * n + j] };
                    mr = mr.max((f64::from(got.to_f32()) - v).abs() / (1.0 + v.abs()));
                }
            }
            let tol = (6e-2 * (k as f64).sqrt() / 4.0).max(6e-2);
            assert!(
                mr < tol,
                "bf16-composable {m}x{n}x{k} r={ur} c={uc} s={us} d={ud} {act:?} cmaj={cmaj}: max_rel={mr}"
            );
        }
    }
}

#[test]
fn gemm_builder() {
    use sme_gemm::{Gemm, prepack_bf16, prepack_f16};
    for &(m, n, k) in SIZES {
        let mut s = 0xc0ff_ee00_1234_abcd ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let bias = fill(&mut s, n);
        // f16 builder: C = relu(A@B + bias), exact reference.
        let wf = prepack_f16(&b, n, k);
        let mut c = vec![f16::ZERO; m * n];
        Gemm::new(&a, &wf, m).add_col(&bias).relu().run(&mut c);
        let base = oracle(&a, &b, m, n, k);
        let want: Vec<f64> = base
            .iter()
            .zip(0..)
            .map(|(&v, idx)| {
                let j = (idx as usize) % n;
                (v + f64::from(bias[j])).max(0.0)
            })
            .collect();
        let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
        assert!(max_rel(&c, &want) < tol, "f16 builder {m}x{n}x{k}");

        // bf16 builder compiles + runs through the same generic path.
        let bb: Vec<bf16> = b.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
        let ab: Vec<bf16> = a.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
        let bias_b: Vec<bf16> = bias.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
        let wb = prepack_bf16(&bb, n, k);
        let mut cb = vec![bf16::ZERO; m * n];
        Gemm::new(&ab, &wb, m).add_col(&bias_b).relu().run(&mut cb);
        let tolb = (6e-2 * (k as f64).sqrt() / 4.0).max(6e-2);
        let mrb = cb
            .iter()
            .zip(&want)
            .map(|(g, w)| (f64::from(g.to_f32()) - w).abs() / (1.0 + w.abs()))
            .fold(0.0, f64::max);
        assert!(mrb < tolb, "bf16 builder {m}x{n}x{k}: {mrb}");

        // beta scaling: D = relu(beta*(A@B) + bias).
        let beta = 0.5f32;
        let mut cbeta = vec![f16::ZERO; m * n];
        Gemm::new(&a, &wf, m)
            .beta(f16::from_f32(beta))
            .add_col(&bias)
            .relu()
            .run(&mut cbeta);
        let want_beta: Vec<f64> = (0..m * n)
            .map(|idx| {
                let j = idx % n;
                (f64::from(beta) * base[idx] + f64::from(bias[j])).max(0.0)
            })
            .collect();
        assert!(
            max_rel(&cbeta, &want_beta) < tol,
            "f16 builder beta {m}x{n}x{k}"
        );

        // column-major output: c[i + j*m] == (A@B)[i,j].
        let mut ccol = vec![f16::ZERO; m * n];
        Gemm::new(&a, &wf, m).col_major_output().run(&mut ccol);
        let mut mrc = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let got = f64::from(ccol[i + j * m]);
                let w = base[i * n + j];
                mrc = mrc.max((got - w).abs() / (1.0 + w.abs()));
            }
        }
        assert!(mrc < tol, "f16 builder col-major {m}x{n}x{k}: {mrc}");
    }
}

// f16 op-graph: mul_col(scale) -> add_col(bias) -> silu(), validated against the
// f64 oracle evaluating the same node sequence. silu is the trailing activation.
#[test]
fn opgraph_f16_mulcol_addcol_silu() {
    use sme_gemm::{Gemm, prepack_f16};
    for &(m, n, k) in SIZES {
        let mut s = 0x0a0a_f16e_1111_2222 ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let scale = fill(&mut s, n);
        let bias = fill(&mut s, n);
        let base = oracle(&a, &b, m, n, k);

        let packed = prepack_f16(&b, n, k);
        let mut c = vec![f16::ZERO; m * n];
        Gemm::new(&a, &packed, m)
            .mul_col(&scale)
            .add_col(&bias)
            .silu()
            .run(&mut c);

        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut v = base[i * n + j];
                v *= f64::from(scale[j]);
                v += f64::from(bias[j]);
                v = silu_f64(v);
                let got = f64::from(c[i * n + j]);
                mr = mr.max((got - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(
            mr < 6e-2,
            "opgraph f16 mulcol+addcol+silu {m}x{n}x{k}: {mr}"
        );
    }
}

// f16 op-graph: clamp(0, 6) -- relu6-style, MAX_SCALAR then MIN_SCALAR. No
// trailing activation; the clamp is order-sensitive nodes.
#[test]
fn opgraph_f16_clamp_relu6() {
    use sme_gemm::{Gemm, prepack_f16};
    for &(m, n, k) in SIZES {
        let mut s = 0x3c3c_6e6e_9090_b1b1 ^ ((m * 131 + n * 17 + k) as u64);
        // scale inputs up so values exercise both clamp bounds.
        let a: Vec<f16> = fill(&mut s, m * k)
            .iter()
            .map(|x| f16::from_f32(x.to_f32() * 4.0))
            .collect();
        let b = fill(&mut s, k * n);
        let base = oracle(&a, &b, m, n, k);

        let packed = prepack_f16(&b, n, k);
        let mut c = vec![f16::ZERO; m * n];
        Gemm::new(&a, &packed, m)
            .clamp(f16::from_f32(0.0), f16::from_f32(6.0))
            .run(&mut c);

        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let v = base[i * n + j].clamp(0.0, 6.0);
                let got = f64::from(c[i * n + j]);
                mr = mr.max((got - v).abs() / (1.0 + v.abs()));
            }
        }
        let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
        assert!(mr < tol, "opgraph f16 clamp(0,6) {m}x{n}x{k}: {mr}");
    }
}

// f16 op-graph: mul_tensor(gate) -> add_scalar -> tanh(), exercised for BOTH a
// row-major and a column-major output (multi-node graph, both store paths).
#[test]
fn opgraph_f16_multensor_addscalar_tanh() {
    use sme_gemm::{Gemm, prepack_f16};
    for &(m, n, k) in SIZES {
        let mut s = 0x55aa_1234_00cd_ef00 ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let gate = fill(&mut s, m * n);
        let scalar = fill(&mut s, 1)[0];
        let base = oracle(&a, &b, m, n, k);

        for &cmaj in &[false, true] {
            let packed = prepack_f16(&b, n, k);
            let mut c = vec![f16::ZERO; m * n];
            let mut g = Gemm::new(&a, &packed, m)
                .mul_tensor(&gate)
                .add_scalar(scalar)
                .tanh();
            if cmaj {
                g = g.col_major_output();
            }
            g.run(&mut c);

            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let mut v = base[i * n + j];
                    v *= f64::from(gate[i * n + j]);
                    v += f64::from(scalar);
                    v = v.tanh();
                    let got = if cmaj { c[i + j * m] } else { c[i * n + j] };
                    mr = mr.max((f64::from(got) - v).abs() / (1.0 + v.abs()));
                }
            }
            assert!(
                mr < 6e-2,
                "opgraph f16 multensor+addscalar+tanh {m}x{n}x{k} cmaj={cmaj}: {mr}"
            );
        }
    }
}

// bf16 op-graph: None/Relu/clamp/affine only (B16B16 has no divide, so no
// gelu/silu/tanh/sigmoid). Validates an affine + relu graph and a clamp graph.
#[test]
fn opgraph_bf16_affine_relu_clamp() {
    use sme_gemm::{Gemm, caps, prepack_bf16};
    if !caps().sme_b16b16 {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0xbf16_0e0e_7a7a_1313 ^ ((m * 131 + n * 17 + k) as u64);
        let rb = |s: &mut u64| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            bf16::from_f32((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        };
        let a: Vec<bf16> = (0..m * k).map(|_| rb(&mut s)).collect();
        let b: Vec<bf16> = (0..k * n).map(|_| rb(&mut s)).collect();
        let scale: Vec<bf16> = (0..n).map(|_| rb(&mut s)).collect();
        let bias: Vec<bf16> = (0..n).map(|_| rb(&mut s)).collect();

        let mut base = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l].to_f32()) * f64::from(b[l * n + j].to_f32());
                }
                base[i * n + j] = acc;
            }
        }

        // affine + relu: mul_col(scale) -> add_col(bias) -> relu()
        let packed = prepack_bf16(&b, n, k);
        let mut c = vec![bf16::ZERO; m * n];
        Gemm::new(&a, &packed, m)
            .mul_col(&scale)
            .add_col(&bias)
            .relu()
            .run(&mut c);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut v = base[i * n + j];
                v *= f64::from(scale[j].to_f32());
                v += f64::from(bias[j].to_f32());
                v = v.max(0.0);
                let got = f64::from(c[i * n + j].to_f32());
                mr = mr.max((got - v).abs() / (1.0 + v.abs()));
            }
        }
        let tol = (8e-2 * (k as f64).sqrt() / 4.0).max(8e-2);
        assert!(mr < tol, "opgraph bf16 affine+relu {m}x{n}x{k}: {mr}");

        // clamp(0, 1)
        let mut c2 = vec![bf16::ZERO; m * n];
        Gemm::new(&a, &packed, m)
            .clamp(bf16::from_f32(0.0), bf16::from_f32(1.0))
            .run(&mut c2);
        let mut mr2 = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let v = base[i * n + j].clamp(0.0, 1.0);
                let got = f64::from(c2[i * n + j].to_f32());
                mr2 = mr2.max((got - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(mr2 < tol, "opgraph bf16 clamp(0,1) {m}x{n}x{k}: {mr2}");
    }
}
