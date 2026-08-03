//! f32/f64 fused-epilogue op-graphs and the closure-based `epilogue_map`.

use crate::{
    F32_EP_SIZES, F64_EP_SIZES, fill, fill_f32, fill_f64, gelu_f64, max_rel, oracle, oracle_f32,
    oracle_f64,
};
use half::f16;
use sme_gemm::{Accuracy, epilogue_map, matmul_f16, matmul_f32};

// f32 op-graph: mul_col(scale) -> add_col(bias) -> gelu(), validated against the
// f64 oracle evaluating the same node sequence. gelu is the trailing activation.
#[test]
fn opgraph_f32_mulcol_addcol_gelu() {
    use sme_gemm::{Gemm, prepack_f32};
    for &(m, n, k) in F32_EP_SIZES {
        let mut s = 0xf320_1111_2222_3333 ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill_f32(&mut s, m * k);
        let b = fill_f32(&mut s, k * n);
        let scale = fill_f32(&mut s, n);
        let bias = fill_f32(&mut s, n);
        let base = oracle_f32(&a, &b, m, n, k);

        let packed = prepack_f32(&b, n, k);
        let mut c = vec![0.0f32; m * n];
        Gemm::new(&a, &packed, m)
            .mul_col(&scale)
            .add_col(&bias)
            .gelu()
            .run(&mut c);

        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut v = base[i * n + j];
                v *= f64::from(scale[j]);
                v += f64::from(bias[j]);
                v = gelu_f64(v);
                mr = mr.max((f64::from(c[i * n + j]) - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(
            mr < 5e-2,
            "opgraph f32 mulcol+addcol+gelu {m}x{n}x{k}: {mr}"
        );
    }
}

// f32 op-graph: clamp(0, 6) -- relu6-style, MAX_SCALAR then MIN_SCALAR, no
// trailing activation.
#[test]
fn opgraph_f32_clamp_relu6() {
    use sme_gemm::{Gemm, prepack_f32};
    for &(m, n, k) in F32_EP_SIZES {
        let mut s = 0xf32c_4444_5555_6666 ^ ((m * 131 + n * 17 + k) as u64);
        // scale inputs up so values exercise both clamp bounds.
        let a: Vec<f32> = fill_f32(&mut s, m * k).iter().map(|x| x * 4.0).collect();
        let b = fill_f32(&mut s, k * n);
        let base = oracle_f32(&a, &b, m, n, k);

        let packed = prepack_f32(&b, n, k);
        let mut c = vec![0.0f32; m * n];
        Gemm::new(&a, &packed, m).clamp(0.0, 6.0).run(&mut c);

        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let v = base[i * n + j].clamp(0.0, 6.0);
                mr = mr.max((f64::from(c[i * n + j]) - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(mr < 1e-4, "opgraph f32 clamp(0,6) {m}x{n}x{k}: {mr}");
    }
}

// f32 op-graph: add_col(bias) ALONE. Exercises the leading-bias fold (rank-1
// FMOPA bias-init -> direct single-instruction store). Exact in the f32
// accumulator domain, so tight tol. Row- and column-major output.
#[test]
fn opgraph_f32_addcol_only() {
    use sme_gemm::{Gemm, prepack_f32};
    for &(m, n, k) in F32_EP_SIZES {
        let mut s = 0xf3b1_aaaa_bbbb_ccccu64.wrapping_add((m * 131 + n * 17 + k) as u64);
        let a = fill_f32(&mut s, m * k);
        let b = fill_f32(&mut s, k * n);
        let bias = fill_f32(&mut s, n);
        let base = oracle_f32(&a, &b, m, n, k);

        for &cmaj in &[false, true] {
            let packed = prepack_f32(&b, n, k);
            let mut c = vec![0.0f32; m * n];
            let mut g = Gemm::new(&a, &packed, m).add_col(&bias);
            if cmaj {
                g = g.col_major_output();
            }
            g.run(&mut c);

            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let v = base[i * n + j] + f64::from(bias[j]);
                    let got = if cmaj { c[i + j * m] } else { c[i * n + j] };
                    mr = mr.max((f64::from(got) - v).abs() / (1.0 + v.abs()));
                }
            }
            assert!(
                mr < 1e-5,
                "opgraph f32 addcol-only {m}x{n}x{k} cmaj={cmaj}: {mr}"
            );
        }
    }
}

// f32 op-graph: mul_tensor(gate) -> add_scalar -> tanh(), exercised for BOTH a
// row-major and a column-major output (multi-node graph, both store paths).
#[test]
fn opgraph_f32_multensor_addscalar_tanh() {
    use sme_gemm::{Gemm, prepack_f32};
    for &(m, n, k) in F32_EP_SIZES {
        let mut s = 0xf3d7_7777_8888_9999u64.wrapping_add((m * 131 + n * 17 + k) as u64);
        let a = fill_f32(&mut s, m * k);
        let b = fill_f32(&mut s, k * n);
        let gate = fill_f32(&mut s, m * n);
        let scalar = fill_f32(&mut s, 1)[0];
        let base = oracle_f32(&a, &b, m, n, k);

        for &cmaj in &[false, true] {
            let packed = prepack_f32(&b, n, k);
            let mut c = vec![0.0f32; m * n];
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
                mr < 5e-2,
                "opgraph f32 multensor+addscalar+tanh {m}x{n}x{k} cmaj={cmaj}: {mr}"
            );
        }
    }
}

// f64 op-graph: mul_col(scale) -> add_col(bias) -> clamp(lo, hi). No activation,
// so the result is near-exact f64 GEMM + exact composable ops -> tight tol.
#[test]
fn opgraph_f64_mulcol_addcol_clamp() {
    use sme_gemm::{Gemm, prepack_f64};
    for &(m, n, k) in F64_EP_SIZES {
        let mut s = 0xf64a_1111_2222_3333 ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill_f64(&mut s, m * k);
        let b = fill_f64(&mut s, k * n);
        let scale = fill_f64(&mut s, n);
        let bias = fill_f64(&mut s, n);
        let base = oracle_f64(&a, &b, m, n, k);

        let packed = prepack_f64(&b, n, k);
        let mut c = vec![0.0f64; m * n];
        Gemm::new(&a, &packed, m)
            .mul_col(&scale)
            .add_col(&bias)
            .clamp(-0.5, 0.5)
            .run(&mut c);

        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let v = (base[i * n + j] * scale[j] + bias[j]).clamp(-0.5, 0.5);
                mr = mr.max((c[i * n + j] - v).abs() / (1.0 + v.abs()));
            }
        }
        assert!(
            mr < 1e-12,
            "opgraph f64 mulcol+addcol+clamp {m}x{n}x{k}: {mr}"
        );
    }
}

// f64 op-graph: add_col(bias) -> tanh(). tanh is an in-register rational
// approximation, so a looser tol applies. Exercised for BOTH row- and
// column-major output (both store paths).
#[test]
fn opgraph_f64_addcol_tanh() {
    use sme_gemm::{Gemm, prepack_f64};
    for &(m, n, k) in F64_EP_SIZES {
        let mut s = 0xf64b_4444_5555_6666u64.wrapping_add((m * 131 + n * 17 + k) as u64);
        let a = fill_f64(&mut s, m * k);
        let b = fill_f64(&mut s, k * n);
        let bias = fill_f64(&mut s, n);
        let base = oracle_f64(&a, &b, m, n, k);

        for &cmaj in &[false, true] {
            let packed = prepack_f64(&b, n, k);
            let mut c = vec![0.0f64; m * n];
            let mut g = Gemm::new(&a, &packed, m).add_col(&bias).tanh();
            if cmaj {
                g = g.col_major_output();
            }
            g.run(&mut c);

            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let v = (base[i * n + j] + bias[j]).tanh();
                    let got = if cmaj { c[i + j * m] } else { c[i * n + j] };
                    mr = mr.max((got - v).abs() / (1.0 + v.abs()));
                }
            }
            assert!(
                mr < 5e-2,
                "opgraph f64 addcol+tanh {m}x{n}x{k} cmaj={cmaj}: {mr}"
            );
        }
    }
}

// f64 op-graph: add_col(bias) ALONE. Exercises the leading-bias fold (rank-1
// FMOPA bias-init -> direct single-instruction store). Exact in the f64
// accumulator domain, so tight tol. Row- and column-major output.
#[test]
fn opgraph_f64_addcol_only() {
    use sme_gemm::{Gemm, prepack_f64};
    for &(m, n, k) in F64_EP_SIZES {
        let mut s = 0xf64c_dddd_eeee_ffffu64.wrapping_add((m * 131 + n * 17 + k) as u64);
        let a = fill_f64(&mut s, m * k);
        let b = fill_f64(&mut s, k * n);
        let bias = fill_f64(&mut s, n);
        let base = oracle_f64(&a, &b, m, n, k);

        for &cmaj in &[false, true] {
            let packed = prepack_f64(&b, n, k);
            let mut c = vec![0.0f64; m * n];
            let mut g = Gemm::new(&a, &packed, m).add_col(&bias);
            if cmaj {
                g = g.col_major_output();
            }
            g.run(&mut c);

            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let v = base[i * n + j] + bias[j];
                    let got = if cmaj { c[i + j * m] } else { c[i * n + j] };
                    mr = mr.max((got - v).abs() / (1.0 + v.abs()));
                }
            }
            assert!(
                mr < 1e-12,
                "opgraph f64 addcol-only {m}x{n}x{k} cmaj={cmaj}: {mr}"
            );
        }
    }
}

// Non-fused arbitrary-closure epilogue: a map that is *not* in the fused node
// set, applied after the GEMM, validated against an independent reference.
#[test]
fn epilogue_map_f16() {
    // 64x64x64 = 262144 flops, clears the SME threshold (1<<18).
    let (m, n, k) = (64usize, 64usize, 64usize);
    let mut s = 0xfeed_face_dead_beef;
    let a = fill(&mut s, m * k);
    let b = fill(&mut s, k * n);

    // Closure not expressible as fused op-graph nodes (sin / abs of the value).
    let map = |x: f32, i: usize, _j: usize| x.sin() + 0.1 * (i as f32) - x.abs();

    let want = oracle(&a, &b, m, n, k); // independent f64 A@B
    let want: Vec<f64> = want
        .iter()
        .enumerate()
        .map(|(idx, &v)| {
            let (i, j) = (idx / n, idx % n);
            // reference in f32 working precision (matches epilogue_map)
            f64::from(map(v as f32, i, j))
        })
        .collect();

    let mut c = vec![f16::ZERO; m * n];
    matmul_f16(&a, &b, &mut c, m, n, k, Accuracy::Accurate);
    epilogue_map(&mut c, m, n, map);

    let mr = max_rel(&c, &want);
    assert!(mr < 5e-3, "epilogue_map f16: max_rel={mr}");
}

#[test]
fn epilogue_map_f32() {
    let (m, n, k) = (3usize, 4usize, 5usize);
    let a: Vec<f32> = (0..m * k).map(|t| (t as f32) * 0.1 - 0.3).collect();
    let b: Vec<f32> = (0..k * n).map(|t| 0.2 - (t as f32) * 0.05).collect();

    let map = |x: f32, i: usize, j: usize| x.tanh() * 1.5 + (i + j) as f32;

    let mut want = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
            }
            want[i * n + j] = f64::from(map(acc as f32, i, j));
        }
    }

    let mut c = vec![0.0f32; m * n];
    matmul_f32(&a, &b, &mut c, m, n, k);
    epilogue_map(&mut c, m, n, map);

    let mr = c
        .iter()
        .zip(&want)
        .map(|(g, w)| (f64::from(*g) - w).abs() / (1.0 + w.abs()))
        .fold(0.0, f64::max);
    assert!(mr < 1e-5, "epilogue_map f32: max_rel={mr}");
}
