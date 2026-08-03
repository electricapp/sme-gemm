//! NaN semantics: which epilogue ops propagate a NaN accumulator and which drop it.

use crate::nan_accumulator_operands;
use sme_gemm::matmul_f32;

#[test]
fn relu_on_a_nan_accumulator_is_zero() {
    // maxNum(NaN, 0) == 0, which is what Rust's `x.max(0.0)` (the scalar
    // fallback and the `epilogue_map` reference) produces. FMAX would give NaN.
    let (m, n, k) = (64, 64, 64); // == 2^18 flops: above the SME floor
    let (a, b) = nan_accumulator_operands(m, n, k);
    let mut plain = vec![0.0f32; m * n];
    matmul_f32(&a, &b, &mut plain, m, n, k);
    assert!(
        plain.iter().all(|x| x.is_nan()),
        "test setup: the bare product must be NaN"
    );

    let w = sme_gemm::prepack_f32(&b, n, k);
    for col_major in [false, true] {
        let mut c = vec![0.0f32; m * n];
        let g = sme_gemm::Gemm::new(&a, &w, m).relu();
        if col_major {
            g.col_major_output().run(&mut c);
        } else {
            g.run(&mut c);
        }
        assert!(
            c.iter().all(|x| *x == 0.0),
            "relu(NaN) must be 0 (maxNum), col_major={col_major}: got {:?}",
            c.iter().find(|x| **x != 0.0)
        );
    }
}

#[test]
fn max_min_scalar_on_a_nan_accumulator_take_the_number() {
    let (m, n, k) = (64, 64, 64);
    let (a, b) = nan_accumulator_operands(m, n, k);
    let w = sme_gemm::prepack_f32(&b, n, k);
    for col_major in [false, true] {
        let mut c = vec![0.0f32; m * n];
        let g = sme_gemm::Gemm::new(&a, &w, m).max(-2.5).min(7.5);
        if col_major {
            g.col_major_output().run(&mut c);
        } else {
            g.run(&mut c);
        }
        // maxNum(NaN, -2.5) = -2.5, then minNum(-2.5, 7.5) = -2.5.
        assert!(
            c.iter().all(|x| (*x - -2.5f32).abs() < 1e-6),
            "clamp(NaN) must take the numeric operand, col_major={col_major}"
        );
    }
}

#[test]
fn max_col_with_nan_operands_takes_the_accumulator() {
    // The other direction, and the one actually reachable with finite inputs: a
    // NaN in a per-N operand vector. maxNum(x, NaN) == x.
    let (m, n, k) = (64, 96, 64);
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.125 - 0.5).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
    let mut base = vec![0.0f32; m * n];
    matmul_f32(&a, &b, &mut base, m, n, k);

    // Every third column's operand is NaN; the rest are a huge finite value that
    // must win, so a wrong lane is unmistakable.
    let v: Vec<f32> = (0..n)
        .map(|j| if j % 3 == 0 { f32::NAN } else { 1e6 })
        .collect();
    let w = sme_gemm::prepack_f32(&b, n, k);
    let mut c = vec![0.0f32; m * n];
    sme_gemm::Gemm::new(&a, &w, m).max_col(&v).run(&mut c);

    for i in 0..m {
        for j in 0..n {
            let got = c[i * n + j];
            if j % 3 == 0 {
                assert!(
                    (got - base[i * n + j]).abs() < 1e-3,
                    "max_col(NaN) must keep the accumulator at ({i},{j}): {got}"
                );
            } else {
                assert!(
                    (got - 1e6).abs() < 1.0,
                    "max_col(1e6) must win at ({i},{j}): {got}"
                );
            }
        }
    }
}

#[test]
fn clamp_activations_still_propagate_nan() {
    // relu6 / hardsigmoid / hardswish are `clamp`, and Rust's `clamp` propagates
    // NaN -- so these must NOT have moved to maxNum with the MAX/MIN nodes.
    let (m, n, k) = (64, 64, 64);
    let (a, b) = nan_accumulator_operands(m, n, k);
    let w = sme_gemm::prepack_f32(&b, n, k);
    for (name, build) in [("relu6", 0u8), ("hardsigmoid", 1), ("hardswish", 2)] {
        let mut c = vec![0.0f32; m * n];
        let g = sme_gemm::Gemm::new(&a, &w, m);
        let g = match build {
            0 => g.relu6(),
            1 => g.hardsigmoid(),
            _ => g.hardswish(),
        };
        g.run(&mut c);
        assert!(
            c.iter().all(|x| x.is_nan()),
            "{name} must propagate NaN (Rust `clamp` semantics)"
        );
    }
}
