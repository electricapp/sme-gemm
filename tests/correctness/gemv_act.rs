//! m <= 7 GEMV paths with a trailing transcendental activation, which runs as a
//! NEON pass after the streaming kernel: Q4 (f16/bf16) and dense packed
//! (f16/bf16), every activation, behind a bias so the split leaves nodes.

use crate::{fill, max_rel, oracle};
use half::{bf16, f16};
use sme_gemm::{
    Epilogue, Gemm, Q4Weights, caps, matmul_q4_bf16_ep, matmul_q4_ep, prepack_bf16, prepack_f16,
};

fn act(i: usize, x: f64) -> f64 {
    match i {
        0 => {
            0.5 * x
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh())
        }
        1 => x / (1.0 + (-x).exp()),
        2 => x.tanh(),
        _ => 1.0 / (1.0 + (-x).exp()),
    }
}

macro_rules! with_act {
    ($e:expr, $i:expr) => {
        match $i {
            0 => $e.gelu(),
            1 => $e.silu(),
            2 => $e.tanh(),
            _ => $e.sigmoid(),
        }
    };
}

fn max_rel_bf16(got: &[bf16], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs() / (1.0 + w.abs()))
        .fold(0.0, f64::max)
}

#[test]
fn gemv_trailing_activation_matches_oracle() {
    for &(n, k) in &[
        (33usize, 40usize),
        (100, 64),
        (1536, 384),
        (384, 1536),
        (2048, 512),
    ] {
        for m in 1..=7 {
            let mut s = 0xac7_0000 ^ ((m * 7919 + n * 31 + k) as u64);
            let a: Vec<f16> = fill(&mut s, m * k)
                .iter()
                .map(|x| f16::from_f32(x.to_f32() * 2.0))
                .collect();
            let quants: Vec<u8> = fill(&mut s, (k * n).div_ceil(2))
                .iter()
                .map(|x| x.to_bits() as u8)
                .collect();
            let nbk = k.div_ceil(32);
            let scales: Vec<f16> = (0..n * nbk)
                .map(|i| f16::from_f32(0.02 + 0.002 * (i % 7) as f32))
                .collect();
            let bias: Vec<f16> = (0..n)
                .map(|j| f16::from_f32(0.01 * (j % 50) as f32 - 0.25))
                .collect();
            // The f16 weights the Q4 kernel multiplies, row-major k x n.
            let b: Vec<f16> = (0..k * n)
                .map(|idx| {
                    let (d, j) = (idx / n, idx % n);
                    let nib = if idx.is_multiple_of(2) {
                        quants[idx / 2] & 0x0f
                    } else {
                        quants[idx / 2] >> 4
                    };
                    let code = i32::from(nib) - if nib < 8 { 0 } else { 16 };
                    f16::from_f32(scales[j * nbk + d / 32].to_f32() * code as f32)
                })
                .collect();
            let base = oracle(&a, &b, m, n, k);
            let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
            let w = Q4Weights::new(&quants, &scales, n, k);
            let pf = prepack_f16(&b, n, k);
            let bb: Vec<bf16> = b.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
            let pb = prepack_bf16(&bb, n, k);
            let ab: Vec<bf16> = a.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
            let biasb: Vec<bf16> = bias.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
            for i in 0..4 {
                let want: Vec<f64> = base
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| act(i, v + f64::from(bias[idx % n])))
                    .collect();
                if caps().sme_f16f16 {
                    let mut c = vec![f16::ZERO; m * n];
                    matmul_q4_ep(
                        &a,
                        &w,
                        &mut c,
                        m,
                        &with_act!(Epilogue::new().add_col(&bias), i),
                    );
                    let e = max_rel(&c, &want);
                    assert!(e < tol, "q4 f16 act {i} {m}x{n}x{k}: max_rel {e}");
                    with_act!(Gemm::new(&a, &pf, m).add_col(&bias), i).run(&mut c);
                    let e = max_rel(&c, &want);
                    assert!(e < tol, "dense f16 act {i} {m}x{n}x{k}: max_rel {e}");
                }
                if caps().sme_b16b16 {
                    let mut c = vec![bf16::ZERO; m * n];
                    matmul_q4_bf16_ep(
                        &ab,
                        &w,
                        &mut c,
                        m,
                        &with_act!(Epilogue::new().add_col(&biasb), i),
                    );
                    let e = max_rel_bf16(&c, &want);
                    assert!(e < 4.0 * tol, "q4 bf16 act {i} {m}x{n}x{k}: max_rel {e}");
                    with_act!(Gemm::new(&ab, &pb, m).add_col(&biasb), i).run(&mut c);
                    let e = max_rel_bf16(&c, &want);
                    assert!(e < 4.0 * tol, "dense bf16 act {i} {m}x{n}x{k}: max_rel {e}");
                }
            }
        }
    }
}
