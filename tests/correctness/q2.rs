//! The 2-bit path: the kernel against an f64 oracle on the unquantized
//! activations, its fused epilogue, and the model layers built on it.

use crate::{SIZES, TAIL_SIZES, fill, fill_f32, gelu_f64};
use half::f16;
use sme_gemm::{
    Epilogue, Gate, GatedLinear, Linear, Mlp, Q2Params, Q2Weights, WeightLayout, matmul_q2,
    matmul_q2_ep,
};

fn bytes(s: &mut u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|_| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (*s >> 40) as u8
        })
        .collect()
}

/// Random 2-bit weights for `p` and their f64 values, row-major `k x n`.
fn weights(s: &mut u64, n: usize, k: usize, p: Q2Params) -> (Q2Weights, Vec<f64>) {
    let nbk = k.div_ceil(p.block());
    let codes = bytes(s, (k * n).div_ceil(4));
    let scales: Vec<f32> = bytes(s, n * nbk)
        .iter()
        .map(|&b| 0.005 + f32::from(b) * 1e-4)
        .collect();
    let mut w = vec![0.0f64; k * n];
    for d in 0..k {
        for j in 0..n {
            let idx = d * n + j;
            let code = usize::from((codes[idx / 4] >> (2 * (idx % 4))) & 3);
            w[idx] = f64::from(scales[j * nbk + d / p.block()]) * f64::from(p.codebook()[code]);
        }
    }
    (Q2Weights::new(&codes, &scales, n, k, p), w)
}

/// Each activation rounds by at most half its block's int8 step, `amax/127`,
/// so an output is off the exact product by at most `sum_d |w[d][j]| *
/// amax/254`, plus its f16 rounding.
#[test]
fn q2_matches_f64_oracle_within_activation_rounding() {
    let params = [
        Q2Params::default(),
        Q2Params::new(32).with_codebook([-2, -1, 1, 2]),
    ];
    for &(m, n, k) in SIZES.iter().chain(TAIL_SIZES) {
        for p in params {
            let mut s = 0x0202_9401_dead_0001 ^ ((m * 131 + n * 17 + k + p.block()) as u64);
            let (w, wf) = weights(&mut s, n, k, p);
            let a = fill(&mut s, m * k);
            let mut c = vec![f16::ZERO; m * n];
            matmul_q2(&a, &w, &mut c, m);
            for i in 0..m {
                let amax: Vec<f64> = a[i * k..(i + 1) * k]
                    .chunks(p.block())
                    .map(|b| b.iter().fold(0.0f64, |x, v| x.max(f64::from(*v).abs())))
                    .collect();
                for j in 0..n {
                    let (mut want, mut bound) = (0.0f64, 0.0f64);
                    for d in 0..k {
                        let wv = wf[d * n + j];
                        want += f64::from(a[i * k + d]) * wv;
                        bound += wv.abs() * amax[d / p.block()] / 254.0;
                    }
                    let got = f64::from(c[i * n + j]);
                    let tol = bound * 1.001 + want.abs() / 1024.0 + 1e-6;
                    assert!(
                        (got - want).abs() <= tol,
                        "{m}x{n}x{k} block {}: c[{i}][{j}] = {got} vs {want} (tol {tol})",
                        p.block()
                    );
                }
            }
        }
    }
}

/// The fused epilogue (bias, row scale, and a trailing gelu that runs as the
/// NEON pass) matches the plain product with the same ops applied after.
#[test]
fn q2_fused_epilogue_matches_plain_then_applied() {
    for &(m, n, k) in &[(1usize, 200usize, 300usize), (4, 1536, 512), (13, 70, 64)] {
        let mut s = 0x0202_e910_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let (w, _) = weights(&mut s, n, k, Q2Params::default());
        let a = fill(&mut s, m * k);
        let (bias, rows) = (fill(&mut s, n), fill(&mut s, m));
        let mut plain = vec![f16::ZERO; m * n];
        matmul_q2(&a, &w, &mut plain, m);
        let mut c = vec![f16::ZERO; m * n];
        let ep = Epilogue::new().add_col(&bias).mul_row(&rows).gelu();
        matmul_q2_ep(&a, &w, &mut c, m, &ep);
        for i in 0..m {
            for j in 0..n {
                let x = (f64::from(plain[i * n + j]) + f64::from(bias[j])) * f64::from(rows[i]);
                let want = gelu_f64(x);
                let got = f64::from(c[i * n + j]);
                assert!(
                    (got - want).abs() <= 1e-2 * (1.0 + want.abs()),
                    "{m}x{n}x{k}: c[{i}][{j}] = {got} vs {want}"
                );
            }
        }
    }
}

/// `Linear`, `GatedLinear` and `Mlp` on 2-bit weights are the 2-bit GEMM
/// with their bias, activation and residual steps, at every row count.
#[test]
fn model_layers_run_2bit_weights() {
    let (d, f) = (160usize, 300usize);
    let lay = WeightLayout::OutIn;
    let p = Q2Params::new(32);
    let mut s = 0x0202_11ea_0001;
    let (w1, b1) = (fill_f32(&mut s, f * d), fill_f32(&mut s, f));
    let (wg, bg) = (fill_f32(&mut s, f * d), fill_f32(&mut s, f));
    let (w2, b2) = (fill_f32(&mut s, d * f), fill_f32(&mut s, d));
    let up = || Linear::quantize_q2(&w1, lay, f, d, p).with_bias(&b1);
    let gated = || GatedLinear::quantize_q2(&wg, &w1, lay, f, d, Gate::Silu, p).with_bias(&bg, &b1);
    let down = || Linear::quantize_q2(&w2, lay, d, f, p).with_bias(&b2);
    let q_up = Q2Weights::quantize(&w1, lay, f, d, p);
    let b1h: Vec<f16> = b1.iter().map(|&v| f16::from_f32(v)).collect();
    for m in [1usize, 3, 9] {
        let x = fill_f32(&mut s, m * d);
        let xh: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();

        // Linear: the 2-bit GEMM with the bias as the epilogue's first step.
        let (mut y, mut want) = (vec![f16::ZERO; m * f], vec![f16::ZERO; m * f]);
        up().forward(&x, &mut y, m);
        matmul_q2_ep(&xh, &q_up, &mut want, m, &Epilogue::new().add_col(&b1h));
        assert_eq!(y, want, "Linear m={m}");

        // GatedLinear: silu(gate) * up from one interleaved 2-bit GEMM.
        let (g, u) = (
            Linear::quantize_q2(&wg, lay, f, d, p).with_bias(&bg),
            Linear::quantize_q2(&w1, lay, f, d, p).with_bias(&b1),
        );
        let (mut h, mut gy, mut uy) = (
            vec![f16::ZERO; m * f],
            vec![f16::ZERO; m * f],
            vec![f16::ZERO; m * f],
        );
        gated().forward(&x, &mut h, m);
        g.forward(&x, &mut gy, m);
        u.forward(&x, &mut uy, m);
        for i in 0..m * f {
            let (a, b) = (gy[i].to_f32(), uy[i].to_f32());
            let want = a / (1.0 + (-a).exp()) * b;
            let got = h[i].to_f32();
            assert!(
                (got - want).abs() <= 2e-3 * (1.0 + want.abs()),
                "GatedLinear m={m} [{i}]: {got} vs {want}"
            );
        }

        // Mlp: up with the activation, then down into the residual.
        let mlp = Mlp::new(up(), Gate::Gelu, down());
        let (mut res, mut want) = (vec![0.25f32; m * d], vec![0.25f32; m * d]);
        mlp.accumulate(&x, &mut res, m);
        let mut hid = vec![f16::ZERO; m * f];
        up().forward_ep(&x, &mut hid, m, &Epilogue::new().gelu());
        down().accumulate(&hid, &mut want, m);
        assert_eq!(res, want, "Mlp m={m}");
    }
}
