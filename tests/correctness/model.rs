//! The model-level helpers: `Linear` against the Q4 / packed kernels it wraps,
//! `prepack` for either weight layout, and the `nn` norms against f64.

use crate::fill_f32;
use half::f16;
use sme_gemm::{
    Epilogue, Gate, GatedLinear, Linear, Q4Params, Q4Weights, WeightLayout, caps,
    matmul_f16_packed, matmul_f32_packed, matmul_q4, matmul_q4_ep, nn, prepack, prepack_f16,
    prepack_f32,
};

fn transpose<T: Copy>(w: &[T], n: usize, k: usize) -> Vec<T> {
    (0..k * n).map(|i| w[(i % n) * k + i / n]).collect()
}

/// Either side of the panel switch, `Linear` gives exactly what `matmul_q4`
/// gives for the same 4-bit weights, plain and with a bias plus activation.
#[test]
fn linear_q4_matches_matmul_q4_on_both_paths() {
    let (n, k) = (200usize, 96usize);
    let mut s = 0x11ea_0001;
    let w = fill_f32(&mut s, n * k);
    let bias = fill_f32(&mut s, n);
    let bias16: Vec<f16> = bias.iter().map(|&b| f16::from_f32(b)).collect();
    let lin = Linear::quantize(&w, WeightLayout::OutIn, n, k);
    let lin_t = Linear::quantize(&transpose(&w, n, k), WeightLayout::InOut, n, k);
    let biased = Linear::quantize(&w, WeightLayout::OutIn, n, k).with_bias(&bias);
    let q4 = Q4Weights::quantize(&w, WeightLayout::OutIn, n, k, Q4Params::default());
    for m in [1usize, 3, 7, 8, 9, 64] {
        let x16: Vec<f16> = fill_f32(&mut s, m * k)
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let x32: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
        let mut want = vec![f16::ZERO; m * n];
        matmul_q4(&x16, &q4, &mut want, m);
        let (mut a, mut b, mut c) = (
            vec![f16::ZERO; m * n],
            vec![f16::ZERO; m * n],
            vec![f16::ZERO; m * n],
        );
        lin.forward(&x16, &mut a, m);
        lin_t.forward(&x16, &mut b, m);
        lin.forward(&x32, &mut c, m);
        assert_eq!(a, want, "Linear vs matmul_q4 at m={m}");
        assert_eq!(b, want, "InOut layout at m={m}");
        assert_eq!(c, want, "f32 input at m={m}");

        let ep = Epilogue::new().add_col(&bias16).gelu();
        matmul_q4_ep(&x16, &q4, &mut want, m, &ep);
        biased.forward_ep(&x16, &mut a, m, &Epilogue::new().gelu());
        if caps().sme_f16f16 {
            assert_eq!(a, want, "bias + gelu at m={m}");
        }

        // f32 output and the residual update agree with the f16 result.
        let mut y32 = vec![0.0f32; m * n];
        lin.forward_f32(&x16, &mut y32, m);
        let mut res = vec![1.0f32; m * n];
        lin.accumulate(&x32, &mut res, m);
        let mut plain = vec![f16::ZERO; m * n];
        lin.forward(&x16, &mut plain, m);
        for i in 0..m * n {
            assert_eq!(
                y32[i].to_bits(),
                plain[i].to_f32().to_bits(),
                "forward_f32 at m={m}, {i}"
            );
            assert_eq!(
                res[i].to_bits(),
                (1.0 + plain[i].to_f32()).to_bits(),
                "accumulate at m={m}, {i}"
            );
        }
    }
}

/// Dense f16 weights from either layout run the packed f16 GEMM.
#[test]
fn linear_f16_matches_packed_gemm() {
    let (n, k) = (72usize, 40usize);
    let mut s = 0x11ea_0002;
    let w = fill_f32(&mut s, n * k);
    let w16: Vec<f16> = transpose(&w, n, k)
        .iter()
        .map(|&v| f16::from_f32(v))
        .collect();
    let packed = prepack_f16(&w16, n, k);
    let lin = Linear::f16(&w, WeightLayout::OutIn, n, k);
    assert!(lin.q4().is_none());
    for m in [1usize, 5, 16] {
        let x: Vec<f16> = fill_f32(&mut s, m * k)
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let (mut got, mut want) = (vec![f16::ZERO; m * n], vec![f16::ZERO; m * n]);
        lin.forward(&x, &mut got, m);
        matmul_f16_packed(&x, &packed, &mut want, m);
        assert_eq!(got, want, "Linear::f16 at m={m}");
    }
}

/// `prepack` of `[out][in]` weights packs what `prepack_*` packs from the
/// transpose.
#[test]
fn prepack_takes_either_layout() {
    let (n, k, m) = (48usize, 24usize, 3usize);
    let mut s = 0x11ea_0003;
    let w = fill_f32(&mut s, n * k);
    let a = fill_f32(&mut s, m * k);
    let (mut got, mut want) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
    matmul_f32_packed(&a, &prepack(&w, WeightLayout::OutIn, n, k), &mut got, m);
    matmul_f32_packed(&a, &prepack_f32(&transpose(&w, n, k), n, k), &mut want, m);
    assert_eq!(got, want, "prepack(OutIn) vs prepack_f32(transpose)");
}

fn norm_oracle(x: &[f32], w: &[f32], b: Option<&[f32]>, eps: f64, rms: bool) -> Vec<f64> {
    let dim = w.len();
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks(dim) {
        let mean = if rms {
            0.0
        } else {
            row.iter().map(|&v| f64::from(v)).sum::<f64>() / dim as f64
        };
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / dim as f64;
        let r = 1.0 / (var + eps).sqrt();
        for (j, &v) in row.iter().enumerate() {
            out.push(
                (f64::from(v) - mean) * r * f64::from(w[j]) + b.map_or(0.0, |b| f64::from(b[j])),
            );
        }
    }
    out
}

#[test]
fn norms_match_f64() {
    for dim in [8usize, 100, 384] {
        let mut s = 0x11ea_0100 ^ dim as u64;
        let x: Vec<f32> = fill_f32(&mut s, 3 * dim)
            .iter()
            .map(|v| v * 4.0 + 1.0)
            .collect();
        let w: Vec<f32> = fill_f32(&mut s, dim).iter().map(|v| v + 1.0).collect();
        let b = fill_f32(&mut s, dim);
        let ln = norm_oracle(&x, &w, Some(&b), 1e-5, false);
        let rms = norm_oracle(&x, &w, None, 1e-6, true);
        let mut y = vec![0.0f32; x.len()];
        let mut y16 = vec![f16::ZERO; x.len()];
        nn::layer_norm(&x, &w, Some(&b), 1e-5, &mut y);
        nn::layer_norm(&x, &w, Some(&b), 1e-5, &mut y16);
        for i in 0..x.len() {
            assert!(
                (f64::from(y[i]) - ln[i]).abs() < 1e-4,
                "layer_norm dim {dim} [{i}]"
            );
            assert!(
                (y16[i].to_f64() - ln[i]).abs() < 4e-3,
                "layer_norm f16 dim {dim} [{i}]"
            );
        }
        nn::rms_norm(&x, &w, 1e-6, &mut y);
        for i in 0..x.len() {
            assert!(
                (f64::from(y[i]) - rms[i]).abs() < 1e-4,
                "rms_norm dim {dim} [{i}]"
            );
        }
    }
}

/// A gated layer equals its two halves run as separate layers, then
/// `act(gate) * up` in f32: interleaving whole 32-column tiles leaves every
/// column's quantization and accumulation unchanged.
#[test]
fn gated_linear_matches_two_linears() {
    let k = 64usize;
    for &(n, gate) in &[(100usize, Gate::Silu), (64, Gate::Gelu)] {
        let mut s = 0x11ea_0200 ^ n as u64;
        let (wg, wu) = (fill_f32(&mut s, n * k), fill_f32(&mut s, n * k));
        let (bg, bu) = (fill_f32(&mut s, n), fill_f32(&mut s, n));
        let gated =
            GatedLinear::quantize(&wg, &wu, WeightLayout::OutIn, n, k, gate).with_bias(&bg, &bu);
        let (g, u) = (
            Linear::quantize(&wg, WeightLayout::OutIn, n, k).with_bias(&bg),
            Linear::quantize(&wu, WeightLayout::OutIn, n, k).with_bias(&bu),
        );
        let dense = GatedLinear::f16(&wg, &wu, WeightLayout::OutIn, n, k, gate);
        let (gd, ud) = (
            Linear::f16(&wg, WeightLayout::OutIn, n, k),
            Linear::f16(&wu, WeightLayout::OutIn, n, k),
        );
        for m in [1usize, 3, 9] {
            let x: Vec<f32> = fill_f32(&mut s, m * k).iter().map(|v| v * 2.0).collect();
            for (layer, (gl, ul)) in [(&gated, (&g, &u)), (&dense, (&gd, &ud))] {
                let (mut y, mut gy, mut uy) = (
                    vec![f16::ZERO; m * n],
                    vec![f16::ZERO; m * n],
                    vec![f16::ZERO; m * n],
                );
                layer.forward(&x, &mut y, m);
                gl.forward(&x, &mut gy, m);
                ul.forward(&x, &mut uy, m);
                for i in 0..m * n {
                    let (a, b) = (gy[i].to_f32(), uy[i].to_f32());
                    let act = match gate {
                        Gate::Silu => a / (1.0 + (-a).exp()),
                        Gate::Gelu => {
                            0.5 * a * (1.0 + (0.797_884_6 * (a + 0.044_715 * a * a * a)).tanh())
                        }
                    };
                    let want = act * b;
                    let got = y[i].to_f32();
                    assert!(
                        (got - want).abs() <= 2e-3 * (1.0 + want.abs()),
                        "{gate:?} n={n} m={m} [{i}]: {got} vs {want}"
                    );
                }
            }
        }
    }
}

/// An `RMSNorm` folded into the weights gives what normalizing first gives.
#[test]
fn rms_norm_fold_matches_normalizing_first() {
    let (n, k, eps) = (96usize, 80usize, 1e-6f32);
    let mut s = 0x11ea_0300;
    let w = fill_f32(&mut s, n * k);
    let gamma: Vec<f32> = fill_f32(&mut s, k).iter().map(|v| v + 1.0).collect();
    let mut folded = w.clone();
    WeightLayout::OutIn.scale_inputs(&mut folded, n, k, &gamma);
    let mut folded_t = transpose(&w, n, k);
    WeightLayout::InOut.scale_inputs(&mut folded_t, n, k, &gamma);
    assert_eq!(
        transpose(&folded, n, k),
        folded_t,
        "scale_inputs layouts agree"
    );
    let fused = Linear::f16(&folded, WeightLayout::OutIn, n, k).rms_norm_input(eps);
    let plain = Linear::f16(&w, WeightLayout::OutIn, n, k);
    for m in [1usize, 4, 12] {
        let x: Vec<f32> = fill_f32(&mut s, m * k)
            .iter()
            .map(|v| v * 6.0 + 0.5)
            .collect();
        let mut h = vec![0.0f32; m * k];
        nn::rms_norm(&x, &gamma, eps, &mut h);
        let (mut a, mut b) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
        fused.forward_f32(&x, &mut a, m);
        plain.forward_f32(&h, &mut b, m);
        for i in 0..m * n {
            assert!(
                (a[i] - b[i]).abs() <= 2e-2 * (1.0 + b[i].abs()),
                "m={m} [{i}]: {} vs {}",
                a[i],
                b[i]
            );
        }
    }
}
