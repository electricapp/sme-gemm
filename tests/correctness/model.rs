//! The model-level helpers: `Linear` against the Q4 / packed kernels it wraps,
//! `prepack` for either weight layout, and the `nn` norms against f64.

use crate::fill_f32;
use half::f16;
use sme_gemm::{
    Epilogue, Gate, GatedLinear, HotPool, KvCache, Linear, Mlp, Q4Params, Q4Weights, SelfAttention,
    WeightLayout, caps, matmul_f16_packed, matmul_f32_packed, matmul_q4, matmul_q4_ep, nn, prepack,
    prepack_f16, prepack_f32,
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

/// `SelfAttention::accumulate` gives the same bits, and leaves the same cache,
/// with a pool alive (one row then overlaps the attention with both GEMVs) as
/// without, and matches its steps run by hand on the unpermuted projection to
/// f16 rounding: a prompt of several rows, then one row at a time, plain
/// multi-head and grouped-query, 4-bit and f16. GPT-2 small's shape runs `qkv`
/// as GEMV passes of 5 and 4 bands, which round differently, so the grouping
/// must not move columns between them on one path only.
#[test]
fn self_attention_matches_its_steps() {
    let lay = WeightLayout::OutIn;
    for &(heads, kv_heads, hd) in &[(4usize, 4usize, 64usize), (6, 2, 32), (6, 6, 64)] {
        let d = heads * hd;
        let wq = (heads + 2 * kv_heads) * hd;
        let mut s = 0x11ea_0400 ^ (heads * 31 + kv_heads) as u64;
        let (w1, b1) = (fill_f32(&mut s, wq * d), fill_f32(&mut s, wq));
        let (w2, b2) = (fill_f32(&mut s, d * d), fill_f32(&mut s, d));
        for q4 in [true, false] {
            let mk = |w: &[f32], n: usize, k: usize, b: &[f32]| {
                if q4 {
                    Linear::quantize(w, lay, n, k).with_bias(b)
                } else {
                    Linear::f16(w, lay, n, k).with_bias(b)
                }
            };
            let attn =
                SelfAttention::new(mk(&w1, wq, d, &b1), mk(&w2, d, d, &b2), heads, kv_heads, hd);
            let (qkv, out) = (mk(&w1, wq, d, &b1), mk(&w2, d, d, &b2));
            let (mut solo_c, mut pool_c) = (attn.cache(32), attn.cache(32));
            let mut hand_c = KvCache::new(heads, kv_heads, hd, 32);
            for (step, m) in [3usize].into_iter().chain([1; 20]).enumerate() {
                let x: Vec<f32> = fill_f32(&mut s, m * d).iter().map(|v| v * 4.0).collect();
                // By hand, on the projection as given.
                let mut want = vec![0.25f32; m * d];
                let mut h = vec![f16::ZERO; m * wq];
                qkv.forward(&x, &mut h, m);
                let (ql, kl) = (heads * hd, kv_heads * hd);
                let mut q = Vec::new();
                for r in 0..m {
                    let row = &h[r * wq..(r + 1) * wq];
                    q.extend_from_slice(&row[..ql]);
                    hand_c.push(&row[ql..ql + kl], &row[ql + kl..]);
                }
                let mut a = vec![0.0f32; m * ql];
                if m == 1 {
                    hand_c.attend(&q, &mut a);
                } else {
                    hand_c.attend_causal(&q, m, &mut a);
                }
                out.accumulate(&a, &mut want, m);
                let mut solo = vec![0.25f32; m * d];
                attn.accumulate(&x, &mut solo_c, &mut solo, m);
                let mut pooled = vec![0.25f32; m * d];
                {
                    let _pool = HotPool::new(2);
                    attn.accumulate(&x, &mut pool_c, &mut pooled, m);
                }
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                let tag = format!("{heads}/{kv_heads}x{hd} q4={q4} step={step}");
                assert_eq!(bits(&pooled), bits(&solo), "{tag}");
                assert_eq!(pool_c.keys(), solo_c.keys(), "{tag} keys");
                assert_eq!(pool_c.values(), solo_c.values(), "{tag} values");
                assert_eq!(solo_c.len(), hand_c.len(), "{tag}");
                // The grouping moves columns between GEMV passes that round
                // differently (~3e-3 relative on q/k/v), which softmax then
                // amplifies on single outputs: compare whole rows.
                let (err, norm) = solo.iter().zip(&want).fold((0f32, 0f32), |(e, n), (g, w)| {
                    (e + (g - w) * (g - w), n + w * w)
                });
                assert!(
                    err.sqrt() <= 2e-2 * norm.sqrt(),
                    "{tag}: |by hand - block| {} vs |by hand| {}",
                    err.sqrt(),
                    norm.sqrt()
                );
            }
        }
    }
}

/// `Mlp::accumulate` gives the same bits as its three calls run one after
/// another: plain and gated, 4-bit and f16, with a pool alive (one row then
/// overlaps the activation with both GEMVs) and without. The hidden width
/// spans several GEMV passes and ends in a partial band, block and chunk.
#[test]
fn mlp_matches_three_calls() {
    let (d, f) = (160usize, 1100usize);
    let mut s = 0x11ea_0300;
    let (w1, b1) = (fill_f32(&mut s, f * d), fill_f32(&mut s, f));
    let (w2, b2) = (fill_f32(&mut s, d * f), fill_f32(&mut s, d));
    let (wg, bg) = (fill_f32(&mut s, f * d), fill_f32(&mut s, f));
    let lay = WeightLayout::OutIn;
    for act in [Gate::Gelu, Gate::Silu] {
        let ep = match act {
            Gate::Gelu => Epilogue::new().gelu(),
            Gate::Silu => Epilogue::new().silu(),
        };
        let up = || Linear::quantize(&w1, lay, f, d).with_bias(&b1);
        let gated = || GatedLinear::quantize(&wg, &w1, lay, f, d, act).with_bias(&bg, &b1);
        let down = || Linear::quantize(&w2, lay, d, f).with_bias(&b2);
        let (u, g, dn) = (up(), gated(), down());
        let blocks = [
            (Mlp::new(up(), act, down()), false),
            (Mlp::gated(gated(), down()), true),
            (
                Mlp::new(
                    Linear::f16(&w1, lay, f, d),
                    act,
                    Linear::f16(&w2, lay, d, f),
                ),
                false,
            ),
        ];
        let (uf, df) = (Linear::f16(&w1, lay, f, d), Linear::f16(&w2, lay, d, f));
        for pool in [false, true] {
            let _pool = pool.then(|| HotPool::new(2));
            for m in [1usize, 3] {
                for rep in 0..if m == 1 { 40 } else { 2 } {
                    let x = fill_f32(&mut s, m * d);
                    for (i, (mlp, is_gated)) in blocks.iter().enumerate() {
                        let mut h = vec![f16::ZERO; m * f];
                        let mut want = vec![0.25f32; m * d];
                        match (i, is_gated) {
                            (2, _) => {
                                uf.forward_ep(&x, &mut h, m, &ep);
                                df.accumulate(&h, &mut want, m);
                            }
                            (_, true) => {
                                g.forward(&x, &mut h, m);
                                dn.accumulate(&h, &mut want, m);
                            }
                            _ => {
                                u.forward_ep(&x, &mut h, m, &ep);
                                dn.accumulate(&h, &mut want, m);
                            }
                        }
                        let mut got = vec![0.25f32; m * d];
                        mlp.accumulate(&x, &mut got, m);
                        let (gb, wb): (Vec<u32>, Vec<u32>) = (
                            got.iter().map(|v| v.to_bits()).collect(),
                            want.iter().map(|v| v.to_bits()).collect(),
                        );
                        assert_eq!(gb, wb, "{act:?} block {i} m={m} pool={pool} rep={rep}");
                    }
                }
            }
        }
    }
}
