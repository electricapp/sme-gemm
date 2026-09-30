//! Flash attention vs. the materialized `softmax(QK^T)V` path, and the f16/bf16
//! variants against the f32 one. Interleaved A/B, median of per-round bests.
//! Half columns use native 16-bit accumulate (`Accum::F16` / `Accum::Bf16`);
//! with `Accum::F32` they land within ~10% of the f32 path instead.
//!   cargo run --release --example attention

use std::time::Instant;

use half::{bf16, f16};
use sme_gemm::{
    Accum, FlashParams, flash_attention_bf16_with, flash_attention_f16_with,
    flash_attention_f32_with, gemm_f32, softmax_rows,
};

/// Materialized reference path: full `m x n` scores, row softmax, then `P @ V`.
#[allow(clippy::too_many_arguments)]
fn materialized(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &mut [f32],
    s: &mut [f32],
    m: usize,
    n: usize,
    d: usize,
    dv: usize,
    scale: f32,
) {
    gemm_f32(m, n, d, s, n, 1, q, d, 1, k, 1, d, 0.0, scale);
    softmax_rows(s, m, n);
    gemm_f32(m, dv, n, o, dv, 1, s, n, 1, v, dv, 1, 0.0, 1.0);
}

fn best<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut b = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn fill(seed: &mut u64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect()
}

fn main() {
    println!("== flash attention: f32 vs the materialized path, and the half dtypes ==");
    println!(
        "  {:>18}  {:>9}  {:>9}  {:>7}  {:>9}  {:>9}  {:>9}",
        "m x n x d", "f32 ms", "matzd ms", "vs mat", "f16 ms", "bf16 ms", "f16 TF/s"
    );

    // (m queries, n keys, d head dim); dv == d as in every real attention layer.
    let shapes = [
        (512usize, 512usize, 64usize),
        (1024, 1024, 64),
        (2048, 2048, 64),
        (4096, 4096, 64),
        (1024, 1024, 128),
        (4096, 4096, 128),
        (8192, 8192, 64),
        (1, 4096, 128),
    ];

    for (m, n, d) in shapes {
        let dv = d;
        let mut seed = 0x243f_6a88_85a3_08d3;
        let q = fill(&mut seed, m * d);
        let k = fill(&mut seed, n * d);
        let v = fill(&mut seed, n * dv);
        let scale = 1.0 / (d as f32).sqrt();
        let mut of = vec![0.0f32; m * dv];
        let mut om = vec![0.0f32; m * dv];
        let mut s = vec![0.0f32; m * n];

        // 2*m*n*d for QK^T plus 2*m*n*dv for PV.
        let flop = 2.0 * (m * n * d) as f64 + 2.0 * (m * n * dv) as f64;
        let big = m * n > 4 << 20;
        let (rounds, reps) = if big { (5, 2) } else { (7, 4) };
        let p = FlashParams::auto(m, n);

        let half = |x: &[f32]| -> Vec<f16> { x.iter().map(|&y| f16::from_f32(y)).collect() };
        let bhalf = |x: &[f32]| -> Vec<bf16> { x.iter().map(|&y| bf16::from_f32(y)).collect() };
        let (qh, kh, vh) = (half(&q), half(&k), half(&v));
        let (qb, kb, vb) = (bhalf(&q), bhalf(&k), bhalf(&v));
        let mut oh = vec![f16::ZERO; m * dv];
        let mut ob = vec![bf16::ZERO; m * dv];
        let (acc16, accb) = (Accum::F16, Accum::Bf16);

        let (mut fs, mut ms, mut hs, mut bs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for _ in 0..rounds {
            fs.push(best(reps, || {
                flash_attention_f32_with(&q, &k, &v, &mut of, m, n, d, dv, scale, p);
            }));
            ms.push(best(reps, || {
                materialized(&q, &k, &v, &mut om, &mut s, m, n, d, dv, scale);
            }));
            hs.push(best(reps, || {
                flash_attention_f16_with(&qh, &kh, &vh, &mut oh, m, n, d, dv, scale, acc16, p);
            }));
            bs.push(best(reps, || {
                flash_attention_bf16_with(&qb, &kb, &vb, &mut ob, m, n, d, dv, scale, accb, p);
            }));
        }
        let (tf, tm, th, tb) = (median(fs), median(ms), median(hs), median(bs));

        let err = of
            .iter()
            .zip(&om)
            .map(|(a, b)| f64::from(a - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-4, "flash vs materialized diverged: {err}");

        println!(
            "  {:>6}x{:<5}x{:<4}  {:>9.3}  {:>9.3}  {:>6.2}x  {:>9.3}  {:>9.3}  {:>9.2}",
            m,
            n,
            d,
            tf * 1e3,
            tm * 1e3,
            tm / tf,
            th * 1e3,
            tb * 1e3,
            flop / th / 1e12
        );
    }
}
