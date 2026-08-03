//! sme-gemm's SME backend vs vanilla candle CPU matmul (candle uses the `gemm`
//! crate). Same tensors, same best-of-N for both.
//!   cargo run --release --features candle --example `vs_candle`

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use sme_gemm::candle::sme_matmul;

/// Best-of-N for both backends, INTERLEAVED: one timed round of each per pass.
/// Run to completion in turn instead, whichever goes first pays the clock ramp
/// and whichever goes second inherits the other's heat.
fn best_interleaved(big: bool, fs: &mut [&mut dyn FnMut()]) -> Vec<f64> {
    for f in fs.iter_mut() {
        for _ in 0..3 {
            f();
        }
    }
    let iters = if big { 4 } else { 40 };
    let mut best = vec![f64::INFINITY; fs.len()];
    for _ in 0..12 {
        for (i, f) in fs.iter_mut().enumerate() {
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            best[i] = best[i].min(t.elapsed().as_secs_f64() / f64::from(iters));
        }
    }
    best
}

fn run(dev: &Device, dt: DType, label: &str) {
    let sizes: &[(usize, usize, usize)] = &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 512, 512),
        (16384, 512, 512),
    ];
    println!("== {label}: sme-gemm backend vs vanilla candle (gemm crate) ==");
    println!(
        "  {:>16}  {:>10}  {:>10}  {:>8}",
        "shape", "sme TF/s", "candle TF/s", "speedup"
    );
    for &(m, n, k) in sizes {
        let a = Tensor::randn(0f32, 1f32, (m, k), dev)
            .expect("candle call failed")
            .to_dtype(dt)
            .expect("candle call failed");
        let b = Tensor::randn(0f32, 1f32, (k, n), dev)
            .expect("candle call failed")
            .to_dtype(dt)
            .expect("candle call failed");
        let big = m * n * k > 64 << 20;
        let flop = 2.0 * (m * n * k) as f64;

        let mut sme = || {
            sme_matmul(&a, &b).expect("candle call failed");
        };
        let mut vanilla = || {
            a.matmul(&b).expect("candle call failed");
        };
        let s = best_interleaved(big, &mut [&mut sme, &mut vanilla]);
        let tf_sme = flop / s[0] / 1e12;
        let tf_candle = flop / s[1] / 1e12;
        println!(
            "  {:>5}x{:<4}x{:<4}  {tf_sme:>10.2}  {tf_candle:>11.2}  {:>7.2}x",
            m,
            n,
            k,
            tf_sme / tf_candle,
        );
    }
}

fn main() {
    let dev = Device::Cpu;
    run(&dev, DType::F32, "f32");
    println!();
    run(&dev, DType::F16, "f16");
}
