//! sme-gemm vs the burn framework's `NdArray` CPU backend, same data/best-of-N.
//! (`NdArray` is f32/f64 only -- no f16 -- so this is the f32 fight.)
//!   cargo run --release --features burn --example `vs_burn`

use std::time::Instant;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Distribution, Tensor};
use sme_gemm::matmul_f32;

/// Best-of-N for both backends, INTERLEAVED: one timed round of each per pass.
/// Run to completion in turn instead, whichever goes first pays the clock ramp
/// and whichever goes second inherits the other's heat.
fn best_interleaved(big: bool, fs: &mut [&mut dyn FnMut()]) -> Vec<f64> {
    for f in fs.iter_mut() {
        for _ in 0..3 {
            f();
        }
    }
    let iters = if big { 4 } else { 30 };
    let mut best = vec![f64::INFINITY; fs.len()];
    for _ in 0..10 {
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

fn main() {
    let sizes: &[(usize, usize, usize)] = &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 512, 512),
    ];
    let dev = NdArrayDevice::default();

    println!("== f32: sme-gemm vs burn NdArray (TF/s) ==");
    println!(
        "  {:>16}  {:>8}  {:>8}  {:>8}",
        "shape", "sme", "burn", "speedup"
    );
    for &(m, n, k) in sizes {
        let big = m * n * k > 64 << 20;
        let flop = 2.0 * (m * n * k) as f64;
        let av = vec![0.02f32; m * k];
        let bv = vec![0.03f32; k * n];
        let mut c = vec![0.0f32; m * n];
        let a = Tensor::<NdArray<f32>, 2>::random([m, k], Distribution::Default, &dev);
        let b = Tensor::<NdArray<f32>, 2>::random([k, n], Distribution::Default, &dev);
        let mut s_sme = || matmul_f32(&av, &bv, &mut c, m, n, k);
        let mut s_burn = || {
            std::hint::black_box(a.clone().matmul(b.clone()));
        };
        let t = best_interleaved(big, &mut [&mut s_sme, &mut s_burn]);
        let (sme, burn) = (t[0], t[1]);
        println!(
            "  {m:>5}x{n:<4}x{k:<4}  {:>8.2}  {:>8.2}  {:>7.2}x",
            flop / sme / 1e12,
            flop / burn / 1e12,
            burn / sme,
        );
    }
}
