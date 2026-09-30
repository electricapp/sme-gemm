//! Batched f16 GEMM (one streaming session for the whole batch) vs a loop of
//! single GEMMs (one streaming session each). For small per-item shapes the
//! per-call streaming entry/exit dominates, so batching amortizes it.
//!
//! Items below `m*n*k = 2^18` are marked `*`: there the single-GEMM path is
//! under the flop floor and runs the scalar reference, so that row measures the
//! floor rather than the cost of batching. Batched has no such floor -- the
//! whole batch clears it -- which is itself the reason to use it, but the
//! speedup on those rows is not "streaming entry amortized".
//!   cargo run --release --example `batched_bench`

use std::time::Instant;

use half::f16;
use sme_gemm::{Accum, matmul_f16, matmul_f16_batched};

/// Best-of-N for both variants, INTERLEAVED: one timed round of each per pass,
/// so neither is measured on the heat the other left behind.
fn best_interleaved(fs: &mut [&mut dyn FnMut()]) -> Vec<f64> {
    for f in fs.iter_mut() {
        for _ in 0..3 {
            f();
        }
    }
    let mut best = vec![f64::INFINITY; fs.len()];
    for _ in 0..12 {
        for (i, f) in fs.iter_mut().enumerate() {
            let t = Instant::now();
            for _ in 0..20 {
                f();
            }
            best[i] = best[i].min(t.elapsed().as_secs_f64() / 20.0);
        }
    }
    best
}

fn main() {
    // (m, n, k, count)
    let cases: &[(usize, usize, usize, usize)] = &[
        (16, 16, 16, 256),
        (32, 32, 32, 128),
        (64, 64, 64, 64),
        (32, 128, 64, 64),
        (96, 96, 96, 32),
    ];
    println!("== batched f16 vs loop of single GEMMs ==");
    println!(
        "  {:>20}  {:>10}  {:>10}  {:>8}",
        "shape x count", "batched", "loop", "speedup"
    );
    for &(m, n, k, count) in cases {
        let a = vec![f16::from_f32(0.01); count * m * k];
        let b = vec![f16::from_f32(0.02); count * k * n];
        let mut c = vec![f16::ZERO; count * m * n];
        let mut c2 = vec![f16::ZERO; count * m * n];

        let mut batched = || matmul_f16_batched(&a, &b, &mut c, m, n, k);
        let mut looped = || {
            for i in 0..count {
                matmul_f16(
                    &a[i * m * k..(i + 1) * m * k],
                    &b[i * k * n..(i + 1) * k * n],
                    &mut c2[i * m * n..(i + 1) * m * n],
                    m,
                    n,
                    k,
                    Accum::F16,
                );
            }
        };
        let s = best_interleaved(&mut [&mut batched, &mut looped]);
        let floor = if m * n * k < 1 << 18 { "*" } else { " " };
        println!(
            "  {:>4}x{:<3}x{:<3} x{:<4}  {:>8.1}us  {:>8.1}us  {:>7.2}x{floor}",
            m,
            n,
            k,
            count,
            s[0] * 1e6,
            s[1] * 1e6,
            s[1] / s[0],
        );
    }
    println!("\n* single-GEMM side is below the 2^18 flop floor (scalar reference),");
    println!("  so that row measures the floor, not the cost of batching.");
}
