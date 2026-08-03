//! 4-bit-resident GEMM (`matmul_q4`) against eager dequant to f16 (`dequant_q4`
//! then `matmul_f16_packed`), which is the same math with the weights expanded
//! up front. Weight packing is outside the timed region for both.
//!
//! For a single weight set, eager leads throughout: residency buys a 4x smaller
//! footprint at a modest throughput cost, and at small M the on-the-fly dequant
//! (O(n*k) whatever M is) has almost no MOPA work to hide behind. Residency
//! turns into a speed win only once the TOTAL weight working set stops fitting
//! -- with several distinct weight sets live, the f16 expansion saturates
//! bandwidth and the ordering inverts.
//!   cargo run --release --example `q4_bench`

use std::time::Instant;

use half::f16;
use sme_gemm::{Q4Weights, caps, dequant_q4, matmul_f16_packed, matmul_q4};

fn best<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut b = f64::INFINITY;
    for _ in 0..12 {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    b
}

fn main() {
    println!("{:?}\n", caps());
    println!("  M     N    K    | q4-resident   eager-f16   4-bit B");
    // Decode-shaped (small M) through prefill-shaped, since the resident path's
    // whole point is holding a large weight set while M is small.
    for &(m, n, k) in &[
        (1usize, 4096usize, 4096usize),
        (16, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (1024, 1024, 1024),
    ] {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut byte = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as u8
        };
        let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
        let scales: Vec<f16> = (0..n * k.div_ceil(32))
            .map(|i| f16::from_f32(0.01 + 0.001 * (i % 5) as f32))
            .collect();
        let a = vec![f16::from_f32(0.01); m * k];
        let mut c = vec![f16::ZERO; m * n];

        let w = Q4Weights::new(&quants, &scales, n, k);
        let q4 = best(|| matmul_q4(&a, &w, &mut c, m));
        let p = dequant_q4(&quants, &scales, n, k);
        let eager = best(|| matmul_f16_packed(&a, &p, &mut c, m));

        println!(
            "  {m:<5} {n:<4} {k:<4} | {:>8.2} ms  {:>8.2} ms  {:>6} KiB",
            q4 * 1e3,
            eager * 1e3,
            k * n / 2 / 1024,
        );
    }
}
