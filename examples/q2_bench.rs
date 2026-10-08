//! One-row 2-bit GEMM (`matmul_q2`) against the Q4 GEMV (`matmul_q4`) on the
//! same shapes: weights hot in the P-cluster's L2 (one matrix, called again
//! and again) and cold (cycling through enough matrices to stream from DRAM).
//! The 2-bit call includes quantizing its rows of A to int8. `Q2_BLOCK` sets
//! its K-block size (default 256), `Q2_ROWS` the rows per call (default 1).
//!   cargo run --release --example `q2_bench`

use std::time::Instant;

use half::f16;
use sme_gemm::{Q2Params, Q2Weights, Q4Weights, caps, matmul_q2, matmul_q4};

fn bytes(s: &mut u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|_| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (*s >> 40) as u8
        })
        .collect()
}

/// Best seconds per call over a few rounds of `calls` calls each.
fn best(calls: usize, mut f: impl FnMut(usize)) -> f64 {
    for i in 0..calls {
        f(i);
    }
    let mut b = f64::INFINITY;
    for _ in 0..8 {
        let t = Instant::now();
        for i in 0..calls {
            f(i);
        }
        b = b.min(t.elapsed().as_secs_f64() / calls as f64);
    }
    b
}

fn main() {
    let m: usize = std::env::var("Q2_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    println!("{:?}\n{m} row(s)\n", caps());
    println!("  n x k        sets | Q4 GEMV          | 2-bit GEMV       | speedup");
    for &(n, k, sets) in &[
        (1536usize, 384usize, 1usize),
        (2048, 2048, 1),
        (4096, 4096, 1),
        (4096, 4096, 8),
    ] {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let q4: Vec<Q4Weights> = (0..sets)
            .map(|_| {
                let scales: Vec<f16> = bytes(&mut s, n * k.div_ceil(32))
                    .iter()
                    .map(|&b| f16::from_f32(0.01 + f32::from(b) * 1e-4))
                    .collect();
                Q4Weights::new(&bytes(&mut s, (k * n).div_ceil(2)), &scales, n, k)
            })
            .collect();
        let p = std::env::var("Q2_BLOCK")
            .ok()
            .and_then(|v| v.parse().ok())
            .map_or_else(Q2Params::default, Q2Params::new);
        let q2: Vec<Q2Weights> = (0..sets)
            .map(|_| {
                let scales: Vec<f32> = bytes(&mut s, n * k.div_ceil(p.block()))
                    .iter()
                    .map(|&b| 0.01 + f32::from(b) * 1e-4)
                    .collect();
                Q2Weights::new(&bytes(&mut s, (k * n).div_ceil(4)), &scales, n, k, p)
            })
            .collect();
        let a: Vec<f16> = bytes(&mut s, m * k)
            .iter()
            .map(|&b| f16::from_f32((f32::from(b) - 128.0) / 64.0))
            .collect();
        let mut c = vec![f16::ZERO; m * n];
        let calls = 64.max(sets * 8);
        let t4 = best(calls, |i| matmul_q4(&a, &q4[i % sets], &mut c, m));
        let t2 = best(calls, |i| matmul_q2(&a, &q2[i % sets], &mut c, m));
        let macs = (n * k) as f64;
        let (b4, b2) = (macs * (0.5 + 2.0 / 32.0), macs * (0.25 + 4.0 / 256.0));
        println!(
            "  {n:>4} x {k:<4}  {sets:>3}  | {:>6.1} us {:>4.0} GB/s | {:>6.1} us {:>4.0} GB/s | {:.2}x",
            t4 * 1e6,
            b4 / t4 / 1e9,
            t2 * 1e6,
            b2 / t2 / 1e9,
            t4 / t2
        );
    }
}
