//! The SME wake cost and `SmeWarm`: an m=1 Q4 GEMV timed back to back, after
//! a microsecond of NEON work, and after the same work with an `SmeWarm` alive.
//! Run with `SME_GEMM_TRACE=1` to see each call reported with its idle gap.
//!   cargo run --release --example `sme_warm`

use std::time::{Duration, Instant};

use half::f16;
use sme_gemm::{Linear, SmeWarm, WeightLayout};

/// Median microseconds of `gemv` over 4000 calls, each after `gap_us` of NEON work.
fn median_us(gemv: &mut impl FnMut(), gap_us: f64, sink: &mut f32) -> f64 {
    let gap = Duration::from_secs_f64(gap_us * 1e-6);
    let mut ts = Vec::with_capacity(4000);
    for i in 0..4000 {
        let t = Instant::now();
        while t.elapsed() < gap {
            *sink = (*sink).mul_add(0.999, 0.001);
        }
        let t = Instant::now();
        gemv();
        if i >= 400 {
            ts.push(t.elapsed().as_secs_f64());
        }
    }
    ts.sort_by(f64::total_cmp);
    ts[ts.len() / 2] * 1e6
}

fn main() {
    let (n, k) = (1152, 384);
    let w: Vec<f32> = (0..n * k).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect();
    let fc = Linear::quantize(&w, WeightLayout::OutIn, n, k);
    let x = vec![f16::from_f32(0.5); k];
    let mut y = vec![f16::ZERO; n];
    let mut gemv = || fc.forward(&x, &mut y, 1);
    let mut sink = 0.0;
    let back_to_back = median_us(&mut gemv, 0.0, &mut sink);
    let after_work = median_us(&mut gemv, 1.0, &mut sink);
    let warm = SmeWarm::new();
    let warmed = median_us(&mut gemv, 1.0, &mut sink);
    drop(warm);
    println!("1x{n}x{k} Q4 GEMV, median us/call");
    println!("  back to back:              {back_to_back:.2}");
    println!("  after 1 us of NEON work:   {after_work:.2}");
    println!("  same, with an SmeWarm:     {warmed:.2}");
    println!("({sink})");
}
