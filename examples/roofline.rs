//! Isolated single-dtype throughput, mirroring phonon's `pure_gemm` methodology
//! (pre-packed B, best-of-12, one dtype at a time, cooldown between) so the
//! shared SME cluster isn't pre-heated by other dtypes. Run one dtype:
//!   cargo run --release --example roofline -- f16
//!   cargo run --release --example roofline -- f32   (etc: bf16 i8)

use std::time::Instant;

use half::{bf16, f16};
use sme_gemm::{
    matmul_bf16_packed, matmul_f16_packed, matmul_f32, matmul_i8_packed, prepack_bf16, prepack_f16,
    prepack_i8,
};

fn best<F: FnMut()>(big: bool, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let batches = 12;
    let iters = if big { 4 } else { 40 };
    let mut best = f64::INFINITY;
    for _ in 0..batches {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / f64::from(iters));
    }
    best
}

fn main() {
    let dt = std::env::args().nth(1).unwrap_or_else(|| "f16".into());
    // Optional 2nd arg: filter to a single square size by M (e.g. `4096`), so a
    // size can be measured in isolation/cold for clean cache-blocking A/B
    // comparison without the 256->4096 walk pre-heating the chip.
    let only: Option<usize> = std::env::args().nth(2).and_then(|s| s.parse().ok());
    let all: &[(usize, usize, usize)] = &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 512, 512),
        (16384, 512, 512),
    ];
    let filtered: Vec<(usize, usize, usize)> = all
        .iter()
        .copied()
        .filter(|&(m, _, _)| only.is_none_or(|o| m == o))
        .collect();
    let sizes: &[(usize, usize, usize)] = &filtered;
    // Apple SME MOPA mac-rate is the same for f16f16 (32x32x1), b16b16
    // (32x32x1) and i8->i32 (16x16x4) -- all 1024 macs/instruction -- so those
    // three share one roofline. i8 is NOT 2x f16 on this uarch (the i32
    // accumulator halves the tile to 16x16, the 4-deep dot restores it).
    //
    // MEASURED ceilings, not estimates: f16/bf16/i8 all plateau at 4.8-4.9 across
    // shapes whose arithmetic intensity differs ~4x (2048^3 vs 16384x512x512),
    // which is the signature of a compute-bound issue limit. f32 FMOPA does a
    // quarter the macs per instruction (16x16x1) but issues ~1.85x faster, so it
    // lands at ~2.13x less throughput -- also its own ceiling, not bandwidth.
    let roof = match dt.as_str() {
        "f32" => 2.25,
        _ => 4.9,
    };
    println!("== {dt} (roofline ~{roof} TF/s) ==");
    for &(m, n, k) in sizes {
        let big = m * n * k > 64 << 20;
        let tf = match dt.as_str() {
            "f16" => {
                let a = vec![f16::from_f32(0.01); m * k];
                let b = vec![f16::from_f32(0.02); k * n];
                let p = prepack_f16(&b, n, k);
                let mut c = vec![f16::ZERO; m * n];
                let s = best(big, || matmul_f16_packed(&a, &p, &mut c, m));
                2.0 * (m * n * k) as f64 / s / 1e12
            }
            "bf16" => {
                let a = vec![bf16::from_f32(0.01); m * k];
                let b = vec![bf16::from_f32(0.02); k * n];
                let p = prepack_bf16(&b, n, k);
                let mut c = vec![bf16::ZERO; m * n];
                let s = best(big, || matmul_bf16_packed(&a, &p, &mut c, m));
                2.0 * (m * n * k) as f64 / s / 1e12
            }
            "f32" => {
                let a = vec![0.01f32; m * k];
                let b = vec![0.02f32; k * n];
                let mut c = vec![0.0f32; m * n];
                let s = best(big, || matmul_f32(&a, &b, &mut c, m, n, k));
                2.0 * (m * n * k) as f64 / s / 1e12
            }
            "i8" => {
                let a = vec![1i8; m * k];
                let b = vec![2i8; k * n];
                let p = prepack_i8(&b, n, k);
                let mut c = vec![0i32; m * n];
                let s = best(big, || matmul_i8_packed(&a, &p, &mut c, m));
                2.0 * (m * n * k) as f64 / s / 1e12
            }
            _ => 0.0,
        };
        println!(
            "  {m:>6}x{n:<5}x{k:<5}  {tf:>6.2} TF/s   {:>3.0}% roof",
            100.0 * tf / roof
        );
    }
}
