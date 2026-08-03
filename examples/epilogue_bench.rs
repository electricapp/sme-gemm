//! Fused op-graph epilogue overhead vs the bare GEMM. The fused store applies
//! bias+activation while the accumulator is still live in registers; the goal is
//! for it to cost barely more than the raw `gemm` store. The headline metric is
//! therefore the fused/gemm OVERHEAD. The `separate` column (gemm + a standalone
//! bias/relu post-pass over the whole output) is kept only as a secondary
//! reference for the work the fusion deletes.
//!   cargo run --release --example `epilogue_bench`

use std::time::Instant;

use half::f16;
use sme_gemm::{Gemm, matmul_f16_packed, prepack_f16};

/// Best-of-N for several variants, INTERLEAVED: one timed round of each variant
/// per pass, rather than running each to completion in turn.
///
/// Measuring the variants sequentially made this bench unusable. Each column
/// heats the machine for the next, and the `separate` column is an order of
/// magnitude slower than the others, so the reported overheads drifted with
/// position rather than with cost -- across two runs of the same binary the
/// bias-only column read +2% and +75%, and sometimes came out FASTER than the
/// bare gemm it is a superset of. Interleaving puts every variant under the same
/// thermal conditions.
fn best_interleaved(iters: usize, rounds: usize, fs: &mut [&mut dyn FnMut()]) -> Vec<f64> {
    for f in fs.iter_mut() {
        for _ in 0..3 {
            f();
        }
    }
    let mut best = vec![f64::INFINITY; fs.len()];
    for _ in 0..rounds {
        for (i, f) in fs.iter_mut().enumerate() {
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            best[i] = best[i].min(t.elapsed().as_secs_f64() / iters as f64);
        }
    }
    best
}

fn main() {
    // Shapes that stress the store (shallow K / wide output, where the epilogue is
    // a larger fraction of the work) plus a square compute-bound one.
    let shapes = [
        (4096, 4096, 512), // square-ish, moderate K
        (4096, 4096, 128), // shallow K, wide output -> store-dominated
        (16384, 512, 512), // tall, store-heavy
        (8192, 8192, 256), // large + shallow K
    ];
    println!("== fused op-graph epilogue overhead vs bare gemm ==");
    println!(
        "  {:>20}  {:>9}  {:>11}  {:>14}  {:>14}  {:>10}",
        "shape (m x n x k)", "gemm", "config", "bias-only", "bias+relu", "separate"
    );
    for &(m, n, k) in &shapes {
        let a = vec![f16::from_f32(0.01); m * k];
        let b = vec![f16::from_f32(0.02); k * n];
        let bias = vec![f16::from_f32(-0.5); n];
        let packed = prepack_f16(&b, n, k);
        let mut c = vec![f16::ZERO; m * n];
        let iters = if m * n * k > 64 << 20 { 4 } else { 40 };

        // Each variant needs its own output buffer so the closures can be live
        // at once; they are the same size, so this does not change what is timed.
        let (mut c1, mut c2, mut c3) = (c.clone(), c.clone(), c.clone());
        let t = {
            let mut f_gemm = || matmul_f16_packed(&a, &packed, &mut c, m);
            // bias+relu: the store does the mandatory ZA-read + relu.
            let mut f_relu = || {
                Gemm::new(&a, &packed, m).add_col(&bias).relu().run(&mut c1);
            };
            // bias-only: a leading ADD_COL folds into the accumulator (FMOPA
            // init) and the store reverts to the direct single-instruction path.
            let mut f_bias = || {
                Gemm::new(&a, &packed, m).add_col(&bias).run(&mut c2);
            };
            let mut f_sep = || {
                matmul_f16_packed(&a, &packed, &mut c3, m);
                // standalone bias + relu post-pass over the whole output
                for i in 0..m {
                    for j in 0..n {
                        let v = c3[i * n + j].to_f32() + bias[j].to_f32();
                        c3[i * n + j] = f16::from_f32(if v < 0.0 { 0.0 } else { v });
                    }
                }
            };
            best_interleaved(
                iters,
                12,
                &mut [&mut f_gemm, &mut f_relu, &mut f_bias, &mut f_sep],
            )
        };
        let (gemm_only, fused, fused_bias, separate) = (t[0], t[1], t[2], t[3]);

        let rb = fused_bias / gemm_only;
        let ratio = fused / gemm_only;
        println!(
            "  {:>6}x{:<5}x{:<5}  {:>7.3}ms  {:>11}  {:>7.3}ms (+{:>3.0}%)  {:>7.3}ms (+{:>3.0}%)  {:>8.3}ms",
            m,
            n,
            k,
            gemm_only * 1e3,
            "",
            fused_bias * 1e3,
            100.0 * (rb - 1.0),
            fused * 1e3,
            100.0 * (ratio - 1.0),
            separate * 1e3,
        );
    }
}
