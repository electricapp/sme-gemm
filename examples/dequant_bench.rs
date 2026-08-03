//! Fused i8/i16 dequant store throughput: the row-major dequant store runs
//! vectorized (i32/i64 -> f32 convert + scale + in-register op-graph) rather than
//! as a per-cell scalar loop. Run: `cargo run --release --example dequant_bench`.

use sme_gemm::{Dequant, caps, matmul_i8_packed_dequant, matmul_i16_dequant, prepack_i8};

fn best<F: FnMut()>(m: usize, n: usize, k: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let iters = if m * n * k > 64 << 20 { 10 } else { 40 };
    let mut b = f64::INFINITY;
    for _ in 0..iters {
        let t = std::time::Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    2.0 * (m * n * k) as f64 / b / 1e12
}

fn main() {
    println!("{:?}\n", caps());
    println!("  M     N    K    | i8->f32 deq(TOPS)  i16->f32 deq(TOPS)  [bias+gelu]");
    // Small shapes (m*n*k < 2^25) exercise the run_small path, which shares the
    // vectorized dequant store with run_streaming.
    for &(m, n, k) in &[
        (128usize, 128, 128),
        (256, 256, 128),
        (1024, 1024, 1024),
        (2048, 2048, 512),
        (4096, 512, 512),
    ] {
        let scale = 0.001f32;
        let bias: Vec<f32> = (0..n).map(|j| 0.01 * (j % 7) as f32).collect();

        let ai = vec![1i8; m * k];
        let bi = vec![2i8; k * n];
        let mut ci = vec![0.0f32; m * n];
        let pk = prepack_i8(&bi, n, k);
        let dq8 = Dequant::new(scale).add_col(&bias).gelu();
        let t8 = best(m, n, k, || {
            matmul_i8_packed_dequant(&ai, &pk, &mut ci, m, &dq8);
        });

        let a16 = vec![1i16; m * k];
        let b16 = vec![2i16; k * n];
        let mut c16 = vec![0.0f32; m * n];
        let dq16 = Dequant::new(scale).add_col(&bias).gelu();
        let t16 = if caps().sme_i16i64 {
            best(m, n, k, || {
                matmul_i16_dequant(&a16, &b16, &mut c16, m, n, k, &dq16);
            })
        } else {
            0.0
        };

        println!("  {m:<5} {n:<4} {k:<4} | {t8:>10.2}        {t16:>10.2}");
    }
    println!("\n(TOPS = 2*M*N*K / time; fused dequant + bias + gelu, row-major f32 out)");
}
