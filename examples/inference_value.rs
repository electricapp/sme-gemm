//! Native f16 GEMM and i8 GEMM + fused dequant vs Accelerate, which has no f16
//! or i8 cblas GEMM. The f16 baseline upcasts f16->f32, runs `cblas_sgemm`, and
//! downcasts f32->f16; B is converted once, A and C per call.
//!   cargo run --release --example `inference_value`

use std::time::Instant;

use half::f16;
use sme_gemm::{Accum, Dequant, matmul_f16, matmul_i8_packed_dequant, prepack_i8};

const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

fn best<F: FnMut()>(big: bool, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let iters = if big { 4 } else { 40 };
    let mut best = f64::INFINITY;
    for _ in 0..12 {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / f64::from(iters));
    }
    best
}

fn main() {
    // FFN-style GEMM shapes (tokens x hidden).
    let sizes: &[(usize, usize, usize)] = &[
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 11008, 4096),
    ];

    println!("== f16: sme-gemm native vs Accelerate (upcast f32 + sgemm + downcast) ==");
    println!(
        "  {:>16}  {:>10}  {:>14}  {:>8}",
        "shape", "sme TF/s", "accel-f16 TF/s", "speedup"
    );
    for &(m, n, k) in sizes {
        let a16 = vec![f16::from_f32(0.01); m * k];
        let b16 = vec![f16::from_f32(0.02); k * n];
        let mut c16 = vec![f16::ZERO; m * n];
        let big = m * n * k > 64 << 20;
        let flop = 2.0 * (m * n * k) as f64;

        let s_sme = best(big, || {
            matmul_f16(&a16, &b16, &mut c16, m, n, k, Accum::F16);
        });

        // B converted once; A and C converted per call around the sgemm.
        let bf: Vec<f32> = b16.iter().map(|x| x.to_f32()).collect();
        let mut af = vec![0.0f32; m * k];
        let mut cf = vec![0.0f32; m * n];
        let s_acc = best(big, || {
            for (d, s) in af.iter_mut().zip(&a16) {
                *d = s.to_f32();
            }
            // SAFETY: row-major f32 buffers sized m*k / k*n / m*n.
            unsafe {
                cblas_sgemm(
                    ROW_MAJOR,
                    NO_TRANS,
                    NO_TRANS,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    af.as_ptr(),
                    k as i32,
                    bf.as_ptr(),
                    n as i32,
                    0.0,
                    cf.as_mut_ptr(),
                    n as i32,
                );
            }
            for (d, s) in c16.iter_mut().zip(&cf) {
                *d = f16::from_f32(*s);
            }
        });
        let tf_sme = flop / s_sme / 1e12;
        let tf_acc = flop / s_acc / 1e12;
        println!(
            "  {:>5}x{:<5}x{:<4}  {tf_sme:>10.2}  {tf_acc:>14.2}  {:>7.2}x",
            m,
            n,
            k,
            tf_sme / tf_acc,
        );
    }

    println!("\n== i8 -> i32 + fused dequant: sme-gemm (Accelerate has no cblas i8 GEMM) ==");
    println!("  {:>16}  {:>10}", "shape", "sme TOP/s");
    for &(m, n, k) in sizes {
        let a = vec![1i8; m * k];
        let b = vec![2i8; k * n];
        let packed = prepack_i8(&b, n, k);
        let bias = vec![0.0f32; n];
        let mut c = vec![0.0f32; m * n];
        let dq = Dequant::new(0.001).add_col(&bias).relu();
        let big = m * n * k > 64 << 20;
        let op = 2.0 * (m * n * k) as f64;
        let s = best(big, || {
            matmul_i8_packed_dequant(&a, &packed, &mut c, m, &dq);
        });
        println!("  {:>5}x{:<5}x{:<4}  {:>10.2}", m, n, k, op / s / 1e12);
    }
}
