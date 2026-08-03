//! Head-to-head vs Apple Accelerate's `cblas_sgemm`/`cblas_dgemm` (f32/f64 --
//! the shared dtypes), the same best-of-N methodology for both.
//!   cargo run --release --example `vs_accelerate`

use std::time::Instant;

use sme_gemm::{matmul_f32, matmul_f64};

// CBLAS constants.
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
    #[allow(clippy::too_many_arguments)]
    fn cblas_dgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

/// Best-of-N for both backends, INTERLEAVED: one timed round of each per pass.
/// Measured to completion in turn, whichever ran first paid the clock ramp and
/// whichever ran second inherited the other's heat, which is worth more than the
/// gap being measured.
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

/// Bring the clocks up so the first shape measured is not reported low.
fn warmup() {
    let (n, k) = (1024, 1024);
    let a = vec![0.01f32; n * k];
    let b = vec![0.02f32; k * n];
    let mut c = vec![0.0f32; n * n];
    for _ in 0..60 {
        matmul_f32(&a, &b, &mut c, n, n, k);
    }
}

fn main() {
    warmup();
    let sizes: &[(usize, usize, usize)] = &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 512, 512),
        (16384, 512, 512),
    ];
    println!("== f32: sme-gemm vs Accelerate cblas_sgemm ==");
    println!(
        "  {:>16}  {:>10}  {:>10}  {:>7}",
        "shape", "sme TF/s", "acc TF/s", "ratio"
    );
    for &(m, n, k) in sizes {
        let a = vec![0.01f32; m * k];
        let b = vec![0.02f32; k * n];
        let mut c = vec![0.0f32; m * n];
        let mut c2 = vec![0.0f32; m * n];
        let big = m * n * k > 64 << 20;
        let flop = 2.0 * (m * n * k) as f64;

        let mut sme = || matmul_f32(&a, &b, &mut c, m, n, k);
        // SAFETY: row-major M*K, K*N, M*N buffers with the matching leading
        // dimensions; Accelerate only reads a/b and writes c2 within bounds.
        let mut acc = || unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                c2.as_mut_ptr(),
                n as i32,
            );
        };
        let s = best_interleaved(big, &mut [&mut sme, &mut acc]);
        let tf_sme = flop / s[0] / 1e12;
        let tf_acc = flop / s[1] / 1e12;
        println!(
            "  {:>5}x{:<4}x{:<4}  {tf_sme:>10.2}  {tf_acc:>10.2}  {:>6.2}x",
            m,
            n,
            k,
            tf_sme / tf_acc,
        );
    }

    println!("\n== f64: sme-gemm vs Accelerate cblas_dgemm ==");
    println!(
        "  {:>16}  {:>10}  {:>10}  {:>7}",
        "shape", "sme TF/s", "acc TF/s", "ratio"
    );
    for &(m, n, k) in sizes {
        let a = vec![0.01f64; m * k];
        let b = vec![0.02f64; k * n];
        let mut c = vec![0.0f64; m * n];
        let mut c2 = vec![0.0f64; m * n];
        let big = m * n * k > 64 << 20;
        let flop = 2.0 * (m * n * k) as f64;

        let mut sme = || matmul_f64(&a, &b, &mut c, m, n, k);
        // SAFETY: as in the f32 loop, f64 buffers.
        let mut acc = || unsafe {
            cblas_dgemm(
                ROW_MAJOR,
                NO_TRANS,
                NO_TRANS,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a.as_ptr(),
                k as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                c2.as_mut_ptr(),
                n as i32,
            );
        };
        let s = best_interleaved(big, &mut [&mut sme, &mut acc]);
        let tf_sme = flop / s[0] / 1e12;
        let tf_acc = flop / s[1] / 1e12;
        println!(
            "  {:>5}x{:<4}x{:<4}  {tf_sme:>10.2}  {tf_acc:>10.2}  {:>6.2}x",
            m,
            n,
            k,
            tf_sme / tf_acc,
        );
    }
}
