//! Throughput across all dtypes. Best-of-N to reject thermal throttling.
//! Run: `cargo run --release --example bench`.

use std::time::Instant;

use half::{bf16, f16};
use sme_gemm::{
    Accuracy, Caps, Gemm, Q4Weights, caps, matmul_bf16, matmul_bf16_packed, matmul_f16,
    matmul_f16_packed, matmul_f32_packed, matmul_i8_packed, matmul_i16_packed, matmul_q4,
    matmul_q4_bf16, prepack_bf16, prepack_f16, prepack_f32, prepack_f64, prepack_i8, prepack_i16,
};

/// Deterministic 4-bit weight set for `k x n`, plus its per-block f16 scales.
fn q4_operands(n: usize, k: usize) -> (Vec<u8>, Vec<f16>) {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut byte = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 40) as u8
    };
    let quants = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
    let scales = (0..n * k.div_ceil(32))
        .map(|i| f16::from_f32(0.01 + 0.001 * (i % 5) as f32))
        .collect();
    (quants, scales)
}

const SIZES: &[(usize, usize, usize)] = &[
    (512, 512, 512),
    (2048, 2048, 2048),
    (16384, 512, 512),
    (1, 4096, 4096),
];

fn best<F: FnMut()>(m: usize, n: usize, k: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let iters = if m * n * k > 64 << 20 { 8 } else { 30 };
    let mut b = f64::INFINITY;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed().as_secs_f64());
    }
    2.0 * (m * n * k) as f64 / b / 1e12
}

/// One printed row. `None` for a cell means the dtype cannot run that shape --
/// either the extension is absent, or (at `m == 1`) it has no pre-packed entry
/// point, so the call would re-pack the whole weight set and report packing cost
/// rather than kernel throughput.
fn row(name: &str, mut f: impl FnMut(usize, usize, usize) -> Option<f64>) {
    print!("{name:<16}");
    for &(m, n, k) in SIZES {
        match f(m, n, k) {
            Some(v) => print!(" {v:>9.2}"),
            None => print!(" {:>9}", "--"),
        }
    }
    println!();
}

/// Bring the clocks up before the first timed row. Without it whichever row is
/// measured first reads ~25% low (f16 and bf16 `Fast` run the same MOPA, and
/// disagreed by that much purely by position).
fn warmup() {
    let (n, k) = (1024, 1024);
    let a = vec![f16::from_f32(0.01); n * k];
    let b = vec![f16::from_f32(0.02); k * n];
    let (p, mut c) = (prepack_f16(&b, n, k), vec![f16::ZERO; n * n]);
    for _ in 0..300 {
        matmul_f16_packed(&a, &p, &mut c, n);
    }
}

fn main() {
    let c = caps();
    println!("{c:?}\n");
    warmup();
    print!("{:<16}", "dtype");
    for &(m, n, k) in SIZES {
        print!(" {:>9}", format!("{m}x{n}x{k}"));
    }
    println!();

    float_rows(c);
    int_rows(c);
    println!("\nTF/s (integer rows are TOPS), same 2*M*N*K op count throughout.");
    println!("Fast = M5 non-widening MOPA; Accurate = widening fp32 accumulate.");
    println!("1x4096x4096 is the decode shape -- one row against a resident weight");
    println!("set, so it is bandwidth-bound and only pre-packed paths can run it.");
}

/// The floating-point dtype rows.
fn float_rows(c: Caps) {
    row("f16 Fast", |m, n, k| {
        if !c.sme_f16f16 {
            return None;
        }
        let a = vec![f16::from_f32(0.01); m * k];
        let b = vec![f16::from_f32(0.02); k * n];
        let (p, mut cc) = (prepack_f16(&b, n, k), vec![f16::ZERO; m * n]);
        Some(best(m, n, k, || matmul_f16_packed(&a, &p, &mut cc, m)))
    });
    row("bf16 Fast", |m, n, k| {
        if !c.sme_b16b16 {
            return None;
        }
        let a = vec![bf16::from_f32(0.01); m * k];
        let b = vec![bf16::from_f32(0.02); k * n];
        let (p, mut cc) = (prepack_bf16(&b, n, k), vec![bf16::ZERO; m * n]);
        Some(best(m, n, k, || matmul_bf16_packed(&a, &p, &mut cc, m)))
    });
    row("f16 Accurate", |m, n, k| {
        if !c.sme || m == 1 {
            return None;
        }
        let a = vec![f16::from_f32(0.01); m * k];
        let b = vec![f16::from_f32(0.02); k * n];
        let mut cc = vec![f16::ZERO; m * n];
        Some(best(m, n, k, || {
            matmul_f16(&a, &b, &mut cc, m, n, k, Accuracy::Accurate);
        }))
    });
    row("bf16 Accurate", |m, n, k| {
        if !c.sme || m == 1 {
            return None;
        }
        let a = vec![bf16::from_f32(0.01); m * k];
        let b = vec![bf16::from_f32(0.02); k * n];
        let mut cc = vec![bf16::ZERO; m * n];
        Some(best(m, n, k, || {
            matmul_bf16(&a, &b, &mut cc, m, n, k, Accuracy::Accurate);
        }))
    });
    row("f32", |m, n, k| {
        if !c.sme {
            return None;
        }
        let a = vec![0.01f32; m * k];
        let b = vec![0.02f32; k * n];
        let (p, mut cc) = (prepack_f32(&b, n, k), vec![0.0f32; m * n]);
        Some(best(m, n, k, || matmul_f32_packed(&a, &p, &mut cc, m)))
    });
    row("f64", |m, n, k| {
        if !c.sme_f64f64 {
            return None;
        }
        let a = vec![0.01f64; m * k];
        let b = vec![0.02f64; k * n];
        let (p, mut cc) = (prepack_f64(&b, n, k), vec![0.0f64; m * n]);
        Some(best(m, n, k, || Gemm::new(&a, &p, m).run(&mut cc)))
    });
}

/// The integer and 4-bit-weight rows.
fn int_rows(c: Caps) {
    row("i8->i32 (TOPS)", |m, n, k| {
        if !c.sme {
            return None;
        }
        let a = vec![1i8; m * k];
        let b = vec![2i8; k * n];
        let (p, mut cc) = (prepack_i8(&b, n, k), vec![0i32; m * n]);
        Some(best(m, n, k, || matmul_i8_packed(&a, &p, &mut cc, m)))
    });
    row("i16->i64 (TOPS)", |m, n, k| {
        if !c.sme_i16i64 {
            return None;
        }
        let a = vec![1i16; m * k];
        let b = vec![2i16; k * n];
        let (p, mut cc) = (prepack_i16(&b, n, k), vec![0i64; m * n]);
        Some(best(m, n, k, || matmul_i16_packed(&a, &p, &mut cc, m)))
    });
    row("Q4->f16", |m, n, k| {
        if !c.sme_f16f16 {
            return None;
        }
        let (q, s) = q4_operands(n, k);
        let w = Q4Weights::new(&q, &s, n, k);
        let a = vec![f16::from_f32(0.01); m * k];
        let mut cc = vec![f16::ZERO; m * n];
        Some(best(m, n, k, || matmul_q4(&a, &w, &mut cc, m)))
    });
    row("Q4->bf16", |m, n, k| {
        if !c.sme_b16b16 {
            return None;
        }
        let (q, s) = q4_operands(n, k);
        let w = Q4Weights::new(&q, &s, n, k);
        let a = vec![bf16::from_f32(0.01); m * k];
        let mut cc = vec![bf16::ZERO; m * n];
        Some(best(m, n, k, || matmul_q4_bf16(&a, &w, &mut cc, m)))
    });
}
