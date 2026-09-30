//! Criterion throughput benchmarks for the SME GEMM kernels.
//!
//! Shapes mirror `examples/bench.rs` and `examples/epilogue_bench.rs` so the
//! numbers are directly comparable. Throughput is reported in elements*2*K (the
//! standard 2*M*N*K FLOP count). On Apple M4+ these hit the SME kernels; on
//! other hosts they exercise the scalar fallback.
//!
//! Sample sizes are kept modest so `cargo bench` stays in the tens-of-seconds
//! range rather than minutes.

// The crate denies `missing_docs`; the `criterion_group!`/`criterion_main!`
// macros generate undocumentable harness items, so allow it for this bench.
#![allow(missing_docs)]

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use half::f16;
use sme_gemm::{Accum, Gemm, matmul_f16, matmul_f32, prepack_f16};

// Square shapes from examples/bench.rs plus a skinny and a shallow-K case.
const SHAPES: &[(usize, usize, usize)] = &[
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 512, 512),  // skinny (tall-ish, narrow N/K)
    (4096, 4096, 128), // shallow K, wide output (store-dominated)
];

const fn flops(m: usize, n: usize, k: usize) -> u64 {
    2 * (m as u64) * (n as u64) * (k as u64)
}

fn label(m: usize, n: usize, k: usize) -> String {
    format!("{m}x{n}x{k}")
}

fn bench_f32(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_f32");
    g.sample_size(20).warm_up_time(Duration::from_millis(500));
    for &(m, n, k) in SHAPES {
        let a = vec![0.01f32; m * k];
        let b = vec![0.02f32; k * n];
        let mut out = vec![0.0f32; m * n];
        g.throughput(Throughput::Elements(flops(m, n, k)));
        g.bench_with_input(
            BenchmarkId::from_parameter(label(m, n, k)),
            &(m, n, k),
            |bn, &(m, n, k)| {
                bn.iter(|| matmul_f32(&a, &b, &mut out, m, n, k));
            },
        );
    }
    g.finish();
}

fn bench_f16(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_f16");
    g.sample_size(20).warm_up_time(Duration::from_millis(500));
    for &(m, n, k) in SHAPES {
        let a = vec![f16::from_f32(0.01); m * k];
        let b = vec![f16::from_f32(0.02); k * n];
        let mut out = vec![f16::ZERO; m * n];
        g.throughput(Throughput::Elements(flops(m, n, k)));
        g.bench_with_input(
            BenchmarkId::from_parameter(label(m, n, k)),
            &(m, n, k),
            |bn, &(m, n, k)| {
                bn.iter(|| matmul_f16(&a, &b, &mut out, m, n, k, Accum::F16));
            },
        );
    }
    g.finish();
}

// Fused op-graph epilogue: relu(A@B + col_bias) on the packed-B f16 path. Same
// store-dominated shapes as examples/epilogue_bench.rs so overhead is comparable.
fn bench_epilogue(c: &mut Criterion) {
    let mut g = c.benchmark_group("gemm_f16_addcol_relu");
    g.sample_size(20).warm_up_time(Duration::from_millis(500));
    let shapes: &[(usize, usize, usize)] =
        &[(4096, 4096, 512), (4096, 4096, 128), (16384, 512, 512)];
    for &(m, n, k) in shapes {
        let a = vec![f16::from_f32(0.01); m * k];
        let b = vec![f16::from_f32(0.02); k * n];
        let bias = vec![f16::from_f32(-0.5); n];
        let packed = prepack_f16(&b, n, k);
        let mut out = vec![f16::ZERO; m * n];
        g.throughput(Throughput::Elements(flops(m, n, k)));
        g.bench_with_input(BenchmarkId::from_parameter(label(m, n, k)), &m, |bn, &m| {
            bn.iter(|| {
                Gemm::new(&a, &packed, m)
                    .add_col(&bias)
                    .relu()
                    .run(&mut out);
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_f32, bench_f16, bench_epilogue);
criterion_main!(benches);
