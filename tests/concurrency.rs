//! Concurrency stress for the GCD-parallel kernel paths.
//!
//! The library's only parallelism lives inside the C kernels: libdispatch
//! `dispatch_apply` over N-tile chunks, plus per-thread A-pack scratch reused
//! across calls. None of that is visible to Rust-side model checkers (loom
//! shims `loom::sync`/`loom::thread` -- there is nothing here for it to
//! instrument), so this is a stress/determinism check instead: several threads
//! hammer the parallel-branch shape concurrently, sharing one prepacked B, and
//! every result must be bit-identical to a quiet single-call baseline. The
//! inputs are small integers with a closed-form product, so for i8 AND f32 the
//! math is exact -- any divergence is a race (scratch corruption, chunk
//! mis-split), not rounding.

// Needs real SME (the parallel branch) and real threads; pointless under Miri.
#![cfg(not(miri))]

use sme_gemm::{Dequant, caps, matmul_f32, matmul_i8_packed, matmul_i8_packed_dequant, prepack_i8};

// Two shapes that drive DIFFERENT GCD-parallel branches (M_CHUNK = 4):
//   * (64, 1024, 1024): m_tiles=2 -> n_chunks=1, so the N-PARALLEL flat-M branch
//     engages (A packed once, read-only shared; disjoint C column bands).
//   * (192, 256, 512): m_tiles=6 -> n_chunks=2, non-narrow n, so the M-PARALLEL
//     branch engages -- each chunk does its OWN PACKA_RANGE write into the shared
//     a_pack buffer concurrently, which is the higher-risk path. Both satisfy
//     m*n*k >= 2^21 so neither falls back to serial.
const SHAPES: &[(usize, usize, usize)] = &[(64, 1024, 1024), (192, 256, 512)];

const THREADS: usize = 4;
const REPS: usize = 4;

const fn av(i: usize) -> i32 {
    (i % 5) as i32 - 2
}
const fn bv(j: usize) -> i32 {
    (j % 7) as i32 - 3
}
const fn cref(i: usize, j: usize, k: usize) -> i32 {
    (k as i32) * av(i) * bv(j)
}

#[test]
fn i8_packed_concurrent_callers_bit_identical() {
    if !caps().sme {
        return;
    }
    for &(m, n, k) in SHAPES {
        let a: Vec<i8> = (0..m * k).map(|x| av(x / k) as i8).collect();
        let b: Vec<i8> = (0..k * n).map(|x| bv(x % n) as i8).collect();
        let packed = prepack_i8(&b, n, k);

        // Quiet baseline, checked against the closed form first.
        let mut base = vec![0i32; m * n];
        matmul_i8_packed(&a, &packed, &mut base, m);
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    base[i * n + j],
                    cref(i, j, k),
                    "baseline i8 at ({i},{j}) {m}x{n}x{k}"
                );
            }
        }

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let (a, packed, base) = (&a, &packed, &base);
                s.spawn(move || {
                    for r in 0..REPS {
                        let mut c = vec![0i32; m * n];
                        matmul_i8_packed(a, packed, &mut c, m);
                        assert!(
                            c == *base,
                            "i8 thread {t} rep {r} {m}x{n}x{k}: result differs"
                        );
                    }
                });
            }
        });
    }
}

#[test]
fn i8_dequant_concurrent_callers_bit_identical() {
    if !caps().sme {
        return;
    }
    for &(m, n, k) in SHAPES {
        let a: Vec<i8> = (0..m * k).map(|x| av(x / k) as i8).collect();
        let b: Vec<i8> = (0..k * n).map(|x| bv(x % n) as i8).collect();
        let packed = prepack_i8(&b, n, k);
        let bias: Vec<f32> = (0..n).map(|j| 0.05 * (j % 11) as f32 - 0.2).collect();
        // ONE shared Dequant across all threads: exercises the `Sync` impl (the
        // op-graph is a read-only view of the bias slice).
        let dq = Dequant::new(0.0009).add_col(&bias).relu();

        let mut base = vec![0.0f32; m * n];
        matmul_i8_packed_dequant(&a, &packed, &mut base, m, &dq);

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let (a, packed, dq, base) = (&a, &packed, &dq, &base);
                s.spawn(move || {
                    for r in 0..REPS {
                        let mut c = vec![0.0f32; m * n];
                        matmul_i8_packed_dequant(a, packed, &mut c, m, dq);
                        // Bit-exact: identical inputs, identical chunk-local
                        // summation order; only a race can differ.
                        assert!(
                            c.iter()
                                .zip(base.iter())
                                .all(|(x, y)| x.to_bits() == y.to_bits()),
                            "i8 dequant thread {t} rep {r} {m}x{n}x{k}: result differs"
                        );
                    }
                });
            }
        });
    }
}

// The Q4 resident path parallelizes over N-tile pairs, and each worker
// dequantizes into its own slice of a per-thread pool that the CALLING thread
// sizes and owns. Two things have to hold under concurrent callers: threads must
// not see each other's pool (it is `_Thread_local`, so each caller has its own),
// and the pool must survive being reused across calls rather than freed with the
// call that allocated it. Each rep runs the shape twice so the compared call is
// always the reusing one, while the other threads are mid-dispatch. Making the
// pool shared instead of thread-local fails this test.
#[test]
fn q4_concurrent_callers_bit_identical() {
    use half::f16;
    use sme_gemm::{Q4Weights, matmul_q4};
    if !caps().sme_f16f16 {
        return;
    }
    // n >= 1024 so n_pairs > 1 and the dispatch_apply branch actually engages.
    for &(m, n, k) in &[(64usize, 1024usize, 256usize), (33, 2048, 512)] {
        let mut s = 0x51ed_270b_7f4a_7c15u64 ^ (k as u64);
        let mut byte = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as u8
        };
        let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
        let scales: Vec<f16> = (0..n * k.div_ceil(32))
            .map(|i| f16::from_f32(0.01 + 0.001 * (i % 5) as f32))
            .collect();
        let w = Q4Weights::new(&quants, &scales, n, k);
        let a: Vec<f16> = (0..m * k)
            .map(|x| f16::from_f32(av(x / k) as f32))
            .collect();

        let mut base = vec![f16::ZERO; m * n];
        matmul_q4(&a, &w, &mut base, m);

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let (a, w, base) = (&a, &w, &base);
                s.spawn(move || {
                    for r in 0..REPS {
                        // A larger call first, so this thread's pool is already
                        // grown when the compared call runs.
                        let mut warm = vec![f16::ZERO; m * n];
                        matmul_q4(a, w, &mut warm, m);
                        let mut c = vec![f16::ZERO; m * n];
                        matmul_q4(a, w, &mut c, m);
                        assert!(
                            c.iter()
                                .zip(base.iter())
                                .all(|(x, y)| x.to_bits() == y.to_bits()),
                            "q4 thread {t} rep {r} {m}x{n}x{k}: result differs"
                        );
                    }
                });
            }
        });
    }
}

#[test]
// Integer-valued inputs make the f32 baseline exact by construction; the
// strict comparison is the point of the test.
#[allow(clippy::float_cmp)]
fn f32_concurrent_callers_bit_identical() {
    if !caps().sme {
        return;
    }
    for &(m, n, k) in SHAPES {
        let a: Vec<f32> = (0..m * k).map(|x| av(x / k) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|x| bv(x % n) as f32).collect();

        let mut base = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut base, m, n, k);
        for i in 0..m {
            for j in 0..n {
                // |C| <= K*2*3 = 6144 < 2^24: integer-valued partial sums, exact in f32.
                assert_eq!(
                    base[i * n + j],
                    cref(i, j, k) as f32,
                    "baseline f32 ({i},{j}) {m}x{n}x{k}"
                );
            }
        }

        std::thread::scope(|s| {
            for t in 0..THREADS {
                let (a, b, base) = (&a, &b, &base);
                s.spawn(move || {
                    for r in 0..REPS {
                        let mut c = vec![0.0f32; m * n];
                        matmul_f32(a, b, &mut c, m, n, k);
                        assert!(
                            c.iter()
                                .zip(base.iter())
                                .all(|(x, y)| x.to_bits() == y.to_bits()),
                            "f32 thread {t} rep {r} {m}x{n}x{k}: result differs"
                        );
                    }
                });
            }
        });
    }
}
