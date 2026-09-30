//! Property-based differential fuzzer for the SME GEMM kernels.
//!
//! Each property generates random shapes (and a few SME-tail-stressing shapes)
//! plus random input data, runs the public kernel, and compares against an
//! **independent** naive triple-loop oracle computed inline with f64 (or exact
//! integer) accumulation. The oracle is deliberately *not* `crate::reference`:
//! it is a separate, dead-simple implementation living in this file so the test
//! is a genuine cross-check of the kernels, not a tautology.
//!
//! On Apple M4+ the public API dispatches to the hand-written SME kernels, so
//! these properties fuzz the real streaming-mode code. Off-Apple (or below the
//! SME FLOP threshold) the same calls fall back to the scalar reference -- still
//! a valid differential check, just against the fallback path.

// The crate denies `missing_docs`; the `proptest!` macro generates test items
// that can't carry doc comments, so allow it for this integration test.
#![allow(missing_docs)]

use half::{bf16, f16};
use proptest::prelude::*;
use sme_gemm::{
    Accum, Gemm, Q4_BLOCK, Q4Weights, gemm_bf16, gemm_f16, gemm_f32, gemm_f64, matmul_bf16,
    matmul_f16, matmul_f32, matmul_f32_batched, matmul_f64, matmul_i8, matmul_i8_packed,
    matmul_i16, matmul_q4, prepack_f32, prepack_i8,
};

// Modest case count: keeps the suite fast while still covering many shapes.
// Under Miri (which interprets MIR orders of magnitude slower than native and
// routes everything through the pure-Rust reference path) a couple of small
// cases per property is enough to surface UB in the glue.
const CASES: u32 = if cfg!(miri) { 2 } else { 96 };

// A dimension strategy biased toward small values (fast) but injecting a few
// larger / non-multiple-of-32 "tail" values (65, 97, 129, 257, 33). When three
// of these are drawn together the product clears the SME_MIN_FLOPS threshold
// (m*n*k >= 2^18) AND leaves M/N/K tails, so the real SME kernels and their
// tail-predicate handling get exercised; most draws stay below the threshold and
// fuzz the small-shape paths.
#[cfg(not(miri))]
fn tail_dim() -> impl Strategy<Value = usize> {
    prop_oneof![
        4 => 1usize..=128,
        1 => prop_oneof![Just(65usize), Just(97), Just(129), Just(257), Just(33)],
    ]
}

// Miri: every call falls back to the scalar reference and the interpreter is
// ~1000x slower, so big tail shapes buy nothing -- keep dims tiny.
#[cfg(miri)]
fn tail_dim() -> impl Strategy<Value = usize> {
    1usize..=8
}

// ---- independent oracles (NOT crate::reference) -----------------------------

/// Naive row-major `A @ B` with f64 accumulation. Independent of the library.
fn oracle_f64_mm(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Max relative error against an f64 oracle: `|got - want| / (1 + |want|)`.
fn max_rel_f64(got: &[f64], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / (1.0 + w.abs()))
        .fold(0.0, f64::max)
}

// f16/bf16 accumulation error grows roughly with sqrt(k): a defensible bound is
// base * max(1, sqrt(k)/4) -- 4 cancels the typical sqrt(k) at the small-k
// shapes where `base` already dominates, and lets the tolerance widen on deep K.
// This mirrors the `(c * sqrt(k)/4).max(c)` pattern used in tests/correctness.rs.
fn half_tol(base: f64, k: usize) -> f64 {
    (base * (k as f64).sqrt() / 4.0).max(base)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: CASES,
        // proptest's file-based failure persistence needs filesystem access,
        // which Miri's isolation rejects at startup; drop it there.
        failure_persistence: if cfg!(miri) {
            None
        } else {
            ProptestConfig::default().failure_persistence
        },
        ..ProptestConfig::default()
    })]

    // f32 row-major matmul vs f64 oracle. f32 single-pass accumulation -> ~1e-4
    // relative (same bound as tests/correctness.rs `f32_full`).
    #[test]
    fn prop_matmul_f32(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_f32(m * k), any_f32(k * n));
        let mut c = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut c, m, n, k);
        let af: Vec<f64> = a.iter().map(|&x| f64::from(x)).collect();
        let bf: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
        let want = oracle_f64_mm(&af, &bf, m, n, k);
        let got: Vec<f64> = c.iter().map(|&x| f64::from(x)).collect();
        let mr = max_rel_f64(&got, &want);
        prop_assert!(mr < 1e-4, "f32 {m}x{n}x{k}: max_rel={mr}");
    }

    // f64 row-major matmul vs f64 oracle: near-exact double GEMM -> ~1e-10.
    #[test]
    fn prop_matmul_f64(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_f64(m * k), any_f64(k * n));
        let mut c = vec![0.0f64; m * n];
        matmul_f64(&a, &b, &mut c, m, n, k);
        let want = oracle_f64_mm(&a, &b, m, n, k);
        let mr = max_rel_f64(&c, &want);
        prop_assert!(mr < 1e-10, "f64 {m}x{n}x{k}: max_rel={mr}");
    }

    // i8 -> i32: exact. Must match an independent i32 triple-loop bit-for-bit.
    #[test]
    fn prop_matmul_i8(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_i8(m * k), any_i8(k * n));
        let mut c = vec![0i32; m * n];
        matmul_i8(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for l in 0..k {
                    acc += i32::from(a[i * k + l]) * i32::from(b[l * n + j]);
                }
                prop_assert_eq!(c[i * n + j], acc, "i8 {}x{}x{} at ({},{})", m, n, k, i, j);
            }
        }
    }

    // i16 -> i64: exact. Must match an independent i64 triple-loop bit-for-bit.
    #[test]
    fn prop_matmul_i16(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_i16(m * k), any_i16(k * n));
        let mut c = vec![0i64; m * n];
        matmul_i16(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i64;
                for l in 0..k {
                    acc += i64::from(a[i * k + l]) * i64::from(b[l * n + j]);
                }
                prop_assert_eq!(c[i * n + j], acc, "i16 {}x{}x{} at ({},{})", m, n, k, i, j);
            }
        }
    }

    // f16 with f32 accumulate (widening): tight ~1e-2 relative (matches
    // `f16_f32_accum`).
    // `half` converts f16<->f32 via inline `fcvt` asm whenever the target has
    // fp16 (always on aarch64-apple-darwin), and Miri cannot interpret inline
    // asm -- so native-aarch64 Miri skips the f16 properties. They still run
    // under `cargo miri test --target x86_64-unknown-linux-gnu`, where `half`
    // takes its pure-Rust soft-float path.
    #[test]
    #[cfg_attr(all(miri, target_arch = "aarch64"), ignore)]
    fn prop_matmul_f16_f32(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_f16(m * k), any_f16(k * n));
        let mut c = vec![f16::ZERO; m * n];
        matmul_f16(&a, &b, &mut c, m, n, k, Accum::F32);
        let af: Vec<f64> = a.iter().map(|x| f64::from(x.to_f32())).collect();
        let bf: Vec<f64> = b.iter().map(|x| f64::from(x.to_f32())).collect();
        let want = oracle_f64_mm(&af, &bf, m, n, k);
        let got: Vec<f64> = c.iter().map(|x| f64::from(x.to_f32())).collect();
        let mr = max_rel_f64(&got, &want);
        prop_assert!(mr < 2e-2, "f16/f32 {m}x{n}x{k}: max_rel={mr}");
    }

    // f16 accumulate (non-widening, M5): error grows ~sqrt(k), so the
    // tolerance scales with sqrt(k) off a 3e-2 base (matches `f16_accum`).
    // See prop_matmul_f16_f32: `half`'s aarch64 fcvt asm is uninterpretable.
    #[test]
    #[cfg_attr(all(miri, target_arch = "aarch64"), ignore)]
    fn prop_matmul_f16(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_f16(m * k), any_f16(k * n));
        let mut c = vec![f16::ZERO; m * n];
        matmul_f16(&a, &b, &mut c, m, n, k, Accum::F16);
        let af: Vec<f64> = a.iter().map(|x| f64::from(x.to_f32())).collect();
        let bf: Vec<f64> = b.iter().map(|x| f64::from(x.to_f32())).collect();
        let want = oracle_f64_mm(&af, &bf, m, n, k);
        let got: Vec<f64> = c.iter().map(|x| f64::from(x.to_f32())).collect();
        let mr = max_rel_f64(&got, &want);
        let tol = half_tol(3e-2, k);
        prop_assert!(mr < tol, "f16 {m}x{n}x{k}: max_rel={mr} tol={tol}");
    }

    // bf16 (8-bit mantissa, fp32 widening accumulate): output rounding dominates;
    // sqrt(k)-scaled tolerance off a 6e-2 base (matches `bf16_widening`).
    #[test]
    fn prop_matmul_bf16(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_bf16(m * k), any_bf16(k * n));
        let mut c = vec![bf16::ZERO; m * n];
        matmul_bf16(&a, &b, &mut c, m, n, k, Accum::F32);
        let af: Vec<f64> = a.iter().map(|x| f64::from(x.to_f32())).collect();
        let bf: Vec<f64> = b.iter().map(|x| f64::from(x.to_f32())).collect();
        let want = oracle_f64_mm(&af, &bf, m, n, k);
        let got: Vec<f64> = c.iter().map(|x| f64::from(x.to_f32())).collect();
        let mr = max_rel_f64(&got, &want);
        let tol = half_tol(6e-2, k);
        prop_assert!(mr < tol, "bf16 {m}x{n}x{k}: max_rel={mr} tol={tol}");
    }

    // Strided gemm_f32: C = alpha*C + beta*(A@B) with non-trivial alpha/beta,
    // checked against the independent f64 oracle of the same accumulate.
    #[test]
    fn prop_gemm_f32_accumulate(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
        alpha in -2.0f32..2.0f32,
        beta in -2.0f32..2.0f32,
    ) {
        let (a, b) = (any_f32(m * k), any_f32(k * n));
        let c_init = any_f32(m * n);
        let mut c = c_init.clone();
        // row-major strides.
        gemm_f32(m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, alpha, beta);

        let af: Vec<f64> = a.iter().map(|&x| f64::from(x)).collect();
        let bf: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
        let base = oracle_f64_mm(&af, &bf, m, n, k);
        let mut mr = 0.0f64;
        for idx in 0..m * n {
            let want = f64::from(alpha) * f64::from(c_init[idx]) + f64::from(beta) * base[idx];
            mr = mr.max((f64::from(c[idx]) - want).abs() / (1.0 + want.abs()));
        }
        prop_assert!(mr < 1e-4, "gemm_f32 accumulate {m}x{n}x{k} a={alpha} b={beta}: max_rel={mr}");
    }

    // Strided gemm_f32 with a TRANSPOSED B (B supplied column-major: b_row=1,
    // b_col=k). The kernel must read B[l,j] = bt[j*k + l]; checked against the
    // f64 oracle reading B the same transposed way. alpha=0 (overwrite), beta=1.
    #[test]
    fn prop_gemm_f32_transposed_b(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let a = any_f32(m * k);
        // bt is n x k row-major (B transposed); B[l,j] = bt[j*k + l].
        let bt = any_f32(n * k);
        let mut c = vec![0.0f32; m * n];
        gemm_f32(m, n, k, &mut c, n, 1, &a, k, 1, &bt, 1, k, 0.0, 1.0);

        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l]) * f64::from(bt[j * k + l]);
                }
                mr = mr.max((f64::from(c[i * n + j]) - acc).abs() / (1.0 + acc.abs()));
            }
        }
        prop_assert!(mr < 1e-4, "gemm_f32 transposed-B {m}x{n}x{k}: max_rel={mr}");
    }

    // Strided gemm_f64: C = alpha*C + beta*(A@B), non-trivial alpha/beta, near
    // exact double GEMM -> ~1e-10.
    #[test]
    fn prop_gemm_f64_accumulate(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
        alpha in -2.0f64..2.0f64,
        beta in -2.0f64..2.0f64,
    ) {
        let (a, b) = (any_f64(m * k), any_f64(k * n));
        let c_init = any_f64(m * n);
        let mut c = c_init.clone();
        gemm_f64(m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, alpha, beta);

        let base = oracle_f64_mm(&a, &b, m, n, k);
        let mut mr = 0.0f64;
        for idx in 0..m * n {
            let want = alpha * c_init[idx] + beta * base[idx];
            mr = mr.max((c[idx] - want).abs() / (1.0 + want.abs()));
        }
        prop_assert!(mr < 1e-10, "gemm_f64 accumulate {m}x{n}x{k} a={alpha} b={beta}: max_rel={mr}");
    }

    // Fused-epilogue f32: relu(A@B + col_bias) via the Gemm op-graph builder,
    // checked against the f64 oracle with the SAME scalar epilogue applied.
    #[test]
    fn prop_gemm_epilogue_f32_addcol_relu(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_f32(m * k), any_f32(k * n));
        let bias = any_f32(n);
        let packed = prepack_f32(&b, n, k);
        let mut c = vec![0.0f32; m * n];
        Gemm::new(&a, &packed, m).add_col(&bias).relu().run(&mut c);

        let af: Vec<f64> = a.iter().map(|&x| f64::from(x)).collect();
        let bf: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
        let base = oracle_f64_mm(&af, &bf, m, n, k);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                // scalar epilogue oracle: relu(base + bias[j]).
                let want = (base[i * n + j] + f64::from(bias[j])).max(0.0);
                mr = mr.max((f64::from(c[i * n + j]) - want).abs() / (1.0 + want.abs()));
            }
        }
        prop_assert!(mr < 1e-4, "f32 epilogue add_col+relu {m}x{n}x{k}: max_rel={mr}");
    }

    // Fused-epilogue f32 via the raw Epilogue + matmul path is exercised in the
    // Gemm builder above; here also check a column-scaled + bias + relu graph to
    // cover mul_col ordering against the oracle.
    #[test]
    fn prop_gemm_epilogue_f32_mulcol_addcol_relu(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_f32(m * k), any_f32(k * n));
        let scale = any_f32(n);
        let bias = any_f32(n);
        let packed = prepack_f32(&b, n, k);
        let mut c = vec![0.0f32; m * n];
        Gemm::new(&a, &packed, m)
            .mul_col(&scale)
            .add_col(&bias)
            .relu()
            .run(&mut c);

        let af: Vec<f64> = a.iter().map(|&x| f64::from(x)).collect();
        let bf: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
        let base = oracle_f64_mm(&af, &bf, m, n, k);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let want =
                    (base[i * n + j] * f64::from(scale[j]) + f64::from(bias[j])).max(0.0);
                mr = mr.max((f64::from(c[i * n + j]) - want).abs() / (1.0 + want.abs()));
            }
        }
        prop_assert!(mr < 1e-4, "f32 epilogue mulcol+addcol+relu {m}x{n}x{k}: max_rel={mr}");
    }

    // Batched f32: `count` independent C_i = A_i @ B_i in one streaming session,
    // each item checked against the f64 oracle.
    #[test]
    fn prop_matmul_f32_batched(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
        count in 1usize..=8,
    ) {
        let (a, b) = (any_f32(count * m * k), any_f32(count * k * n));
        let mut c = vec![0.0f32; count * m * n];
        matmul_f32_batched(&a, &b, &mut c, m, n, k);
        for bi in 0..count {
            let ai: Vec<f64> =
                a[bi * m * k..(bi + 1) * m * k].iter().map(|&x| f64::from(x)).collect();
            let bb: Vec<f64> =
                b[bi * k * n..(bi + 1) * k * n].iter().map(|&x| f64::from(x)).collect();
            let want = oracle_f64_mm(&ai, &bb, m, n, k);
            let got: Vec<f64> = c[bi * m * n..(bi + 1) * m * n]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            let mr = max_rel_f64(&got, &want);
            prop_assert!(mr < 1e-4, "f32-batched item {bi} {m}x{n}x{k}: max_rel={mr}");
        }
    }

    // Strided gemm_f16: C = alpha*C + beta*(A@B) with non-trivial alpha/beta,
    // checked against the f64 oracle of the same accumulate. f16 storage +
    // accumulation -> sqrt(k)-scaled tol off a 3e-2 base (mirrors gemm_f16_accumulate
    // and prop_gemm_f32_accumulate). See prop_matmul_f16_f32: aarch64 Miri
    // can't interpret `half`'s fcvt asm, so skip there.
    #[test]
    #[cfg_attr(all(miri, target_arch = "aarch64"), ignore)]
    fn prop_gemm_f16_accumulate(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
        alpha in -2.0f32..2.0f32,
        beta in -2.0f32..2.0f32,
    ) {
        let (a, b) = (any_f16(m * k), any_f16(k * n));
        let c_init = any_f16(m * n);
        let (alpha, beta) = (f16::from_f32(alpha), f16::from_f32(beta));
        let mut c = c_init.clone();
        gemm_f16(m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, alpha, beta, Accum::F32);

        let af: Vec<f64> = a.iter().map(|x| f64::from(x.to_f32())).collect();
        let bf: Vec<f64> = b.iter().map(|x| f64::from(x.to_f32())).collect();
        let base = oracle_f64_mm(&af, &bf, m, n, k);
        let mut mr = 0.0f64;
        for idx in 0..m * n {
            let want = f64::from(alpha.to_f32()) * f64::from(c_init[idx].to_f32())
                + f64::from(beta.to_f32()) * base[idx];
            mr = mr.max((f64::from(c[idx].to_f32()) - want).abs() / (1.0 + want.abs()));
        }
        let tol = half_tol(3e-2, k);
        prop_assert!(mr < tol, "gemm_f16 accumulate {m}x{n}x{k} a={alpha} b={beta}: max_rel={mr} tol={tol}");
    }

    // Strided gemm_bf16: C = alpha*C + beta*(A@B), non-trivial alpha/beta, checked
    // against the f64 oracle. bf16 (8-bit mantissa) -> sqrt(k)-scaled tol off a
    // 6e-2 base (mirrors gemm_bf16_accumulate and prop_matmul_bf16).
    #[test]
    fn prop_gemm_bf16_accumulate(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
        alpha in -2.0f32..2.0f32,
        beta in -2.0f32..2.0f32,
    ) {
        let (a, b) = (any_bf16(m * k), any_bf16(k * n));
        let c_init = any_bf16(m * n);
        let (alpha, beta) = (bf16::from_f32(alpha), bf16::from_f32(beta));
        let mut c = c_init.clone();
        gemm_bf16(m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, alpha, beta, Accum::F32);

        let af: Vec<f64> = a.iter().map(|x| f64::from(x.to_f32())).collect();
        let bf: Vec<f64> = b.iter().map(|x| f64::from(x.to_f32())).collect();
        let base = oracle_f64_mm(&af, &bf, m, n, k);
        let mut mr = 0.0f64;
        for idx in 0..m * n {
            let want = f64::from(alpha.to_f32()) * f64::from(c_init[idx].to_f32())
                + f64::from(beta.to_f32()) * base[idx];
            mr = mr.max((f64::from(c[idx].to_f32()) - want).abs() / (1.0 + want.abs()));
        }
        let tol = half_tol(6e-2, k);
        prop_assert!(mr < tol, "gemm_bf16 accumulate {m}x{n}x{k} a={alpha} b={beta}: max_rel={mr} tol={tol}");
    }

    // Pre-packed i8 -> i32: exact. matmul_i8_packed on prepack_i8(B) must match an
    // independent i32 triple-loop bit-for-bit (mirrors prop_matmul_i8). On M5 this
    // fuzzes the real SME packb + run-packed path; off-SME it is the row-major
    // fallback -- both a valid differential check.
    #[test]
    fn prop_matmul_i8_packed(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let (a, b) = (any_i8(m * k), any_i8(k * n));
        let packed = prepack_i8(&b, n, k);
        let mut c = vec![0i32; m * n];
        matmul_i8_packed(&a, &packed, &mut c, m);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for l in 0..k {
                    acc += i32::from(a[i * k + l]) * i32::from(b[l * n + j]);
                }
                prop_assert_eq!(c[i * n + j], acc, "i8-packed {}x{}x{} at ({},{})", m, n, k, i, j);
            }
        }
    }

    // 4-bit resident Q4: matmul_q4 dequantizes B on the fly. Checked against an
    // f64 oracle over the same dequantized weights (scale * signed-4-bit code),
    // f16 A. sqrt(k)-scaled tol off a 3e-2 base (mirrors q4_resident in
    // correctness.rs). On M5 this fuzzes the resident Q4 SME kernel; off-SME the
    // dequant-to-rowmajor fallback -- both valid. aarch64 Miri can't interpret
    // `half`'s fcvt asm.
    #[test]
    #[cfg_attr(all(miri, target_arch = "aarch64"), ignore)]
    fn prop_matmul_q4(
        (m, n, k) in (tail_dim(), tail_dim(), tail_dim()),
    ) {
        let a = any_f16(m * k);
        let nbk = k.div_ceil(Q4_BLOCK);
        let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| (next_u64() >> 56) as u8).collect();
        let scales: Vec<f16> = (0..n * nbk)
            .map(|i| f16::from_f32(0.01 + 0.001 * (i % 5) as f32))
            .collect();

        // f64 dequant oracle weights: scale[col,blk] * sign-extend(4-bit code).
        let mut bf = vec![0.0f64; k * n];
        for d in 0..k {
            for j in 0..n {
                let idx = d * n + j;
                let nib = if idx.is_multiple_of(2) {
                    quants[idx / 2] & 0x0f
                } else {
                    quants[idx / 2] >> 4
                };
                let code = i32::from(nib) - if nib < 8 { 0 } else { 16 };
                bf[idx] = f64::from(scales[j * nbk + d / Q4_BLOCK].to_f32()) * f64::from(code);
            }
        }

        let w = Q4Weights::new(&quants, &scales, n, k);
        let mut c = vec![f16::ZERO; m * n];
        matmul_q4(&a, &w, &mut c, m);

        let af: Vec<f64> = a.iter().map(|x| f64::from(x.to_f32())).collect();
        let want = oracle_f64_mm(&af, &bf, m, n, k);
        let got: Vec<f64> = c.iter().map(|x| f64::from(x.to_f32())).collect();
        let mr = max_rel_f64(&got, &want);
        let tol = half_tol(3e-2, k);
        prop_assert!(mr < tol, "q4 {m}x{n}x{k}: max_rel={mr} tol={tol}");
    }
}

// proptest's `in` clause needs strategies, but for the input *data* (whose length
// depends on the already-drawn shape) it is cleaner to sample directly from a
// fresh runner. These helpers draw a deterministic-but-varied vector of the right
// length using `proptest`'s own value tree, keeping the data path independent of
// the shape strategy. They are seeded by a process-global counter so successive
// calls within one case differ.
//
// NOTE: we use a simple SplitMix64 PRNG here rather than threading a second
// proptest input, because the data length is a runtime function of (m,n,k). This
// keeps the properties readable while still feeding the kernels pseudo-random
// inputs. Shrinking still works on the shape; the data is regenerated per case.
use std::cell::Cell;

thread_local! {
    static RNG: Cell<u64> = const { Cell::new(0x9e37_79b9_7f4a_7c15) };
}

fn next_u64() -> u64 {
    RNG.with(|r| {
        let mut z = r.get().wrapping_add(0x9e37_79b9_7f4a_7c15);
        r.set(z);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    })
}

// uniform f32 in [-1, 1)
fn rand_unit() -> f32 {
    (next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
}

fn any_f32(len: usize) -> Vec<f32> {
    (0..len).map(|_| rand_unit()).collect()
}
fn any_f64(len: usize) -> Vec<f64> {
    (0..len).map(|_| f64::from(rand_unit())).collect()
}
fn any_f16(len: usize) -> Vec<f16> {
    (0..len).map(|_| f16::from_f32(rand_unit())).collect()
}
fn any_bf16(len: usize) -> Vec<bf16> {
    (0..len).map(|_| bf16::from_f32(rand_unit())).collect()
}
fn any_i8(len: usize) -> Vec<i8> {
    (0..len).map(|_| (next_u64() >> 56) as i8).collect()
}
fn any_i16(len: usize) -> Vec<i16> {
    (0..len).map(|_| (next_u64() >> 48) as i16).collect()
}
