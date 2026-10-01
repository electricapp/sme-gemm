//! Batched GEMM (`*_batched`, `*_batched_ep`, `*_batched_dequant`).

use crate::{
    fill, fill_f32, fill_f64, gelu_f64, max_rel, oracle, oracle_f32, oracle_f64, silu_f64,
};
use half::{bf16, f16};
use sme_gemm::caps;

#[test]
fn f16_batched() {
    use sme_gemm::matmul_f16_batched;
    if !caps().sme_f16f16 {
        return;
    }
    // (m, n, k, count)
    let cases: &[(usize, usize, usize, usize)] = &[
        (16, 16, 16, 50),
        (32, 48, 24, 20),
        (8, 33, 17, 13),
        (31, 31, 31, 7),
        (70, 100, 75, 3),
        (33, 65, 64, 4),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0xba7c_0ed0_1111_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let a = fill(&mut s, count * m * k);
        let b = fill(&mut s, count * k * n);
        let mut c = vec![f16::ZERO; count * m * n];
        matmul_f16_batched(&a, &b, &mut c, m, n, k);
        let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
        for bi in 0..count {
            let ai = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let ci = &c[bi * m * n..(bi + 1) * m * n];
            let mr = max_rel(ci, &oracle(ai, bb, m, n, k));
            assert!(mr < tol, "batched item {bi} {m}x{n}x{k}: max_rel={mr}");
        }
    }
}

#[test]
fn bf16_batched() {
    use sme_gemm::matmul_bf16_batched;
    if !caps().sme_b16b16 {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (16, 16, 16, 40),
        (32, 48, 24, 16),
        (31, 31, 31, 6),
        (70, 100, 75, 3),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0xbf16_bf16_3333_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let rb = |s: &mut u64| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            bf16::from_f32((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        };
        let a: Vec<bf16> = (0..count * m * k).map(|_| rb(&mut s)).collect();
        let b: Vec<bf16> = (0..count * k * n).map(|_| rb(&mut s)).collect();
        let mut c = vec![bf16::ZERO; count * m * n];
        matmul_bf16_batched(&a, &b, &mut c, m, n, k);
        let tol = (6e-2 * (k as f64).sqrt() / 4.0).max(6e-2);
        for bi in 0..count {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for l in 0..k {
                        acc += f64::from(a[bi * m * k + i * k + l].to_f32())
                            * f64::from(b[bi * k * n + l * n + j].to_f32());
                    }
                    let got = f64::from(c[bi * m * n + i * n + j].to_f32());
                    let mr = (got - acc).abs() / (1.0 + acc.abs());
                    assert!(mr < tol, "bf16-batched {bi} {m}x{n}x{k}: {mr}");
                }
            }
        }
    }
}

// Batched f32: count independent C_i = A_i @ B_i, one streaming session.
// Validated against the f64 oracle per item. Sizes clear the SME threshold.
#[test]
fn f32_batched() {
    use sme_gemm::matmul_f32_batched;
    if !caps().sme {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (32, 32, 32, 20),
        (33, 17, 9, 13),
        (16, 48, 24, 17),
        (31, 31, 31, 7),
        (70, 100, 75, 3),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0xf32b_a7c0_5555_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let a = fill_f32(&mut s, count * m * k);
        let b = fill_f32(&mut s, count * k * n);
        let mut c = vec![0.0f32; count * m * n];
        matmul_f32_batched(&a, &b, &mut c, m, n, k);
        for bi in 0..count {
            let ai = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let ci = &c[bi * m * n..(bi + 1) * m * n];
            let base = oracle_f32(ai, bb, m, n, k);
            let mut mr = 0.0f64;
            for idx in 0..m * n {
                mr = mr.max((f64::from(ci[idx]) - base[idx]).abs() / (1.0 + base[idx].abs()));
            }
            assert!(mr < 1e-4, "f32-batched item {bi} {m}x{n}x{k}: max_rel={mr}");
        }
    }
}

// Batched f64: count independent C_i = A_i @ B_i, one streaming session (M5-only).
// Validated against the f64 oracle per item -- near-exact double GEMM.
#[test]
fn f64_batched() {
    use sme_gemm::matmul_f64_batched;
    if !caps().sme_f64f64 {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (16, 16, 16, 20),
        (17, 9, 5, 13),
        (16, 48, 24, 11),
        (15, 15, 15, 7),
        (40, 37, 29, 3),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0xf64b_a7c0_7777_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let a = fill_f64(&mut s, count * m * k);
        let b = fill_f64(&mut s, count * k * n);
        let mut c = vec![0.0f64; count * m * n];
        matmul_f64_batched(&a, &b, &mut c, m, n, k);
        for bi in 0..count {
            let ai = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let ci = &c[bi * m * n..(bi + 1) * m * n];
            let base = oracle_f64(ai, bb, m, n, k);
            let mut mr = 0.0f64;
            for idx in 0..m * n {
                mr = mr.max((ci[idx] - base[idx]).abs() / (1.0 + base[idx].abs()));
            }
            assert!(
                mr < 1e-12,
                "f64-batched item {bi} {m}x{n}x{k}: max_rel={mr}"
            );
        }
    }
}

#[test]
fn i8_batched() {
    use sme_gemm::matmul_i8_batched;
    let cases: &[(usize, usize, usize, usize)] = &[
        (16, 16, 16, 40),
        (32, 48, 24, 16),
        (8, 33, 17, 11),
        (31, 31, 31, 6),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0x18b1_18b1_2222_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 48) as i32 % 255 - 127) as i8
        };
        let a: Vec<i8> = (0..count * m * k).map(|_| rnd()).collect();
        let b: Vec<i8> = (0..count * k * n).map(|_| rnd()).collect();
        let mut c = vec![0i32; count * m * n];
        matmul_i8_batched(&a, &b, &mut c, m, n, k);
        for bi in 0..count {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0i32;
                    for l in 0..k {
                        acc += i32::from(a[bi * m * k + i * k + l])
                            * i32::from(b[bi * k * n + l * n + j]);
                    }
                    assert_eq!(
                        c[bi * m * n + i * n + j],
                        acc,
                        "i8-batched {bi} {m}x{n}x{k} ({i},{j})"
                    );
                }
            }
        }
    }
}

// Batched i16 -> i64 (raw, exact). count > 1; validated against the exact i64
// oracle per item. M5-only (FEAT_SME_I16I64); a no-op otherwise (scalar path).
#[test]
fn i16_batched() {
    use sme_gemm::{caps, matmul_i16_batched};
    if !caps().sme_i16i64 {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (65, 257, 257, 4),
        (33, 512, 257, 3),
        (97, 300, 131, 3),
        (7, 400, 800, 5),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0x1616_ba70_8888_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as i32 % 65535 - 32767) as i16
        };
        let a: Vec<i16> = (0..count * m * k).map(|_| rnd()).collect();
        let b: Vec<i16> = (0..count * k * n).map(|_| rnd()).collect();
        let mut c = vec![0i64; count * m * n];
        matmul_i16_batched(&a, &b, &mut c, m, n, k);
        for bi in 0..count {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0i64;
                    for l in 0..k {
                        acc += i64::from(a[bi * m * k + i * k + l])
                            * i64::from(b[bi * k * n + l * n + j]);
                    }
                    assert_eq!(
                        c[bi * m * n + i * n + j],
                        acc,
                        "i16-batched {bi} {m}x{n}x{k} ({i},{j})"
                    );
                }
            }
        }
    }
}

// Batched f16 with one shared fused epilogue applied to every item:
//   add_col(bias) -> relu(), and mul_col(scale) -> add_col(bias) -> silu().
// Each item validated against the f64 oracle applying the same epilogue.
#[test]
fn f16_batched_ep() {
    use sme_gemm::{Epilogue, matmul_f16_batched_ep};
    if !caps().sme_f16f16 {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (16, 16, 16, 30),
        (32, 48, 24, 12),
        (8, 33, 17, 9),
        (31, 31, 31, 5),
        (70, 100, 75, 3),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0xe9_f16e_be11_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let a = fill(&mut s, count * m * k);
        let b = fill(&mut s, count * k * n);
        let scale = fill(&mut s, n);
        let bias = fill(&mut s, n);
        let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(6e-2);

        // add_col(bias) -> relu()
        let mut c = vec![f16::ZERO; count * m * n];
        matmul_f16_batched_ep(
            &a,
            &b,
            &mut c,
            count,
            m,
            n,
            k,
            &Epilogue::<f16>::new().add_col(&bias).relu(),
        );
        for bi in 0..count {
            let ai = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let base = oracle(ai, bb, m, n, k);
            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let v = (base[i * n + j] + f64::from(bias[j])).max(0.0);
                    let got = f64::from(c[bi * m * n + i * n + j]);
                    mr = mr.max((got - v).abs() / (1.0 + v.abs()));
                }
            }
            assert!(mr < tol, "f16 batched-ep relu {bi} {m}x{n}x{k}: {mr}");
        }

        // mul_col(scale) -> add_col(bias) -> silu()
        let mut c2 = vec![f16::ZERO; count * m * n];
        matmul_f16_batched_ep(
            &a,
            &b,
            &mut c2,
            count,
            m,
            n,
            k,
            &Epilogue::<f16>::new().mul_col(&scale).add_col(&bias).silu(),
        );
        for bi in 0..count {
            let ai = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let base = oracle(ai, bb, m, n, k);
            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let v = silu_f64(base[i * n + j] * f64::from(scale[j]) + f64::from(bias[j]));
                    let got = f64::from(c2[bi * m * n + i * n + j]);
                    mr = mr.max((got - v).abs() / (1.0 + v.abs()));
                }
            }
            assert!(mr < tol, "f16 batched-ep silu {bi} {m}x{n}x{k}: {mr}");
        }
    }
}

// Batched bf16 with one shared fused epilogue: add_col(bias) -> relu() (B16B16
// allows only None/ReLU/clamp/affine). Validated against the f64 oracle per item.
#[test]
fn bf16_batched_ep() {
    use sme_gemm::{Epilogue, matmul_bf16_batched_ep};
    if !caps().sme_b16b16 {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (16, 16, 16, 24),
        (32, 48, 24, 10),
        (31, 31, 31, 5),
        (33, 65, 64, 3),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0xbf16_be11_4444_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let rb = |s: &mut u64| {
            *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            bf16::from_f32((*s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        };
        let a: Vec<bf16> = (0..count * m * k).map(|_| rb(&mut s)).collect();
        let b: Vec<bf16> = (0..count * k * n).map(|_| rb(&mut s)).collect();
        let bias: Vec<bf16> = (0..n).map(|_| rb(&mut s)).collect();
        let mut c = vec![bf16::ZERO; count * m * n];
        matmul_bf16_batched_ep(
            &a,
            &b,
            &mut c,
            count,
            m,
            n,
            k,
            &Epilogue::<bf16>::new().add_col(&bias).relu(),
        );
        let tol = (6e-2 * (k as f64).sqrt() / 4.0).max(6e-2);
        for bi in 0..count {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for l in 0..k {
                        acc += f64::from(a[bi * m * k + i * k + l].to_f32())
                            * f64::from(b[bi * k * n + l * n + j].to_f32());
                    }
                    let v = (acc + f64::from(bias[j].to_f32())).max(0.0);
                    let got = f64::from(c[bi * m * n + i * n + j].to_f32());
                    let mr = (got - v).abs() / (1.0 + v.abs());
                    assert!(mr < tol, "bf16 batched-ep {bi} {m}x{n}x{k}: {mr}");
                }
            }
        }
    }
}

// Batched f32 with one shared fused epilogue applied to every item:
//   mul_col(scale) -> add_col(bias) -> gelu()  (loose tol, rational activation)
//   clamp(0, 6)                                (tight tol, exact ops)
// Each item validated against the f64 oracle applying the same epilogue.
#[test]
fn f32_batched_ep() {
    use sme_gemm::{Epilogue, matmul_f32_batched_ep};
    if !caps().sme {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (32, 32, 32, 16),
        (33, 17, 9, 11),
        (16, 48, 24, 13),
        (31, 31, 31, 5),
        (40, 70, 33, 3),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0xf32e_be11_6666_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let a = fill_f32(&mut s, count * m * k);
        let b = fill_f32(&mut s, count * k * n);
        let scale = fill_f32(&mut s, n);
        let bias = fill_f32(&mut s, n);

        // mul_col(scale) -> add_col(bias) -> gelu()
        let mut c = vec![0.0f32; count * m * n];
        matmul_f32_batched_ep(
            &a,
            &b,
            &mut c,
            count,
            m,
            n,
            k,
            &Epilogue::<f32>::new().mul_col(&scale).add_col(&bias).gelu(),
        );
        for bi in 0..count {
            let ai = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let base = oracle_f32(ai, bb, m, n, k);
            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let v = gelu_f64(base[i * n + j] * f64::from(scale[j]) + f64::from(bias[j]));
                    let got = f64::from(c[bi * m * n + i * n + j]);
                    mr = mr.max((got - v).abs() / (1.0 + v.abs()));
                }
            }
            assert!(mr < 5e-2, "f32 batched-ep gelu {bi} {m}x{n}x{k}: {mr}");
        }

        // clamp(0, 6) -- exact composable ops, tight tol. Scale inputs up to
        // exercise both clamp bounds.
        let aw: Vec<f32> = a.iter().map(|x| x * 4.0).collect();
        let mut c2 = vec![0.0f32; count * m * n];
        matmul_f32_batched_ep(
            &aw,
            &b,
            &mut c2,
            count,
            m,
            n,
            k,
            &Epilogue::<f32>::new().clamp(0.0, 6.0),
        );
        for bi in 0..count {
            let ai = &aw[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let base = oracle_f32(ai, bb, m, n, k);
            let mut mr = 0.0f64;
            for idx in 0..m * n {
                let v = base[idx].clamp(0.0, 6.0);
                mr = mr.max((f64::from(c2[bi * m * n + idx]) - v).abs() / (1.0 + v.abs()));
            }
            assert!(mr < 1e-4, "f32 batched-ep clamp {bi} {m}x{n}x{k}: {mr}");
        }
    }
}

// Batched f64 with one shared fused epilogue applied to every item:
//   mul_col(scale) -> add_col(bias) -> clamp(lo, hi)  (tight tol, exact ops)
//   add_col(bias) -> tanh()                           (loose tol, rational tanh)
// Each item validated against the f64 oracle applying the same epilogue.
#[test]
fn f64_batched_ep() {
    use sme_gemm::{Epilogue, matmul_f64_batched_ep};
    if !caps().sme_f64f64 {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (16, 16, 16, 16),
        (17, 9, 5, 11),
        (16, 48, 24, 9),
        (40, 37, 29, 3),
        (15, 15, 15, 5),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0xf64e_be11_8888_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let a = fill_f64(&mut s, count * m * k);
        let b = fill_f64(&mut s, count * k * n);
        let scale = fill_f64(&mut s, n);
        let bias = fill_f64(&mut s, n);

        // mul_col(scale) -> add_col(bias) -> clamp(-0.5, 0.5): exact, tight tol.
        let mut c = vec![0.0f64; count * m * n];
        matmul_f64_batched_ep(
            &a,
            &b,
            &mut c,
            count,
            m,
            n,
            k,
            &Epilogue::<f64>::new()
                .mul_col(&scale)
                .add_col(&bias)
                .clamp(-0.5, 0.5),
        );
        for bi in 0..count {
            let ai = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let base = oracle_f64(ai, bb, m, n, k);
            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let v = (base[i * n + j] * scale[j] + bias[j]).clamp(-0.5, 0.5);
                    mr = mr.max((c[bi * m * n + i * n + j] - v).abs() / (1.0 + v.abs()));
                }
            }
            assert!(mr < 1e-12, "f64 batched-ep clamp {bi} {m}x{n}x{k}: {mr}");
        }

        // add_col(bias) -> tanh(): rational tanh, loose tol.
        let mut c2 = vec![0.0f64; count * m * n];
        matmul_f64_batched_ep(
            &a,
            &b,
            &mut c2,
            count,
            m,
            n,
            k,
            &Epilogue::<f64>::new().add_col(&bias).tanh(),
        );
        for bi in 0..count {
            let ai = &a[bi * m * k..(bi + 1) * m * k];
            let bb = &b[bi * k * n..(bi + 1) * k * n];
            let base = oracle_f64(ai, bb, m, n, k);
            let mut mr = 0.0f64;
            for i in 0..m {
                for j in 0..n {
                    let v = (base[i * n + j] + bias[j]).tanh();
                    mr = mr.max((c2[bi * m * n + i * n + j] - v).abs() / (1.0 + v.abs()));
                }
            }
            assert!(mr < 5e-2, "f64 batched-ep tanh {bi} {m}x{n}x{k}: {mr}");
        }
    }
}

// Batched i8 with one shared fused dequant + op-graph: scale*acc -> add_col(bias)
// -> relu(), f32 output. Validated against the exact i32 oracle dequantized the
// same way per item.
#[test]
fn i8_batched_dequant() {
    use sme_gemm::{Dequant, has_sme, matmul_i8_batched_dequant};
    if !has_sme() {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (16, 16, 16, 20),
        (32, 48, 24, 10),
        (8, 33, 17, 7),
        (31, 31, 31, 5),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0x18b1_be11_5555_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 48) as i32 % 255 - 127) as i8
        };
        let a: Vec<i8> = (0..count * m * k).map(|_| rnd()).collect();
        let b: Vec<i8> = (0..count * k * n).map(|_| rnd()).collect();
        let scale = 0.005_f32;
        let bias: Vec<f32> = (0..n).map(|j| 0.1 * (j as f32) - 0.3).collect();
        let mut c = vec![0.0f32; count * m * n];
        matmul_i8_batched_dequant(
            &a,
            &b,
            &mut c,
            count,
            m,
            n,
            k,
            &Dequant::new(scale).add_col(&bias).relu(),
        );
        for bi in 0..count {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0i32;
                    for l in 0..k {
                        acc += i32::from(a[bi * m * k + i * k + l])
                            * i32::from(b[bi * k * n + l * n + j]);
                    }
                    let v = ((scale * acc as f32) + bias[j]).max(0.0);
                    let got = c[bi * m * n + i * n + j];
                    let mr = (f64::from(got) - f64::from(v)).abs() / (1.0 + f64::from(v.abs()));
                    assert!(
                        mr < 1e-5,
                        "i8 batched-dequant {bi} {m}x{n}x{k} ({i},{j}): {mr}"
                    );
                }
            }
        }
    }
}

// Batched i16 with one shared fused dequant + op-graph (per-tensor and per-N
// scale x add_col(bias)->relu and a clamp), f32 output. Validated against the
// exact i64 oracle dequantized identically per item. Bounded inputs keep the
// i64 -> f32 store conversion within the i16_dequant tolerance.
#[test]
fn i16_batched_dequant() {
    use sme_gemm::{Dequant, caps, matmul_i16_batched_dequant};
    if !caps().sme_i16i64 {
        return;
    }
    let cases: &[(usize, usize, usize, usize)] = &[
        (65, 257, 257, 3),
        (33, 512, 257, 2),
        (97, 300, 131, 3),
        (7, 400, 800, 4),
    ];
    for &(m, n, k, count) in cases {
        let mut s = 0x16d6_ba70_9999_0001 ^ ((m * 131 + n * 17 + k * 7 + count) as u64);
        // |a|,|b| <= 64 keeps the i64 accumulator small enough for an effectively
        // exact i64 -> f32 store.
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 40) as i32 % 129 - 64) as i16
        };
        let a: Vec<i16> = (0..count * m * k).map(|_| rnd()).collect();
        let b: Vec<i16> = (0..count * k * n).map(|_| rnd()).collect();
        let scale = 0.000_037_f32;
        let scale_n: Vec<f32> = (0..n).map(|j| 0.00003 * (1.0 + (j % 7) as f32)).collect();
        let bias: Vec<f32> = (0..n).map(|j| 0.05 * (j as f32 - n as f32 / 2.0)).collect();

        // (use_per_n_scale, clamp): both run add_col(bias) -> relu().
        let variants: &[(bool, bool)] = &[(false, false), (true, false), (false, true)];
        for &(pn, clamp) in variants {
            let mut c = vec![0.0f32; count * m * n];
            let mut dq = Dequant::new(scale);
            if pn {
                dq = dq.scale_per_n(&scale_n);
            }
            dq = dq.add_col(&bias);
            if clamp {
                dq = dq.clamp(-1.0, 1.0);
            }
            dq = dq.relu();
            matmul_i16_batched_dequant(&a, &b, &mut c, count, m, n, k, &dq);

            let mut mr = 0.0f64;
            for bi in 0..count {
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0i64;
                        for l in 0..k {
                            acc += i64::from(a[bi * m * k + i * k + l])
                                * i64::from(b[bi * k * n + l * n + j]);
                        }
                        let sc = if pn { scale_n[j] } else { scale };
                        let mut want = f64::from(sc) * acc as f64 + f64::from(bias[j]);
                        if clamp {
                            want = want.clamp(-1.0, 1.0);
                        }
                        want = want.max(0.0);
                        let got = f64::from(c[bi * m * n + i * n + j]);
                        mr = mr.max((got - want).abs() / (1.0 + want.abs()));
                    }
                }
            }
            assert!(
                mr < 1e-4,
                "i16 batched-dequant {m}x{n}x{k} pn={pn} clamp={clamp}: {mr}"
            );
        }
    }
}
