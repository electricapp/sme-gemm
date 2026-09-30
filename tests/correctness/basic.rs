//! Per-dtype exactness and prepacked-weight parity against the f64/i32 oracle.

use crate::{SIZES, check, fill, max_rel, oracle};
use half::{bf16, f16};
use sme_gemm::{
    Accum, caps, matmul_bf16, matmul_bf16_packed, matmul_f16_packed, matmul_f32, matmul_i8,
    prepack_bf16, prepack_f16,
};

#[test]
fn f16_f32_accum() {
    for &(m, n, k) in SIZES {
        check(m, n, k, Accum::F32, 2e-2);
    }
}

#[test]
fn f16_accum() {
    // fp16 accumulation: error grows ~sqrt(k); only runs the SME path if M5.
    if !caps().sme_f16f16 {
        return;
    }
    for &(m, n, k) in SIZES {
        let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
        check(m, n, k, Accum::F16, tol);
    }
}

#[test]
fn f32_full() {
    for &(m, n, k) in SIZES {
        let mut s = 0xabcd_ef01_2345_6789 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
        let mut c = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut c, m, n, k);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
                }
                mr = mr.max((f64::from(c[i * n + j]) - acc).abs() / (1.0 + acc.abs()));
            }
        }
        assert!(mr < 1e-4, "f32 {m}x{n}x{k}: max_rel={mr}");
    }
}

#[test]
fn f64_full() {
    use sme_gemm::matmul_f64;
    for &(m, n, k) in SIZES {
        let mut s = 0xf64f_64f6_4f64_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as f64 / (1u64 << 24) as f64 - 0.5
        };
        let a: Vec<f64> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<f64> = (0..k * n).map(|_| rnd()).collect();
        let mut c = vec![0.0f64; m * n];
        matmul_f64(&a, &b, &mut c, m, n, k);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += a[i * k + l] * b[l * n + j];
                }
                mr = mr.max((c[i * n + j] - acc).abs() / (1.0 + acc.abs()));
            }
        }
        assert!(mr < 1e-12, "f64 {m}x{n}x{k}: max_rel={mr}");
    }
}

#[test]
fn bf16_widening() {
    // bf16 has an 8-bit mantissa; output rounding dominates -> coarse tol.
    for &(m, n, k) in SIZES {
        let mut s = 0xfeed_face_0000_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let a: Vec<bf16> = (0..m * k)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                bf16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            })
            .collect();
        let b: Vec<bf16> = (0..k * n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                bf16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            })
            .collect();
        let mut c = vec![bf16::ZERO; m * n];
        matmul_bf16(&a, &b, &mut c, m, n, k, Accum::F32);
        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l].to_f32()) * f64::from(b[l * n + j].to_f32());
                }
                want[i * n + j] = acc;
            }
        }
        let mr = c
            .iter()
            .zip(&want)
            .map(|(g, w)| (f64::from(g.to_f32()) - w).abs() / (1.0 + w.abs()))
            .fold(0.0, f64::max);
        assert!(mr < 6e-2, "bf16 {m}x{n}x{k}: max_rel={mr}");
    }
}

#[test]
fn i8_exact() {
    // Integer GEMM is exact: must match the i32 reference bit-for-bit.
    for &(m, n, k) in SIZES {
        let mut s = 0x0bad_c0de_dead_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 48) as i32 % 255 - 127) as i8
        };
        let a: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i8> = (0..k * n).map(|_| rnd()).collect();
        let mut c = vec![0i32; m * n];
        matmul_i8(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for l in 0..k {
                    acc += i32::from(a[i * k + l]) * i32::from(b[l * n + j]);
                }
                assert_eq!(c[i * n + j], acc, "i8 {m}x{n}x{k} at ({i},{j})");
            }
        }
    }
}

#[test]
fn i16_exact() {
    use sme_gemm::matmul_i16;
    // i16 x i16 -> i64 is exact: must match the i64 reference bit-for-bit.
    for &(m, n, k) in SIZES {
        let mut s = 0x1616_1616_2727_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as i32 % 65535 - 32767) as i16
        };
        let a: Vec<i16> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i16> = (0..k * n).map(|_| rnd()).collect();
        let mut c = vec![0i64; m * n];
        matmul_i16(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i64;
                for l in 0..k {
                    acc += i64::from(a[i * k + l]) * i64::from(b[l * n + j]);
                }
                assert_eq!(c[i * n + j], acc, "i16 {m}x{n}x{k} at ({i},{j})");
            }
        }
    }
}

#[test]
fn f16_prepacked() {
    if !caps().sme_f16f16 {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0x9e37_79b9_0000_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let packed = prepack_f16(&b, n, k);
        let mut c = vec![f16::ZERO; m * n];
        matmul_f16_packed(&a, &packed, &mut c, m);
        let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
        let mr = max_rel(&c, &oracle(&a, &b, m, n, k));
        assert!(mr < tol, "prepacked {m}x{n}x{k}: max_rel={mr}");
    }
}

#[test]
fn bf16_prepacked() {
    // Non-widening B16B16: bf16 accumulate, error grows ~sqrt(k); coarse tol
    // (bf16 8-bit mantissa). Only runs the SME path on M5.
    if !caps().sme_b16b16 {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0xb16b_16b1_0000_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let a: Vec<bf16> = (0..m * k)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                bf16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            })
            .collect();
        let b: Vec<bf16> = (0..k * n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                bf16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
            })
            .collect();
        let packed = prepack_bf16(&b, n, k);
        let mut c = vec![bf16::ZERO; m * n];
        matmul_bf16_packed(&a, &packed, &mut c, m);
        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l].to_f32()) * f64::from(b[l * n + j].to_f32());
                }
                want[i * n + j] = acc;
            }
        }
        // bf16 accumulation: per-step rounding accrues over k -> looser tol.
        let tol = (6e-2 * (k as f64).sqrt() / 4.0).max(6e-2);
        let mr = c
            .iter()
            .zip(&want)
            .map(|(g, w)| (f64::from(g.to_f32()) - w).abs() / (1.0 + w.abs()))
            .fold(0.0, f64::max);
        assert!(mr < tol, "bf16-fast {m}x{n}x{k}: max_rel={mr}");
    }
}

#[test]
fn i8_prepacked() {
    // Pre-packed B path: still exact integer GEMM.
    for &(m, n, k) in SIZES {
        let mut s = 0x1234_5678_9abc_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 48) as i32 % 255 - 127) as i8
        };
        let a: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i8> = (0..k * n).map(|_| rnd()).collect();
        let packed = sme_gemm::prepack_i8(&b, n, k);
        let mut c = vec![0i32; m * n];
        sme_gemm::matmul_i8_packed(&a, &packed, &mut c, m);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for l in 0..k {
                    acc += i32::from(a[i * k + l]) * i32::from(b[l * n + j]);
                }
                assert_eq!(c[i * n + j], acc, "i8-packed {m}x{n}x{k} at ({i},{j})");
            }
        }
    }
}

#[test]
fn f32_packed() {
    for &(m, n, k) in SIZES {
        let mut s = 0x5151_2323_4545_6767 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
        let packed = sme_gemm::prepack_f32(&b, n, k);
        let mut c = vec![0.0f32; m * n];
        sme_gemm::matmul_f32_packed(&a, &packed, &mut c, m);
        let mut mr = 0.0f64;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
                }
                mr = mr.max((f64::from(c[i * n + j]) - acc).abs() / (1.0 + acc.abs()));
            }
        }
        assert!(mr < 1e-4, "f32-packed {m}x{n}x{k}: max_rel={mr}");
    }
}

// i16 packed-B: exact against a closed-form i64 oracle (raw), and against the
// scalar dequant for the fused f32 path. Shapes straddle the 16-wide super-tile
// and the 4-deep SMOPA group so the pad lanes and partial k-groups are covered.
#[test]
fn i16_packed_matches_reference() {
    use sme_gemm::{matmul_i16, matmul_i16_packed, prepack_i16};
    if !caps().sme_i16i64 {
        return;
    }
    for &(m, n, k) in &[
        (1usize, 1usize, 1usize),
        (7, 5, 3),
        (16, 16, 16),
        (33, 17, 9),
        (65, 97, 129),
        (64, 8, 512),
    ] {
        let mut s = 0x1616_0f0f_0000_0001u64 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as i32 % 4001 - 2000) as i16
        };
        let a: Vec<i16> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i16> = (0..k * n).map(|_| rnd()).collect();

        let mut want = vec![0i64; m * n];
        matmul_i16(&a, &b, &mut want, m, n, k);
        let w = prepack_i16(&b, n, k);
        let mut got = vec![0i64; m * n];
        matmul_i16_packed(&a, &w, &mut got, m);
        assert_eq!(got, want, "i16 packed {m}x{n}x{k}");
    }
}

#[test]
fn i16_packed_dequant_matches_unpacked() {
    use sme_gemm::{Dequant, matmul_i16_dequant, matmul_i16_packed_dequant, prepack_i16};
    if !caps().sme_i16i64 {
        return;
    }
    for &(m, n, k) in &[(33usize, 17usize, 9usize), (65, 97, 129), (64, 8, 512)] {
        let mut s = 0x1616_dede_0000_0001u64 ^ ((m * 131 + n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as i32 % 501 - 250) as i16
        };
        let a: Vec<i16> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<i16> = (0..k * n).map(|_| rnd()).collect();
        let scale: Vec<f32> = (0..n).map(|j| 1e-5 + (j % 7) as f32 * 1e-6).collect();
        let bias: Vec<f32> = (0..n).map(|j| 0.25 - (j % 5) as f32 * 0.1).collect();
        let dq = Dequant::new(1e-5).scale_per_n(&scale).add_col(&bias).relu();

        let mut want = vec![0.0f32; m * n];
        matmul_i16_dequant(&a, &b, &mut want, m, n, k, &dq);
        let w = prepack_i16(&b, n, k);
        let mut got = vec![0.0f32; m * n];
        matmul_i16_packed_dequant(&a, &w, &mut got, m, &dq);
        for (idx, (g, wv)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - wv).abs() <= 1e-5 * (1.0 + wv.abs()),
                "i16 packed dequant {m}x{n}x{k} idx={idx}: {g} vs {wv}"
            );
        }
    }
}
