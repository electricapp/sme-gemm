//! m == 1 GEMV kernels: every N-tile count and K mod 4, serial and parallel sizes.

use crate::{fill, max_rel, oracle};
use half::{bf16, f16};
use sme_gemm::{
    Accum, Gemm, caps, gemm_f16, matmul_bf16_packed, matmul_f16_packed, prepack_bf16, prepack_f16,
};

const GEMV_SIZES: &[(usize, usize)] = &[
    (1, 1),
    (31, 3),
    (32, 4),
    (33, 5),
    (100, 6),
    (127, 64),
    (128, 257),
    (129, 1024),
    (200, 7),
    (1000, 1023),
    (4109, 4099),
    (2048, 2050),
];

fn half_tol(k: usize) -> f64 {
    (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2)
}

#[test]
fn gemv_f16_bf16_packed() {
    for (si, &(n, k)) in GEMV_SIZES.iter().enumerate() {
        let m = 1 + si % 4;
        let mut s = 0x6e37_1111_0000_0001 ^ ((n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let want = oracle(&a, &b, m, n, k);
        if caps().sme_f16f16 {
            let mut c = vec![f16::ZERO; m * n];
            matmul_f16_packed(&a, &prepack_f16(&b, n, k), &mut c, m);
            let mr = max_rel(&c, &want);
            assert!(mr < half_tol(k), "f16 gemv {m}x{n}x{k}: max_rel={mr}");
        }
        if caps().sme_b16b16 {
            let ab: Vec<bf16> = a.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
            let bb: Vec<bf16> = b.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
            let mut c = vec![bf16::ZERO; m * n];
            matmul_bf16_packed(&ab, &prepack_bf16(&bb, n, k), &mut c, m);
            let mr = c
                .iter()
                .zip(&want)
                .map(|(g, w)| (f64::from(g.to_f32()) - w).abs() / (1.0 + w.abs()))
                .fold(0.0, f64::max);
            assert!(
                mr < 2.0 * half_tol(k),
                "bf16 gemv {m}x{n}x{k}: max_rel={mr}"
            );
        }
    }
}

#[test]
fn gemv_f16_epilogue() {
    if !caps().sme_f16f16 {
        return;
    }
    for (m, n, k) in [(1usize, 33usize, 5usize), (3, 129, 1024), (4, 4109, 4099)] {
        let mut s = 0x6e37_e9e9_0000_0001 ^ ((n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let scale = fill(&mut s, n);
        let bias = fill(&mut s, n);
        let row = fill(&mut s, m);
        let res = fill(&mut s, m * n);
        let base = oracle(&a, &b, m, n, k);
        let packed = prepack_f16(&b, n, k);
        let mut c = vec![f16::ZERO; m * n];
        Gemm::new(&a, &packed, m)
            .mul_col(&scale)
            .add_col(&bias)
            .add_row(&row)
            .add_tensor(&res)
            .relu()
            .run(&mut c);
        let want: Vec<f64> = (0..m * n)
            .map(|ij| {
                let (i, j) = (ij / n, ij % n);
                let v = base[ij] * f64::from(scale[j])
                    + f64::from(bias[j])
                    + f64::from(row[i])
                    + f64::from(res[ij]);
                v.max(0.0)
            })
            .collect();
        let mr = max_rel(&c, &want);
        assert!(
            mr < 2.0 * half_tol(k),
            "f16 gemv epilogue {m}x{n}x{k}: max_rel={mr}"
        );

        let mut c = vec![f16::ZERO; m * n];
        Gemm::new(&a, &packed, m).add_col(&bias).run(&mut c);
        let want: Vec<f64> = (0..m * n)
            .map(|ij| base[ij] + f64::from(bias[ij % n]))
            .collect();
        let mr = max_rel(&c, &want);
        assert!(mr < half_tol(k), "f16 gemv bias {m}x{n}x{k}: max_rel={mr}");
    }
}

#[test]
fn gemv_f16_strided_alpha_beta() {
    if !caps().sme_f16f16 {
        return;
    }
    for &(n, k) in &[(33usize, 7usize), (1000, 1023)] {
        let mut s = 0x6e37_abab_0000_0001 ^ ((n * 17 + k) as u64);
        let a_wide = fill(&mut s, 2 * k); // A[0, d] = a_wide[2*d]
        let b = fill(&mut s, k * n);
        let c0 = fill(&mut s, n);
        let a: Vec<f16> = (0..k).map(|d| a_wide[2 * d]).collect();
        let base = oracle(&a, &b, 1, n, k);
        let (alpha, beta) = (f16::from_f32(0.5), f16::from_f32(-1.25));
        let mut c = c0.clone();
        gemm_f16(
            1,
            n,
            k,
            &mut c,
            n,
            1,
            &a_wide,
            2 * k,
            2,
            &b,
            n,
            1,
            alpha,
            beta,
            Accum::F16,
        );
        let want: Vec<f64> = (0..n)
            .map(|j| 0.5 * f64::from(c0[j]) - 1.25 * base[j])
            .collect();
        let mr = max_rel(&c, &want);
        assert!(
            mr < 2.0 * half_tol(k),
            "f16 gemv alpha/beta 1x{n}x{k}: max_rel={mr}"
        );
    }
}

fn i8_operands(m: usize, n: usize, k: usize, seed: u64) -> (Vec<i8>, Vec<i8>) {
    let mut s = seed ^ ((n * 17 + k) as u64);
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 33) as i32 % 255 - 127) as i8
    };
    let a = (0..m * k).map(|_| rnd()).collect();
    let b = (0..k * n).map(|_| rnd()).collect();
    (a, b)
}

fn i8_rows(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
    let mut want = vec![0i32; m * n];
    for i in 0..m {
        for l in 0..k {
            let x = i32::from(a[i * k + l]);
            for j in 0..n {
                want[i * n + j] += x * i32::from(b[l * n + j]);
            }
        }
    }
    want
}

#[test]
fn gemv_i8_packed() {
    use sme_gemm::{matmul_i8_packed, prepack_i8};
    for (si, &(n, k)) in [
        (1usize, 1usize),
        (15, 3),
        (16, 4),
        (17, 5),
        (40, 13),
        (63, 16),
        (64, 17),
        (65, 33),
        (100, 64),
        (1000, 1023),
        (4109, 4099),
        (2048, 2050),
    ]
    .iter()
    .enumerate()
    {
        let m = 1 + si % 4;
        let (a, b) = i8_operands(m, n, k, 0x18e1_0000_0001);
        let mut c = vec![0i32; m * n];
        matmul_i8_packed(&a, &prepack_i8(&b, n, k), &mut c, m);
        assert_eq!(c, i8_rows(&a, &b, m, n, k), "i8 gemv {m}x{n}x{k}");
    }
}

#[test]
fn gemv_i8_packed_dequant() {
    use sme_gemm::{Dequant, matmul_i8_packed_dequant, prepack_i8};
    for (m, n, k) in [(1usize, 17usize, 5usize), (2, 1000, 1023), (4, 4109, 4099)] {
        let (a, b) = i8_operands(m, n, k, 0x18e1_dede_0001);
        let scale: Vec<f32> = (0..n).map(|j| 1e-3 + (j % 7) as f32 * 1e-4).collect();
        let bias: Vec<f32> = (0..n).map(|j| 0.25 - (j % 5) as f32 * 0.1).collect();
        let dq = Dequant::new(1.0).scale_per_n(&scale).add_col(&bias).relu();
        let mut c = vec![0.0f32; m * n];
        matmul_i8_packed_dequant(&a, &prepack_i8(&b, n, k), &mut c, m, &dq);
        let raw = i8_rows(&a, &b, m, n, k);
        for ij in 0..m * n {
            let j = ij % n;
            let want = (raw[ij] as f32 * scale[j] + bias[j]).max(0.0);
            let err = (c[ij] - want).abs() / (1.0 + want.abs());
            assert!(
                err < 1e-5,
                "i8 gemv dequant {m}x{n}x{k} at {ij}: got {} want {want}",
                c[ij]
            );
        }
    }
}

// Q4 GEMV against f64 dequant: ragged N and K, partial last K-block, both forms.
#[test]
fn gemv_q4() {
    use sme_gemm::{Epilogue, Q4Params, Q4Weights, matmul_q4, matmul_q4_ep};
    if !caps().sme_f16f16 {
        return;
    }
    for (m, n, k) in [
        (1usize, 33usize, 256usize),
        (1, 389, 1027),
        (1, 1100, 2051),
        (2, 100, 70),
        (3, 129, 1027),
        (4, 4109, 4099),
        (5, 70, 513),
        (7, 1030, 2050),
    ] {
        for &block in &[16usize, 32, 64] {
            for affine in [false, true] {
                let p = if affine {
                    Q4Params::new(block).affine()
                } else {
                    Q4Params::new(block)
                };
                let nbk = k.div_ceil(block);
                let mut s = 0x6e37_0404_0001u64 ^ ((n * 17 + k + block) as u64);
                let mut byte = || {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (s >> 40) as u8
                };
                let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
                let scales: Vec<f16> = (0..n * nbk)
                    .map(|i| f16::from_f32(0.01 + 0.001 * (i % 7) as f32))
                    .collect();
                let mins: Vec<f16> = (0..n * nbk)
                    .map(|i| f16::from_f32(0.05 * (i % 5) as f32 - 0.1))
                    .collect();
                let mv = affine.then_some(&mins[..]);
                let a: Vec<f16> = (0..m * k)
                    .map(|i| f16::from_f32((i % 13) as f32 * 0.05 - 0.3))
                    .collect();
                let want: Vec<f64> = (0..m * n)
                    .map(|ij| {
                        let (i, j) = (ij / n, ij % n);
                        (0..k)
                            .map(|d| {
                                let e = d * n + j;
                                let nib = if e % 2 == 0 {
                                    quants[e / 2] & 15
                                } else {
                                    quants[e / 2] >> 4
                                };
                                let code = f64::from(nib) - if nib < 8 { 0.0 } else { 16.0 };
                                let bi = j * nbk + d / block;
                                let mut w = f64::from(scales[bi]) * code;
                                if affine {
                                    w += f64::from(mins[bi]);
                                }
                                f64::from(a[i * k + d]) * w
                            })
                            .sum()
                    })
                    .collect();
                let w = Q4Weights::with_params(&quants, &scales, mv, n, k, p);
                let mut got = vec![f16::ZERO; m * n];
                matmul_q4(&a, &w, &mut got, m);
                let mr = max_rel(&got, &want);
                assert!(
                    mr < half_tol(k),
                    "q4 gemv {m}x{n}x{k} block={block} affine={affine}: {mr}"
                );

                let bias: Vec<f16> = (0..n)
                    .map(|j| f16::from_f32(0.1 * (j % 3) as f32 - 0.1))
                    .collect();
                let mut got = vec![f16::ZERO; m * n];
                matmul_q4_ep(&a, &w, &mut got, m, &Epilogue::new().add_col(&bias).relu());
                let want_ep: Vec<f64> = want
                    .iter()
                    .enumerate()
                    .map(|(ij, v)| (v + f64::from(bias[ij % n])).max(0.0))
                    .collect();
                let mr = max_rel(&got, &want_ep);
                assert!(
                    mr < half_tol(k),
                    "q4 gemv ep {m}x{n}x{k} block={block} affine={affine}: {mr}"
                );
            }
        }
    }
}

fn max_rel_bf(got: &[bf16], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(g.to_f32()) - w).abs() / (1.0 + w.abs()))
        .fold(0.0, f64::max)
}

#[test]
fn gemv_q4_bf16() {
    use sme_gemm::{Epilogue, Q4Params, Q4Weights, matmul_q4_bf16, matmul_q4_bf16_ep};
    if !caps().sme_b16b16 {
        return;
    }
    for (m, n, k) in [
        (1usize, 33usize, 256usize),
        (1, 389, 1027),
        (1, 1100, 2051),
        (2, 100, 70),
        (3, 129, 1027),
        (4, 4109, 4099),
        (5, 70, 513),
        (7, 1030, 2050),
    ] {
        for &block in &[16usize, 32, 64] {
            for affine in [false, true] {
                let p = if affine {
                    Q4Params::new(block).affine()
                } else {
                    Q4Params::new(block)
                };
                let nbk = k.div_ceil(block);
                let mut s = 0x6e37_0404_0001u64 ^ ((n * 17 + k + block) as u64);
                let mut byte = || {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (s >> 40) as u8
                };
                let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
                let scales: Vec<f16> = (0..n * nbk)
                    .map(|i| f16::from_f32(0.01 + 0.001 * (i % 7) as f32))
                    .collect();
                let mins: Vec<f16> = (0..n * nbk)
                    .map(|i| f16::from_f32(0.05 * (i % 5) as f32 - 0.1))
                    .collect();
                let mv = affine.then_some(&mins[..]);
                let a: Vec<bf16> = (0..m * k)
                    .map(|i| bf16::from_f32((i % 13) as f32 * 0.05 - 0.3))
                    .collect();
                let want: Vec<f64> = (0..m * n)
                    .map(|ij| {
                        let (i, j) = (ij / n, ij % n);
                        (0..k)
                            .map(|d| {
                                let e = d * n + j;
                                let nib = if e % 2 == 0 {
                                    quants[e / 2] & 15
                                } else {
                                    quants[e / 2] >> 4
                                };
                                let code = f64::from(nib) - if nib < 8 { 0.0 } else { 16.0 };
                                let bi = j * nbk + d / block;
                                let mut w = f64::from(scales[bi]) * code;
                                if affine {
                                    w += f64::from(mins[bi]);
                                }
                                f64::from(a[i * k + d].to_f32()) * w
                            })
                            .sum()
                    })
                    .collect();
                let w = Q4Weights::with_params(&quants, &scales, mv, n, k, p);
                let mut got = vec![bf16::ZERO; m * n];
                matmul_q4_bf16(&a, &w, &mut got, m);
                let mr = max_rel_bf(&got, &want);
                assert!(
                    mr < 2.0 * half_tol(k),
                    "q4 bf16 gemv {m}x{n}x{k} block={block} affine={affine}: {mr}"
                );

                let bias: Vec<bf16> = (0..n)
                    .map(|j| bf16::from_f32(0.1 * (j % 3) as f32 - 0.1))
                    .collect();
                let mut got = vec![bf16::ZERO; m * n];
                matmul_q4_bf16_ep(&a, &w, &mut got, m, &Epilogue::new().add_col(&bias).relu());
                let want_ep: Vec<f64> = want
                    .iter()
                    .enumerate()
                    .map(|(ij, v)| (v + f64::from(bias[ij % n].to_f32())).max(0.0))
                    .collect();
                let mr = max_rel_bf(&got, &want_ep);
                assert!(
                    mr < 2.0 * half_tol(k),
                    "q4 bf16 gemv ep {m}x{n}x{k} block={block} affine={affine}: {mr}"
                );
            }
        }
    }
}

#[test]
fn gemv_f32_packed() {
    use sme_gemm::{matmul_f32_packed, prepack_f32};
    for (si, &(n, k)) in GEMV_SIZES.iter().enumerate() {
        let m = 1 + si % 4;
        let mut s = 0x6e37_f32f_0000_0001u64 ^ ((n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
        let bias: Vec<f32> = (0..n).map(|_| rnd()).collect();
        let packed = prepack_f32(&b, n, k);
        let want: Vec<f64> = (0..m * n)
            .map(|ij| {
                let (i, j) = (ij / n, ij % n);
                (0..k)
                    .map(|l| f64::from(a[i * k + l]) * f64::from(b[l * n + j]))
                    .sum()
            })
            .collect();
        let rel = |c: &[f32], w: &dyn Fn(usize) -> f64| {
            c.iter()
                .enumerate()
                .map(|(ij, &g)| (f64::from(g) - w(ij)).abs() / (1.0 + w(ij).abs()))
                .fold(0.0, f64::max)
        };
        let mut c = vec![0.0f32; m * n];
        matmul_f32_packed(&a, &packed, &mut c, m);
        let mr = rel(&c, &|ij| want[ij]);
        assert!(mr < 1e-4, "f32 gemv {m}x{n}x{k}: {mr}");
        let mut c = vec![0.0f32; m * n];
        Gemm::new(&a, &packed, m).add_col(&bias).relu().run(&mut c);
        let mr = rel(&c, &|ij| (want[ij] + f64::from(bias[ij % n])).max(0.0));
        assert!(mr < 1e-4, "f32 gemv epilogue {m}x{n}x{k}: {mr}");
    }
}

#[test]
fn gemv_i16_packed() {
    use sme_gemm::{Dequant, matmul_i16_packed, matmul_i16_packed_dequant, prepack_i16};
    if !caps().sme_i16i64 {
        return;
    }
    for &(n, k) in &[
        (1usize, 1usize),
        (7, 3),
        (16, 16),
        (17, 17),
        (40, 70),
        (100, 1023),
        (4109, 4099),
    ] {
        let mut s = 0x6e37_1616_0001u64 ^ ((n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as i32 % 4001 - 2000) as i16
        };
        let a: Vec<i16> = (0..k).map(|_| rnd()).collect();
        let b: Vec<i16> = (0..k * n).map(|_| rnd()).collect();
        let want: Vec<i64> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|l| i64::from(a[l]) * i64::from(b[l * n + j]))
                    .sum()
            })
            .collect();
        let w = prepack_i16(&b, n, k);
        let mut c = vec![0i64; n];
        matmul_i16_packed(&a, &w, &mut c, 1);
        assert_eq!(c, want, "i16 gemv 1x{n}x{k}");
        let scale: Vec<f32> = (0..n).map(|j| 1e-6 + (j % 7) as f32 * 1e-7).collect();
        let dq = Dequant::new(1.0).scale_per_n(&scale).relu();
        let mut c = vec![0.0f32; n];
        matmul_i16_packed_dequant(&a, &w, &mut c, 1, &dq);
        for j in 0..n {
            let wv = (want[j] as f32 * scale[j]).max(0.0);
            assert!(
                (c[j] - wv).abs() <= 1e-5 * (1.0 + wv.abs()),
                "i16 gemv dq 1x{n}x{k} at {j}"
            );
        }
    }
}

#[test]
fn gemv_f64_packed() {
    use sme_gemm::prepack_f64;
    if !caps().sme_f64f64 {
        return;
    }
    for &(n, k) in &[
        (1usize, 1usize),
        (7, 3),
        (16, 16),
        (17, 17),
        (40, 70),
        (100, 1023),
        (4109, 4099),
    ] {
        let mut s = 0x6e37_f64f_0001u64 ^ ((n * 17 + k) as u64);
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let a: Vec<f64> = (0..k).map(|_| rnd()).collect();
        let b: Vec<f64> = (0..k * n).map(|_| rnd()).collect();
        let bias: Vec<f64> = (0..n).map(|_| rnd()).collect();
        let want: Vec<f64> = (0..n)
            .map(|j| (0..k).map(|l| a[l] * b[l * n + j]).sum())
            .collect();
        let packed = prepack_f64(&b, n, k);
        let mut c = vec![0.0f64; n];
        Gemm::new(&a, &packed, 1).run(&mut c);
        for j in 0..n {
            assert!(
                (c[j] - want[j]).abs() <= 1e-12 * (1.0 + want[j].abs()),
                "f64 gemv 1x{n}x{k} at {j}"
            );
        }
        let mut c = vec![0.0f64; n];
        Gemm::new(&a, &packed, 1).add_col(&bias).relu().run(&mut c);
        for j in 0..n {
            let wv = (want[j] + bias[j]).max(0.0);
            assert!(
                (c[j] - wv).abs() <= 1e-12 * (1.0 + wv.abs()),
                "f64 gemv ep 1x{n}x{k} at {j}"
            );
        }
    }
}
