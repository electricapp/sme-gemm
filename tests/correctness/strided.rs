//! Strided / transposed / column-major views, accumulation, and tail shapes.

use crate::{TAIL_SIZES, check, gemm_oracle};
use half::{bf16, f16};
use sme_gemm::{Accum, caps, matmul_f32, matmul_i8};

#[test]
fn gemm_f32_accumulate() {
    use sme_gemm::gemm_f32;
    // 65^3-ish reads B in place; 40x4100x1030 (B past 14 MB) builds it per call.
    for &(m, n, k) in &[(65, 257, 257), (40, 4100, 1030)] {
        let mut s = 0xacc0_f32f_0000_0001u64 ^ (m * 131 + n * 17 + k) as u64;
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
        let c_init: Vec<f32> = (0..m * n).map(|_| rnd()).collect();
        let (alpha, beta) = (0.5f32, 2.0f32);

        let mut c = c_init.clone();
        // row-major strides: c_row=n,c_col=1 ; a_row=k,a_col=1 ; b_row=n,b_col=1.
        gemm_f32(m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, alpha, beta);

        let af: Vec<f64> = a.iter().map(|&x| f64::from(x)).collect();
        let bf: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
        let cf: Vec<f64> = c_init.iter().map(|&x| f64::from(x)).collect();
        let want = gemm_oracle(&af, &bf, &cf, m, n, k, f64::from(alpha), f64::from(beta));
        let mut mr = 0.0f64;
        for (g, w) in c.iter().zip(&want) {
            mr = mr.max((f64::from(*g) - w).abs() / (1.0 + w.abs()));
        }
        assert!(mr < 1e-4, "gemm_f32 accumulate {m}x{n}x{k}: max_rel={mr}");
    }
}

// Covers each route without pre-packed B: per-worker columns (65), in-place
// GEMV (3; 2 dispatched), and the panel ring (4200, A past 8 MB).
const ACC_SIZES: &[(usize, usize, usize)] = &[
    (65, 257, 257),
    (3, 300, 257),
    (2, 2100, 1030),
    (4200, 40, 1030),
];

#[test]
fn gemm_f16_accumulate() {
    use sme_gemm::gemm_f16;
    for &(m, n, k) in ACC_SIZES {
        let mut s = 0xacc0_f16f_0000_0001u64 ^ (m * 131 + n * 17 + k) as u64;
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            f16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        };
        let a: Vec<f16> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<f16> = (0..k * n).map(|_| rnd()).collect();
        let c_init: Vec<f16> = (0..m * n).map(|_| rnd()).collect();
        let (alpha, beta) = (f16::from_f32(0.5), f16::from_f32(2.0));
        let af: Vec<f64> = a.iter().map(|&x| f64::from(x)).collect();
        let bf: Vec<f64> = b.iter().map(|&x| f64::from(x)).collect();
        let cf: Vec<f64> = c_init.iter().map(|&x| f64::from(x)).collect();
        let want = gemm_oracle(
            &af,
            &bf,
            &cf,
            m,
            n,
            k,
            f64::from(alpha.to_f32()),
            f64::from(beta.to_f32()),
        );
        // Native f16 accumulation drifts ~sqrt(k), so that mode is held to the
        // packed entry's product (same K-order) instead of the f64 oracle.
        let mut ab = vec![f16::ZERO; m * n];
        sme_gemm::matmul_f16_packed(&a, &sme_gemm::prepack_f16(&b, n, k), &mut ab, m);
        let (fa, fb) = (f64::from(alpha.to_f32()), f64::from(beta.to_f32()));
        let native: Vec<f64> = cf
            .iter()
            .zip(&ab)
            .map(|(&c0, &p)| fa * c0 + fb * f64::from(p))
            .collect();
        for (accum, want) in [(Accum::F32, &want), (Accum::F16, &native)] {
            if matches!(accum, Accum::F16) && !caps().sme_f16f16 {
                continue;
            }
            let mut c = c_init.clone();
            gemm_f16(
                m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, alpha, beta, accum,
            );
            let mut mr = 0.0f64;
            for (g, w) in c.iter().zip(want) {
                mr = mr.max((f64::from(*g) - w).abs() / (1.0 + w.abs()));
            }
            assert!(
                mr < 2e-2,
                "gemm_f16 accumulate {m}x{n}x{k} {accum:?}: max_rel={mr}"
            );
        }
    }
}

#[test]
fn gemm_bf16_accumulate() {
    use sme_gemm::gemm_bf16;
    for &(m, n, k) in ACC_SIZES {
        let mut s = 0xacc0_bf16_0000_0001u64 ^ (m * 131 + n * 17 + k) as u64;
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            bf16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
        };
        let a: Vec<bf16> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<bf16> = (0..k * n).map(|_| rnd()).collect();
        let c_init: Vec<bf16> = (0..m * n).map(|_| rnd()).collect();
        let (alpha, beta) = (bf16::from_f32(0.5), bf16::from_f32(2.0));
        let af: Vec<f64> = a.iter().map(|x| f64::from(x.to_f32())).collect();
        let bf: Vec<f64> = b.iter().map(|x| f64::from(x.to_f32())).collect();
        let cf: Vec<f64> = c_init.iter().map(|x| f64::from(x.to_f32())).collect();
        let want = gemm_oracle(
            &af,
            &bf,
            &cf,
            m,
            n,
            k,
            f64::from(alpha.to_f32()),
            f64::from(beta.to_f32()),
        );
        // As gemm_f16_accumulate: native bf16 accumulation is held to the packed
        // entry's product. bf16 storage (8-bit mantissa) -> loose tol.
        let mut ab = vec![bf16::ZERO; m * n];
        sme_gemm::matmul_bf16_packed(&a, &sme_gemm::prepack_bf16(&b, n, k), &mut ab, m);
        let (fa, fb) = (f64::from(alpha.to_f32()), f64::from(beta.to_f32()));
        let native: Vec<f64> = cf
            .iter()
            .zip(&ab)
            .map(|(&c0, p)| fa * c0 + fb * f64::from(p.to_f32()))
            .collect();
        let tol = (6e-2 * (k as f64).sqrt() / 4.0).max(6e-2);
        for (accum, want, tol) in [(Accum::F32, &want, tol), (Accum::Bf16, &native, 6e-2)] {
            if matches!(accum, Accum::Bf16) && !caps().sme_b16b16 {
                continue;
            }
            let mut c = c_init.clone();
            gemm_bf16(
                m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, alpha, beta, accum,
            );
            let mut mr = 0.0f64;
            for (g, w) in c.iter().zip(want) {
                mr = mr.max((f64::from(g.to_f32()) - w).abs() / (1.0 + w.abs()));
            }
            assert!(
                mr < tol,
                "gemm_bf16 accumulate {m}x{n}x{k} {accum:?}: max_rel={mr}"
            );
        }
    }
}

#[test]
fn gemm_f32_transposed_b() {
    // B supplied column-major (B^T row-major): b_row=1, b_col=k. The kernel must
    // read B[l,j] = bt[j*k + l], giving the same A @ B as the row-major oracle.
    use sme_gemm::gemm_f32;
    let (m, n, k) = (33, 512, 257); // 4_342_272 >= 2^18
    let mut s = 0x7a17_b00b_0000_0001u64;
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
    // bt is n x k row-major (i.e. B transposed); B[l,j] = bt[j*k + l].
    let bt: Vec<f32> = (0..n * k).map(|_| rnd()).collect();

    let mut c = vec![0.0f32; m * n];
    // a row-major (a_row=k,a_col=1); b via strides (b_row=1, b_col=k); overwrite.
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
    assert!(mr < 1e-4, "gemm_f32 transposed-B {m}x{n}x{k}: max_rel={mr}");
}

// Transposed B for the 16-bit dtypes, both accumulators -- the four pack
// paths that read a column-major B. Native 16-bit takes the non-widening
// kernels' 8x8-transpose pack, f32-accum the widening kernels' transpose-and-zip
// pack into the pair-interleaved layout; each is a distinct routine.
//
// The shape has to clear SME_MIN_FLOPS (2^18) or the whole thing runs the
// scalar reference and proves nothing. flash_half exercises transposed B too,
// but only one of its tile configurations clears the floor, and that one missed
// a deliberate half-vector fault in the widening pack -- hence this direct test.
/// A (`m x k`) and the same logical B in both layouts: `bt` is `n x k` row-major
/// (B transposed, so `B[l,j] = bt[j*k+l]`) and `br` is the `k x n` row-major
/// form. The shape clears `SME_MIN_FLOPS` so the kernels actually run.
fn transposed_b_operands() -> (usize, usize, usize, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (m, n, k) = (33usize, 129usize, 257usize); // 1_094_049 >= 2^18
    let mut s = 0x51ed_270b_c0ff_ee01u64;
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
    let bt: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
    let br: Vec<f32> = (0..k * n).map(|i| bt[(i % n) * k + i / n]).collect();
    (m, n, k, a, bt, br)
}

#[test]
fn gemm_f16_transposed_b() {
    use sme_gemm::gemm_f16;
    if !caps().sme {
        return;
    }
    let (m, n, k, af, btf, brf) = transposed_b_operands();
    let a: Vec<f16> = af.iter().map(|&x| f16::from_f32(x)).collect();
    let bt: Vec<f16> = btf.iter().map(|&x| f16::from_f32(x)).collect();
    let br: Vec<f16> = brf.iter().map(|&x| f16::from_f32(x)).collect();
    for accum in [Accum::F32, Accum::F16] {
        let (mut ct, mut cr) = (vec![f16::ZERO; m * n], vec![f16::ZERO; m * n]);
        let z = f16::ZERO;
        let o = f16::ONE;
        gemm_f16(m, n, k, &mut ct, n, 1, &a, k, 1, &bt, 1, k, z, o, accum);
        gemm_f16(m, n, k, &mut cr, n, 1, &a, k, 1, &br, n, 1, z, o, accum);
        assert!(
            ct.iter().zip(&cr).all(|(x, y)| x.to_bits() == y.to_bits()),
            "gemm_f16 transposed-B {accum:?} differs from the row-major layout"
        );
    }
}

#[test]
fn gemm_bf16_transposed_b() {
    use sme_gemm::gemm_bf16;
    if !caps().sme {
        return;
    }
    let (m, n, k, af, btf, brf) = transposed_b_operands();
    let a: Vec<bf16> = af.iter().map(|&x| bf16::from_f32(x)).collect();
    let bt: Vec<bf16> = btf.iter().map(|&x| bf16::from_f32(x)).collect();
    let br: Vec<bf16> = brf.iter().map(|&x| bf16::from_f32(x)).collect();
    for accum in [Accum::F32, Accum::Bf16] {
        let (mut ct, mut cr) = (vec![bf16::ZERO; m * n], vec![bf16::ZERO; m * n]);
        let z = bf16::ZERO;
        let o = bf16::ONE;
        gemm_bf16(m, n, k, &mut ct, n, 1, &a, k, 1, &bt, 1, k, z, o, accum);
        gemm_bf16(m, n, k, &mut cr, n, 1, &a, k, 1, &br, n, 1, z, o, accum);
        assert!(
            ct.iter().zip(&cr).all(|(x, y)| x.to_bits() == y.to_bits()),
            "gemm_bf16 transposed-B {accum:?} differs from the row-major layout"
        );
    }
}

#[test]
fn gemm_f32_col_major_output() {
    // C column-major via strides: c_row=1, c_col=m. C[i,j] = c[i + j*m].
    use sme_gemm::gemm_f32;
    let (m, n, k) = (97, 300, 131); // 3_812_100 >= 2^18
    let mut s = 0xc011_f32f_0000_0001u64;
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
    let mut c = vec![0.0f32; m * n];
    gemm_f32(m, n, k, &mut c, 1, m, &a, k, 1, &b, n, 1, 0.0, 1.0);

    let mut mr = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..k {
                acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
            }
            let got = f64::from(c[i + j * m]);
            mr = mr.max((got - acc).abs() / (1.0 + acc.abs()));
        }
    }
    assert!(
        mr < 1e-4,
        "gemm_f32 col-major-out {m}x{n}x{k}: max_rel={mr}"
    );
}

#[test]
fn gemm_i8_strided_and_colmajor() {
    // gemm_i8 is overwrite-only (no alpha/beta), so exercise transposed B and
    // column-major C only. Exact integer result.
    use sme_gemm::gemm_i8;
    let (m, n, k) = (65, 257, 257);
    let mut s = 0x7a17_18c8_0000_0001u64;
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 48) as i32 % 255 - 127) as i8
    };
    let a: Vec<i8> = (0..m * k).map(|_| rnd()).collect();
    // bt is n x k row-major; B[l,j] = bt[j*k + l].
    let bt: Vec<i8> = (0..n * k).map(|_| rnd()).collect();

    // transposed B, row-major C.
    let mut c = vec![0i32; m * n];
    gemm_i8(m, n, k, &mut c, n, 1, &a, k, 1, &bt, 1, k);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for l in 0..k {
                acc += i32::from(a[i * k + l]) * i32::from(bt[j * k + l]);
            }
            assert_eq!(c[i * n + j], acc, "i8 transposed-B {m}x{n}x{k} ({i},{j})");
        }
    }

    // row-major B, column-major C (c_row=1, c_col=m).
    let b: Vec<i8> = (0..k * n).map(|_| rnd()).collect();
    let mut cc = vec![0i32; m * n];
    gemm_i8(m, n, k, &mut cc, 1, m, &a, k, 1, &b, n, 1);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for l in 0..k {
                acc += i32::from(a[i * k + l]) * i32::from(b[l * n + j]);
            }
            assert_eq!(cc[i + j * m], acc, "i8 col-major {m}x{n}x{k} ({i},{j})");
        }
    }
}

#[test]
fn gemm_i16_strided_and_colmajor() {
    // gemm_i16 is overwrite-only (no alpha/beta): transposed B + column-major C.
    use sme_gemm::gemm_i16;
    if !caps().sme_i16i64 {
        return;
    }
    let (m, n, k) = (65, 257, 257);
    let mut s = 0x7a17_1616_0000_0001u64;
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 33) as i32 % 65535 - 32767) as i16
    };
    let a: Vec<i16> = (0..m * k).map(|_| rnd()).collect();
    let bt: Vec<i16> = (0..n * k).map(|_| rnd()).collect();

    // transposed B, row-major C.
    let mut c = vec![0i64; m * n];
    gemm_i16(m, n, k, &mut c, n, 1, &a, k, 1, &bt, 1, k);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i64;
            for l in 0..k {
                acc += i64::from(a[i * k + l]) * i64::from(bt[j * k + l]);
            }
            assert_eq!(c[i * n + j], acc, "i16 transposed-B {m}x{n}x{k} ({i},{j})");
        }
    }

    // row-major B, column-major C.
    let b: Vec<i16> = (0..k * n).map(|_| rnd()).collect();
    let mut cc = vec![0i64; m * n];
    gemm_i16(m, n, k, &mut cc, 1, m, &a, k, 1, &b, n, 1);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i64;
            for l in 0..k {
                acc += i64::from(a[i * k + l]) * i64::from(b[l * n + j]);
            }
            assert_eq!(cc[i + j * m], acc, "i16 col-major {m}x{n}x{k} ({i},{j})");
        }
    }
}

#[test]
fn sme_tail_shapes_clear_threshold() {
    // Guard the comment math: every TAIL_SIZES entry must cross the SME path.
    for &(m, n, k) in TAIL_SIZES {
        assert!(m * n * k >= 1 << 18, "{m}x{n}x{k} below 2^18");
    }
}

#[test]
fn f16_tails() {
    for &(m, n, k) in TAIL_SIZES {
        // f32 accum (f16xf16->f32) is tight; f16 accum grows ~sqrt(k).
        check(m, n, k, Accum::F32, 2e-2);
        if caps().sme_f16f16 {
            let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
            check(m, n, k, Accum::F16, tol);
        }
    }
}

#[test]
fn f32_tails() {
    for &(m, n, k) in TAIL_SIZES {
        let mut s = 0x7a17_f32f_0000_0001 ^ ((m * 131 + n * 17 + k) as u64);
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
        assert!(mr < 1e-4, "f32-tail {m}x{n}x{k}: max_rel={mr}");
    }
}

#[test]
fn i8_tails() {
    // Integer GEMM is exact even on the tail predicates: match i32 reference.
    for &(m, n, k) in TAIL_SIZES {
        let mut s = 0x7a17_18b1_0000_0001 ^ ((m * 131 + n * 17 + k) as u64);
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
                assert_eq!(c[i * n + j], acc, "i8-tail {m}x{n}x{k} at ({i},{j})");
            }
        }
    }
}

/// Without pre-packed B the 16-bit entries pack B per call (per worker, or into a
/// shared panel ring once A is large) or, for m <= 4, read it in place. Every
/// route runs each output's K-sum in the packed entry's order, so they must agree
/// bit for bit -- across row-major, transposed and strided B, odd and partial
/// N-tiles, K tails, and more ring blocks than slots. Column-major C is checked
/// past m = 4 only: there small m takes MOPA instead of the GEMV.
macro_rules! unpacked_matches_packed {
    ($name:ident, $t:ty, $gemm:ident, $prepack:ident, $packed:ident, $accum:expr, $cap:ident) => {
        #[test]
        fn $name() {
            use sme_gemm::{$gemm, $packed, $prepack};
            if !caps().$cap {
                return;
            }
            for &(m, n, k) in &[
                (96, 4100, 2050),
                (200, 4100, 1027),
                (40, 1000, 300),
                (130, 64, 4096),
                (520, 1050, 8200),
                (1, 4100, 2050),
                (3, 1000, 301),
                (4, 70, 513),
            ] {
                let mut s = (m * 31 + n * 7 + k) as u64;
                let mut rnd = || {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    <$t>::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                };
                let a: Vec<$t> = (0..m * k).map(|_| rnd()).collect();
                let b: Vec<$t> = (0..k * n).map(|_| rnd()).collect();
                let mut want = vec![<$t>::ZERO; m * n];
                $packed(&a, &$prepack(&b, n, k), &mut want, m);
                let (z, o) = (<$t>::ZERO, <$t>::ONE);
                let same = |c: &[$t], cs: usize, rs: usize, what: &str| {
                    for i in 0..m {
                        for j in 0..n {
                            let (g, w) = (c[i * rs + j * cs], want[i * n + j]);
                            assert_eq!(g.to_bits(), w.to_bits(), "{what} {m}x{n}x{k} at ({i},{j})");
                        }
                    }
                };
                let mut c = vec![<$t>::ZERO; m * n];
                $gemm(m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, z, o, $accum);
                same(&c, 1, n, "row-major B");
                if m > 4 {
                    $gemm(m, n, k, &mut c, 1, m, &a, k, 1, &b, n, 1, z, o, $accum);
                    same(&c, m, 1, "col-major C");
                }
                let bt: Vec<$t> = (0..n * k).map(|i| b[(i % k) * n + i / k]).collect();
                $gemm(m, n, k, &mut c, n, 1, &a, k, 1, &bt, 1, k, z, o, $accum);
                same(&c, 1, n, "transposed B");
                let mut b2 = vec![<$t>::ZERO; 2 * k * n];
                for (i, &x) in b.iter().enumerate() {
                    b2[2 * i] = x;
                }
                $gemm(m, n, k, &mut c, n, 1, &a, k, 1, &b2, 2 * n, 2, z, o, $accum);
                same(&c, 1, n, "strided B");
            }
        }
    };
}

unpacked_matches_packed!(
    f16_unpacked_matches_packed,
    f16,
    gemm_f16,
    prepack_f16,
    matmul_f16_packed,
    Accum::F16,
    sme_f16f16
);
unpacked_matches_packed!(
    bf16_unpacked_matches_packed,
    bf16,
    gemm_bf16,
    prepack_bf16,
    matmul_bf16_packed,
    Accum::Bf16,
    sme_b16b16
);

/// f32 without pre-packed B, past the 14 MB direct-B limit: per-worker columns
/// (40), the panel ring with more blocks than slots (300), the in-place GEMV
/// (1, 2; K tails), and transposed and strided B, all bit for bit against the
/// packed entry (same K-order). Column-major C is checked past m = 2 only: there
/// small m takes MOPA instead of the GEMV.
#[test]
fn f32_unpacked_matches_packed() {
    use sme_gemm::{gemm_f32, matmul_f32_packed, prepack_f32};
    if !caps().sme {
        return;
    }
    for &(m, n, k) in &[
        (40, 4100, 1030),
        (300, 1500, 8200),
        (1, 4100, 1030),
        (2, 1000, 301),
        (1, 70, 513),
    ] {
        let mut s = (m * 31 + n * 7 + k) as u64;
        let mut rnd = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let a: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
        let mut want = vec![0.0f32; m * n];
        matmul_f32_packed(&a, &prepack_f32(&b, n, k), &mut want, m);
        let same = |c: &[f32], cs: usize, rs: usize, what: &str| {
            for i in 0..m {
                for j in 0..n {
                    let (g, w) = (c[i * rs + j * cs], want[i * n + j]);
                    assert_eq!(g.to_bits(), w.to_bits(), "{what} {m}x{n}x{k} at ({i},{j})");
                }
            }
        };
        let mut c = vec![0.0f32; m * n];
        gemm_f32(m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, 0.0, 1.0);
        same(&c, 1, n, "row-major B");
        if m > 2 {
            gemm_f32(m, n, k, &mut c, 1, m, &a, k, 1, &b, n, 1, 0.0, 1.0);
            same(&c, m, 1, "col-major C");
        }
        let bt: Vec<f32> = (0..n * k).map(|i| b[(i % k) * n + i / k]).collect();
        gemm_f32(m, n, k, &mut c, n, 1, &a, k, 1, &bt, 1, k, 0.0, 1.0);
        same(&c, 1, n, "transposed B");
        let mut b2 = vec![0.0f32; 2 * k * n];
        for (i, &x) in b.iter().enumerate() {
            b2[2 * i] = x;
        }
        gemm_f32(m, n, k, &mut c, n, 1, &a, k, 1, &b2, 2 * n, 2, 0.0, 1.0);
        same(&c, 1, n, "strided B");
    }
}

/// The widening drivers read a row-major B in place (zipping row pairs into the
/// MOPA's interleaved layout) but pack any other layout first. Same per-output
/// K-order either way, so row-major and transposed B must agree bit for bit:
/// one A band (16), a full M-tile against a line pair (31), more M-tiles than
/// the first pass (40, 200), odd k and ragged N.
macro_rules! widening_direct_matches_packed {
    ($name:ident, $t:ty, $gemm:ident) => {
        #[test]
        fn $name() {
            use sme_gemm::$gemm;
            if !caps().sme {
                return;
            }
            for &(m, n, k) in &[
                (16, 4100, 1031),
                (31, 4100, 513),
                (40, 1000, 301),
                (200, 2050, 257),
            ] {
                let mut s = (m * 31 + n * 7 + k) as u64;
                let mut rnd = || {
                    s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    <$t>::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                };
                let a: Vec<$t> = (0..m * k).map(|_| rnd()).collect();
                let b: Vec<$t> = (0..k * n).map(|_| rnd()).collect();
                let bt: Vec<$t> = (0..n * k).map(|i| b[(i % k) * n + i / k]).collect();
                let (z, o) = (<$t>::ZERO, <$t>::ONE);
                let (mut cr, mut ct) = (vec![z; m * n], vec![z; m * n]);
                $gemm(m, n, k, &mut cr, n, 1, &a, k, 1, &b, n, 1, z, o, Accum::F32);
                $gemm(
                    m,
                    n,
                    k,
                    &mut ct,
                    n,
                    1,
                    &a,
                    k,
                    1,
                    &bt,
                    1,
                    k,
                    z,
                    o,
                    Accum::F32,
                );
                for (i, (x, y)) in cr.iter().zip(&ct).enumerate() {
                    assert_eq!(x.to_bits(), y.to_bits(), "{m}x{n}x{k} at {i}");
                }
            }
        }
    };
}

widening_direct_matches_packed!(f16_widening_direct_matches_packed, f16, gemm_f16);
widening_direct_matches_packed!(bf16_widening_direct_matches_packed, bf16, gemm_bf16);
