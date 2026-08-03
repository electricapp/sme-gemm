//! Strided / transposed / column-major views, accumulation, and tail shapes.

use crate::{TAIL_SIZES, check, gemm_oracle};
use half::{bf16, f16};
use sme_gemm::{Accuracy, caps, matmul_f32, matmul_i8};

#[test]
fn gemm_f32_accumulate() {
    use sme_gemm::gemm_f32;
    let (m, n, k) = (65, 257, 257); // 4_293_185 >= 2^18
    let mut s = 0xacc0_f32f_0000_0001u64;
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

#[test]
fn gemm_f16_accumulate() {
    use sme_gemm::gemm_f16;
    let (m, n, k) = (65, 257, 257);
    let mut s = 0xacc0_f16f_0000_0001u64;
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        f16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
    };
    let a: Vec<f16> = (0..m * k).map(|_| rnd()).collect();
    let b: Vec<f16> = (0..k * n).map(|_| rnd()).collect();
    let c_init: Vec<f16> = (0..m * n).map(|_| rnd()).collect();
    let (alpha, beta) = (f16::from_f32(0.5), f16::from_f32(2.0));

    let mut c = c_init.clone();
    gemm_f16(
        m,
        n,
        k,
        &mut c,
        n,
        1,
        &a,
        k,
        1,
        &b,
        n,
        1,
        alpha,
        beta,
        Accuracy::Accurate,
    );

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
    // f16 storage + accumulation -> loose tolerance.
    let mut mr = 0.0f64;
    for (g, w) in c.iter().zip(&want) {
        mr = mr.max((f64::from(*g) - w).abs() / (1.0 + w.abs()));
    }
    assert!(mr < 2e-2, "gemm_f16 accumulate {m}x{n}x{k}: max_rel={mr}");
}

#[test]
fn gemm_bf16_accumulate() {
    use sme_gemm::gemm_bf16;
    let (m, n, k) = (65, 257, 257);
    let mut s = 0xacc0_bf16_0000_0001u64;
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        bf16::from_f32((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
    };
    let a: Vec<bf16> = (0..m * k).map(|_| rnd()).collect();
    let b: Vec<bf16> = (0..k * n).map(|_| rnd()).collect();
    let c_init: Vec<bf16> = (0..m * n).map(|_| rnd()).collect();
    let (alpha, beta) = (bf16::from_f32(0.5), bf16::from_f32(2.0));

    let mut c = c_init.clone();
    gemm_bf16(
        m,
        n,
        k,
        &mut c,
        n,
        1,
        &a,
        k,
        1,
        &b,
        n,
        1,
        alpha,
        beta,
        Accuracy::Accurate,
    );

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
    // bf16 storage (8-bit mantissa) + accumulation -> loose, sqrt(k)-scaled tol.
    let tol = (6e-2 * (k as f64).sqrt() / 4.0).max(6e-2);
    let mut mr = 0.0f64;
    for (g, w) in c.iter().zip(&want) {
        mr = mr.max((f64::from(g.to_f32()) - w).abs() / (1.0 + w.abs()));
    }
    assert!(mr < tol, "gemm_bf16 accumulate {m}x{n}x{k}: max_rel={mr}");
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

// Transposed B for the 16-bit dtypes, in BOTH accuracy modes -- the four pack
// paths that read a column-major B. `Fast` takes the non-widening kernels'
// 8x8-transpose pack, `Accurate` the widening kernels' transpose-and-zip pack
// into the pair-interleaved layout; each is a distinct routine.
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
    for mode in [Accuracy::Accurate, Accuracy::Fast] {
        let (mut ct, mut cr) = (vec![f16::ZERO; m * n], vec![f16::ZERO; m * n]);
        let z = f16::ZERO;
        let o = f16::ONE;
        gemm_f16(m, n, k, &mut ct, n, 1, &a, k, 1, &bt, 1, k, z, o, mode);
        gemm_f16(m, n, k, &mut cr, n, 1, &a, k, 1, &br, n, 1, z, o, mode);
        assert!(
            ct.iter().zip(&cr).all(|(x, y)| x.to_bits() == y.to_bits()),
            "gemm_f16 transposed-B {mode:?} differs from the row-major layout"
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
    for mode in [Accuracy::Accurate, Accuracy::Fast] {
        let (mut ct, mut cr) = (vec![bf16::ZERO; m * n], vec![bf16::ZERO; m * n]);
        let z = bf16::ZERO;
        let o = bf16::ONE;
        gemm_bf16(m, n, k, &mut ct, n, 1, &a, k, 1, &bt, 1, k, z, o, mode);
        gemm_bf16(m, n, k, &mut cr, n, 1, &a, k, 1, &br, n, 1, z, o, mode);
        assert!(
            ct.iter().zip(&cr).all(|(x, y)| x.to_bits() == y.to_bits()),
            "gemm_bf16 transposed-B {mode:?} differs from the row-major layout"
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
        // Accurate (f16xf16->f32) is tight; Fast (f16 accum) grows ~sqrt(k).
        check(m, n, k, Accuracy::Accurate, 2e-2);
        if caps().sme_f16f16 {
            let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
            check(m, n, k, Accuracy::Fast, tol);
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
