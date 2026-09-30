//! Flash attention vs. a materialized softmax(QK^T)V reference.

use crate::{fill_f32, oracle_f32};
use sme_gemm::{FlashParams, flash_attention_f32, flash_attention_f32_with};

/// f64 reference: materialize the scores, row-softmax, multiply by V.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    m: usize,
    n: usize,
    d: usize,
    dv: usize,
    scale: f32,
) -> Vec<f64> {
    let mut s = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for l in 0..d {
                acc += f64::from(q[i * d + l]) * f64::from(k[j * d + l]);
            }
            s[i * n + j] = acc * f64::from(scale);
        }
    }
    for row in s.chunks_exact_mut(n) {
        let mx = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for x in row.iter_mut() {
            *x = (*x - mx).exp();
            sum += *x;
        }
        for x in row.iter_mut() {
            *x /= sum;
        }
    }
    let mut o = vec![0.0f64; m * dv];
    for i in 0..m {
        for j in 0..dv {
            let mut acc = 0.0f64;
            for l in 0..n {
                acc += s[i * n + l] * f64::from(v[l * dv + j]);
            }
            o[i * dv + j] = acc;
        }
    }
    o
}

fn max_abs(got: &[f32], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (f64::from(*g) - w).abs())
        .fold(0.0, f64::max)
}

#[test]
fn flash_matches_materialized_reference() {
    // Tiles chosen so every case exercises multiple key blocks and, where m
    // allows, multiple query blocks -- including ragged trailing tiles.
    let cases = [
        (1usize, 1usize, 1usize, 1usize),
        (7, 5, 3, 4),
        (32, 64, 16, 16),
        (33, 129, 17, 9),
        (64, 512, 64, 64),
        (129, 300, 40, 72),
    ];
    for (m, n, d, dv) in cases {
        let mut s = 0x51ed_270b_1234_5678 ^ ((m * 131 + n * 17 + d) as u64);
        let q = fill_f32(&mut s, m * d);
        let k = fill_f32(&mut s, n * d);
        let v = fill_f32(&mut s, n * dv);
        let scale = 1.0 / (d as f32).sqrt();
        let want = reference(&q, &k, &v, m, n, d, dv, scale);

        for (bm, bn) in [(16, 16), (7, 33), (256, 512)] {
            let mut o = vec![0.0f32; m * dv];
            let p = FlashParams {
                block_m: bm,
                block_n: bn,
            };
            flash_attention_f32_with(&q, &k, &v, &mut o, m, n, d, dv, scale, p);
            let e = max_abs(&o, &want);
            assert!(e < 1e-5, "{m}x{n}x{d}x{dv} tiles {bm}x{bn}: max_abs={e}");
        }

        let mut o = vec![0.0f32; m * dv];
        flash_attention_f32(&q, &k, &v, &mut o, m, n, d, dv, scale);
        let e = max_abs(&o, &want);
        assert!(e < 1e-5, "{m}x{n}x{d}x{dv} default tiles: max_abs={e}");
    }
}

#[test]
fn flash_half_matches_materialized_reference() {
    use half::{bf16, f16};
    use sme_gemm::{Accum, flash_attention_bf16_with, flash_attention_f16_with};

    let cases = [
        (7usize, 5usize, 3usize, 4usize),
        (33, 129, 17, 9),
        (64, 300, 64, 40),
    ];
    for (m, n, d, dv) in cases {
        let mut s = 0x1f83_d9ab_fb41_bd6b ^ ((m * 131 + n * 17 + d) as u64);
        let q = fill_f32(&mut s, m * d);
        let k = fill_f32(&mut s, n * d);
        let v = fill_f32(&mut s, n * dv);
        let scale = 1.0 / (d as f32).sqrt();
        let want = reference(&q, &k, &v, m, n, d, dv, scale);

        for (bm, bn) in [(16, 16), (7, 33), (256, 512)] {
            let p = FlashParams {
                block_m: bm,
                block_n: bn,
            };

            let (qh, kh, vh) = (to_half::<f16>(&q), to_half::<f16>(&k), to_half::<f16>(&v));
            let mut o = vec![f16::ZERO; m * dv];
            flash_attention_f16_with(&qh, &kh, &vh, &mut o, m, n, d, dv, scale, Accum::F32, p);
            let e = half_max_abs(o.iter().map(|x| f64::from(x.to_f32())), &want);
            assert!(
                e < 5e-3,
                "f16 {m}x{n}x{d}x{dv} tiles {bm}x{bn}: max_abs={e}"
            );

            let (qb, kb, vb) = (
                to_half::<bf16>(&q),
                to_half::<bf16>(&k),
                to_half::<bf16>(&v),
            );
            let mut o = vec![bf16::ZERO; m * dv];
            flash_attention_bf16_with(&qb, &kb, &vb, &mut o, m, n, d, dv, scale, Accum::F32, p);
            let e = half_max_abs(o.iter().map(|x| f64::from(x.to_f32())), &want);
            assert!(
                e < 4e-2,
                "bf16 {m}x{n}x{d}x{dv} tiles {bm}x{bn}: max_abs={e}"
            );
        }
    }
}

/// Round an f32 buffer into a half storage type.
fn to_half<T: HalfCast>(v: &[f32]) -> Vec<T> {
    v.iter().map(|&x| T::cast(x)).collect()
}

trait HalfCast: Copy {
    fn cast(x: f32) -> Self;
}
impl HalfCast for half::f16 {
    fn cast(x: f32) -> Self {
        Self::from_f32(x)
    }
}
impl HalfCast for half::bf16 {
    fn cast(x: f32) -> Self {
        Self::from_f32(x)
    }
}

fn half_max_abs(got: impl Iterator<Item = f64>, want: &[f64]) -> f64 {
    got.zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f64::max)
}

#[test]
fn flash_degenerate_shapes() {
    let q = vec![0.5f32; 4 * 8];
    let k = vec![0.25f32; 6 * 8];
    let v = vec![1.5f32; 6 * 3];

    // n == 0: no keys to attend to, output is zeroed.
    let mut o = vec![9.0f32; 4 * 3];
    flash_attention_f32(&q, &[], &[], &mut o, 4, 0, 8, 3, 0.5);
    assert!(o.iter().all(|&x| x == 0.0), "n == 0 zeroes the output");

    // m == 0 / dv == 0: nothing to write, must not panic.
    flash_attention_f32(&[], &k, &v, &mut [], 0, 6, 8, 3, 0.5);
    flash_attention_f32(&q, &k, &[], &mut [], 4, 6, 8, 0, 0.5);

    // Uniform scores: every row is a flat average of V.
    let mut o = vec![0.0f32; 4 * 3];
    flash_attention_f32(&q, &k, &v, &mut o, 4, 6, 8, 3, 0.5);
    for &x in &o {
        assert!((x - 1.5).abs() < 1e-5, "uniform attention averages V: {x}");
    }
}

#[test]
fn softmax_rows_matches_the_gemm_oracle() {
    // The vectorized softmax pass vs the f64 oracle, on a real GEMM output.
    let (m, n, k) = (64, 96, 48);
    let mut s = 0x9e37_79b9_7f4a_7c15;
    let a = fill_f32(&mut s, m * k);
    let b = fill_f32(&mut s, k * n);
    let mut c = vec![0.0f32; m * n];
    sme_gemm::matmul_f32(&a, &b, &mut c, m, n, k);
    sme_gemm::softmax_rows(&mut c, m, n);

    let mut want = oracle_f32(&a, &b, m, n, k);
    for row in want.chunks_exact_mut(n) {
        let mx = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0f64;
        for x in row.iter_mut() {
            *x = (*x - mx).exp();
            sum += *x;
        }
        for x in row.iter_mut() {
            *x /= sum;
        }
    }
    let e = max_abs(&c, &want);
    assert!(e < 1e-6, "softmax_rows vs oracle: max_abs={e}");
}

// The half-precision entry points that pick their own tiles must equal the
// explicit-tile form fed `FlashParams::auto`. Only the `_with` variants were
// covered before, so the default-parameter path -- the one most callers use --
// was untested. Shapes straddle the auto-tiler's single-tile/blocked branch.
#[test]
fn flash_half_auto_params_match_the_explicit_form() {
    use half::{bf16, f16};
    use sme_gemm::{
        Accum, FlashParams, flash_attention_bf16, flash_attention_bf16_with, flash_attention_f16,
        flash_attention_f16_with,
    };

    for (m, n, d, dv) in [(33usize, 129usize, 17usize, 9usize), (64, 300, 64, 40)] {
        let mut s = 0x51ed_270b_1234_9abc ^ ((m * 131 + n * 17 + d) as u64);
        let q = fill_f32(&mut s, m * d);
        let k = fill_f32(&mut s, n * d);
        let v = fill_f32(&mut s, n * dv);
        let scale = 1.0 / (d as f32).sqrt();
        let p = FlashParams::auto(m, n);

        for accum in [Accum::F32, Accum::F16] {
            let (qh, kh, vh) = (to_half::<f16>(&q), to_half::<f16>(&k), to_half::<f16>(&v));
            let mut got = vec![f16::ZERO; m * dv];
            let mut want = vec![f16::ZERO; m * dv];
            flash_attention_f16(&qh, &kh, &vh, &mut got, m, n, d, dv, scale, accum);
            flash_attention_f16_with(&qh, &kh, &vh, &mut want, m, n, d, dv, scale, accum, p);
            assert_eq!(got, want, "f16 auto vs explicit {m}x{n}x{d}x{dv} {accum:?}");
        }
        for accum in [Accum::F32, Accum::Bf16] {
            let (qb, kb, vb) = (
                to_half::<bf16>(&q),
                to_half::<bf16>(&k),
                to_half::<bf16>(&v),
            );
            let mut gotb = vec![bf16::ZERO; m * dv];
            let mut wantb = vec![bf16::ZERO; m * dv];
            flash_attention_bf16(&qb, &kb, &vb, &mut gotb, m, n, d, dv, scale, accum);
            flash_attention_bf16_with(&qb, &kb, &vb, &mut wantb, m, n, d, dv, scale, accum, p);
            assert_eq!(
                gotb, wantb,
                "bf16 auto vs explicit {m}x{n}x{d}x{dv} {accum:?}"
            );
        }
    }
}
