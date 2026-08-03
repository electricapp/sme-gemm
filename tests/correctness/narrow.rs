//! Narrow-N / narrow-band shapes and the flat-M N-parallel branch.

use half::{bf16, f16};
use sme_gemm::{caps, matmul_bf16_packed, matmul_f16_packed, matmul_f32};

// The flat-M N-parallel dispatch branch (m small, n/k large -> the work is split
// over N-tile chunks across both clusters instead of running serially on one) is
// NOT reached by the default SIZES, which top out at nn_chunks==1. Structured
// inputs give a closed-form reference C[i][j] = k*(i%5-2)*(j%7-3) that depends on
// BOTH i and j, so a column-misindexing bug in the N-range split is caught.
// m=64 -> m_tiles=2 (n_chunks=1); n=1024 -> n_tiles=32 (nn_chunks=8);
// m*n*k=67M (>= 2^25 so f32 skips run_small; >= 2^21 so the int path is "big").
#[test]
// Integer-valued inputs make the f32 result exact by construction; the strict
// comparison is the point of the test.
#[allow(clippy::float_cmp)]
fn flat_m_n_parallel_branch() {
    if !caps().sme {
        return;
    }
    let (m, n, k) = (64usize, 1024, 1024);
    let av = |i: usize| (i % 5) as i32 - 2;
    let bv = |j: usize| (j % 7) as i32 - 3;
    let cref = |i: usize, j: usize| (k as i32) * av(i) * bv(j);

    // i8 raw: exact integer GEMM.
    let ai: Vec<i8> = (0..m * k).map(|x| av(x / k) as i8).collect();
    let bi: Vec<i8> = (0..k * n).map(|x| bv(x % n) as i8).collect();
    let packed = sme_gemm::prepack_i8(&bi, n, k);
    let mut ci = vec![0i32; m * n];
    sme_gemm::matmul_i8_packed(&ai, &packed, &mut ci, m);
    for i in 0..m {
        for j in 0..n {
            assert_eq!(ci[i * n + j], cref(i, j), "i8 flat-M N-parallel ({i},{j})");
        }
    }

    // i8 dequant: scale + per-column bias + relu, row-major f32 out (the dq store
    // writes disjoint column ranges per N-chunk).
    {
        use sme_gemm::{Dequant, matmul_i8_packed_dequant};
        let scale = 0.0009f32;
        let bias: Vec<f32> = (0..n).map(|j| 0.05 * (j % 11) as f32 - 0.2).collect();
        let dq = Dequant::new(scale).add_col(&bias).relu();
        let mut cd = vec![0.0f32; m * n];
        matmul_i8_packed_dequant(&ai, &packed, &mut cd, m, &dq);
        for i in 0..m {
            for j in 0..n {
                let want = (cref(i, j) as f32 * scale + bias[j]).max(0.0);
                let got = cd[i * n + j];
                assert!(
                    (got - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "i8 dequant flat-M ({i},{j}): got {got} want {want}"
                );
            }
        }
    }

    // f32: exact (|C| <= 6144 < 2^24, integer partial sums).
    let af: Vec<f32> = (0..m * k).map(|x| av(x / k) as f32).collect();
    let bf: Vec<f32> = (0..k * n).map(|x| bv(x % n) as f32).collect();
    let mut cf = vec![0.0f32; m * n];
    matmul_f32(&af, &bf, &mut cf, m, n, k);
    for i in 0..m {
        for j in 0..n {
            assert_eq!(
                cf[i * n + j],
                cref(i, j) as f32,
                "f32 flat-M N-parallel ({i},{j})"
            );
        }
    }

    // f64: same shape hits the f64 N-parallel branch (m=64 -> m_tiles=4,
    // n_chunks=1; n_tiles=64, nn_chunks=16; big). Exact: |C| <= 6144 << 2^53.
    {
        use sme_gemm::matmul_f64;
        let ad: Vec<f64> = (0..m * k).map(|x| f64::from(av(x / k))).collect();
        let bd: Vec<f64> = (0..k * n).map(|x| f64::from(bv(x % n))).collect();
        let mut cd = vec![0.0f64; m * n];
        matmul_f64(&ad, &bd, &mut cd, m, n, k);
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    cd[i * n + j],
                    f64::from(cref(i, j)),
                    "f64 flat-M N-parallel ({i},{j})"
                );
            }
        }
    }
}

// Narrow-N decode path: n_tiles==1 (n<=32) row-major routes to run_narrow_rowmajor
// (pairs two M-tiles against the single B-tile). The pure store is covered by the
// default SIZES; this also covers the node-major EPILOGUE store in the narrow path,
// for f16 AND bf16. m=96 -> 3 M-tiles: one pair + one unpaired (exercises the
// have1 true AND false arms). k=128 keeps |C| <= 128, exact in f16 and bf16, so the
// structured reference C[i][j] = k*(i%3-1)*(j%3-1) is exact (and j-dependent, so a
// column/tile-misindexing bug is caught).
#[test]
fn narrow_n_decode() {
    use sme_gemm::{Gemm, prepack_bf16, prepack_f16};
    if !caps().sme_f16f16 {
        return;
    }
    let (m, n, k) = (96usize, 32, 128);
    let av = |i: usize| (i % 3) as i32 - 1;
    let bv = |j: usize| (j % 3) as i32 - 1;
    let bias = |j: usize| (j % 5) as i32 - 2;
    let cpure = |i: usize, j: usize| (k as i32) * av(i) * bv(j);

    // f16
    {
        let a: Vec<f16> = (0..m * k)
            .map(|x| f16::from_f32(av(x / k) as f32))
            .collect();
        let b: Vec<f16> = (0..k * n)
            .map(|x| f16::from_f32(bv(x % n) as f32))
            .collect();
        let w = prepack_f16(&b, n, k);
        let mut c = vec![f16::ZERO; m * n];
        matmul_f16_packed(&a, &w, &mut c, m);
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[i * n + j].to_f32() - cpure(i, j) as f32).abs() <= 0.5,
                    "f16 narrow pure ({i},{j})"
                );
            }
        }
        let biasv: Vec<f16> = (0..n).map(|j| f16::from_f32(bias(j) as f32)).collect();
        let mut ce = vec![f16::ZERO; m * n];
        Gemm::new(&a, &w, m).add_col(&biasv).relu().run(&mut ce);
        for i in 0..m {
            for j in 0..n {
                let want = (cpure(i, j) + bias(j)).max(0) as f32;
                assert!(
                    (ce[i * n + j].to_f32() - want).abs() <= 0.5,
                    "f16 narrow ep ({i},{j}): got {} want {want}",
                    ce[i * n + j].to_f32()
                );
            }
        }
    }
    // bf16
    {
        let a: Vec<bf16> = (0..m * k)
            .map(|x| bf16::from_f32(av(x / k) as f32))
            .collect();
        let b: Vec<bf16> = (0..k * n)
            .map(|x| bf16::from_f32(bv(x % n) as f32))
            .collect();
        let w = prepack_bf16(&b, n, k);
        let mut c = vec![bf16::ZERO; m * n];
        matmul_bf16_packed(&a, &w, &mut c, m);
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[i * n + j].to_f32() - cpure(i, j) as f32).abs() <= 0.5,
                    "bf16 narrow pure ({i},{j})"
                );
            }
        }
        let biasv: Vec<bf16> = (0..n).map(|j| bf16::from_f32(bias(j) as f32)).collect();
        let mut ce = vec![bf16::ZERO; m * n];
        Gemm::new(&a, &w, m).add_col(&biasv).relu().run(&mut ce);
        for i in 0..m {
            for j in 0..n {
                let want = (cpure(i, j) + bias(j)).max(0) as f32;
                assert!(
                    (ce[i * n + j].to_f32() - want).abs() <= 0.5,
                    "bf16 narrow ep ({i},{j}): got {} want {want}",
                    ce[i * n + j].to_f32()
                );
            }
        }
    }
}

// i8 narrow-N (n <= 16: a single ZA32 N-band) routes to run_narrow_rowmajor
// (pairs two M-tiles against the one N-band). SIZES covers n=1,5,8 (partial band);
// this pins n=16 (full band, nc=16 predicate) with multiple M-pairs (m=96 -> 3
// M-tiles: one pair + one unpaired), for raw i32 AND dequant. Exact integer
// reference (|C| <= 128), j-dependent so a column/tile-misindex is caught.
#[test]
fn i8_narrow_decode() {
    use sme_gemm::{Dequant, matmul_i8_packed, matmul_i8_packed_dequant, prepack_i8};
    if !caps().sme {
        return;
    }
    let (m, n, k) = (96usize, 16, 128);
    let av = |i: usize| (i % 3) as i32 - 1;
    let bv = |j: usize| (j % 3) as i32 - 1;
    let cpure = |i: usize, j: usize| (k as i32) * av(i) * bv(j);
    let a: Vec<i8> = (0..m * k).map(|x| av(x / k) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|x| bv(x % n) as i8).collect();
    let w = prepack_i8(&b, n, k);

    // raw i32
    let mut c = vec![0i32; m * n];
    matmul_i8_packed(&a, &w, &mut c, m);
    for i in 0..m {
        for j in 0..n {
            assert_eq!(c[i * n + j], cpure(i, j), "i8 narrow raw ({i},{j})");
        }
    }

    // dequant: scale + per-column bias + relu (row-major f32)
    let scale = 0.01f32;
    let bias: Vec<f32> = (0..n).map(|j| 0.1 * (j % 5) as f32 - 0.2).collect();
    let dq = Dequant::new(scale).add_col(&bias).relu();
    let mut cf = vec![0.0f32; m * n];
    matmul_i8_packed_dequant(&a, &w, &mut cf, m, &dq);
    for i in 0..m {
        for j in 0..n {
            let want = (cpure(i, j) as f32 * scale + bias[j]).max(0.0);
            assert!(
                (cf[i * n + j] - want).abs() <= 1e-3 * want.abs().max(1.0),
                "i8 narrow dq ({i},{j}): got {} want {want}",
                cf[i * n + j]
            );
        }
    }
}

// A bare `add_col(bias)` (no other op) triggers the rank-1 FMOPA bias-fold path,
// where `bias_init_za` seeds ONLY the live ZA quadrants (narrow-N nc<=16 leaves
// za1/za3 dead; narrow-M mr<=16 leaves za2/za3 dead). A dead-tile mis-gate would
// drop the bias on a tile the store actually reads -> a wrong row/column band.
// Closed form C[i][j] = k*av(i)*bv(j) + bias[j], exact in f32 (small-int
// accumulators stay < 2^24). Shapes hit narrow-N, narrow-M, and a full control,
// all clearing 2^18 flops so the SME fold path runs.
#[test]
#[allow(clippy::float_cmp)] // integer-valued accumulators: the f32 result is exact
fn fold_bias_narrow_bands_f32() {
    use sme_gemm::{Gemm, prepack_f32};
    if !caps().sme {
        return;
    }
    let av = |i: usize| (i % 5) as f32 - 2.0;
    let bv = |j: usize| (j % 7) as f32 - 3.0;
    let bias_at = |j: usize| (j % 4) as f32 - 1.0;
    let shapes: &[(usize, usize, usize)] = &[(64, 8, 512), (8, 256, 512), (64, 256, 64)];
    for &(m, n, k) in shapes {
        let a: Vec<f32> = (0..m * k).map(|x| av(x / k)).collect();
        let b: Vec<f32> = (0..k * n).map(|x| bv(x % n)).collect();
        let bias: Vec<f32> = (0..n).map(bias_at).collect();
        let packed = prepack_f32(&b, n, k);
        let mut c = vec![0.0f32; m * n];
        Gemm::new(&a, &packed, m).add_col(&bias).run(&mut c);
        for i in 0..m {
            for j in 0..n {
                let want = k as f32 * av(i) * bv(j) + bias_at(j);
                assert_eq!(c[i * n + j], want, "f32 fold_bias ({i},{j}) {m}x{n}x{k}");
            }
        }
    }
}

// f64 analog of `fold_bias_narrow_bands_f32` (8-wide ZA quadrants: narrow-N nc<=8,
// narrow-M mr<=8). Same closed form, exact in f64.
#[test]
#[allow(clippy::float_cmp)]
fn fold_bias_narrow_bands_f64() {
    use sme_gemm::{Gemm, prepack_f64};
    if !caps().sme_f64f64 {
        return;
    }
    let av = |i: usize| (i % 5) as f64 - 2.0;
    let bv = |j: usize| (j % 7) as f64 - 3.0;
    let bias_at = |j: usize| (j % 4) as f64 - 1.0;
    let shapes: &[(usize, usize, usize)] = &[(64, 4, 2048), (4, 256, 1024), (64, 256, 64)];
    for &(m, n, k) in shapes {
        let a: Vec<f64> = (0..m * k).map(|x| av(x / k)).collect();
        let b: Vec<f64> = (0..k * n).map(|x| bv(x % n)).collect();
        let bias: Vec<f64> = (0..n).map(bias_at).collect();
        let packed = prepack_f64(&b, n, k);
        let mut c = vec![0.0f64; m * n];
        Gemm::new(&a, &packed, m).add_col(&bias).run(&mut c);
        for i in 0..m {
            for j in 0..n {
                let want = k as f64 * av(i) * bv(j) + bias_at(j);
                assert_eq!(c[i * n + j], want, "f64 fold_bias ({i},{j}) {m}x{n}x{k}");
            }
        }
    }
}

// f64 narrow-N / narrow-M band-skip paths: the f64 SME kernel takes a dedicated
// branch when a tile's column count `nc <= 8` (narrow-N: the hi-N ZA bands are
// pad) or row count `mr <= 8` (narrow-M: the hi-M bands are pad). The default
// SIZES never both clear the SME flop threshold AND land in a <=8 band, so these
// branches were untested. Both shapes give m*n*k = 524288 >= 2^18 (SME path
// runs) with a single 8-wide tile, hitting nc==8 / mr==8. Integer-valued inputs
// keep |C| <= 256*4 = 1024 << 2^53, so the f64 result is exact and the structured
// oracle C[i][j] = sum_d A[i][d]*B[d][j] (i- AND j-dependent, catching a
// row/column-misindex in the band-skip) holds bit-for-bit.
#[test]
#[allow(clippy::float_cmp)]
fn f64_narrow_band() {
    use sme_gemm::matmul_f64;
    if !caps().sme_f64f64 {
        return;
    }
    let av = |i: usize, d: usize| ((i + 2 * d) % 5) as f64 - 2.0;
    let bv = |d: usize, j: usize| ((3 * d + j) % 7) as f64 - 3.0;
    let oracle_at = |a: &[f64], b: &[f64], n: usize, k: usize, i: usize, j: usize| {
        let mut acc = 0.0f64;
        for d in 0..k {
            acc += a[i * k + d] * b[d * n + j];
        }
        acc
    };
    // (m, n, k): narrow-N (n=8) and narrow-M (m=8).
    for &(m, n, k) in &[(256usize, 8usize, 256usize), (8, 256, 256)] {
        let a: Vec<f64> = (0..m * k).map(|x| av(x / k, x % k)).collect();
        let b: Vec<f64> = (0..k * n).map(|x| bv(x / n, x % n)).collect();
        let mut c = vec![0.0f64; m * n];
        matmul_f64(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    c[i * n + j],
                    oracle_at(&a, &b, n, k, i, j),
                    "f64 narrow band {m}x{n}x{k} at ({i},{j})"
                );
            }
        }
    }
}

// i16 narrow-N / narrow-M band-skip paths (mirror of `f64_narrow_band`). i16xi16
// -> i64 is exact, so the same structured oracle holds bit-for-bit. Bounded inputs
// keep the i64 accumulator small for the dequant variant's i64->f32 store. Covers
// BOTH the raw i64 path (`matmul_i16`) and the dequant store (`matmul_i16_dequant`).
#[test]
fn i16_narrow_band() {
    use sme_gemm::{Dequant, matmul_i16, matmul_i16_dequant};
    if !caps().sme_i16i64 {
        return;
    }
    let av = |i: usize, d: usize| (((i + 2 * d) % 5) as i32 - 2) as i16;
    let bv = |d: usize, j: usize| (((3 * d + j) % 7) as i32 - 3) as i16;
    // (m, n, k): narrow-N (n=8) and narrow-M (m=8).
    for &(m, n, k) in &[(256usize, 8usize, 256usize), (8, 256, 256)] {
        let a: Vec<i16> = (0..m * k).map(|x| av(x / k, x % k)).collect();
        let b: Vec<i16> = (0..k * n).map(|x| bv(x / n, x % n)).collect();
        // exact i64 oracle, i- and j-dependent.
        let mut acc = vec![0i64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut t = 0i64;
                for d in 0..k {
                    t += i64::from(a[i * k + d]) * i64::from(b[d * n + j]);
                }
                acc[i * n + j] = t;
            }
        }

        // raw i64: exact.
        let mut c = vec![0i64; m * n];
        matmul_i16(&a, &b, &mut c, m, n, k);
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    c[i * n + j],
                    acc[i * n + j],
                    "i16 narrow raw {m}x{n}x{k} ({i},{j})"
                );
            }
        }

        // dequant store: scale + per-column bias + relu (row-major f32). |acc| is
        // small here so the i64->f32 store is exact for the tolerance.
        let scale = 0.01f32;
        let bias: Vec<f32> = (0..n).map(|j| 0.1 * (j % 5) as f32 - 0.2).collect();
        let dq = Dequant::new(scale).add_col(&bias).relu();
        let mut cf = vec![0.0f32; m * n];
        matmul_i16_dequant(&a, &b, &mut cf, m, n, k, &dq);
        for i in 0..m {
            for j in 0..n {
                let want = (acc[i * n + j] as f32 * scale + bias[j]).max(0.0);
                assert!(
                    (cf[i * n + j] - want).abs() <= 1e-3 * want.abs().max(1.0),
                    "i16 narrow dq {m}x{n}x{k} ({i},{j}): got {} want {want}",
                    cf[i * n + j]
                );
            }
        }
    }
}

// Closed-form f64/i16 narrow-band oracles use integer inputs; assert the shape
// math (these MUST clear the SME flop threshold for the band-skip branches to run).
#[test]
fn narrow_band_shapes_clear_threshold() {
    for &(m, n, k) in &[(256usize, 8usize, 256usize), (8, 256, 256)] {
        assert!(m * n * k >= 1 << 18, "{m}x{n}x{k} below 2^18");
        assert!(m <= 8 || n <= 8, "{m}x{n}x{k} hits neither narrow band");
    }
}
