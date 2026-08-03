//! Fused row reductions and the softmax entry points.

// The kernel-computed per-row reductions must equal a scalar sweep of the
// finished C. Shapes straddle the SME floor, the 32-wide N-tile (partial tiles
// exercise the predicated tail), and multiple M-chunks.
#[test]
fn row_reductions_match_scalar_sweep() {
    use sme_gemm::{Gemm, RowReduce, prepack_f32};
    for &(m, n, k) in &[
        (7usize, 5usize, 3usize),
        (33, 17, 9),
        (64, 96, 128),
        (129, 257, 64),
        (256, 48, 256),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.07 - 0.3).collect();
        let bias: Vec<f32> = (0..n).map(|j| (j % 5) as f32 * 0.1 - 0.2).collect();
        let w = prepack_f32(&b, n, k);

        let mut c = vec![0.0f32; m * n];
        let (mut sum, mut max) = (vec![0.0f32; m], vec![0.0f32; m]);
        Gemm::new(&a, &w, m)
            .add_col(&bias)
            .run_reduce(&mut c, RowReduce::new().sum(&mut sum).max(&mut max));

        for i in 0..m {
            let row = &c[i * n..(i + 1) * n];
            let want_sum: f32 = row.iter().sum();
            let want_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(
                (sum[i] - want_sum).abs() <= 1e-3 * (1.0 + want_sum.abs()),
                "row_sum {m}x{n}x{k} row {i}: {} vs {want_sum}",
                sum[i]
            );
            // A max is a selection, not an arithmetic combination: exact.
            assert!(
                max[i].to_bits() == want_max.to_bits(),
                "row_max {m}x{n}x{k} row {i}: {} vs {want_max}",
                max[i]
            );
        }
    }
}

// The fused softmax entry point uses the kernel's VECTORIZED exp, so it is
// checked against an f64 reference (not against softmax_rows, which would only
// prove the two agree). Tolerance covers the vector exp's ~1e-6 relative error.
#[test]
fn softmax_gemm_matches_reference() {
    use sme_gemm::{matmul_f32, prepack_f32, softmax_gemm_f32};
    for &(m, n, k) in &[
        (7usize, 5usize, 3usize),
        (33, 17, 9),
        (64, 96, 128),
        (129, 257, 64),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.15 - 1.2).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.2 - 1.0).collect();
        let mut raw = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut raw, m, n, k);

        let mut got = vec![0.0f32; m * n];
        softmax_gemm_f32(&a, &prepack_f32(&b, n, k), &mut got, m);

        for i in 0..m {
            let row = &raw[i * n..(i + 1) * n];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f64> = row.iter().map(|&x| f64::from(x - mx).exp()).collect();
            let denom: f64 = exps.iter().sum();
            let mut total = 0.0f64;
            for (j, e) in exps.iter().enumerate() {
                let want = (e / denom) as f32;
                let g = got[i * n + j];
                assert!(
                    (g - want).abs() <= 1e-5 * (1.0 + want.abs()),
                    "softmax_gemm {m}x{n}x{k} ({i},{j}): {g} vs {want}"
                );
                total += f64::from(g);
            }
            assert!((total - 1.0).abs() < 1e-4, "row {i} must sum to 1: {total}");
        }
    }
}

// Row softmax over the GEMM output: must match an f64 reference, and each row
// must sum to 1.
#[test]
fn softmax_rows_matches_reference() {
    use sme_gemm::{matmul_f32, softmax_rows};
    for &(m, n, k) in &[(33usize, 17usize, 9usize), (64, 96, 128), (129, 257, 64)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.15 - 1.2).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.2 - 1.0).collect();
        let mut c = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut c, m, n, k);
        let raw = c.clone();
        softmax_rows(&mut c, m, n);

        for i in 0..m {
            let row = &raw[i * n..(i + 1) * n];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f64> = row.iter().map(|&x| f64::from(x - mx).exp()).collect();
            let denom: f64 = exps.iter().sum();
            let mut total = 0.0f64;
            for (j, e) in exps.iter().enumerate() {
                let want = (e / denom) as f32;
                let got = c[i * n + j];
                assert!(
                    (got - want).abs() <= 1e-5,
                    "softmax {m}x{n}x{k} ({i},{j}): {got} vs {want}"
                );
                total += f64::from(got);
            }
            assert!((total - 1.0).abs() < 1e-4, "row {i} must sum to 1: {total}");
        }
    }
}

// `Gemm::run_softmax` takes the maxima from the GEMM's fused reduction instead
// of sweeping for them, so it must agree with the separate run + softmax_rows
// exactly -- including under an epilogue, which softmax_gemm_f32 cannot carry.
#[test]
fn run_softmax_matches_run_then_softmax_rows() {
    use sme_gemm::{Gemm, prepack_f32, softmax_rows};
    for &(m, n, k) in &[(33usize, 17usize, 9usize), (64, 96, 128), (129, 257, 64)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.15 - 1.2).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 11) as f32 * 0.2 - 1.0).collect();
        let bias: Vec<f32> = (0..n).map(|j| (j % 7) as f32 * 0.1 - 0.3).collect();
        let w = prepack_f32(&b, n, k);

        let mut want = vec![0.0f32; m * n];
        Gemm::new(&a, &w, m).add_col(&bias).relu().run(&mut want);
        softmax_rows(&mut want, m, n);

        let mut got = vec![0.0f32; m * n];
        Gemm::new(&a, &w, m)
            .add_col(&bias)
            .relu()
            .run_softmax(&mut got);

        for (idx, (&g, &wv)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - wv).abs() <= 1e-6,
                "run_softmax {m}x{n}x{k} [{idx}]: {g} vs {wv}"
            );
        }
    }
}

// The softmax entry points must agree when the supplied maxima are the true row
// maxima -- the only case in which skipping the max sweep is defined.
#[test]
fn softmax_rows_with_max_matches_the_sweeping_form() {
    use sme_gemm::{matmul_f32, softmax_rows, softmax_rows_with_max};
    for &(m, n, k) in &[(33usize, 17usize, 9usize), (128, 129, 64)] {
        let a: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.2 - 1.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i % 19) as f32 * 0.1 - 0.9).collect();
        let mut raw = vec![0.0f32; m * n];
        matmul_f32(&a, &b, &mut raw, m, n, k);

        let rmax: Vec<f32> = raw
            .chunks_exact(n)
            .map(|r| r.iter().copied().fold(f32::NEG_INFINITY, f32::max))
            .collect();
        let mut want = raw.clone();
        softmax_rows(&mut want, m, n);
        let mut got = raw;
        softmax_rows_with_max(&mut got, m, n, &rmax);

        assert_eq!(
            got, want,
            "supplied-max softmax must be bit-identical {m}x{n}"
        );
    }
}

// `Dequant::run_packed` is the quantized counterpart of `Gemm::run`: it must be
// exactly the free function it dispatches to, for both integer widths.
#[test]
fn dequant_run_packed_matches_the_free_functions() {
    use sme_gemm::{
        Dequant, caps, matmul_i8_packed_dequant, matmul_i16_packed_dequant, prepack_i8, prepack_i16,
    };
    let (m, n, k) = (48usize, 40usize, 64usize);
    let mut s = 0xc0de_0001u64;
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 48) as i32 % 17 - 8
    };
    let ai: Vec<i32> = (0..m * k).map(|_| rnd()).collect();
    let bi: Vec<i32> = (0..k * n).map(|_| rnd()).collect();
    let bias: Vec<f32> = (0..n).map(|j| 0.05 * (j as f32) - 0.4).collect();

    if caps().sme {
        let a: Vec<i8> = ai.iter().map(|&v| v as i8).collect();
        let b: Vec<i8> = bi.iter().map(|&v| v as i8).collect();
        let w = prepack_i8(&b, n, k);
        let dq = Dequant::new(0.003).add_col(&bias).relu();
        let mut want = vec![0.0f32; m * n];
        matmul_i8_packed_dequant(&a, &w, &mut want, m, &dq);
        let mut got = vec![0.0f32; m * n];
        dq.run_packed(&a, &w, &mut got, m);
        assert_eq!(got, want, "i8 run_packed must equal the free function");
    }

    if caps().sme_i16i64 {
        let a: Vec<i16> = ai.iter().map(|&v| v as i16).collect();
        let b: Vec<i16> = bi.iter().map(|&v| v as i16).collect();
        let w = prepack_i16(&b, n, k);
        let dq = Dequant::new(0.002).add_col(&bias).relu();
        let mut want = vec![0.0f32; m * n];
        matmul_i16_packed_dequant(&a, &w, &mut want, m, &dq);
        let mut got = vec![0.0f32; m * n];
        dq.run_packed(&a, &w, &mut got, m);
        assert_eq!(got, want, "i16 run_packed must equal the free function");
    }
}
