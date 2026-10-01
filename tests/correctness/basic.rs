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

// Two N-blocks and two M-chunks; odd (67) and even (68) tile counts.
#[test]
fn half_packed_multi_nblock() {
    let (m, k) = (70usize, 1024usize);
    for n in [67 * 32 - 5, 68 * 32 - 3] {
        let mut s = 0x16b1_0c16_0000_0001u64 ^ n as u64;
        let a = fill(&mut s, m * k);
        let b = fill(&mut s, k * n);
        let mut bt = vec![0.0f64; n * k];
        for l in 0..k {
            for j in 0..n {
                bt[j * k + l] = f64::from(b[l * n + j]);
            }
        }
        let want: Vec<f64> = (0..m * n)
            .map(|ij| {
                let (i, j) = (ij / n, ij % n);
                a[i * k..(i + 1) * k]
                    .iter()
                    .zip(&bt[j * k..(j + 1) * k])
                    .map(|(&x, &y)| f64::from(x) * y)
                    .sum()
            })
            .collect();
        let tol = 3e-2 * (k as f64).sqrt() / 4.0;
        if caps().sme_f16f16 {
            let mut c = vec![f16::ZERO; m * n];
            matmul_f16_packed(&a, &prepack_f16(&b, n, k), &mut c, m);
            let mr = max_rel(&c, &want);
            assert!(mr < tol, "f16 multi-block {m}x{n}x{k}: max_rel={mr}");
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
            assert!(mr < 2.0 * tol, "bf16 multi-block {m}x{n}x{k}: max_rel={mr}");
        }
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

// Two N-blocks and two M-chunks, ragged in every dim.
#[test]
fn i16_packed_multi_nblock() {
    use sme_gemm::{matmul_i16_packed, prepack_i16};
    if !caps().sme_i16i64 {
        return;
    }
    let (m, n, k) = (40usize, 520usize, 8196usize);
    let mut s = 0x1616_b10c_0000_0001u64;
    let mut rnd = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 33) as i32 % 4001 - 2000) as i16
    };
    let a: Vec<i16> = (0..m * k).map(|_| rnd()).collect();
    let b: Vec<i16> = (0..k * n).map(|_| rnd()).collect();
    let mut bt = vec![0i16; n * k];
    for l in 0..k {
        for j in 0..n {
            bt[j * k + l] = b[l * n + j];
        }
    }
    let w = prepack_i16(&b, n, k);
    let mut got = vec![0i64; m * n];
    matmul_i16_packed(&a, &w, &mut got, m);
    for i in 0..m {
        let ar = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let want: i64 = ar
                .iter()
                .zip(&bt[j * k..(j + 1) * k])
                .map(|(&x, &y)| i64::from(x) * i64::from(y))
                .sum();
            assert_eq!(
                got[i * n + j],
                want,
                "i16 multi-block {m}x{n}x{k} at ({i},{j})"
            );
        }
    }
}

// Multi-N-block i8 and widening drivers. B = h[l]*p[j] keeps the oracle O(mk+mn).
#[test]
fn int8_and_widening_multi_nblock() {
    let mut s = 0x0b10_c8a1_0000_0001u64;
    let mut next = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 33) as i32
    };

    // 34 N-tiles at k=8192: two blocks.
    let (m, n, k) = (70usize, 1061usize, 8192usize);
    let a: Vec<i8> = (0..m * k).map(|_| (next() % 255 - 127) as i8).collect();
    let h: Vec<i32> = (0..k).map(|_| next() % 23 - 11).collect();
    let p: Vec<i32> = (0..n).map(|_| next() % 23 - 11).collect();
    let b: Vec<i8> = (0..k * n).map(|x| (h[x / n] * p[x % n]) as i8).collect();
    let mut c = vec![0i32; m * n];
    sme_gemm::matmul_i8_packed(&a, &sme_gemm::prepack_i8(&b, n, k), &mut c, m);
    for i in 0..m {
        let r: i32 = (0..k).map(|l| i32::from(a[i * k + l]) * h[l]).sum();
        for j in 0..n {
            assert_eq!(
                c[i * n + j],
                r * p[j],
                "i8 multi-block {m}x{n}x{k} at ({i},{j})"
            );
        }
    }

    // 66 N-tiles at k=2048: two blocks. B is exact in f16 and bf16.
    let (m, n, k) = (70usize, 2087usize, 2048usize);
    let a: Vec<f32> = (0..m * k)
        .map(|_| (next() % 1024) as f32 / 1024.0 - 0.5)
        .collect();
    let h: Vec<f32> = (0..k).map(|_| (next() % 17 - 8) as f32 / 8.0).collect();
    let p: Vec<f32> = (0..n).map(|_| (next() % 16 + 1) as f32 / 16.0).collect();
    let b: Vec<f32> = (0..k * n).map(|x| h[x / n] * p[x % n]).collect();
    // bf16 rounds A, so each format gets its own oracle.
    let rows = |av: &dyn Fn(usize) -> f64| -> Vec<f64> {
        (0..m)
            .map(|i| (0..k).map(|l| av(i * k + l) * f64::from(h[l])).sum())
            .collect()
    };
    let rel = |r: &[f64], got: f32, i: usize, j: usize| {
        let w = r[i] * f64::from(p[j]);
        (f64::from(got) - w).abs() / (1.0 + w.abs())
    };
    let ah: Vec<f16> = a.iter().map(|&x| f16::from_f32(x)).collect();
    let bh: Vec<f16> = b.iter().map(|&x| f16::from_f32(x)).collect();
    let mut ch = vec![f16::ZERO; m * n];
    sme_gemm::matmul_f16(&ah, &bh, &mut ch, m, n, k, Accum::F32);
    let rh = rows(&|x| f64::from(ah[x].to_f32()));
    let ab: Vec<bf16> = a.iter().map(|&x| bf16::from_f32(x)).collect();
    let bb: Vec<bf16> = b.iter().map(|&x| bf16::from_f32(x)).collect();
    let mut cb = vec![bf16::ZERO; m * n];
    matmul_bf16(&ab, &bb, &mut cb, m, n, k, Accum::F32);
    let rb = rows(&|x| f64::from(ab[x].to_f32()));
    for i in 0..m {
        for j in 0..n {
            let eh = rel(&rh, ch[i * n + j].to_f32(), i, j);
            assert!(eh < 2e-3, "f16 widening multi-block at ({i},{j}): rel={eh}");
            let eb = rel(&rb, cb[i * n + j].to_f32(), i, j);
            assert!(
                eb < 1e-2,
                "bf16 widening multi-block at ({i},{j}): rel={eb}"
            );
        }
    }
}

// Multi-N-block f32 and f64 drivers, rank-1 B as above.
#[test]
fn f32_f64_multi_nblock() {
    let mut s = 0x0f32_f64b_0000_0001u64;
    let mut next = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 33) as i32
    };
    let k = 2048usize;
    let h: Vec<f64> = (0..k).map(|_| f64::from(next() % 17 - 8) / 8.0).collect();
    for (m, n, f64_path) in [(70usize, 1575usize, false), (40, 789, true)] {
        let a: Vec<f64> = (0..m * k)
            .map(|_| f64::from(next() % 1024) / 1024.0 - 0.5)
            .collect();
        let p: Vec<f64> = (0..n).map(|_| f64::from(next() % 16 + 1) / 16.0).collect();
        let r: Vec<f64> = (0..m)
            .map(|i| (0..k).map(|l| a[i * k + l] * h[l]).sum())
            .collect();
        let got: Vec<f64> = if f64_path {
            if !caps().sme_f64f64 {
                continue;
            }
            let b: Vec<f64> = (0..k * n).map(|x| h[x / n] * p[x % n]).collect();
            let mut c = vec![0.0f64; m * n];
            sme_gemm::Gemm::new(&a, &sme_gemm::prepack_f64(&b, n, k), m).run(&mut c);
            c
        } else {
            let af: Vec<f32> = a.iter().map(|&x| x as f32).collect();
            let b: Vec<f32> = (0..k * n).map(|x| (h[x / n] * p[x % n]) as f32).collect();
            let mut c = vec![0.0f32; m * n];
            sme_gemm::matmul_f32_packed(&af, &sme_gemm::prepack_f32(&b, n, k), &mut c, m);
            c.iter().map(|&x| f64::from(x)).collect()
        };
        for i in 0..m {
            for j in 0..n {
                let w = r[i] * p[j];
                let e = (got[i * n + j] - w).abs() / (1.0 + w.abs());
                assert!(
                    e < 1e-4,
                    "f{} multi-block {m}x{n}x{k} at ({i},{j}): {e}",
                    if f64_path { 64 } else { 32 }
                );
            }
        }
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
