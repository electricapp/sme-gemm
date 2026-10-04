//! The 4-bit block-quantized paths: eager dequant, the 4-bit-resident
//! kernel (f16 and bf16), and their fused epilogues.

use crate::{SIZES, fill, gelu_f64, max_rel, oracle};
use half::f16;
use sme_gemm::caps;

#[test]
fn q4_path() {
    use sme_gemm::{Q4_BLOCK, dequant_q4, matmul_f16_packed};
    if !caps().sme_f16f16 {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0x9401_9401_dead_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let nbk = k.div_ceil(Q4_BLOCK);
        let mut byte = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as u8
        };
        let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
        let scales: Vec<f16> = (0..n * nbk)
            .map(|i| f16::from_f32(0.01 + 0.001 * (i % 5) as f32))
            .collect();

        // f64 dequant oracle weights.
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

        let w = dequant_q4(&quants, &scales, n, k);
        let mut c = vec![f16::ZERO; m * n];
        matmul_f16_packed(&a, &w, &mut c, m);

        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l]) * bf[l * n + j];
                }
                want[i * n + j] = acc;
            }
        }
        let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
        assert!(
            max_rel(&c, &want) < tol,
            "q4 {m}x{n}x{k}: max_rel={}",
            max_rel(&c, &want)
        );
    }
}

#[test]
fn q4_resident() {
    use sme_gemm::{Q4_BLOCK, Q4Weights, matmul_q4};
    if !caps().sme_f16f16 {
        return;
    }
    for &(m, n, k) in SIZES {
        let mut s = 0x4444_9999_dead_0001 ^ ((m * 131 + n * 17 + k) as u64);
        let a = fill(&mut s, m * k);
        let nbk = k.div_ceil(Q4_BLOCK);
        let mut byte = || {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (s >> 40) as u8
        };
        let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
        let scales: Vec<f16> = (0..n * nbk)
            .map(|i| f16::from_f32(0.01 + 0.001 * (i % 5) as f32))
            .collect();

        // f64 dequant oracle (same row-major Q4 input).
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

        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l]) * bf[l * n + j];
                }
                want[i * n + j] = acc;
            }
        }
        let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);
        assert!(
            max_rel(&c, &want) < tol,
            "q4-resident {m}x{n}x{k}: {}",
            max_rel(&c, &want)
        );
    }
}

// Q4 on-the-fly dequant kernel with a non-default block size and the affine
// (Q4_1) code form: the 4-bit-resident path must agree with eagerly
// dequantizing the same weights and running the plain f16 GEMM. The Q4 paths
// carry no flop gate, so the C kernel runs at every shape here.
#[test]
fn q4_blocks_and_affine_match_eager_dequant() {
    use sme_gemm::{Q4Params, Q4Weights, dequant_q4_with, matmul_f16_packed, matmul_q4};
    if !caps().sme_f16f16 {
        return;
    }
    for &(m, n, k) in &[(64usize, 96usize, 128usize), (65, 33, 256), (800, 520, 512)] {
        for &block in &[16usize, 32, 64] {
            for affine in [false, true] {
                let p = if affine {
                    Q4Params::new(block).affine()
                } else {
                    Q4Params::new(block)
                };
                let nbk = k.div_ceil(block);
                let mut s = 0x4a4a_0001u64 ^ ((m * 131 + n * 17 + k + block) as u64);
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

                let mut want = vec![f16::ZERO; m * n];
                matmul_f16_packed(
                    &a,
                    &dequant_q4_with(&quants, &scales, mv, n, k, p),
                    &mut want,
                    m,
                );
                let mut got = vec![f16::ZERO; m * n];
                matmul_q4(
                    &a,
                    &Q4Weights::with_params(&quants, &scales, mv, n, k, p),
                    &mut got,
                    m,
                );
                // Both paths round the same f32 products to f16 and accumulate in
                // the same tile order, so this is exact rather than approximate.
                for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "q4 {m}x{n}x{k} block={block} affine={affine} idx={idx}: {g} vs {w}"
                    );
                }
            }
        }
    }
}

// More N-blocks than panel slots (slot reuse), the N-split GEMM items at small m,
// a K tail off the 4-row unpack group, block < 4 and an odd tile count; f16 and
// bf16, each exact against eager dequant through the dense kernel.
#[test]
fn q4_panel_ring_matches_eager_dequant() {
    use half::bf16;
    use sme_gemm::{
        Q4Params, Q4Weights, dequant_q4_bf16_with, dequant_q4_with, matmul_bf16_packed,
        matmul_f16_packed, matmul_q4, matmul_q4_bf16,
    };
    if !caps().sme_f16f16 {
        return;
    }
    for &(m, n, k, block) in &[
        (40usize, 4200usize, 2050usize, 2usize),
        (200, 4100, 2048, 32),
        (96, 4130, 1027, 1),
    ] {
        for affine in [false, true] {
            let p = if affine {
                Q4Params::new(block).affine()
            } else {
                Q4Params::new(block)
            };
            let nbk = k.div_ceil(block);
            let mut s = 0x5a5a_0001u64 ^ ((m * 131 + n * 17 + k + block) as u64);
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
            let w = Q4Weights::with_params(&quants, &scales, mv, n, k, p);
            let mut want = vec![f16::ZERO; m * n];
            matmul_f16_packed(
                &a,
                &dequant_q4_with(&quants, &scales, mv, n, k, p),
                &mut want,
                m,
            );
            let mut got = vec![f16::ZERO; m * n];
            matmul_q4(&a, &w, &mut got, m);
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "q4 {m}x{n}x{k} block={block} affine={affine} idx={idx}: {g} vs {w}"
                );
            }
            if !caps().sme_b16b16 {
                continue;
            }
            let ab: Vec<bf16> = a.iter().map(|x| bf16::from_f32(x.to_f32())).collect();
            let mut want = vec![bf16::ZERO; m * n];
            let bp = dequant_q4_bf16_with(&quants, &scales, mv, n, k, p);
            matmul_bf16_packed(&ab, &bp, &mut want, m);
            let mut got = vec![bf16::ZERO; m * n];
            matmul_q4_bf16(&ab, &w, &mut got, m);
            for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "q4 bf16 {m}x{n}x{k} block={block} affine={affine} idx={idx}: {g} vs {w}"
                );
            }
        }
    }
}

// The bf16 Q4 kernel, over the same block sizes and both code forms: the
// 4-bit-resident bf16 path must agree with eagerly dequantizing the same
// weights to bf16 and running the plain bf16 GEMM. The same Q4Weights (f16
// scales) feeds both, which is the point -- one packed weight set, two compute
// dtypes. Both round the same f32 products to bf16 and accumulate in the same
// tile order, so this is exact, not approximate -- which makes it the sensitive
// test of the two (any drift in the tile dequant shows up immediately).
#[test]
fn q4_bf16_matches_eager_dequant() {
    use half::bf16;
    use sme_gemm::{Q4Params, Q4Weights, dequant_q4_bf16_with, matmul_bf16_packed, matmul_q4_bf16};
    if !caps().sme_b16b16 {
        return;
    }
    for &(m, n, k) in &[(64usize, 96usize, 128usize), (65, 33, 256), (800, 520, 512)] {
        for &block in &[16usize, 32, 64] {
            for affine in [false, true] {
                let p = if affine {
                    Q4Params::new(block).affine()
                } else {
                    Q4Params::new(block)
                };
                let nbk = k.div_ceil(block);
                let mut s = 0x5b5b_0001u64 ^ ((m * 131 + n * 17 + k + block) as u64);
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

                let mut want = vec![bf16::ZERO; m * n];
                matmul_bf16_packed(
                    &a,
                    &dequant_q4_bf16_with(&quants, &scales, mv, n, k, p),
                    &mut want,
                    m,
                );
                let mut got = vec![bf16::ZERO; m * n];
                matmul_q4_bf16(
                    &a,
                    &Q4Weights::with_params(&quants, &scales, mv, n, k, p),
                    &mut got,
                    m,
                );
                for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "q4 bf16 {m}x{n}x{k} block={block} affine={affine} idx={idx}: {g} vs {w}"
                    );
                }
            }
        }
    }
}

// The bf16 Q4 kernel against an exact f64 oracle over the dequantized weights,
// not just against the eager bf16 path -- so a shared bug in the pack could not
// hide by cancelling out. Scale-only and affine, default block. The tolerance is
// bf16 accumulation over k=128 (measured max 1.5e-2, so 2e-2 is ~1.3x slack).
#[test]
fn q4_bf16_matches_f64_oracle() {
    use half::bf16;
    use sme_gemm::{Q4Params, Q4Weights, matmul_q4_bf16};
    if !caps().sme_b16b16 {
        return;
    }
    let (m, n, k, block) = (64usize, 96usize, 128usize, 32usize);
    let nbk = k.div_ceil(block);
    for affine in [false, true] {
        let p = if affine {
            Q4Params::new(block).affine()
        } else {
            Q4Params::new(block)
        };
        let mut s = 0x6c6c_0001u64 ^ u64::from(affine);
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

        let mut got = vec![bf16::ZERO; m * n];
        matmul_q4_bf16(
            &a,
            &Q4Weights::with_params(&quants, &scales, mv, n, k, p),
            &mut got,
            m,
        );

        // Oracle: dequantize row-major exactly as the format defines, round to
        // bf16 (which is what the kernel's tile scratch holds), accumulate f64.
        let w: Vec<f64> = (0..k * n)
            .map(|idx| {
                let (d, j) = (idx / n, idx % n);
                let nib = if idx.is_multiple_of(2) {
                    quants[idx / 2] & 0x0f
                } else {
                    quants[idx / 2] >> 4
                };
                let code = i32::from(nib) - if nib < 8 { 0 } else { 16 };
                let bi = j * nbk + d / block;
                let mut v = scales[bi].to_f32() * code as f32;
                if let Some(mv) = mv {
                    v += mv[bi].to_f32();
                }
                f64::from(bf16::from_f32(v).to_f32())
            })
            .collect();
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for d in 0..k {
                    acc += f64::from(a[i * k + d].to_f32()) * w[d * n + j];
                }
                let g = f64::from(got[i * n + j].to_f32());
                assert!(
                    (g - acc).abs() <= 2e-2 * (1.0 + acc.abs()),
                    "q4 bf16 oracle affine={affine} ({i},{j}): {g} vs {acc}"
                );
            }
        }
    }
}

// The Q4 fused epilogue, against an f64 oracle over the dequantized weights.
//
// Deliberately NOT compared bit-for-bit against the dense packed path: that one
// folds an `add_col` bias into the ZA init with a rank-1 MOPA, so the bias lands
// in the accumulator before the K-loop rather than at the store, and the two
// round differently. Both are correct; only the oracle arbitrates. The empty
// op-graph IS pinned exactly, since it must leave the plain store untouched.
#[test]
fn q4_fused_epilogue_matches_oracle() {
    use sme_gemm::{Epilogue, Q4Params, Q4Weights, matmul_q4, matmul_q4_ep};
    if !caps().sme_f16f16 {
        return;
    }
    for &(m, n, k) in &[(64usize, 96usize, 128usize), (65, 33, 256), (800, 520, 512)] {
        for affine in [false, true] {
            let p = if affine {
                Q4Params::new(32).affine()
            } else {
                Q4Params::new(32)
            };
            let nbk = k.div_ceil(32);
            let mut s = 0x7e7e_0001u64 ^ ((m * 131 + n * 17 + k) as u64);
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
            let col: Vec<f16> = (0..n)
                .map(|j| f16::from_f32(0.02 * (j as f32) - 0.4))
                .collect();
            let row: Vec<f16> = (0..m)
                .map(|i| f16::from_f32(0.01 * (i as f32) - 0.2))
                .collect();

            let w = Q4Weights::with_params(&quants, &scales, mv, n, k, p);
            // The weights the kernel actually multiplies: f16, row-major k x n.
            let b: Vec<f16> = (0..k * n)
                .map(|idx| {
                    let (d, j) = (idx / n, idx % n);
                    let nib = if idx.is_multiple_of(2) {
                        quants[idx / 2] & 0x0f
                    } else {
                        quants[idx / 2] >> 4
                    };
                    let code = i32::from(nib) - if nib < 8 { 0 } else { 16 };
                    let bi = j * nbk + d / 32;
                    let mut v = scales[bi].to_f32() * code as f32;
                    if let Some(mv) = mv {
                        v += mv[bi].to_f32();
                    }
                    f16::from_f32(v)
                })
                .collect();
            let base = oracle(&a, &b, m, n, k);
            let tol = (3e-2 * (k as f64).sqrt() / 4.0).max(3e-2);

            for case in 0..4 {
                let ep = match case {
                    0 => Epilogue::new(),
                    1 => Epilogue::new().add_col(&col),
                    2 => Epilogue::new().add_col(&col).relu(),
                    _ => Epilogue::new().add_row(&row).add_col(&col).gelu(),
                };
                let mut got = vec![f16::ZERO; m * n];
                matmul_q4_ep(&a, &w, &mut got, m, &ep);

                let mut want = base.clone();
                for i in 0..m {
                    for j in 0..n {
                        let v = &mut want[i * n + j];
                        if case >= 1 {
                            *v += f64::from(col[j]);
                        }
                        if case == 3 {
                            *v += f64::from(row[i]);
                        }
                        match case {
                            2 => *v = v.max(0.0),
                            3 => *v = gelu_f64(*v),
                            _ => {}
                        }
                    }
                }
                let mr = max_rel(&got, &want);
                assert!(
                    mr < tol,
                    "q4 ep {m}x{n}x{k} affine={affine} case={case}: max_rel={mr} >= {tol}"
                );

                // The empty op-graph must leave the plain store untouched.
                if case == 0 {
                    let mut plain = vec![f16::ZERO; m * n];
                    matmul_q4(&a, &w, &mut plain, m);
                    assert_eq!(plain, got, "empty epilogue must equal matmul_q4");
                }
            }
        }
    }
}

// The bf16 twin of `q4_fused_epilogue_matches_oracle`.
#[test]
fn q4_bf16_fused_epilogue_matches_oracle() {
    use half::bf16;
    use sme_gemm::{Epilogue, Q4Params, Q4Weights, matmul_q4_bf16, matmul_q4_bf16_ep};
    if !caps().sme_b16b16 {
        return;
    }
    let (m, n, k) = (64usize, 96usize, 128usize);
    let nbk = k.div_ceil(32);
    for affine in [false, true] {
        let p = if affine {
            Q4Params::new(32).affine()
        } else {
            Q4Params::new(32)
        };
        let mut s = 0x8f8f_0001u64 ^ u64::from(affine);
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
        let col: Vec<bf16> = (0..n)
            .map(|j| bf16::from_f32(0.02 * (j as f32) - 0.4))
            .collect();

        let w = Q4Weights::with_params(&quants, &scales, mv, n, k, p);
        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for d in 0..k {
                    let idx = d * n + j;
                    let nib = if idx.is_multiple_of(2) {
                        quants[idx / 2] & 0x0f
                    } else {
                        quants[idx / 2] >> 4
                    };
                    let code = i32::from(nib) - if nib < 8 { 0 } else { 16 };
                    let bi = j * nbk + d / 32;
                    let mut v = scales[bi].to_f32() * code as f32;
                    if let Some(mv) = mv {
                        v += mv[bi].to_f32();
                    }
                    acc += f64::from(a[i * k + d].to_f32()) * f64::from(bf16::from_f32(v).to_f32());
                }
                want[i * n + j] = (acc + f64::from(col[j].to_f32())).max(0.0);
            }
        }

        let mut got = vec![bf16::ZERO; m * n];
        matmul_q4_bf16_ep(&a, &w, &mut got, m, &Epilogue::new().add_col(&col).relu());
        for (idx, (gv, wv)) in got.iter().zip(&want).enumerate() {
            let g = f64::from(gv.to_f32());
            assert!(
                (g - wv).abs() <= 8e-2 * (1.0 + wv.abs()),
                "q4 bf16 ep affine={affine} idx={idx}: {g} vs {wv}"
            );
        }

        let mut plain = vec![bf16::ZERO; m * n];
        matmul_q4_bf16(&a, &w, &mut plain, m);
        let mut empty = vec![bf16::ZERO; m * n];
        matmul_q4_bf16_ep(&a, &w, &mut empty, m, &Epilogue::new());
        assert_eq!(plain, empty, "empty epilogue must equal matmul_q4_bf16");
    }
}

// `dequant_q4_bf16` must equal `dequant_q4_bf16_with` at the default params
// (Q4_0: 32-value blocks, scale-only). Only the `_with` form was covered, so the
// default-parameter wrapper -- what most callers reach for -- was untested.
#[test]
fn dequant_q4_bf16_defaults_match_the_explicit_form() {
    use half::bf16;
    use sme_gemm::{Q4Params, dequant_q4_bf16, dequant_q4_bf16_with, matmul_bf16_packed};
    if !caps().sme_b16b16 {
        return;
    }
    let (m, n, k) = (64usize, 96usize, 128usize);
    let nbk = k.div_ceil(32);
    let mut s = 0xd0d0_0001u64;
    let mut byte = || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (s >> 40) as u8
    };
    let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|_| byte()).collect();
    let scales: Vec<f16> = (0..n * nbk)
        .map(|i| f16::from_f32(0.01 + 0.001 * (i % 7) as f32))
        .collect();
    let a: Vec<bf16> = (0..m * k)
        .map(|i| bf16::from_f32((i % 13) as f32 * 0.05 - 0.3))
        .collect();

    // Compare through a GEMM: `Packed` holds no public accessor for its panel.
    let mut got = vec![bf16::ZERO; m * n];
    matmul_bf16_packed(&a, &dequant_q4_bf16(&quants, &scales, n, k), &mut got, m);
    let mut want = vec![bf16::ZERO; m * n];
    matmul_bf16_packed(
        &a,
        &dequant_q4_bf16_with(&quants, &scales, None, n, k, Q4Params::default()),
        &mut want,
        m,
    );
    assert_eq!(got, want, "dequant_q4_bf16 must equal the explicit form");
}
