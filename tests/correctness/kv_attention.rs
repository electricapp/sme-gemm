//! Single-query attention over an f16 KV cache against an f64 oracle: every
//! length mod 4, grouped-query and multi-head, `head_dim` 8..256.

use crate::{fill, fill_f32};
use sme_gemm::{attention_kv_causal_f16, attention_kv_f16};

#[allow(clippy::too_many_arguments)]
fn oracle(
    q: &[f32],
    k: &[half::f16],
    v: &[half::f16],
    len: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    scale: f32,
) -> Vec<f64> {
    let ld = kv_heads * hd;
    let group = heads / kv_heads;
    let mut out = vec![0.0; heads * hd];
    for h in 0..heads {
        let off = (h / group) * hd;
        let s: Vec<f64> = (0..len)
            .map(|t| {
                (0..hd)
                    .map(|d| {
                        f64::from(q[h * hd + d]) * f64::from(scale) * k[t * ld + off + d].to_f64()
                    })
                    .sum()
            })
            .collect();
        let mx = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
        let sum: f64 = e.iter().sum();
        for t in 0..len {
            for d in 0..hd {
                out[h * hd + d] += e[t] / sum * v[t * ld + off + d].to_f64();
            }
        }
    }
    out
}

#[test]
fn kv_f16_matches_oracle() {
    for &(heads, kv_heads, hd) in &[
        (1, 1, 8),
        (6, 6, 64),
        (8, 2, 64),
        (4, 1, 128),
        (2, 2, 256),
        (12, 4, 72),
    ] {
        for len in [1usize, 2, 3, 4, 5, 7, 31, 128, 257] {
            let mut s = 0x5eed_0000 ^ ((heads * 1000 + hd * 10 + len) as u64);
            let ld = kv_heads * hd;
            // Larger q spreads the softmax so it is not near-uniform.
            let q: Vec<f32> = fill_f32(&mut s, heads * hd)
                .iter()
                .map(|x| x * 8.0)
                .collect();
            let k = fill(&mut s, len * ld);
            let v = fill(&mut s, len * ld);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut out = vec![0.0f32; heads * hd];
            attention_kv_f16(&q, &k, &v, len, heads, kv_heads, hd, scale, &mut out);
            let want = oracle(&q, &k, &v, len, heads, kv_heads, hd, scale);
            let err = out
                .iter()
                .zip(&want)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0.0, f64::max);
            // q rounds to f16 (~5e-4 relative) and |v| <= 0.5.
            assert!(
                err < 2e-3,
                "kv attention {heads}/{kv_heads}x{hd} len {len}: max abs err {err}"
            );
        }
    }
}

#[test]
fn kv_causal_rows_match_one_row() {
    for &(heads, kv_heads, hd) in &[(6, 6, 64), (8, 2, 64), (4, 4, 128)] {
        for &(start, rows) in &[(0usize, 1usize), (0, 7), (5, 3), (0, 256), (100, 300)] {
            let mut s = 0xca05_0000 ^ ((heads * 100 + start * 7 + rows) as u64);
            let ld = kv_heads * hd;
            let q: Vec<f32> = fill_f32(&mut s, rows * heads * hd)
                .iter()
                .map(|x| x * 8.0)
                .collect();
            let k = fill(&mut s, (start + rows) * ld);
            let v = fill(&mut s, (start + rows) * ld);
            let scale = 1.0 / (hd as f32).sqrt();
            let mut out = vec![0.0f32; rows * heads * hd];
            attention_kv_causal_f16(
                &q, &k, &v, start, rows, heads, kv_heads, hd, scale, &mut out,
            );
            let qr = heads * hd;
            let mut one = vec![0.0f32; qr];
            for i in 0..rows {
                let n = start + i + 1;
                attention_kv_f16(
                    &q[i * qr..(i + 1) * qr],
                    &k[..n * ld],
                    &v[..n * ld],
                    n,
                    heads,
                    kv_heads,
                    hd,
                    scale,
                    &mut one,
                );
                assert_eq!(
                    &out[i * qr..(i + 1) * qr],
                    &one[..],
                    "causal row {i} of {start}+{rows}, {heads}/{kv_heads}x{hd}"
                );
            }
        }
    }
}

/// `KvCache` gives exactly the free functions' results on the same rows,
/// pushed one at a time or in blocks, from f16 or f32.
#[test]
fn kv_cache_matches_free_functions() {
    use sme_gemm::KvCache;
    let (heads, kv_heads, hd) = (8usize, 2usize, 64usize);
    let ld = kv_heads * hd;
    let mut s = 0xcac4_e001;
    let len = 37;
    let k = fill(&mut s, len * ld);
    let v = fill(&mut s, len * ld);
    let scale = 1.0 / (hd as f32).sqrt();
    let mut one = KvCache::new(heads, kv_heads, hd, 64);
    for t in 0..len {
        one.push(&k[t * ld..(t + 1) * ld], &v[t * ld..(t + 1) * ld]);
    }
    let mut block = KvCache::new(heads, kv_heads, hd, 64);
    let k32: Vec<f32> = k.iter().map(|x| x.to_f32()).collect();
    let v32: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
    block.extend(&k32[..10 * ld], &v32[..10 * ld]);
    block.extend(&k[10 * ld..], &v[10 * ld..]);
    assert_eq!(one.keys(), block.keys());
    assert_eq!(one.values(), block.values());
    assert_eq!(one.len(), len);

    let q: Vec<f32> = fill_f32(&mut s, heads * hd)
        .iter()
        .map(|x| x * 8.0)
        .collect();
    let (mut got, mut want) = (vec![0.0f32; heads * hd], vec![0.0f32; heads * hd]);
    one.attend(&q, &mut got);
    attention_kv_f16(&q, &k, &v, len, heads, kv_heads, hd, scale, &mut want);
    assert_eq!(got, want, "attend");

    let rows = 5;
    let qr: Vec<f32> = fill_f32(&mut s, rows * heads * hd)
        .iter()
        .map(|x| x * 8.0)
        .collect();
    let (mut got, mut want) = (
        vec![0.0f32; rows * heads * hd],
        vec![0.0f32; rows * heads * hd],
    );
    one.attend_causal(&qr, rows, &mut got);
    attention_kv_causal_f16(
        &qr,
        &k,
        &v,
        len - rows,
        rows,
        heads,
        kv_heads,
        hd,
        scale,
        &mut want,
    );
    assert_eq!(got, want, "attend_causal");

    one.truncate(3);
    assert_eq!(one.len(), 3);
    one.clear();
    assert!(one.is_empty());
}

#[test]
#[should_panic(expected = "KvCache full")]
fn kv_cache_refuses_to_overflow() {
    let mut c = sme_gemm::KvCache::new(1, 1, 8, 1);
    let row = vec![half::f16::ZERO; 8];
    c.push(&row, &row);
    c.push(&row, &row);
}

/// With a `HotPool` alive the heads split across workers, and every output is
/// bit for bit the one-thread result (each KV-head group runs the same kernel).
#[test]
fn hot_pool_attention_matches_one_thread() {
    for &(heads, kv_heads, hd, len) in &[
        (6usize, 6usize, 64usize, 200usize),
        (8, 2, 64, 300),
        (12, 4, 128, 64),
    ] {
        let ld = kv_heads * hd;
        let mut s = 0x9001_0000 ^ (heads * 1000 + len) as u64;
        let q: Vec<f32> = fill_f32(&mut s, heads * hd)
            .iter()
            .map(|x| x * 8.0)
            .collect();
        let k = fill(&mut s, len * ld);
        let v = fill(&mut s, len * ld);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut one = vec![0.0f32; heads * hd];
        attention_kv_f16(&q, &k, &v, len, heads, kv_heads, hd, scale, &mut one);
        let pool = sme_gemm::HotPool::with_workers(2);
        for _ in 0..50 {
            let mut many = vec![0.0f32; heads * hd];
            attention_kv_f16(&q, &k, &v, len, heads, kv_heads, hd, scale, &mut many);
            assert_eq!(one, many, "{heads}/{kv_heads}x{hd} len {len}");
        }
        drop(pool);
    }
}
