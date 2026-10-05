//! `KvCache::attend` cost by cached length, on the calling thread and split
//! across a `HotPool` sized for this machine (with the SME unit kept awake
//! throughout, as in a model loop).
//!   cargo run --release --example `kv_attention`

use std::time::Instant;

use half::f16;
use sme_gemm::{HotPool, KvCache, SmeWarm};

fn median_us(cache: &KvCache, q: &[f32], out: &mut [f32]) -> f64 {
    let mut ts = Vec::with_capacity(20_000);
    for i in 0..20_000 {
        let t = Instant::now();
        cache.attend(q, out);
        if i >= 2000 {
            ts.push(t.elapsed().as_secs_f64());
        }
    }
    ts.sort_by(f64::total_cmp);
    ts[ts.len() / 2] * 1e6
}

fn main() {
    let (heads, hd) = (6, 64);
    let row = heads * hd;
    let _warm = SmeWarm::new();
    let q: Vec<f32> = (0..row)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
        .collect();
    let mut out = vec![0.0f32; row];
    println!("{heads} heads x {hd}: us per attend");
    println!("  len | 1 thread | HotPool::new()");
    for len in [16, 64, 128, 256, 512, 1024] {
        let mut cache = KvCache::new(heads, heads, hd, len);
        for t in 0..len {
            let k: Vec<f16> = (0..row)
                .map(|i| f16::from_f32(((t * 13 + i) % 97) as f32 / 96.0 - 0.5))
                .collect();
            let v: Vec<f16> = (0..row)
                .map(|i| f16::from_f32(((t * 7 + i) % 89) as f32 / 88.0 - 0.5))
                .collect();
            cache.push(&k, &v);
        }
        let one = median_us(&cache, &q, &mut out);
        let pool = HotPool::new();
        let many = median_us(&cache, &q, &mut out);
        drop(pool);
        println!("{len:>5} | {one:>8.2} | {many:>10.2}");
    }
}
