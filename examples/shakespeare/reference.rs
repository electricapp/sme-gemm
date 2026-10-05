//! An unquantized f32 forward pass (exact erf GELU, plain loops) to check the
//! Q4/SME model against, plus the accuracy and speed report behind `--bench`.

use std::time::Instant;

use crate::model::{
    CTX, D, F32Weights, FF, HD, Model, NH, NL, Rng, Sampler, Session, V, encode, sample,
};

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn layernorm(x: &[f32], w: &[f32], out: &mut [f32]) {
    sme_gemm::nn::layer_norm(x, w, None, 1e-5, out);
}

struct RefState {
    k: Vec<f32>,
    v: Vec<f32>,
    logits: Vec<f32>,
}

fn erf(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26, |err| < 1.5e-7, in f64.
    let (s, x) = (f64::from(x.signum()), f64::from(x.abs()));
    let t = 1.0 / 0.327_591_1f64.mul_add(x, 1.0);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    (s * (1.0 - poly * (-x * x).exp())) as f32
}

fn step_ref(w: &F32Weights, st: &mut RefState, tok: usize, pos: usize) {
    let mv = |m: &[f32], x: &[f32], n: usize, k: usize, out: &mut [f32]| {
        for j in 0..n {
            out[j] = m[j * k..(j + 1) * k]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum();
        }
    };
    let mut x: Vec<f32> = (0..D)
        .map(|c| w.wte[tok * D + c] + w.wpe[pos * D + c])
        .collect();
    let (mut h, mut qkv, mut y, mut o, mut f) = (
        vec![0.0; D],
        vec![0.0; 3 * D],
        vec![0.0; D],
        vec![0.0; D],
        vec![0.0; FF],
    );
    for (l, ly) in w.layers.iter().enumerate() {
        layernorm(&x, &ly.ln1, &mut h);
        mv(&ly.attn, &h, 3 * D, D, &mut qkv);
        let base = l * CTX * D;
        st.k[base + pos * D..base + (pos + 1) * D].copy_from_slice(&qkv[D..2 * D]);
        st.v[base + pos * D..base + (pos + 1) * D].copy_from_slice(&qkv[2 * D..]);
        let scale = 1.0 / (HD as f32).sqrt();
        y.fill(0.0);
        for hh in 0..NH {
            let q = &qkv[hh * HD..(hh + 1) * HD];
            let s: Vec<f32> = (0..=pos)
                .map(|t| {
                    q.iter()
                        .zip(&st.k[base + t * D + hh * HD..])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let mx = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = s.iter().map(|v| (v - mx).exp()).collect();
            let sum: f32 = e.iter().sum();
            for (t, et) in e.iter().enumerate() {
                for d in 0..HD {
                    y[hh * HD + d] += et / sum * st.v[base + t * D + hh * HD + d];
                }
            }
        }
        mv(&ly.proj, &y, D, D, &mut o);
        for (a, b) in x.iter_mut().zip(&o) {
            *a += b;
        }
        layernorm(&x, &ly.ln2, &mut h);
        mv(&ly.fc, &h, FF, D, &mut f);
        for v in &mut f {
            *v = 0.5 * *v * (1.0 + erf(*v / std::f32::consts::SQRT_2));
        }
        mv(&ly.fcp, &f, D, FF, &mut o);
        for (a, b) in x.iter_mut().zip(&o) {
            *a += b;
        }
    }
    layernorm(&x.clone(), &w.lnf, &mut x);
    for c in 0..V {
        st.logits[c] = dot(&x, &w.wte[c * D..(c + 1) * D]);
    }
}

fn xent(logits: &[f32], y: usize) -> f64 {
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = logits
        .iter()
        .map(|v| f64::from(v - mx).exp())
        .sum::<f64>()
        .ln()
        + f64::from(mx);
    lse - f64::from(logits[y])
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len())
        .max_by(|&a, &b| v[a].total_cmp(&v[b]))
        .unwrap_or(0)
}

/// Validation loss of the f32 reference and the Q4/SME model, then generation
/// and prompt throughput.
pub(crate) fn bench(w: &F32Weights, m: &Model, text: &str) {
    let toks: Vec<usize> = text.chars().filter_map(encode).collect();
    let val = &toks[toks.len() * 9 / 10..];
    let mut sess = Session::new(m);
    let mut rs = RefState {
        k: vec![0.0; NL * CTX * D],
        v: vec![0.0; NL * CTX * D],
        logits: vec![0.0; V],
    };
    let (mut lr, mut lq, mut agree, mut n) = (0f64, 0f64, 0usize, 0usize);
    for c in 0..8 {
        let off = c * 1300;
        sess.reset();
        for pos in 0..CTX {
            step_ref(w, &mut rs, val[off + pos], pos);
            sess.forward(&[val[off + pos]]);
            lr += xent(&rs.logits, val[off + pos + 1]);
            lq += xent(&sess.logits, val[off + pos + 1]);
            agree += usize::from(argmax(&rs.logits) == argmax(&sess.logits));
            n += 1;
        }
    }
    println!(
        "val loss over {n} chars: f32 reference {:.4}, Q4/SME {:.4}; argmax agreement {:.1}%",
        lr / n as f64,
        lq / n as f64,
        100.0 * agree as f64 / n as f64
    );
    speed(w, val, &mut sess, &mut rs);
}

/// Generation and prompt throughput, and batched vs token-by-token logits.
fn speed(w: &F32Weights, val: &[usize], sess: &mut Session<'_>, rs: &mut RefState) {
    // Generation: full 256-token windows from "\n", median of 50.
    let s = Sampler {
        temperature: 0.8,
        top_k: 0,
        seed: 7,
    };
    let mut rng = Rng::new(s.seed);
    let mut times = vec![];
    for i in 0..60 {
        sess.reset();
        let mut tok = 0;
        let t0 = Instant::now();
        for _ in 0..CTX {
            sess.forward(&[tok]);
            tok = sample(&sess.logits, &s, &mut rng);
        }
        if i >= 10 {
            times.push(t0.elapsed().as_secs_f64());
        }
    }
    times.sort_by(f64::total_cmp);
    let t = times[times.len() / 2];
    println!(
        "generate: {:.1} us/token -> {:.0} tok/s (256-token windows, mean context 128)",
        t / CTX as f64 * 1e6,
        CTX as f64 / t
    );

    // Short context: the first 16 positions only.
    let mut times = vec![];
    for i in 0..400 {
        sess.reset();
        let mut tok = 0;
        let t0 = Instant::now();
        for _ in 0..16 {
            sess.forward(&[tok]);
            tok = sample(&sess.logits, &s, &mut rng);
        }
        if i >= 40 {
            times.push(t0.elapsed().as_secs_f64());
        }
    }
    times.sort_by(f64::total_cmp);
    let t = times[times.len() / 2];
    println!(
        "generate, context <= 16: {:.1} us/token -> {:.0} tok/s",
        t / 16.0 * 1e6,
        16.0 / t
    );

    // Prompts: one batch each.
    for len in [64, 128, CTX] {
        let prompt = &val[..len];
        let mut times = vec![];
        for i in 0..30 {
            sess.reset();
            let t0 = Instant::now();
            sess.forward(prompt);
            if i >= 5 {
                times.push(t0.elapsed().as_secs_f64());
            }
        }
        times.sort_by(f64::total_cmp);
        let t = times[times.len() / 2];
        println!(
            "prompt: {len} tokens in {:.2} ms -> {:.0} tok/s",
            t * 1e3,
            len as f64 / t
        );
    }

    // Batched prompt must match token-by-token (up to rounding: rows <= 7 run
    // the GEMV kernels, more run the MOPA path).
    for r in [2usize, 7, 8, 64, 256] {
        let p = &val[..r];
        sess.reset();
        sess.forward(p);
        let batched = sess.logits.clone();
        sess.reset();
        for &t in p {
            sess.forward(&[t]);
        }
        let diff = batched
            .iter()
            .zip(&sess.logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        for (pos, &t) in p.iter().enumerate() {
            step_ref(w, rs, t, pos);
        }
        let d = |a: &[f32]| {
            a.iter()
                .zip(&rs.logits)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        println!(
            "prompt of {r:>3}: batched vs token-by-token max logit diff {diff:.4}; vs f32: batched {:.4}, token-by-token {:.4}",
            d(&batched),
            d(&sess.logits)
        );
    }
}
