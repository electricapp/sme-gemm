//! `PrismML`'s Bonsai (1-bit Qwen3) on the SME unit, from its `Q1_0` GGUF.
//!
//! Every projection and the tied head run `matmul_q1` through `Linear`; the
//! rest is the library's norms, `KvCache` and `SwiGLU` `Mlp` (`model.rs`). One
//! stream, CPU only.
//!
//!   examples/bonsai/fetch.sh                      # once: the 1.7B GGUF, 248 MB
//!   cargo run --release --example bonsai -- "Why is the sky blue?"
//!   cargo run --release --example bonsai -- --bench     # decode tok/s, as llama-bench tg128
//!
//! Options: `-n N` tokens (default 256), `--temp T` (0.5; 0 is greedy),
//! `--top-k K` (20), `--top-p P` (0.85), `--seed S`, `--raw` (no chat
//! template), `--model PATH`.

mod gguf;
mod model;
mod tokenizer;

use std::io::Write;
use std::time::Instant;

const IM_END: u32 = 151_645;
const END_OF_TEXT: u32 = 151_643;

struct Opts {
    prompt: String,
    n: usize,
    temp: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    raw: bool,
    bench: bool,
    model: std::path::PathBuf,
}

/// Option `name`'s value `v` parsed.
fn num<T: std::str::FromStr>(name: &str, v: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    v.parse().map_err(|e| format!("{name}: {e}"))
}

fn opts() -> Result<Opts, String> {
    let mut o = Opts {
        prompt: String::new(),
        n: 256,
        temp: 0.5,
        top_k: 20,
        top_p: 0.85,
        seed: 0x5eed,
        raw: false,
        bench: false,
        model: std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples/bonsai/data/Bonsai-1.7B-Q1_0.gguf"),
    };
    let mut args = std::env::args().skip(1);
    let mut words = Vec::new();
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "-n" => o.n = num(&a, &val()?)?,
            "--temp" => o.temp = num(&a, &val()?)?,
            "--top-k" => o.top_k = num(&a, &val()?)?,
            "--top-p" => o.top_p = num(&a, &val()?)?,
            "--seed" => o.seed = num(&a, &val()?)?,
            "--model" => o.model = val()?.into(),
            "--raw" => o.raw = true,
            "--bench" => o.bench = true,
            _ => words.push(a),
        }
    }
    o.prompt = words.join(" ");
    if o.prompt.is_empty() && !o.bench {
        o.prompt = "Write a short poem about the sea.".into();
    }
    Ok(o)
}

/// xorshift64*: a uniform f32 in [0, 1).
fn uniform(s: &mut u64) -> f32 {
    *s ^= *s >> 12;
    *s ^= *s << 25;
    *s ^= *s >> 27;
    (s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32
}

/// Greedy at `temp == 0`, else top-k then top-p sampling at `temp`.
fn sample(logits: &[f32], o: &Opts, rng: &mut u64) -> u32 {
    if o.temp <= 0.0 || o.top_k <= 1 {
        // A max reduction vectorizes; finding its position is then one scan.
        let best = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let at = logits.iter().position(|v| v.to_bits() == best.to_bits());
        return at.unwrap_or(0) as u32;
    }
    // The k largest, kept sorted descending.
    let mut top: Vec<(f32, u32)> = Vec::with_capacity(o.top_k + 1);
    for (i, &v) in logits.iter().enumerate() {
        if top.len() < o.top_k || v > top[top.len() - 1].0 {
            let at = top.partition_point(|&(t, _)| t >= v);
            top.insert(at, (v, i as u32));
            top.truncate(o.top_k);
        }
    }
    let mut p: Vec<f32> = top
        .iter()
        .map(|&(v, _)| ((v - top[0].0) / o.temp).exp())
        .collect();
    let total: f32 = p.iter().sum();
    let (mut acc, mut keep) = (0.0, p.len());
    for (i, x) in p.iter_mut().enumerate() {
        *x /= total;
        acc += *x;
        if acc >= o.top_p {
            keep = i + 1;
            break;
        }
    }
    let mass: f32 = p[..keep].iter().sum();
    let mut u = uniform(rng) * mass;
    for (i, &x) in p[..keep].iter().enumerate() {
        u -= x;
        if u <= 0.0 {
            return top[i].1;
        }
    }
    top[keep - 1].1
}

/// Writes the longest valid UTF-8 prefix of `pending` and keeps the rest.
fn flush_utf8(out: &mut impl Write, pending: &mut Vec<u8>) -> std::io::Result<()> {
    let valid = match std::str::from_utf8(pending) {
        Ok(s) => s.len(),
        Err(e) => e.valid_up_to(),
    };
    out.write_all(&pending[..valid])?;
    out.flush()?;
    pending.drain(..valid);
    Ok(())
}

/// As llama-bench tg128: tokens generated greedily from an empty context, one
/// at a time; the mean and spread of three timed rounds after a warmup one.
/// `BONSAI_PROFILE=1` adds the last round's time per stage.
fn bench(model: &model::Model, tok: &tokenizer::Tokenizer, o: &Opts, logits: &mut [f32]) {
    let n = if o.n == 256 { 128 } else { o.n };
    let greedy = Opts {
        temp: 0.0,
        prompt: String::new(),
        model: o.model.clone(),
        ..*o
    };
    let mut rates = Vec::new();
    for round in 0..4 {
        let mut sess = model::Session::new(model, n + 1);
        if std::env::var_os("BONSAI_PROFILE").is_some() && round == 3 {
            sess.prof = Some([0.0; model::STAGES.len()]);
        }
        let mut t = tok.encode("<|im_start|>")[0];
        let start = Instant::now();
        for _ in 0..n {
            sess.forward(&[t], logits);
            t = sample(logits, &greedy, &mut 0);
        }
        let s = start.elapsed().as_secs_f64();
        if round > 0 {
            rates.push(n as f64 / s);
        }
        eprintln!(
            "[round {round}: {n} tokens in {:.1} ms = {:.1} tok/s]",
            s * 1e3,
            n as f64 / s
        );
        if let Some(p) = sess.prof {
            for (name, t) in model::STAGES.iter().zip(p) {
                eprintln!("  {name:14} {:7.1} us/token", t / n as f64 * 1e6);
            }
        }
    }
    let mean = rates.iter().sum::<f64>() / rates.len() as f64;
    let sd = (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64).sqrt();
    println!("tg{n}: {mean:.1} ± {sd:.1} tok/s");

    // As llama-bench pp512: one 512-token prompt in a single forward pass.
    let prompt: Vec<u32> = tok
        .encode(&"The quick brown fox jumps over the lazy dog. ".repeat(60))
        .into_iter()
        .take(512)
        .collect();
    let mut best = f64::INFINITY;
    for round in 0..3 {
        let mut sess = model::Session::new(model, prompt.len() + 1);
        if std::env::var_os("BONSAI_PROFILE").is_some() && round == 2 {
            sess.prof = Some([0.0; model::STAGES.len()]);
        }
        let start = Instant::now();
        sess.forward(&prompt, logits);
        best = best.min(start.elapsed().as_secs_f64());
        if let Some(p) = sess.prof {
            for (name, t) in model::STAGES.iter().zip(p) {
                eprintln!("  {name:14} {:7.1} ms", t * 1e3);
            }
        }
    }
    println!(
        "pp{}: {:.1} tok/s",
        prompt.len(),
        prompt.len() as f64 / best
    );
}

fn run() -> Result<(), String> {
    let o = opts()?;
    let t0 = Instant::now();
    let g = gguf::Gguf::load(&o.model).map_err(|e| {
        format!(
            "{}: {e} (run examples/bonsai/fetch.sh first)",
            o.model.display()
        )
    })?;
    let tok = tokenizer::Tokenizer::from_gguf(&g).map_err(|e| e.to_string())?;
    let model = model::Model::load(&g).map_err(|e| e.to_string())?;
    drop(g);
    let vocab = model.cfg.vocab;
    eprintln!(
        "[loaded {} layers, d={}, vocab {} in {:.1} s]",
        model.cfg.layers,
        model.cfg.d,
        vocab,
        t0.elapsed().as_secs_f64()
    );
    let mut logits = vec![0.0f32; vocab];
    if o.bench {
        bench(&model, &tok, &o, &mut logits);
        return Ok(());
    }

    let text = if o.raw {
        o.prompt.clone()
    } else {
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            o.prompt
        )
    };
    let prompt = tok.encode(&text);
    let mut sess = model::Session::new(&model, prompt.len() + o.n + 1);
    let mut rng = o.seed;
    let start = Instant::now();
    sess.forward(&prompt, &mut logits);
    let prefill = start.elapsed().as_secs_f64();
    let mut stdout = std::io::stdout().lock();
    let mut pending = Vec::new();
    let (mut generated, mut forward_s) = (0usize, 0.0f64);
    let gen_start = Instant::now();
    for _ in 0..o.n {
        let t = sample(&logits, &o, &mut rng);
        if t == IM_END || t == END_OF_TEXT {
            break;
        }
        generated += 1;
        pending.extend_from_slice(tok.bytes(t));
        flush_utf8(&mut stdout, &mut pending).map_err(|e| e.to_string())?;
        let f = Instant::now();
        sess.forward(&[t], &mut logits);
        forward_s += f.elapsed().as_secs_f64();
    }
    pending.extend_from_slice(b"\n");
    flush_utf8(&mut stdout, &mut pending).map_err(|e| e.to_string())?;
    let total = gen_start.elapsed().as_secs_f64();
    eprintln!(
        "[prompt: {} tokens in {:.1} ms = {:.0} tok/s | generated: {generated} tokens, \
         {:.1} tok/s model ({:.1} tok/s with sampling and output)]",
        prompt.len(),
        prefill * 1e3,
        prompt.len() as f64 / prefill,
        generated as f64 / forward_s,
        generated as f64 / total
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("bonsai: {e}");
        std::process::exit(1);
    }
}
