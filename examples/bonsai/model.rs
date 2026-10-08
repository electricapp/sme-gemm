//! Qwen3 (Bonsai's architecture) from the library's parts and a GGUF of 1-bit
//! `Q1_0` weights: `Q1Weights` behind every projection and the tied head,
//! `nn` norms (QK-norm included), `KvCache`, a `SwiGLU` `Mlp`, and `YaRN`
//! `RoPE`.

use std::io;
use std::time::Instant;

use half::f16;
use sme_gemm::nn::{self, Norm};
use sme_gemm::{Gate, GatedLinear, HotPool, KvCache, Linear, Mlp, Q1Weights, WeightLayout};

use crate::gguf::{self, Gguf};

/// `Q1_0`'s block: 128 values, an f16 scale then 16 bytes of signs.
const QK: usize = 128;
const BLOCK_BYTES: usize = 18;

pub(crate) struct Config {
    pub(crate) d: usize,
    pub(crate) layers: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab: usize,
    pub(crate) eps: f32,
}

/// Rotary embedding with `YaRN` frequency scaling, as Hugging Face's
/// `_compute_yarn_parameters` and llama.cpp compute it.
struct Rope {
    inv_freq: Vec<f32>,
    /// Multiplies cos and sin (`YaRN`'s attention factor; 1 without scaling).
    mscale: f32,
}

impl Rope {
    fn new(g: &Gguf, arch: &str, dim: usize) -> io::Result<Self> {
        let base = g
            .float(&format!("{arch}.rope.freq_base"))
            .unwrap_or(10_000.0);
        let half = dim / 2;
        let pos_freq = |i: usize| base.powf((2 * i) as f64 / dim as f64);
        let yarn = g.str(&format!("{arch}.rope.scaling.type")) == Some("yarn");
        if !yarn {
            let inv_freq = (0..half).map(|i| (1.0 / pos_freq(i)) as f32).collect();
            return Ok(Self {
                inv_freq,
                mscale: 1.0,
            });
        }
        let factor = g.float(&format!("{arch}.rope.scaling.factor"))?;
        let orig = g.int(&format!("{arch}.rope.scaling.original_context_length"))? as f64;
        let corr = |rot: f64| {
            dim as f64 * (orig / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
        };
        let low = corr(32.0).floor().max(0.0);
        let mut high = corr(1.0).ceil().min(dim as f64 - 1.0);
        if (high - low).abs() < f64::EPSILON {
            high += 0.001;
        }
        let inv_freq = (0..half)
            .map(|i| {
                let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                let extrap = 1.0 - ramp;
                let f = pos_freq(i);
                (1.0 / (factor * f) * (1.0 - extrap) + 1.0 / f * extrap) as f32
            })
            .collect();
        Ok(Self {
            inv_freq,
            mscale: (0.1 * factor.ln() + 1.0) as f32,
        })
    }

    /// Position `pos`'s (cos, sin) per frequency, scaled, into `cs`.
    fn angles(&self, pos: usize, cs: &mut Vec<(f32, f32)>) {
        cs.clear();
        cs.extend(self.inv_freq.iter().map(|&f| {
            let (s, c) = (pos as f32 * f).sin_cos();
            (c * self.mscale, s * self.mscale)
        }));
    }

    /// Rotates each `2 * cs.len()`-wide head of `x` by the angles `cs`
    /// (halves rotated against each other, the `NeoX` / Hugging Face form).
    fn apply(x: &mut [f32], cs: &[(f32, f32)]) {
        let half = cs.len();
        for head in x.chunks_exact_mut(2 * half) {
            let (a, b) = head.split_at_mut(half);
            for ((x1, x2), &(c, s)) in a.iter_mut().zip(b.iter_mut()).zip(cs) {
                let (u, v) = (*x1, *x2);
                *x1 = u * c - v * s;
                *x2 = v * c + u * s;
            }
        }
    }
}

struct Layer {
    attn_norm: Norm,
    qkv: Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    out: Linear,
    ffn_norm: Norm,
    mlp: Mlp,
}

pub(crate) struct Model {
    pub(crate) cfg: Config,
    /// `token_embd` as stored: one row of `Q1_0` blocks per token.
    embd: Vec<u8>,
    layers: Vec<Layer>,
    norm: Norm,
    head: Linear,
    rope: Rope,
}

fn f32s(g: &Gguf, name: &str) -> io::Result<Vec<f32>> {
    let (t, b) = g.tensor(name)?;
    if t.ty != gguf::F32 {
        return Err(io::Error::other(format!("{name}: expected F32")));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// The named `Q1_0` tensors' rows stacked (`OutIn`, rows in order) as one
/// weight set.
fn q1(g: &Gguf, names: &[&str]) -> io::Result<Q1Weights> {
    let (mut signs, mut scales, mut n, mut k) = (Vec::new(), Vec::new(), 0, None);
    for name in names {
        let (t, b) = g.tensor(name)?;
        if t.ty != gguf::Q1_0 || t.dims.len() != 2 {
            return Err(io::Error::other(format!(
                "{name}: expected a 2-D Q1_0 tensor"
            )));
        }
        if *k.get_or_insert(t.dims[0]) != t.dims[0] {
            return Err(io::Error::other(format!("{name}: input width differs")));
        }
        n += t.dims[1];
        for blk in b.chunks_exact(BLOCK_BYTES) {
            scales.push(f16::from_le_bytes([blk[0], blk[1]]));
            signs.extend_from_slice(&blk[2..]);
        }
    }
    let k = k.unwrap_or(0);
    Ok(Q1Weights::new(
        &signs,
        &scales,
        WeightLayout::OutIn,
        n,
        k,
        QK,
    ))
}

impl Model {
    pub(crate) fn load(g: &Gguf) -> io::Result<Self> {
        let arch = g.str("general.architecture").unwrap_or("qwen3").to_string();
        let key = |k: &str| format!("{arch}.{k}");
        let d = g.int(&key("embedding_length"))?;
        let heads = g.int(&key("attention.head_count"))?;
        let head_dim = g.int(&key("attention.key_length")).unwrap_or(d / heads);
        let (embd_t, embd) = g.tensor("token_embd.weight")?;
        let cfg = Config {
            d,
            layers: g.int(&key("block_count"))?,
            heads,
            kv_heads: g.int(&key("attention.head_count_kv"))?,
            head_dim,
            vocab: embd_t.dims[1],
            eps: g.float(&key("attention.layer_norm_rms_epsilon"))? as f32,
        };
        let rope = Rope::new(g, &arch, head_dim)?;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let t = |s: &str| format!("blk.{i}.{s}.weight");
            layers.push(Layer {
                attn_norm: Norm::rms(f32s(g, &t("attn_norm"))?, cfg.eps),
                qkv: Linear::from_q1(q1(g, &[&t("attn_q"), &t("attn_k"), &t("attn_v")])?),
                q_norm: f32s(g, &t("attn_q_norm"))?,
                k_norm: f32s(g, &t("attn_k_norm"))?,
                out: Linear::from_q1(q1(g, &[&t("attn_output")])?),
                ffn_norm: Norm::rms(f32s(g, &t("ffn_norm"))?, cfg.eps),
                mlp: Mlp::gated(
                    GatedLinear::from_q1(
                        &q1(g, &[&t("ffn_gate")])?,
                        &q1(g, &[&t("ffn_up")])?,
                        Gate::Silu,
                    ),
                    Linear::from_q1(q1(g, &[&t("ffn_down")])?),
                ),
            });
        }
        // Qwen3-1.7B ties the head to the embedding; larger ones store it.
        let head = if g.tensors.contains_key("output.weight") {
            q1(g, &["output.weight"])?
        } else {
            q1(g, &["token_embd.weight"])?
        };
        Ok(Self {
            embd: embd.to_vec(),
            layers,
            norm: Norm::rms(f32s(g, "output_norm.weight")?, cfg.eps),
            head: Linear::from_q1(head),
            rope,
            cfg,
        })
    }

    /// Token `t`'s embedding row, dequantized.
    fn embed(&self, t: u32, out: &mut [f32]) {
        let per_row = self.cfg.d / QK * BLOCK_BYTES;
        let row = &self.embd[t as usize * per_row..][..per_row];
        for (blk, o) in row.chunks_exact(BLOCK_BYTES).zip(out.chunks_exact_mut(QK)) {
            let s = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            for (i, v) in o.iter_mut().enumerate() {
                *v = if blk[2 + i / 8] >> (i % 8) & 1 == 1 {
                    s
                } else {
                    -s
                };
            }
        }
    }
}

/// A generation in progress: the KV caches, scratch, and the pool that
/// splits attention across cores and keeps the SME unit awake.
pub(crate) struct Session<'m> {
    m: &'m Model,
    caches: Vec<KvCache>,
    x: Vec<f32>,
    h: Vec<f16>,
    qkv: Vec<f16>,
    q: Vec<f32>,
    kv: (Vec<f32>, Vec<f16>),
    angles: Vec<Vec<(f32, f32)>>,
    att: Vec<f32>,
    last: Vec<f16>,
    /// Seconds per stage (`STAGES`), when profiling.
    pub(crate) prof: Option<[f64; STAGES.len()]>,
    _pool: HotPool,
}

pub(crate) const STAGES: [&str; 9] = [
    "attn norm",
    "qkv",
    "qk-norm+rope",
    "kv append",
    "attention",
    "out proj",
    "ffn norm",
    "mlp",
    "norm+head",
];

impl<'m> Session<'m> {
    pub(crate) fn new(m: &'m Model, capacity: usize) -> Self {
        let c = &m.cfg;
        Self {
            m,
            caches: (0..c.layers)
                .map(|_| KvCache::new(c.heads, c.kv_heads, c.head_dim, capacity))
                .collect(),
            x: Vec::new(),
            h: Vec::new(),
            qkv: Vec::new(),
            q: Vec::new(),
            kv: (Vec::new(), Vec::new()),
            angles: Vec::new(),
            att: Vec::new(),
            prof: None,
            last: vec![f16::ZERO; c.d],
            _pool: HotPool::new(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.caches[0].len()
    }

    /// Runs `tokens` at the next positions and writes the last one's logits.
    pub(crate) fn forward(&mut self, tokens: &[u32], logits: &mut [f32]) {
        let (m, c) = (self.m, &self.m.cfg);
        let (r, d) = (tokens.len(), c.d);
        let (qw, kw) = (c.heads * c.head_dim, c.kv_heads * c.head_dim);
        let pos0 = self.len();
        self.x.resize(r * d, 0.0);
        for (t, row) in tokens.iter().zip(self.x.chunks_exact_mut(d)) {
            m.embed(*t, row);
        }
        self.h.resize(r * d, f16::ZERO);
        self.qkv.resize(r * (qw + 2 * kw), f16::ZERO);
        self.q.resize(r * qw, 0.0);
        self.kv.0.resize(r * kw, 0.0);
        self.kv.1.resize(r * kw, f16::ZERO);
        self.att.resize(r * qw, 0.0);
        self.angles.resize_with(r, Vec::new);
        for (i, cs) in self.angles.iter_mut().enumerate() {
            m.rope.angles(pos0 + i, cs);
        }
        let mut t = Instant::now();
        let mut lap = |prof: &mut Option<[f64; STAGES.len()]>, stage: usize| {
            if let Some(p) = prof {
                let now = Instant::now();
                p[stage] += (now - t).as_secs_f64();
                t = now;
            }
        };
        for (l, cache) in m.layers.iter().zip(&mut self.caches) {
            l.attn_norm.forward(&self.x, &mut self.h);
            lap(&mut self.prof, 0);
            l.qkv.forward(&self.h, &mut self.qkv, r);
            lap(&mut self.prof, 1);
            for (i, row) in self.qkv.chunks_exact(qw + 2 * kw).enumerate() {
                let (q, k) = (&mut self.q[i * qw..][..qw], &mut self.kv.0[i * kw..][..kw]);
                nn::rms_norm(&row[..qw], &l.q_norm, c.eps, q);
                nn::rms_norm(&row[qw..qw + kw], &l.k_norm, c.eps, k);
                Rope::apply(q, &self.angles[i]);
                Rope::apply(k, &self.angles[i]);
                self.kv.1[i * kw..][..kw].copy_from_slice(&row[qw + kw..]);
            }
            lap(&mut self.prof, 2);
            cache.extend(&self.kv.0, &self.kv.1);
            lap(&mut self.prof, 3);
            if r == 1 {
                cache.attend(&self.q, &mut self.att);
            } else {
                cache.attend_causal(&self.q, r, &mut self.att);
            }
            lap(&mut self.prof, 4);
            l.out.accumulate(&self.att, &mut self.x, r);
            lap(&mut self.prof, 5);
            l.ffn_norm.forward(&self.x, &mut self.h);
            lap(&mut self.prof, 6);
            l.mlp.accumulate(&self.h, &mut self.x, r);
            lap(&mut self.prof, 7);
        }
        m.norm.forward(&self.x[(r - 1) * d..], &mut self.last);
        m.head.forward_f32(&self.last, logits, 1);
        lap(&mut self.prof, 8);
    }
}
