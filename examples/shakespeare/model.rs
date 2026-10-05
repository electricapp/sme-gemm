//! nanoGPT shakespeare-char (6 layers, 384 wide, 6 heads of 64, 65-character
//! vocabulary, 256-token context) composed from sme-gemm's model-level pieces:
//! a [`Block`] per layer (layer norms, [`SelfAttention`] and [`Mlp`] over
//! 4-bit [`Linear`]s made from the checkpoint's `[out][in]` weights), a
//! [`KvCache`] per layer, and a [`HotPool`] that keeps the SME unit awake and
//! runs attention and the MLP's gelu on spare cores beside the matmuls.
//! Embeddings and sampling are the only model code left here.

use std::collections::HashMap;
use std::path::Path;

use safetensors::{Dtype, SafeTensors};
use sme_gemm::nn::Norm;
use sme_gemm::{Block, Gate, HotPool, KvCache, Linear, Mlp, Q4Params, SelfAttention, WeightLayout};

pub(crate) const D: usize = 384;
pub(crate) const NL: usize = 6;
pub(crate) const NH: usize = 6;
pub(crate) const HD: usize = 64;
pub(crate) const V: usize = 65;
pub(crate) const CTX: usize = 256;
pub(crate) const FF: usize = 4 * D;
const EPS: f32 = 1e-5;

/// The model's characters, in token order.
pub(crate) const VOCAB: &str = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// The checkpoint's tensors by name, as f32 with their shapes (torch layout,
/// `[out][in]` for linear weights), from the `model.safetensors` that
/// `convert.py` writes.
pub(crate) struct Checkpoint(HashMap<String, (Vec<usize>, Vec<f32>)>);

impl Checkpoint {
    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        let at = |e: &dyn std::fmt::Display| format!("{}: {e}", path.display());
        let bytes = std::fs::read(path).map_err(|e| at(&e))?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| at(&e))?;
        st.tensors()
            .into_iter()
            .map(|(name, t)| {
                if t.dtype() != Dtype::F32 {
                    return Err(at(&format!("{name} is {:?}, not F32", t.dtype())));
                }
                let data = t.data().chunks_exact(4);
                let data = data.map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                Ok((name, (t.shape().to_vec(), data.collect())))
            })
            .collect::<Result<_, _>>()
            .map(Self)
    }

    /// The tensor `name`'s values.
    pub(crate) fn get(&self, name: &str) -> Result<&[f32], String> {
        self.0
            .get(name)
            .map(|(_, v)| &v[..])
            .ok_or_else(|| format!("the checkpoint has no {name}"))
    }

    /// The matrix `name`: its values, rows and columns.
    pub(crate) fn matrix(&self, name: &str) -> Result<(&[f32], usize, usize), String> {
        match self.0.get(name) {
            Some((s, v)) if s.len() == 2 => Ok((v, s[0], s[1])),
            Some((s, _)) => Err(format!("{name} is {s:?}, not a matrix")),
            None => Err(format!("the checkpoint has no {name}")),
        }
    }
}

/// The model: projections 4-bit, the tied output head f16, embeddings and norms
/// f32.
pub(crate) struct Model {
    wte: Vec<f32>,
    wpe: Vec<f32>,
    blocks: Vec<Block>,
    lnf: Norm,
    head: Linear,
}

impl Model {
    /// Builds the model from the checkpoint, every projection quantized to Q4
    /// with `q4_block`-deep K-blocks (32 = `Q4_0`), with the f16 panels prompts
    /// run on built now rather than on the first prompt (~8 ms for all 24).
    pub(crate) fn load(ck: &Checkpoint, q4_block: usize) -> Result<Self, String> {
        let linear = |name: &str| -> Result<Linear, String> {
            let (w, rows, cols) = ck.matrix(name)?;
            let l =
                Linear::quantize_with(w, WeightLayout::OutIn, rows, cols, Q4Params::new(q4_block));
            l.build_panel();
            Ok(l)
        };
        let norm = |name: &str| Ok::<_, String>(Norm::layer(ck.get(name)?.to_vec(), None, EPS));
        let blocks = (0..NL)
            .map(|i| {
                let p = |s: &str| format!("transformer.h.{i}.{s}");
                Ok(Block::new(
                    norm(&p("ln_1.weight"))?,
                    SelfAttention::new(
                        linear(&p("attn.c_attn.weight"))?,
                        linear(&p("attn.c_proj.weight"))?,
                        NH,
                        NH,
                        HD,
                    ),
                    norm(&p("ln_2.weight"))?,
                    Mlp::new(
                        linear(&p("mlp.c_fc.weight"))?,
                        Gate::Gelu,
                        linear(&p("mlp.c_proj.weight"))?,
                    ),
                ))
            })
            .collect::<Result<_, String>>()?;
        let (wte, v, d) = ck.matrix("transformer.wte.weight")?;
        Ok(Self {
            wte: wte.to_vec(),
            wpe: ck.get("transformer.wpe.weight")?.to_vec(),
            blocks,
            lnf: norm("transformer.ln_f.weight")?,
            head: Linear::f16(wte, WeightLayout::OutIn, v, d),
        })
    }
}

/// Character <-> token.
pub(crate) fn encode(c: char) -> Option<usize> {
    VOCAB.chars().position(|x| x == c)
}
pub(crate) fn decode(t: usize) -> char {
    VOCAB.chars().nth(t).unwrap_or('?')
}

/// How to pick the next character from the logits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sampler {
    /// 0 picks the most likely character every time.
    pub(crate) temperature: f32,
    /// Keep only the k most likely characters (0 = all 65).
    pub(crate) top_k: usize,
    pub(crate) seed: u64,
}

pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub(crate) fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub(crate) fn sample(logits: &[f32], s: &Sampler, rng: &mut Rng) -> usize {
    let argmax = (0..V)
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap_or(0);
    if s.temperature <= 0.0 {
        return argmax;
    }
    let cut = if s.top_k > 0 && s.top_k < V {
        let mut sorted: Vec<f32> = logits[..V].to_vec();
        sorted.sort_by(|a, b| b.total_cmp(a));
        sorted[s.top_k - 1]
    } else {
        f32::NEG_INFINITY
    };
    let mx = logits[argmax];
    let mut p = [0f32; V];
    let mut sum = 0.0;
    for c in 0..V {
        if logits[c] >= cut {
            p[c] = ((logits[c] - mx) / s.temperature).exp();
            sum += p[c];
        }
    }
    let mut u = rng.next_f32() * sum;
    for (c, pc) in p.iter().enumerate() {
        u -= pc;
        if u <= 0.0 && *pc > 0.0 {
            return c;
        }
    }
    argmax
}

/// The stages `--profile` times, in `Session::forward` order.
pub(crate) const STAGES: [&str; 3] = ["embed", "blocks", "ln_f + head"];

/// Generation state: a KV cache per layer, the residual rows, and the pool.
pub(crate) struct Session<'m> {
    /// Seconds per stage, accumulated while `Some` (`--profile`).
    pub(crate) prof: Option<[f64; 3]>,
    m: &'m Model,
    caches: Vec<KvCache>,
    // [rows][D], sized for a full-context batch.
    x: Vec<f32>,
    last: Vec<f32>,
    pub(crate) logits: Vec<f32>,
    _pool: HotPool,
}

impl<'m> Session<'m> {
    pub(crate) fn new(m: &'m Model) -> Self {
        Self {
            prof: None,
            m,
            caches: m.blocks.iter().map(|b| b.cache(CTX)).collect(),
            x: vec![0.0; CTX * D],
            last: vec![0.0; D],
            logits: vec![0.0; V],
            // NO_POOL=1 runs everything on the calling thread (still keeping
            // the SME unit awake).
            _pool: if std::env::var_os("NO_POOL").is_some() {
                HotPool::with_workers(0)
            } else {
                HotPool::new()
            },
        }
    }

    /// Positions already run.
    pub(crate) fn pos(&self) -> usize {
        self.caches[0].len()
    }

    pub(crate) fn reset(&mut self) {
        self.caches.iter_mut().for_each(KvCache::clear);
    }

    /// Runs `toks` at the next positions (which must fit in the context) and
    /// leaves the logits for the last one in `self.logits`. One token is a
    /// generation step; several run as one batch per matmul.
    pub(crate) fn forward(&mut self, toks: &[usize]) {
        let (r, pos) = (toks.len(), self.pos());
        assert!(
            r > 0 && pos + r <= CTX,
            "forward: {r} tokens at {pos} overflow the context"
        );
        let m = self.m;
        let mut t0 = std::time::Instant::now();
        let prof = &mut self.prof;
        let mut lap = |i: usize| {
            if let Some(p) = prof.as_mut() {
                let now = std::time::Instant::now();
                p[i] += (now - t0).as_secs_f64();
                t0 = now;
            }
        };
        for (i, &t) in toks.iter().enumerate() {
            let x = &mut self.x[i * D..(i + 1) * D];
            let (te, pe) = (&m.wte[t * D..][..D], &m.wpe[(pos + i) * D..][..D]);
            for ((x, &a), &b) in x.iter_mut().zip(te).zip(pe) {
                *x = a + b;
            }
        }
        lap(0);
        let x = &mut self.x[..r * D];
        for (b, c) in m.blocks.iter().zip(&mut self.caches) {
            b.accumulate(x, c, r);
        }
        lap(1);
        m.lnf.forward(&x[(r - 1) * D..], &mut self.last);
        m.head.forward_f32(&self.last, &mut self.logits, 1);
        lap(2);
    }
}
