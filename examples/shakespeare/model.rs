//! nanoGPT shakespeare-char (6 layers, 384 wide, 6 heads of 64, 65-character
//! vocabulary, 256-token context) composed from sme-gemm's model-level pieces:
//! a [`Linear`] per projection (4-bit, from the checkpoint's `[out][in]`
//! weights), a [`KvCache`] per layer, [`nn::layer_norm`], an [`SmeWarm`]
//! keeping the SME unit awake across the work between calls, and a [`HotPool`]
//! splitting attention's heads across two more cores. Embeddings and sampling
//! are the only model code left here.

use half::f16;
use sme_gemm::{Epilogue, HotPool, KvCache, Linear, Q4Params, SmeWarm, WeightLayout, nn};

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

/// Checkpoint tensors as f32, torch layout (`[out][in]` for linear weights).
pub(crate) struct F32Weights {
    pub(crate) wte: Vec<f32>,
    pub(crate) wpe: Vec<f32>,
    pub(crate) lnf: Vec<f32>,
    pub(crate) layers: Vec<F32Layer>,
}

pub(crate) struct F32Layer {
    pub(crate) ln1: Vec<f32>,
    pub(crate) ln2: Vec<f32>,
    pub(crate) attn: Vec<f32>,
    pub(crate) proj: Vec<f32>,
    pub(crate) fc: Vec<f32>,
    pub(crate) fcp: Vec<f32>,
}

impl F32Weights {
    /// Reads `weights.f32` + `index.txt` (written by `convert.py`) from `dir`.
    pub(crate) fn load(dir: &std::path::Path) -> Result<Self, String> {
        let raw = std::fs::read(dir.join("weights.f32"))
            .map_err(|e| format!("{}: {e}", dir.join("weights.f32").display()))?;
        let all: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let idx = std::fs::read_to_string(dir.join("index.txt"))
            .map_err(|e| format!("{}: {e}", dir.join("index.txt").display()))?;
        let get = |name: &str| -> Result<Vec<f32>, String> {
            let line = idx
                .lines()
                .find(|l| l.split(' ').next() == Some(name))
                .ok_or_else(|| format!("index.txt has no {name}"))?;
            let f: Vec<usize> = line
                .split(' ')
                .skip(1)
                .filter_map(|x| x.parse().ok())
                .collect();
            let n: usize = f[1..].iter().product();
            all.get(f[0]..f[0] + n)
                .map(<[f32]>::to_vec)
                .ok_or_else(|| format!("{name} is truncated"))
        };
        let layers = (0..NL)
            .map(|i| {
                let p = |s: &str| get(&format!("transformer.h.{i}.{s}"));
                Ok(F32Layer {
                    ln1: p("ln_1.weight")?,
                    ln2: p("ln_2.weight")?,
                    attn: p("attn.c_attn.weight")?,
                    proj: p("attn.c_proj.weight")?,
                    fc: p("mlp.c_fc.weight")?,
                    fcp: p("mlp.c_proj.weight")?,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            wte: get("transformer.wte.weight")?,
            wpe: get("transformer.wpe.weight")?,
            lnf: get("transformer.ln_f.weight")?,
            layers,
        })
    }
}

struct Layer {
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    attn: Linear,
    proj: Linear,
    fc: Linear,
    fcp: Linear,
}

/// The model: projections 4-bit, the tied output head f16, embeddings and norms
/// f32.
pub(crate) struct Model {
    wte: Vec<f32>,
    wpe: Vec<f32>,
    lnf: Vec<f32>,
    head: Linear,
    layers: Vec<Layer>,
}

impl Model {
    /// Quantizes every projection to Q4 with `block`-deep K-blocks (32 = `Q4_0`),
    /// and builds the f16 panels prompts run on now rather than on the first
    /// prompt (~8 ms for all 24).
    pub(crate) fn quantize(w: &F32Weights, block: usize) -> Self {
        let q = |m: &[f32], n: usize, k: usize| {
            let l = Linear::quantize_with(m, WeightLayout::OutIn, n, k, Q4Params::new(block));
            l.build_panel();
            l
        };
        let layers = w
            .layers
            .iter()
            .map(|l| Layer {
                ln1: l.ln1.clone(),
                ln2: l.ln2.clone(),
                attn: q(&l.attn, 3 * D, D),
                proj: q(&l.proj, D, D),
                fc: q(&l.fc, FF, D),
                fcp: q(&l.fcp, D, FF),
            })
            .collect();
        Self {
            wte: w.wte.clone(),
            wpe: w.wpe.clone(),
            lnf: w.lnf.clone(),
            head: Linear::f16(&w.wte, WeightLayout::OutIn, V, D),
            layers,
        }
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

/// Generation state: a KV cache per layer, row scratch, and the SME guard.
pub(crate) struct Session<'m> {
    m: &'m Model,
    caches: Vec<KvCache>,
    // [rows][...] scratch, sized for a full-context batch.
    x: Vec<f32>,
    h16: Vec<f16>,
    qkv: Vec<f16>,
    q: Vec<f16>,
    y: Vec<f32>,
    ff: Vec<f16>,
    last: Vec<f32>,
    pub(crate) logits: Vec<f32>,
    gelu: Epilogue<'static, f16>,
    _warm: SmeWarm,
    _pool: Option<HotPool>,
}

impl<'m> Session<'m> {
    pub(crate) fn new(m: &'m Model) -> Self {
        Self {
            m,
            caches: (0..NL).map(|_| KvCache::new(NH, NH, HD, CTX)).collect(),
            x: vec![0.0; CTX * D],
            h16: vec![f16::ZERO; CTX * D],
            qkv: vec![f16::ZERO; CTX * 3 * D],
            q: vec![f16::ZERO; CTX * D],
            y: vec![0.0; CTX * D],
            ff: vec![f16::ZERO; CTX * FF],
            last: vec![0.0; D],
            logits: vec![0.0; V],
            gelu: Epilogue::new().gelu(),
            _warm: SmeWarm::new(),
            // NO_POOL=1 runs attention on the calling thread alone.
            _pool: std::env::var_os("NO_POOL")
                .is_none()
                .then(|| HotPool::new(2)),
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
        for (i, &t) in toks.iter().enumerate() {
            let x = &mut self.x[i * D..(i + 1) * D];
            for (c, x) in x.iter_mut().enumerate() {
                *x = m.wte[t * D + c] + m.wpe[(pos + i) * D + c];
            }
        }
        let (x, h16) = (&mut self.x[..r * D], &mut self.h16[..r * D]);
        for (ly, cache) in m.layers.iter().zip(&mut self.caches) {
            // attention
            nn::layer_norm(&*x, &ly.ln1, None, EPS, &mut *h16);
            ly.attn.forward(&*h16, &mut self.qkv[..r * 3 * D], r);
            for (row, q) in self.qkv[..r * 3 * D]
                .chunks_exact(3 * D)
                .zip(self.q.chunks_exact_mut(D))
            {
                q.copy_from_slice(&row[..D]);
                cache.push(&row[D..2 * D], &row[2 * D..]);
            }
            if r == 1 {
                cache.attend(&self.q[..D], &mut self.y[..D]);
            } else {
                cache.attend_causal(&self.q[..r * D], r, &mut self.y[..r * D]);
            }
            ly.proj.accumulate(&self.y[..r * D], &mut *x, r);
            // MLP
            nn::layer_norm(&*x, &ly.ln2, None, EPS, &mut *h16);
            ly.fc
                .forward_ep(&*h16, &mut self.ff[..r * FF], r, &self.gelu);
            ly.fcp.accumulate(&self.ff[..r * FF], &mut *x, r);
        }
        nn::layer_norm(&x[(r - 1) * D..], &m.lnf, None, EPS, &mut self.last);
        m.head.forward_f32(&self.last, &mut self.logits, 1);
    }
}
