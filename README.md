# sme-gemm

Apple-Silicon **SME** GEMM for Rust. Hand-written against `arm_sme.h`, SVL=512,
dispatched across both P- and E-cluster SME units via GCD.

## SME on Apple Silicon

SVE is length-agnostic SIMD. SME adds **ZA** (a 2-D accumulator) and `*MOPA`
outer products. Compute runs only in [streaming mode][streaming] (`SMSTART` /
`SMSTOP`); SVL is 512 bits on M4/M5. The transition is fixed-cost, so small
problems skip it.

ZA is 64×64 bytes (4 KB). Tile shape follows the element: 64×64 i8, two 32×32
f16/bf16/i16, four 16×16 f32/i32, eight 8×8 f64/i64. Apple shares **one SME unit
per cluster**, not per core:

```text
        P-cluster                       E-cluster            (Apple M5)
    ┌───┬───┬───┬───┐          ┌───┬───┬───┬───┬───┬───┐
    │ P │ P │ P │ P │          │ E │ E │ E │ E │ E │ E │      cores
    └─┬─┴─┬─┴─┬─┴─┬─┘          └─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┘
      └───┴─┬─┴───┘              └───┴───┴─┬─┴───┴───┘
       ┌────┴────┐                    ┌────┴────┐
       │   SME   │                    │   SME   │             one unit per
       │ ZA 4 KB │                    │ ZA 4 KB │             cluster, shared
       └─────────┘                    └─────────┘
```

A MOPA is a rank-1 update — 1024 MACs/instr for f16. Widening f16→f32 takes two
K-steps into a 16×16 f32 tile (512 MACs, ~½ the rate). i8→i32 takes four (1024,
matching f16).

```text
     Zm →   b0   b1   b2   ..  b31      one streaming-SVE vector
             │    │    │        │       (32 lanes of f16 at SVL=512)
           ┌─┴────┴────┴────────┴─┐
  Zn  a0 ─►│  ·    ·    ·       · │
      a1 ─►│  ·    ·    ·       · │     ZA[i][j] += a[i] * b[j]
      ..   │                      │
     a31 ─►│  ·    ·    ·       · │
           └──────────────────────┘
              ZA0.H — 32×32 f16

  GEMM tile:  for k in 0..K:  ZA += A[:,k] ⊗ B[k,:] ;  then store ZA → C
```

[sve] · [sme] · [streaming] · [acle] · [spec]

[sve]: https://developer.arm.com/documentation/102476/latest/
[sme]: https://developer.arm.com/documentation/109246/0101/
[streaming]:
  https://developer.arm.com/documentation/109246/0101/SME-Overview/Streaming-SVE-mode
[acle]: https://arm-software.github.io/acle/main/acle.html
[spec]: https://developer.arm.com/documentation/ddi0616/latest/

## Kernels

| dtype       | accumulate                      | output   | min CPU |
| ----------- | ------------------------------- | -------- | ------- |
| f16         | f32 (FMOPA) / f16 (`F16F16`)    | f16      | M4 / M5 |
| bf16        | f32 (BFMOPA) / bf16 (`B16B16`)  | bf16     | M4 / M5 |
| f32         | f32                             | f32      | M4      |
| **f64**     | f64                             | f64      | M5      |
| i8→i32      | i32 (SMOPA)                     | i32      | M4      |
| **i16→i64** | i64 (SMOPA)                     | i64      | M5      |
| **Q4**      | 4-bit block-quant → f16 or bf16 | f16/bf16 | M5      |

- **`Accum`** picks the compute type on half GEMMs: `Accum::F32` (widening, M4+,
  default), `Accum::F16` (`FEAT_SME_F16F16`, M5), `Accum::Bf16`
  (`FEAT_SME_B16B16`, M5). Missing the feature, or a mismatched 16-bit choice,
  falls back to `F32`. Native 16-bit is ~2× the widening rate; error grows ~√K.
- **Fused epilogue** — in-register op-graph at the store, any composition:
  `Gemm::new(&a, &w, m).mul_col(&s).add_col(&b).silu().clamp(0., 6.).run(&mut c)`.
  Elementwise `add`/`sub`/`mul`/`div`/`max`/`min` (scalar, per-row, per-col, or
  M×N); activations relu, relu6, leaky_relu, gelu, gelu_exact, silu, sigmoid,
  tanh, softplus, mish, elu, selu, hardswish, hardsigmoid, softsign; unary abs,
  neg, sign, square, sqrt, rsqrt, recip, exp, log; clamp. Full op parity across
  f16/bf16/f32/f64 and i8/i16→f32 dequant (bf16 runs the graph in f32, then
  rounds). On the 16-bit paths a trailing gelu/silu/sigmoid/tanh runs as a NEON
  pass after the kernel, since streaming mode is slow at it (ARCHITECTURE.md
  §2.5). A leading per-column bias folds into the accumulator. `.beta(b)`
  scales the product; `.col_major_output()` stores C column-major.
  `epilogue_map` is the non-fused escape hatch.
- **Strided GEMM** — `gemm_{f16,bf16,f32,f64,i8,i16}`: `C = αC + β(A @ B)` with
  per-matrix row/col strides. Strides encode transpose, column-major, and
  accumulate-into-C.
- **i8/i16 → f32 dequant** — `Dequant::new(scale)` or `.scale_per_n(&scales)`,
  then any epilogue (`matmul_i8_packed_dequant` / `matmul_i16_dequant`).
- **Batched** — many small `C_i = A_i @ B_i` in one streaming session
  (`matmul_*_batched` / `_ep`).
- **Pre-packed weights** — `prepack_*` once, reuse. Calls that skip it pack B
  inside the parallel GEMM (or, for m ≤ 4, read it in place) and land within a
  few percent of the pre-packed rate on large shapes.
- **Model building blocks** — the pieces between "a GEMM" and "a model":
  - `Linear`: a layer's weights, 4-bit (`Linear::quantize`, from f32) or f16,
    from either `WeightLayout` (`OutIn` is PyTorch's `[out][in]`), with an
    optional bias. It takes f32 or f16 rows, writes f16, f32, or adds into an f32
    residual stream (`accumulate`), and picks the kernel by row count: the Q4
    GEMV under 8 rows, an f16 panel built once on first use from 8.
  - `GatedLinear`: the gate and up projections of a SwiGLU / GeGLU MLP as one
    matmul (interleaved 32 columns at a time) plus one NEON `act(g)·u` pass.
  - An input RMSNorm folds into the layer after it: `WeightLayout::scale_inputs`
    multiplies the norm's weight into W once, and `.rms_norm_input(eps)` applies
    the per-row `1/rms(x)` as the epilogue's first step, so no normalized copy of
    `x` is written.
  - `Q4Weights::quantize` (`Q4_0` / `Q4_1`, any block) and `prepack(w, layout,
    n, k)` for every dtype, so weights go in as they are stored.
  - `KvCache`: an f16 key/value cache with `push` / `attend` / `attend_causal`
    (grouped-query aware; NEON, since one row has no GEMM in it), over
    `attention_kv_f16` / `attention_kv_causal_f16`.
  - `nn::layer_norm` / `nn::rms_norm`, writing f16 straight for the next matmul.
  - `SmeWarm` keeps the SME unit awake between calls: it idles within a few
    hundred ns, and a loop alternating short SME calls with NEON work otherwise
    pays ~1 µs per call to wake it. `SME_GEMM_TRACE=1` prints every SME call
    with its shape, time and the idle gap before it, flagging those that paid
    the wake.
  - `HotPool::new(workers)` keeps worker threads spinning so short NEON passes
    can split across cores without GCD's microseconds of wake-up: `KvCache`
    attention splits its heads across them from 64 cached keys (2x at 128 keys,
    2.9x at 1024 on M5).
- **candle** / **burn** adapters (optional features).
- Scalar reference fallback off Apple.

## Benchmarks (Apple M5, best-of-N)

TF/s (10¹² FLOP/s); integer rows are TOPS. Same `2·M·N·K` throughout. Every
number is from `examples/` on this machine. Cross-backend runs interleave, one
timed round each per pass.

**sme-gemm** (`examples/bench`). Pre-packed B except `/f32`, which has no packed
entry point. Native 16-bit is the non-widening MOPA; `/f32` is widening (~½ the
MACs/instr):

| dtype    | 256³ | 512³ | 1024³ | 2048³ | 4096³ | 4096×512×512 | 16384×512×512 | 1×4096×4096 ¹ |
| -------- | ---- | ---- | ----- | ----- | ----- | ------------ | ------------- | ------------- |
| f16      | 2.51 | 4.10 | 4.76  | 4.93  | 4.77  | 4.78         | 4.83          | 0.16          |
| bf16     | 2.46 | 3.20 | 4.75  | 4.94  | 4.75  | 4.77         | 4.82          | 0.16          |
| f16/f32  | 1.30 | 1.82 | 2.34  | 2.45  | 2.48  | 2.31         | 2.39          | — ²           |
| bf16/f32 | 1.30 | 1.81 | 2.34  | 2.44  | 2.48  | 2.34         | 2.40          | — ²           |
| f32      | 1.45 | 2.00 | 2.37  | 2.31  | 2.29  | 2.39         | 2.40          | 0.06          |
| f64      | 0.41 | 0.58 | 0.60  | 0.56  | 0.58  | 0.61         | 0.61          | 0.03          |
| i8→i32   | 2.42 | 3.41 | 4.75  | 4.98  | 4.99  | 4.66         | 4.88          | 0.56          |
| i16→i64  | 1.02 | 2.06 | 2.41  | 2.48  | 2.43  | 2.38         | 2.32          | 0.14          |
| Q4→f16   | 1.80 | 3.63 | 4.65  | 4.83  | 4.93  | 4.65         | 4.79          | 0.50          |
| Q4→bf16  | 1.98 | 3.52 | 4.52  | 4.83  | 4.92  | 4.72         | 4.79          | 0.47          |

**vs. other backends**, unpacked strided GEMM
(`examples/vs_{accelerate,candle,burn}`, `inference_value`). Accel has no
f16/`cblas`; † is upcast + `sgemm` + downcast. B is packed inside the parallel
GEMM (ARCHITECTURE.md §2.10), so f16 here is within ~3% of pre-packed from
2048³. Accelerate leads only at 256³ f32 (see dispatch table: the call is ~20
µs, so streaming entry is a real slice):

| shape           | sme f32 | accel | candle | burn | sme f16 | accel † | candle | sme f64 | accel |
| --------------- | ------- | ----- | ------ | ---- | ------- | ------- | ------ | ------- | ----- |
| 256³            | 1.50    | 1.64  | 0.24   | 0.11 | 1.56    | —       | 0.31   | 0.46    | 0.45  |
| 512³            | 1.91    | 1.61  | 0.39   | 0.11 | 3.69    | 0.95    | 0.74   | 0.57    | 0.46  |
| 1024³           | 2.34    | 2.05  | 0.45   | 0.12 | 4.38    | 1.43    | 0.86   | 0.55    | 0.53  |
| 2048³           | 2.29    | 2.08  | 0.45   | 0.12 | 4.83    | 1.73    | 0.88   | 0.55    | 0.52  |
| 4096³           | 2.33    | 2.06  | 0.45   | —    | 4.88    | 1.85    | 0.86   | 0.55    | 0.52  |
| 4096×512×512    | 2.30    | 1.93  | 0.41   | 0.11 | 4.05    | —       | 0.81   | 0.61    | 0.52  |
| 16384×512×512   | 2.30    | 2.01  | 0.42   | —    | 4.22    | —       | 0.84   | 0.61    | 0.52  |
| 4096×11008×4096 | —       | —     | —      | —    | 4.88    | 1.91    | —      | —       | —     |

f16 sme at 256³ / 4096×512 / 16384×512 is the candle adapter (no accel† run
there); the other f16 sme cells are `inference_value`.

**Dispatch overhead** (`examples/dispatch --features dispatch-cmp`), µs/call,
f32. Submit + wait; compile, Metal buffer alloc, and SME `prepack_f32` are
outside the timer. P-core is Accelerate `cblas_sgemm` at `USER_INTERACTIVE` (may
use AMX). GPU is MPS, shared storage, timed through `waitUntilCompleted`. ANE is
MLCompute MatMul on `aneDevice` — the MatMul layer stayed on ANE for every shape
here:

|                | 8³   | 16³  | 32³  | 64³  | 128³ | 256³ | 1×64×64 | 1×256×256 | 1×1024×1024 | 1×4096×4096 |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- | ------- | --------- | ----------- | ----------- |
| P-core (cblas) | 0.05 | 0.19 | 0.27 | 0.49 | 3.3  | 20.2 | 0.05    | 0.35      | 4.5         | 804         |
| SME (packed)   | 0.37 | 0.42 | 0.61 | 1.8  | 7.6  | 24.8 | 1.1     | 1.3       | 9.9         | 523         |
| GPU (MPS)      | 195  | 194  | 191  | 203  | 189  | 202  | 194     | 196       | 225         | 751         |
| ANE (MLC)      | 59   | 77   | 61   | 57   | 66   | 66   | 77      | 61        | 169         | 1685        |

CPU/SME dispatch is sub-µs until the math shows up (~128³). GPU's floor is ~190
µs of command-buffer + `waitUntilCompleted`; ANE's is ~60–80 µs. At
`1×4096×4096` (one row, resident B) it is bandwidth: packed SME beats both
Accelerate and a round-trip MPS GEMM. At `1×1024×1024` B is cache-resident and
the packed f32 GEMV is bound by the one SME unit's FMLA rate (ARCHITECTURE.md
§2.8); a row-major B that was not pre-packed runs it in ~7 µs.

**Flash attention** (`examples/attention`, `m×n×d`). vs materialized is a
footprint win (`O(m·d + n·d + n·dv)`), not a speed win, until 4096²:

|          | 512²×64 | 1024²×64 | 2048²×64 | 4096²×64 | 1024²×128 | 4096²×128 | 8192²×64 | 1×4096×128 |
| -------- | ------- | -------- | -------- | -------- | --------- | --------- | -------- | ---------- |
| f32 ms   | 0.18    | 0.41     | 1.30     | 4.66     | 0.48      | 6.54      | 18.6     | 0.10       |
| matzd ms | 0.17    | 0.37     | 1.17     | 4.96     | 0.47      | 6.52      | 22.7     | 0.10       |
| vs matzd | 0.96×   | 0.89×    | 0.90×    | 1.06×    | 0.97×     | 1.00×     | 1.22×    | 1.00×      |
| f16 ms   | 0.14    | 0.29     | 0.82     | 2.82     | 0.35      | 3.89      | 11.9     | 0.12       |
| bf16 ms  | 0.15    | 0.33     | 0.82     | 3.12     | 0.38      | 3.95      | 13.0     | 0.13       |
| f16 TF/s | 0.47    | 0.94     | 1.30     | 1.52     | 1.54      | 2.21      | 1.44     | 0.02       |

|                       | 4096×4096×512 | 4096×4096×128 | 16384×512×512 | 8192×8192×256 |
| --------------------- | ------------- | ------------- | ------------- | ------------- |
| epilogue gemm ms      | 3.46          | 0.96          | 1.79          | 7.03          |
| +bias                 | +0%           | +2%           | +0%           | +0%           |
| +bias+relu            | +5%           | +27%          | +5%           | +8%           |
| separate bias+relu ms | 14.4          | 12.0          | 7.34          | 51.1          |

|                | m=1 ×4096² | m=16 | m=128 | m=256 | 1024³ |
| -------------- | ---------- | ---- | ----- | ----- | ----- |
| q4-resident ms | 0.07       | 0.47 | 1.10  | 1.99  | 0.47  |
| eager-f16 ms   | 0.26       | 0.35 | 0.94  | 1.78  | 0.46  |

Q4 holds a 4× smaller resident weight set. Up to `m=7` it runs the LUTI4 GEMV
and reads a quarter of eager's bytes (3× at `m=1`). Above that it unpacks each
weight block once into ZA-built panels, an `O(n·k)` cost whatever `m`: it trails
eager by 8–40% at `m=16..256`, where eager's 32 MB of f16 stays cache-resident
between calls, and matches it by 1024³.

|                  | 16³×256 | 32³×128 | 64³×64 | 32×128×64 ×64 | 96³×32 |
| ---------------- | ------- | ------- | ------ | ------------- | ------ |
| batched f16 µs   | 42      | 27      | 45     | 56            | 62     |
| loop of GEMMs µs | 180 \*  | 86 \*   | 93     | 106           | 102    |

`*`: loop side is under the 2¹⁸ flop floor (scalar reference), so that speedup
is the floor, not streaming-entry amortization.

|                      | 128×128×128 | 256×256×128 | 1024³ | 2048×2048×512 | 4096×512×512 |
| -------------------- | ----------- | ----------- | ----- | ------------- | ------------ |
| i8→f32 dequant TOPS  | 0.08        | 0.16        | 1.87  | 1.38          | 1.36         |
| i16→f32 dequant TOPS | 0.10        | 0.17        | 1.27  | 0.99          | 1.02         |

Fused i8/i16→f32 dequant + bias + gelu (`examples/dequant_bench`).

- ¹ m = 1: one row vs a resident weight set, bandwidth-bound. m ≤ 4 (Q4: m ≤ 7)
  runs a ZA-vector GEMV (SME2 multi-vector FMLA/SDOT, LUTI4 for Q4); Q4 at m = 1
  reads ~135 GB/s with the weights in L2 and ~80 GB/s from DRAM. The TF/s column
  understates that.
- ² No pre-packed entry point — the call would re-pack the whole weight set.
- † No native f16 GEMM; f16 inputs run through f32.

`examples/vs_ort` omitted: no `coreml+xnnpack` `ort` distro for this target. A
prior run had the three ORT EPs within 1% at 2048³ (~2.0 f32 / 1.8 f16) — the
`MatMul` node stays on CPU.

## Usage

```rust
use half::f16;
use sme_gemm::{
    matmul_f16, matmul_i8, matmul_i8_packed_dequant, gemm_f32,
    Accum, Dequant, Gemm, prepack_f16, prepack_i8,
};

let (m, n, k) = (256, 256, 256);
let a = vec![f16::from_f32(0.1); m * k];
let b = vec![f16::from_f32(0.2); k * n];
let mut c = vec![f16::ZERO; m * n];
matmul_f16(&a, &b, &mut c, m, n, k, Accum::F16); // f16 accumulate, M5
// matmul_f16(..., Accum::F32);                 // widening f32 accumulate, M4+

let w = prepack_f16(&b, n, k);
let bias = vec![f16::ZERO; n];
Gemm::new(&a, &w, m).add_col(&bias).gelu().run(&mut c);

let (ai, bi) = (vec![1i8; m * k], vec![2i8; k * n]);
let mut ci = vec![0i32; m * n];
matmul_i8(&ai, &bi, &mut ci, m, n, k);

let qw = prepack_i8(&bi, n, k);
let mut cf = vec![0f32; m * n];
matmul_i8_packed_dequant(&ai, &qw, &mut cf, m, &Dequant::new(0.02).relu());

// C = αC + β(Aᵀ @ B)
let af = vec![0.1f32; k * m];
let bf = vec![0.2f32; k * n];
gemm_f32(m, n, k, &mut cf, /*c_row*/ n, /*c_col*/ 1,
    &af, /*a_row*/ 1, /*a_col*/ m,
    &bf, /*b_row*/ n, /*b_col*/ 1, /*alpha*/ 1.0, /*beta*/ 1.0);
```

A transformer block from the building blocks (weights as PyTorch stores them):

```rust
use half::f16;
use sme_gemm::{Epilogue, KvCache, Linear, SmeWarm, WeightLayout, nn};

let (d, heads) = (384, 6);
let w_qkv = vec![0.01f32; 3 * d * d]; // [out][in]
let w_fc = vec![0.01f32; 4 * d * d];
let qkv = Linear::quantize(&w_qkv, WeightLayout::OutIn, 3 * d, d);
let proj = Linear::quantize(&vec![0.01; d * d], WeightLayout::OutIn, d, d);
let fc = Linear::quantize(&w_fc, WeightLayout::OutIn, 4 * d, d);
let fc2 = Linear::quantize(&vec![0.01; d * 4 * d], WeightLayout::OutIn, d, 4 * d);
let (ln1, ln2) = (vec![1.0f32; d], vec![1.0f32; d]);
let mut cache = KvCache::new(heads, heads, d / heads, 256);
let _warm = SmeWarm::new(); // keep the SME unit awake across the NEON work

let mut x = vec![0.1f32; d]; // one position's residual stream, f32
let (mut h, mut t) = (vec![f16::ZERO; d], vec![f16::ZERO; 3 * d]);
let (mut y, mut ff) = (vec![0.0f32; d], vec![f16::ZERO; 4 * d]);
nn::layer_norm(&x, &ln1, None, 1e-5, &mut h);
qkv.forward(&h, &mut t, 1);
cache.push(&t[d..2 * d], &t[2 * d..]);
cache.attend(&t[..d], &mut y);
proj.accumulate(&y, &mut x, 1); // x += y @ W
nn::layer_norm(&x, &ln2, None, 1e-5, &mut h);
fc.forward_ep(&h, &mut ff, 1, &Epilogue::new().gelu());
fc2.accumulate(&ff, &mut x, 1);
```

Also: `matmul_{bf16,f32,f64,i16}`, `gemm_*`, `matmul_*_batched`,
`matmul_i16_dequant`, `dequant_q4` / `matmul_q4`, optional
`sme_gemm::candle::sme_matmul`. Caps: `sme_gemm::caps()`.

**End to end.** `examples/shakespeare` runs nanoGPT shakespeare-char (10.7M
parameters) from these blocks alone (`Linear`, `KvCache`, `nn::layer_norm`,
`SmeWarm`, `HotPool`; the model file adds only embeddings and sampling) as a
CLI that continues a prompt. On an M5 it generates ~12k characters/s (~13.5k
at short context) with the validation loss of the f32 model:

```sh
examples/shakespeare/fetch.sh             # once: checkpoint + text, ~45 MB (curl, python3)
cargo run --release --example shakespeare -- "ROMEO:"
cargo run --release --example shakespeare -- --bench   # accuracy vs f32, speed
cargo install --path . --example shakespeare           # optional: `shakespeare` on PATH
```

`examples/sme_warm` shows the wake cost `SmeWarm` removes.

## Design

- **One SME unit per cluster.** Hot paths `dispatch_apply` M-tiles across both
  cluster units; small problems take a serial / direct-load path.
- **Streaming mode, SVL=512.** Compiled `-mcpu=apple-m4`. M5 features
  (`+sme-f16f16`, `+sme-b16b16`, `+sme-f64f64`, `+sme-i16i64`) are separate TUs
  behind a runtime probe, so the M4 floor stays.
- **No stable Rust SME intrinsics** — kernels are C (`cc` + `arm_sme.h`).

Full design: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## License

Dual-licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.

Unless stated otherwise, any contribution intentionally submitted for inclusion
in this crate, as defined in the Apache-2.0 license, is dual-licensed as above
with no additional terms or conditions.
