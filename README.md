# sme-gemm

Apple-Silicon-tuned **SME** (Scalable Matrix Extension) GEMM kernels for Rust.

Hand-written against `arm_sme.h`, tuned for Apple's per-cluster shared SME unit
(streaming-mode, SVL=512), dispatched across both the P- and E-cluster SME units
via GCD.

## SVE, SME, streaming mode

[SVE][sve] is Arm's vector-length-agnostic SIMD: one binary runs at any hardware
vector length, with predication in place of scalar tail loops. [SME][sme] builds
on it with a two-dimensional accumulator, **ZA**, and outer-product instructions
(`*MOPA`) that feed it. The C surface for both is [ACLE][acle]'s `arm_sve.h` and
`arm_sme.h`; the normative definition is the [architecture supplement][spec].

SME's compute instructions execute only in [streaming mode][streaming], entered
and left with `SMSTART` / `SMSTOP`. Inside it the SVE registers take the
_streaming_ vector length **SVL** (512 bits on Apple M4 and M5) and ZA is live;
outside it ZA does not exist and part of the SVE and NEON instruction space is
unavailable. The transition costs fixed time, so a kernel does as much work as
it can per region, and small problems are routed around it.

ZA is a 64×64-byte array (4 KB at SVL=512), addressed as tiles whose shape
follows the element size: one 64×64 i8 tile, two 32×32 f16/bf16/i16, four 16×16
f32/i32, eight 8×8 f64/i64. Apple gives each CPU _cluster_ one SME unit shared
by its cores, rather than one per core, so the parallelism on offer is two
units:

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

A MOPA reads two streaming-SVE vectors, forms their outer product, and
accumulates it into a ZA tile — 1024 multiply-accumulates in one instruction for
f16:

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

Widening forms take more of K per instruction into a wider tile: f16→f32 takes
two K-steps into a 16×16 f32 tile (512 MACs, hence about half the rate of the
f16→f16 form), i8→i32 takes four (1024, matching f16).

[sve]: https://developer.arm.com/documentation/102476/latest/
[sme]: https://developer.arm.com/documentation/109246/0101/
[streaming]:
  https://developer.arm.com/documentation/109246/0101/SME-Overview/Streaming-SVE-mode
[acle]: https://arm-software.github.io/acle/main/acle.html
[spec]: https://developer.arm.com/documentation/ddi0616/latest/

## Kernels

| dtype          | path                                  | output | min CPU |
| -------------- | ------------------------------------- | ------ | ------- |
| f16            | widening FMOPA / non-widening (Fast)  | f16    | M4 / M5 |
| bf16           | widening BFMOPA / non-widening (Fast) | bf16   | M4 / M5 |
| f32            | FMOPA single-precision                | f32    | M4      |
| **f64**        | FMOPA double-precision                | f64    | M5      |
| i8→i32         | SMOPA                                 | i32    | M4      |
| **i16→i64**    | SMOPA                                 | i64    | M5      |
| **Q4 weights** | 4-bit block-quant → f16 packed        | f16    | M5      |

- Fused arbitrary epilogue — an in-register op-graph evaluated at the store, any
  composition, zero extra passes. e.g.
  `Gemm::new(&a, &w, m).mul_col(&s).add_col(&b).silu().clamp(0.,6.).run(&mut c)`.
  - elementwise `add`/`sub`/`mul`/`div`/`max`/`min` by scalar, per-row, per-col,
    or full M×N tensor.
  - activations: relu, relu6, leaky_relu, gelu, gelu_exact, silu, sigmoid, tanh,
    softplus, mish, elu, selu, hardswish, hardsigmoid, softsign.
  - unary math: abs, neg, sign, square, sqrt, rsqrt, recip, exp, log; clamp.
  - Covers f16/bf16/f32/f64 and i8/i16→f32 dequant — full op parity across
    dtypes (bf16 evaluates the graph in f32, then rounds back, so it has no
    divide/transcendental restriction). A leading per-column bias folds into the
    accumulator (free). `epilogue_map` is the non-fused escape hatch for
    arbitrary closures.
  - `.beta(b)` scales the product (`D = act(b·(A@B) + bias)`);
    `.col_major_output()` stores `C` column-major natively (no transpose pass).
- Strided / BLAS-style GEMM — `gemm_{f16,bf16,f32,f64,i8,i16}` compute
  `C = alpha·C + beta·(A @ B)` with arbitrary per-matrix row/col strides (in
  elements). Strides encode transpose (swap row/col), column-major operands, and
  accumulate-into-`C` (`alpha != 0`) — a single-call analog of `cblas_*gemm`.
- Accuracy modes (f16) — `Accuracy::Accurate` (fp32 accumulate, M4+, default) vs
  `Accuracy::Fast` (fp16 accumulate, M5 `FEAT_SME_F16F16`, ~2× faster, error
  grows ~√K; falls back to `Accurate` without the extension).
- Quantized i8/i16 → f32 dequant — the `Dequant` builder dequantizes in-register
  at the store: per-tensor `Dequant::new(scale)` or per-output-channel
  `.scale_per_n(&scales)`, then any epilogue op-graph on top
  (`matmul_i8_packed_dequant` / `matmul_i16_dequant`).
- Batched GEMM (f16/bf16/f32/f64/i8/i16) — many small `C_i = A_i @ B_i` in one
  streaming session, with the same fused epilogue (`matmul_f16_batched` /
  `_ep`).
- Pre-packed weights — `prepack_*` once, reuse across calls.
- candle backend — optional `candle` feature: a `CustomOp2` for CPU f32/f16/bf16
  tensors.
- burn adapter — optional `burn` feature: `sme_matmul` over `Tensor<B, 2>` for
  CPU f32/f16 tensors.

Falls back to a scalar reference on non-Apple targets.

## Benchmarks (Apple M5, best-of-N)

Throughput in TF/s (10¹² FLOP/s); integer rows are TOPS, same `2·M·N·K` op count
throughout. Every number comes from an example in `examples/`. Cross-backend
comparisons interleave the backends, one timed round each per pass, so neither
side is measured on the heat or the clock ramp left by the other.

**Per dtype** (`examples/bench`), pre-packed B except the `Accurate` rows, which
have no packed entry point. `Fast` is the M5 non-widening MOPA, `Accurate` the
widening one — half the MACs per instruction, hence half the rate:

| dtype           | 512³ | 2048³ | 16384×512×512 | 1×4096×4096 ¹ |
| --------------- | ---- | ----- | ------------- | ------------- |
| f16 `Fast`      | 3.8  | 4.9   | 4.8           | 0.10          |
| bf16 `Fast`     | 3.9  | 4.9   | 4.8           | 0.10          |
| f16 `Accurate`  | 1.8  | 2.3   | 2.4           | — ²           |
| bf16 `Accurate` | 1.9  | 2.3   | 2.4           | — ²           |
| f32             | 2.0  | 2.3   | 2.3           | 0.06          |
| f64             | 0.58 | 0.59  | 0.61          | 0.03          |
| i8→i32 (TOPS)   | 3.8  | 5.0   | 4.9           | 0.14          |
| i16→i64 (TOPS)  | 2.1  | 2.5   | 2.3           | 0.12          |
| Q4→f16          | 2.5  | 4.2   | 3.1           | 0.06          |
| Q4→bf16         | 2.6  | 4.2   | 3.3           | 0.04          |

**vs. other backends — 2048³** (`examples/vs_{accelerate,ort,candle,burn}`).
These are full strided GEMMs on both sides, so B is packed per call — hence f16
at 4.2 here against 4.9 pre-packed above:

| backend                     | f32     | f16     |
| --------------------------- | ------- | ------- |
| **sme-gemm** (this crate)   | **2.1** | **4.2** |
| Apple Accelerate (`cblas`)  | 2.2     | — ³     |
| ONNX Runtime · MLAS (CPU)   | 2.0     | 1.8 ³   |
| ONNX Runtime · XNNPACK      | 2.0     | — ³     |
| ONNX Runtime · CoreML       | 2.0     | 1.8 ³   |
| candle (`gemm` crate)       | 0.6     | 1.4     |
| burn (`NdArray`, pure-Rust) | 0.1     | — ³     |

**vs. Accelerate across shapes** (`examples/vs_accelerate`, ratio > 1 = sme-gemm
faster). Accelerate leads at 256³, where a call is a few tens of microseconds
and streaming-mode entry plus the cross-cluster dispatch are most of it, and at
2048³, where B has just outgrown the footprint that lets the kernel skip packing
it:

| shape (both dtypes) | f32       | f64       |
| ------------------- | --------- | --------- |
| 256×256×256         | 0.90×     | 0.99×     |
| 512×512×512         | 1.19×     | **1.26×** |
| 1024×1024×1024      | 1.13×     | 1.09×     |
| 2048×2048×2048      | 0.96×     | 1.06×     |
| 4096×4096×4096      | 1.05×     | 1.01×     |
| 4096×512×512        | **1.20×** | 1.17×     |
| 16384×512×512       | 1.15×     | 1.17×     |

The f16 margin on skinny shapes is wider, since no other backend here has a
native f16 kernel: 4.3 at 4096×512×512 against ORT's 1.2.

**Flash attention** (`examples/attention`, `m×n×d`, ms): 4096×4096×64 runs 2.8
(f16) / 3.2 (bf16) / 4.6 (f32). Against materializing `softmax(QK^T)V` it is
1.08× at 4096², 1.20× at 8192², and 0.93–1.06× below that, where the score
matrix still fits cache and the advantage is the `O(m·d + n·d + n·dv)` footprint
rather than speed.

**Fused epilogue** (`examples/epilogue_bench`): a bias adds +0%, bias+ReLU +4%
to +10% — rising to +30% only at shallow `K` (4096×4096×128), where the store is
most of the work. Doing the same bias+ReLU as a separate pass over the output
instead costs 4–12× the bare GEMM.

**Batched** (`examples/batched_bench`): 1.8–2.1× over a loop of single GEMMs for
per-item shapes at the flop floor, which is where the per-call streaming
entry/exit is worth amortizing.

**Q4** (`examples/q4_bench`): the 4-bit-resident path is within ~7% of eagerly
dequantizing to f16 at `m ≥ 128`, and ~1.6× slower at `m = 1`, since the
on-the-fly dequant costs `O(n·k)` regardless of `m` and small-`m` calls have
little MOPA work to overlap it with. It trades throughput for a 4× smaller
resident weight set.

- ¹ The decode shape: one row against a resident weight set, so it is
  bandwidth-bound rather than compute-bound. f16 there moves its 32 MB of
  weights at ~100 GB/s of the machine's ~150 GB/s, which the TF/s column
  understates.
- ² No pre-packed entry point, so the call would re-pack the whole weight set
  and report packing cost rather than kernel throughput.
- ³ No native f16 GEMM; f16 inputs run through f32.

ORT's three execution providers land within 1% of each other on this graph,
consistent with the single `MatMul` node staying on CPU rather than being
offloaded to XNNPACK or the ANE. Per-shape numbers and methodology:
`examples/{vs_accelerate, vs_ort, vs_candle, vs_burn, inference_value}`.

## Usage

```rust
use half::f16;
use sme_gemm::{
    matmul_f16, matmul_i8, matmul_i8_packed_dequant, gemm_f32,
    Accuracy, Dequant, Gemm, prepack_f16, prepack_i8,
};

// row-major C = A @ B
let (m, n, k) = (256, 256, 256);
let a = vec![f16::from_f32(0.1); m * k];
let b = vec![f16::from_f32(0.2); k * n];
let mut c = vec![f16::ZERO; m * n];
matmul_f16(&a, &b, &mut c, m, n, k, Accuracy::Fast); // fp16 accumulate, ~2x, M5

// pre-packed weights + fused bias/activation epilogue
let w = prepack_f16(&b, n, k);
let bias = vec![f16::ZERO; n];
Gemm::new(&a, &w, m).add_col(&bias).gelu().run(&mut c);

// quantized i8 x i8 -> i32 (raw), or fused dequant -> f32
let (ai, bi) = (vec![1i8; m * k], vec![2i8; k * n]);
let mut ci = vec![0i32; m * n];
matmul_i8(&ai, &bi, &mut ci, m, n, k);

// i8 weights, dequant -> f32 in-register (per-tensor scale) + relu
// (or .scale_per_n(&scales) for a per-output-channel scale vector)
let qw = prepack_i8(&bi, n, k);
let mut cf = vec![0f32; m * n];
matmul_i8_packed_dequant(&ai, &qw, &mut cf, m, &Dequant::new(0.02).relu());

// strided GEMM: C = alpha*C + beta*(Aᵀ @ B), BLAS-style (alpha=1 accumulates)
let af = vec![0.1f32; k * m]; // k x m, read transposed as A (m x k)
let bf = vec![0.2f32; k * n];
gemm_f32(m, n, k, &mut cf, /*c_row*/ n, /*c_col*/ 1,
    &af, /*a_row*/ 1, /*a_col*/ m,        // swap A strides => transpose
    &bf, /*b_row*/ n, /*b_col*/ 1, /*alpha*/ 1.0, /*beta*/ 1.0);
```

Other entry points: `matmul_{bf16,f32,f64,i16}`, the strided
`gemm_{f16,bf16,f32,f64,i8,i16}` (alpha/beta/transpose/col-major),
`matmul_*_batched`, `matmul_i16_dequant`, `dequant_q4` / `matmul_q4` (4-bit
resident), and the optional `sme_gemm::candle::sme_matmul`. Runtime
capabilities: `sme_gemm::caps()`.

## Design notes

- **Per-cluster shared unit, multi-cluster dispatch.** SME is one matrix unit
  per CPU cluster, not per-core. The hot paths `dispatch_apply` their M-tiles
  across _both_ cluster SME units; small problems use a low-overhead serial /
  direct-load path instead.
- **Streaming mode, SVL=512.** Compiled `-mcpu=apple-m4`; the M5 extensions
  (`+sme-f16f16`, `+sme-b16b16`, `+sme-f64f64`, `+sme-i16i64`) are separate TUs
  gated behind a runtime probe, so building them keeps the M4 floor.
- **No stable Rust SME intrinsics exist**, so kernels are C (`cc` +
  `arm_sme.h`).

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full design: the SME execution
model, TU/feature gating, module layout, the fused op-graph epilogue, and the
correctness/fallback contract.

## Verification

Beyond `cargo test` (independent-oracle correctness suite, differential
proptests, footprint validation, concurrent-caller stress for the
`dispatch_apply` paths):

```sh
# Miri: UB-checks the pure-Rust path (stride/pointer glue, packing, reference).
# The SME kernels are FFI and never run under the interpreter. On aarch64 the
# f16 properties are skipped (half's fcvt inline asm); CI runs them on linux.
cargo +nightly miri test --test proptest --test validation

# The in-crate unit tests are NOT gated that way, so on aarch64 they need half
# pushed onto its software conversion path or Miri aborts on the fcvt asm:
RUSTFLAGS="-C target-feature=-fp16" cargo +nightly miri test --lib

# Coverage-guided differential fuzzing against naive oracles, on the REAL SME
# kernels when run on M4+ (i8 is bit-exact; f32 uses a condition-aware bound
# and also feeds NaN/Inf/subnormal bit patterns as a crash detector):
cargo +nightly fuzz run diff_i8  -- -max_total_time=300
cargo +nightly fuzz run diff_f32 -- -max_total_time=300

# Overnight proptest soak (more cases through every dtype property):
PROPTEST_CASES=20000 cargo test --release --test proptest

# Guard pages (also part of plain `cargo test`): every kernel operand sits
# flush against a PROT_NONE page in both directions, so an unpredicated SME
# load/store even one element past a buffer edge faults loudly instead of
# corrupting the heap -- the OOB class ASan cannot see through intrinsics.
cargo test --release --test guard_page

# AddressSanitizer over the C kernels (experimental knob; verified to catch
# heap OOB inside __arm_locally_streaming functions on Apple clang 17).
# Slow; do not combine with `cargo fuzz` (conflicting ASan runtimes).
SME_GEMM_C_ASAN=1 cargo test --release
```

There is deliberately no loom setup: the crate's only concurrency is libdispatch
inside the C kernels plus per-thread scratch, which Rust-side model checkers
cannot instrument; `tests/concurrency.rs` stress-checks those paths for
bit-identical results under concurrent callers instead.

## License

Dual-licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.

Unless stated otherwise, any contribution intentionally submitted for inclusion
in this crate, as defined in the Apache-2.0 license, is dual-licensed as above
with no additional terms or conditions.
