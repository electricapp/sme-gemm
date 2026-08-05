# Architecture

`sme-gemm` is an Apple-Silicon GEMM library: streaming-mode **SME** (Scalable
Matrix Extension) kernels written in C against `arm_sme.h`, driven from a thin
Rust crate. It exists because the Rust ecosystem (gemm, faer, candle, ndarray)
has no on-CPU SME path on Apple M4+.

This document is the design overview. The public API is documented in the
rustdoc on `lib.rs`, the in-kernel epilogue interpreter in the header comment of
`csrc/epilogue.h`, and the SME hot loop in the comment above `run_streaming` in
`csrc/gemm_f16f16.c`.

---

## 1. The Apple SME model

SME on Apple Silicon is not the server-SME model, and most design decisions
follow from the differences.

- **Per-cluster shared unit, not per-core.** Each CPU cluster (P and E) has one
  SME matrix unit shared by all its cores, not one pipe per thread. Multiple
  threads on a single cluster do not scale; the only real parallelism is across
  clusters (§2.1).
- **Streaming mode.** SME runs inside a region bracketed by `SMSTART`/`SMSTOP`
  (`__arm_streaming` / `__arm_locally_streaming`). Inside it the SVE registers
  become streaming SVE at the streaming vector length and the ZA accumulator is
  live. Entering and leaving the region, and the `dispatch` hop to reach a
  cluster, cost fixed time, which is why small problems take a different path
  (§2.3).
- **SVL = 512 bits**, giving 32 fp16/bf16 lanes, 16 f32, 8 f64. The kernels are
  therefore built around 32-wide tiles: a 32×32 ZA tile is the natural MOPA
  output, packed panels are `[K, 32]` tile-major, and M/N are cut into 32-row
  and 32-column tiles.
- **ZA accumulator tiles.** MOPA reads two streaming-SVE vectors and accumulates
  a rank-1 update into a 2-D ZA tile. A GEMM tile loops MOPA over K, then reads
  the tile back out in horizontal or vertical slices and stores it to C.
- **ZA tile indices are instruction immediates.** `svmopa_za16_m(0, ...)`,
  `svread_hor_za16_m(.., 0, ..)` and `svst1_hor_za16(0, ..)` encode the tile
  number in the instruction, so it cannot be a runtime value. A kernel keeping
  two tiles in flight — `za0` for N-tile `nt`, `za1` for `nt+1`, run in lockstep
  so consecutive MOPAs never collide — must hard-code both. This is why the hot
  loop cannot be parameterized over "which tile" and is one large function; see
  the `run_streaming` header comment.

---

## 2. Dispatch, blocking, and packing

### 2.1 Multi-cluster dispatch

Since the SME unit is per-cluster, the kernels parallelize across clusters with
`dispatch_apply`: M is chunked into tiles and the chunks go to
`dispatch_get_global_queue`, which schedules them on both cluster units.

Chunks are finer than pure L2 residency would pick (`M_CHUNK = 2` tiles) so
`dispatch_apply` can work-steal instead of stalling on the slower E-cluster: ~5%
geomean, and 9–13% at 256³/512³/4096³, against 4 tiles per chunk. The value must
stay even, because `run_narrow_rowmajor` pairs M-tiles and an odd chunk would
idle `za1`.

### 2.2 Flat-M parallelism

M is normally the only parallel axis, so a wide flat-M problem — few M-tiles,
large N and K, the decode and small-batch shape — would run on one cluster. The
f16f16 and b16b16 packed paths detect this and parallelize over N instead: pack
all of A once into a shared buffer, then hand even-aligned N-tile-pair chunks
(`N_CHUNK = 4`) to both clusters. C columns per chunk are disjoint, A and B are
read-only and shared, and the even alignment keeps a dual-tile pair intact.
`run_streaming` carries an `[nt_lo, nt_hi)` N-tile range for this.

The split needs at least three chunks to pay. A 2-way split is slower than no
split: running serially instead measures 1.03–1.15× across n=160..256,
k=512..16384, m=1..64 on both dtypes, reaching parity by k=32768 where
B-streaming dominates. With two chunks the makespan is the E-cluster half, which
exceeds the cost of running everything on the caller's cluster; from three
chunks up, work-stealing balances the halves. Hence the `nn_chunks >= 3` gate.
The affected band is narrow: `nn_chunks == 2` means `n_tiles_pad` in [5,8], so n
in [129,256].

This does not generalize to the M-chunk path, where a 2-way split wins — forcing
it serial measures 0.78–0.97×. M-chunks pack their own A slice, so a second
cluster overlaps NEON packing with MOPA issue, while flat-M chunks are pure MOPA
against an A-pack built up front. The allocator is not the cause either:
replacing the flat-M `malloc` with the thread-local scratch used by the serial
path measures 1.00–1.01×.

### 2.3 Small- and large-problem paths

`dispatch_apply` entry and exit and the streaming-mode transition cost a few µs
each, which dominates small problems. Each kernel forks:

- **Direct-load B.** B is read straight from a row-major `rhs`, skipping the
  pack entirely — worth more than the packing it saves, since packing reads and
  writes all of B before a single MOPA issues. The cost is that B is re-read
  once per M-tile, so the gate is B's own footprint (`n*k*elem`), not the
  problem size: at equal flop counts a small-B shape prefers this path and a
  large-B one collapses on it. Blocking the path over N does not raise the
  limit, because a block still walks K at stride `n` and so covers B's whole
  address range; making that range contiguous is what packing is for. The f64
  budget is well under f32's, its N-tile being 8 lanes rather than 32.
- **Packed B, multi-cluster.** B packed once, a shared `malloc`'d A-pack, and
  `dispatch_apply` over M-chunks.

Within the direct-load path the split is serial vs. `dispatch_apply` over
M-chunks, on a flop floor rather than a chunk count: below a few MFLOP the
dispatch costs more than the second cluster returns, and two shapes that chunk
identically can want opposite answers.

Rust has an analogous floor. `sme_worth_it(m,n,k)` requires
`has_sme() && k >= 2` plus a flop minimum that depends on how much of a ZA tile
the output fills: MOPA accumulates into a 32×32 tile whatever the shape, so a
1×1 product still pays for 1024 lanes, and vector-shaped work loses to the
scalar reference no matter how large `k` is. From `m*n >= 8` upward the padding
is amortized and the floor drops to `1<<12`; below it the original `1<<18`
stands.

The floor applies only where the fallback costs O(m·n·k). For the pre-packed
weight types, `Packed<T>` and `Q4Weights`, the fallback must unpack the weights
first, which costs O(n·k) regardless of m — so gating them makes the small
shapes the floor protects slower, not faster. Those paths dispatch on capability
alone (`packed.sme`, `caps().sme_f16f16`) with no flop test. Dropping the gate
from Q4 measured 2.3–3.9× on decode-shaped calls.

### 2.4 Cache blocking

The large path adds BLIS-style `jc→ic→mt→nt` blocking so each packed panel
streams from DRAM once rather than once per opposing tile.

In f32 and f64 the `jc` loop sits **outside** the `dispatch_apply` over
M-chunks, so one B-block serves every M-chunk while it is still L2-resident, and
A is packed on the first block only. Nested the other way each chunk sweeps all
of B and B leaves L2 before the next chunk reaches it, so packed B is
re-streamed from DRAM once per chunk — gigabytes of it on a large square — and
the inner blocking cannot help, since it only ever sees one chunk. A reduction
epilogue spans all of N and so keeps a single block.

Block sizes come from a per-dtype byte budget: `nc_blk`/`mc_blk` count whole
32-wide super-tiles, and `tile_bytes` is the full super-tile — two packed
16-lane bands in the widening/int kernels, or the single tile in f16f16/b16b16
where the tile is the super-tile — so the budget equals total resident bytes
(A-block + B-block).

| kernels              | budget |
| -------------------- | ------ |
| f16f16, b16b16       | 8 MB   |
| widening 16-bit, int | 16 MB  |
| f32, f64             | 24 MB  |
| Q4                   | 16 MB  |

The budget is sized to the smaller cluster. Chunks dispatch to both, and on M5
the E-cluster L2 is 6 MB against the P-cluster's 16
(`hw.perflevel1/0.l2cachesize`), so a P-sized block thrashes the E workers. For
MOPA-bound dense shapes the constant barely matters: sweeping f16f16/b16b16 over
8/16/24 MB from 2048³ through 1024×8192×8192 moves nothing outside ±3% noise,
and not monotonically, and f32 behaves the same over 12–24 MB once the `jc` loop
is hoisted. Q4 is the exception, because there the budget also sets how many
full dequant passes a shape pays — 16 MB is worth 2–6% for m ≥ 992, where 8 MB
forces a second M-block and therefore a second dequant of the whole weight set.

When the whole problem fits, the blocks span everything and the outer loops run
once, at no cost over the flat path.

### 2.5 Streaming regions contain only MOPA

Streaming-mode SVE is tuned for MOPA, and scalar code inside a streaming region
runs at roughly a tenth of its normal rate. The measured cost, per site:

| site                            | scalar in streaming | vectorized outside it                      |
| ------------------------------- | ------------------- | ------------------------------------------ |
| `softmax_rows`                  | 21 ms / 16.8M f32   | 1.2 ms — parallel NEON, `csrc/attention.c` |
| batched fused dequant, i8 / i16 | 1678 / 1720 µs      | 52 / 75 µs — vectorized ZA store           |
| Q4 tile dequant                 | 44.4 ms             | 2.05 ms — hoisted out, NEON, threaded      |

A `__arm_streaming` or `__arm_locally_streaming` function should contain MOPAs,
ZA reads and writes, and the vectorized store epilogue, and nothing else.
Per-cell scalar loops (`ep_dequant_cell*`) survive only where no alternative
exists: streaming mode has no scatter store, so a strided `dst` still needs a
scalar write, and even there the dequant and op-graph run vectorized into a
contiguous row first.

Work that cannot be vectorized in place belongs outside the region rather than
inside it. `run_q4` is an ordinary function that dequantizes a tile pair and
calls `run_q4_pair`, which is locally-streaming and does only MOPA and store.
Hoisting it also makes it threadable across independent N-tile pairs.

Vectorizing such a loop with SVE in place is not an option: with ZA live across
the call, Apple clang 21 aborts with
`Invalid size request on a scalable vector`, and `noinline` does not avoid it.

### 2.6 Operand packing

Both operands land in the same tile-major panel, so each kernel needs two pack
routines, chosen by which stride is 1. A and B differ only in which stride that
is — A's lanes are rows, B's are columns.

| operand layout           | lanes at each depth | routine                                   |
| ------------------------ | ------------------- | ----------------------------------------- |
| row-major A, col-major B | strided             | transpose (lane-contiguous → depth-major) |
| col-major A, row-major B | contiguous          | direct copy / interleave                  |

Both directions are vectorized in every dtype. The transposing side uses NEON
block transposes: `transpose_8x8_u16`, shared by all four 16-bit drivers via
`csrc/transpose16.h`, 4×4 for f32, 2×2 for f64. The widening kernels want a
pair-interleaved destination (`dst[p*32 + 2*lane + s]`), which is compatible
with a block transpose: an 8×8 transpose yields one depth per vector, and
zipping the two depth vectors of a p-group is exactly that interleave.

Two rules hold across the layer:

- **Walk the panel depth-outer.** A store that strides one lane-row is a cache
  line, so a lane-outer loop re-walks the whole `k × lanes` panel once per lane.
  At m=1 that is 32 passes over a 256 KB panel to place a single row, 31 of them
  writing zero padding.
- **Scalar fillers are for edges only** — ragged bands, odd-K slices, and
  layouts with neither stride equal to 1. A fast path gated on the full-band
  case sends every narrow-M call down the scalar path, which is where latency
  matters most.

Packing only shows up in a measurement when the packed operand is a large
fraction of the traffic: probe the A pack with `m ≫ n` and the B pack with
`n ≫ m`. A square or B-dominated shape reports no change for an A-pack fix worth
1.4× where it applies.

### 2.7 Flash attention

`flash_attention_{f32,f16,bf16}` run the online-softmax schedule: for each query
block, loop over key blocks computing `S = QKᵀ` with a strided GEMM, fold `S`
into the running per-row max and sum, then accumulate `O += P·V` with a second
GEMM. Only a `block_m × block_n` score tile is live at a time, so traffic is
O(m·d + n·d + n·dv) rather than O(m·n). The softmax passes live in
`csrc/attention.c` and are parallel NEON, not streaming SVE (§2.5).

`FlashParams::auto` keeps the whole score matrix in one tile while it fits a 4M
f32 budget, then uses key tiles of 1024 with as many queries as fit. Query
tiling is the last thing to give, since it re-reads K and V once per query
block.

The score block and accumulators are thread-local buffers reused across calls.
At the auto cap the score block is 16 MB, so allocating per call costs an mmap
and its page faults: 0.09–0.15 ms regardless of shape, which weighs most where
flash is cheapest (1.04–1.13× at 512²–2048², ~1.01× by 8192²). Requests above
the cap allocate locally, so a caller cannot pin an arbitrarily large buffer to
a thread for the process lifetime. The buffer is taken out of its cell rather
than borrowed, so a re-entrant call allocates instead of panicking.

The buffers are never cleared, because every element is written before it is
read. Debug builds poison a reused buffer with all-ones — a NaN for f32, f16 and
bf16 — to keep that invariant enforced, since the natural way to break it is
silent: on the first key block `corr` is 0, so `acc = corr*acc + t` discards
finite garbage and only a non-finite value would surface. Removing the `acc`
fill passes the suite in release and fails three tests in debug.

Fusing the row-max into the first GEMM does not pay. Skipping the max sweep
entirely, an upper bound on any fusion, measures 1.00–1.06×, mostly 1.00×: its
loads are already hidden behind the memory-bound exp pass. For the same reason,
giving that sweep four accumulators rather than one changes nothing — it is
bound by load throughput, not by the serial `vmaxnm` chain.

---

## 3. Translation units and feature gating

There are no stable Rust SME intrinsics — `core::arch` does not expose
`svmopa_za16_m` and inline asm for the full streaming ABI is impractical — so
all kernels are C, compiled by `build.rs` with `cc` against `arm_sme.h` on Apple
aarch64 and skipped elsewhere.

The gating scheme keeps the M4 floor while shipping M5-only instructions in the
same binary:

- **Base `FEAT_SME` TUs (M4+).** `gemm_f16f32.c`, `gemm_bf16f32.c`,
  `gemm_i8i32.c`, `gemm_f32.c`, `attention.c`, `sme_probe.c` and
  `sme_runtime_shims.c` compile together with
  `-mcpu=apple-m4 -ffp-contract=fast -O3` into `sme_gemm_base`, using only
  instructions present on every M4.
- **M5 extension TUs.** Each M5-only kernel is its own TU with its target
  feature layered on the same base flags, linked as its own static lib:
  `gemm_f16f16.c` → `+sme-f16f16`; `gemm_b16b16.c` → `+sme-b16b16 +sve-b16b16`
  (the SVE feature covers `svmul_bf16`/`svmla_bf16` in the bf16 epilogue);
  `gemm_f64.c` → `+sme-f64f64`; `gemm_i16i64.c` → `+sme-i16i64`. Each is called
  only behind a runtime probe, so compiling it with newer features does not
  raise the binary's floor — an M4 never enters those symbols.

`build.rs` also links `clang_rt.osx` from `clang --print-runtime-dir`, since
streaming functions call `__arm_tpidr2_{save,restore}` from compiler-rt.

### 3.1 Fragment headers

Several kernels are split across `.h` fragments `#include`d into the TU that
owns them:

| TU              | fragments                              |
| --------------- | -------------------------------------- |
| `epilogue.h`    | `epilogue_{scalar,f16,bf16,f32,f64}.h` |
| `gemm_f32.c`    | `gemm_f32_{small,batched}.h`           |
| `gemm_f64.c`    | `gemm_f64_{small,batched}.h`           |
| `gemm_f16f16.c` | `gemm_f16f16_{batched,q4}.h`           |
| `gemm_b16b16.c` | `gemm_b16b16_q4.h`                     |
| `gemm_i16i64.c` | `gemm_i16i64_batched.h`                |

The split is for navigability, not compilation. Each fragment closes over its
parent's `static` packing helpers, per-thread scratch and typedefs, so it cannot
be its own TU without exporting all of that, and each is included at exactly the
point its code would otherwise occupy, leaving the preprocessed TU and the
generated code unchanged. `csrc/transpose16.h` is the exception: it is a
genuinely shared header, included by all four 16-bit drivers. Every fragment
needs an entry in `build.rs`'s `CSRC` list for rerun tracking, since `cc` never
sees it.

### 3.2 The runtime probe

`src/probe.rs` and `csrc/sme_probe.c` detect capabilities via `sysctlbyname` on
the `hw.optional.arm.FEAT_SME*` flags, cached once in a `OnceLock`. `Caps` holds
one bool per group: `sme` (base, M4+), and `sme_f16f16`, `sme_b16b16`,
`sme_i16i64`, `sme_f64f64` (M5+). On non-Apple targets every flag is false and
everything falls back to the scalar reference. Per-dtype entry points consult
the matching flag before dispatching to an M5 kernel.

---

## 4. Rust module layout

`lib.rs` is a thin re-export root: crate docs and `pub use`.

- **`element.rs`** — the `Accuracy` mode (Accurate = fp32-accumulate widening
  MOPA on M4+; Fast = fp16-accumulate on M5 `FEAT_SME_F16F16`), the sealed
  `Element` trait and its dtype impls, and the reusable `Packed<T>` weight
  panel.
- **`epilogue/`** — the fused-epilogue builder API. `mod.rs` holds the shared
  `EpOp` opcodes, the `#[repr(C)]` `EpNode` mirroring the C struct, the
  `EP_ACT_*` activation table and the scalar f32/f64 activation evaluators.
  `graph.rs` is the element-typed `Epilogue<T>`; `dequant.rs` the i8/i16
  `Dequant`, whose operands are all f32 (the post-dequant domain).
- **`exec/`** — execution glue, one concern per file. `builder.rs`: the
  `Gemm<T>` builder and `RowReduce`. `packed.rs`: the f32/f64 fused
  packed-epilogue paths and `softmax_gemm_f32`/`softmax_rows`. `scalar.rs`: node
  resolution (`resolve_nodes`) and the scalar evaluators (`ep_apply_cell`,
  `apply_ep_scalar`, f64 analogs). `guard.rs`: `sme_worth_it` and the pre-FFI
  footprint checks (`check_strided`, `checked_dims`). `unpack.rs`: the
  SME-packed-B unpackers for the OOM fallback. `macros.rs`: the
  `batched_ep_impl!`/`packed_ep_impl!` macros generating the 16-bit
  batched/packed paths, where f16 and bf16 differ only in symbol, caps and
  fallback. `mod.rs`: the `PackedEpilogue`/`MapElem` dispatch traits.
- **`ffi.rs`** — the `unsafe extern "C"` declarations for every `gemm_sme_*`
  kernel and the `#[repr(C)]` descriptor mirrors (`EpDesc16`, `EpDescF32`,
  `EpDescF64`). Apple aarch64 only.
- **`kernels/`** — public per-dtype entry points: `f16.rs`, `bf16.rs`, `f32.rs`,
  `f64.rs`, `int/` (`i8.rs`, `i16.rs`), and `q4/` (`weights.rs` builds and
  dequantizes `Q4Weights`, `matmul.rs` the GEMM entry points). Each wires the
  worth-it check, the FFI call and the scalar fallback. `attention.rs` and
  `attention_half.rs` hold the flash-attention drivers (§2.7).
- **`reference.rs`** — the portable scalar GEMM oracle (§6).
- **`probe.rs`** — runtime capability detection (§3.2).
- **`burn.rs`, `candle.rs`** — optional framework adapters, feature-gated.

---

## 5. The fused-epilogue op-graph

A GEMM epilogue (`C = act(beta*(A@B) + bias) ...`) is expressed at runtime as an
ordered op-graph, the runtime analog of a CUTLASS EVT. Rather than a separate
pass over C, the graph is applied in-register at the ZA→C store, between the ZA
read and the store.

### EpNode encoding

```c
typedef struct {
    uint32_t op;      // enum ep_op: which operation
    uint32_t aux;     // activation kind for EP_OP_ACT, else unused
    float    scalar;  // f32 operand for *_SCALAR ops (and leaky/elu alpha)
    const void *ptr;  // operand base for ROW/COL/TENSOR ops
    size_t   ld;      // row stride (elements) for TENSOR ops
} EpNode;
```

`Epilogue` and `Dequant` hold a `Vec<EpNode>`; the FFI descriptor
(`EpDesc16`/`EpDescF32`/`EpDescF64` ↔ `ep_desc16`/`ep_desc_f32`/`ep_desc_f64`)
is `{ n_nodes, nodes* }`. The Rust and C `EpNode` layouts must match exactly.

### Op kinds

- **Elementwise binary** × {SCALAR, ROW (per-M, length m), COL (per-N, length
  n), TENSOR (full m×n at stride `ld`)} for ADD, MUL, SUB, DIV, MAX, MIN.
- **`EP_OP_ACT`** — an activation selected by `aux` (`enum ep_act`): ReLU, GELU,
  SiLU, tanh, sigmoid; the Group-A exact ops (leaky-relu, relu6, hard-sigmoid,
  hard-swish, abs, neg, square, sign, sqrt, softsign, recip, rsqrt); and the
  Group-B transcendentals (exp, log, elu, selu, softplus, mish, exact GELU via
  erf).

### Builder lowering

Each builder method pushes one node (`push_scalar`, `push_vec`, `push_act`) in
call order, and the running value is mutated in that order at the store. Two
exceptions:

- The five named activation shortcuts (`.relu`, `.gelu`, `.silu`, `.tanh`,
  `.sigmoid`) set a trailing `act: Option<u32>` instead of pushing an in-order
  node, so they always apply last, CUTLASS-style and independent of the
  additive/scaling ops. Every other activation (`.leaky_relu`, `.exp`, …) pushes
  an in-order `EP_OP_ACT`. `resolve_nodes` appends the trailing `act` as a final
  node when building the FFI array, and resolves `ld == 0` on TENSOR ops to `n`.
- bf16 evaluates its graph in f32: accumulator and operands upcast, the graph
  runs in f32, and the result rounds back to bf16. bf16 therefore has the same
  op set as f16 and f32, with no divide or transcendental restriction.

### Node-major register-resident store

The store is node-major rather than row-major. A row-major store — a ZA read
plus an out-of-line interpreted node loop per output row, with jump-table
dispatch and operands reloaded each row — costs ~500K dispatches for a 4096²
output, enough to make a fused epilogue 1.7–4.5× the bare GEMM. Node-major
instead reads a 4-row block of the live ZA tile into Z registers and dispatches
each node once per block across all its rows, hoisting invariant operands (COL
vectors, scalars, activation constants) out of the per-row loop with no memory
bounce. The gelu/silu/tanh rational approximation sits in a `noinline` helper so
the common add/mul/relu nodes still inline. Fused-vs-bare overhead is roughly
+2..11% when compute-bound and +35% at the shallowest store-bound K.

### Bias-init MOPA fold

The fastest epilogue is none: a pure overwrite `C = A@B` stores ZA→C in one
`st1h {zaNv}` per slice, with no read, scale or store, which dominates at small
K. Any fused epilogue forces the accumulator into registers and loses that.

A leading per-column bias — a single `ADD_COL` node with `beta == 1` and
`!read_dst` — is therefore folded into the ZA accumulator before the K-loop via
a rank-1 FMOPA (`ones[r] * bias[c]`) and dropped from the store-time graph. With
the only node folded away the store reverts to the single-instruction fast path,
making a bias-only epilogue essentially free. The fold adds bias in the element
domain (bias first, then accumulate), a rounding-order change that stays inside
the √K tolerance. It is restricted to the bias-only case: folding a bias
followed by an activation perturbs the layout-sensitive store codegen and
regresses the relu store, so bias+activation keeps the in-store add.

---

## 6. Correctness

- **Scalar oracle (`reference.rs`).** A portable triple-loop GEMM per dtype,
  accumulating in a wider type — f32 for the 16-bit floats, f64 for f32 and f64,
  i32/i64 for the ints — so it bounds the SME kernels' accuracy from above. It
  is both the non-Apple fallback and the test oracle. The int oracles use
  wrapping adds to match SMOPA's modular accumulation, so a debug build does not
  panic where the SME path wraps silently. The fused-epilogue fallback pairs it
  with the scalar op-graph evaluators in `exec/scalar.rs`.
- **Pre-FFI footprint validation.** The strided `gemm_*` entry points assert
  each slice covers its `(rows-1)*row_stride + (cols-1)*col_stride + 1`
  footprint with checked arithmetic before raw pointers cross to C, which honors
  strides literally and cannot bounds-check. This is what makes those `unsafe`
  FFI calls sound from safe Rust (`exec::check_strided`, tested in
  `tests/validation.rs`). The `EpNode` FFI layout is pinned with const asserts
  on both sides.
- **SIMD vs scalar divergence**, all bounded and within the √K tolerance. `log`
  and `exp` are specified at the boundaries (`log(0) = -inf`, `log(x<0) = NaN`,
  `exp` overflow `= +inf`) and both paths honor them. Three deliberate tradeoffs
  differ: the five named activations relu/gelu/silu/tanh/ sigmoid share one
  in-register rational tanh, whose ~4e-2 error is masked by the envelope for
  gelu and silu but exposed by a bare `.tanh()` or `.sigmoid()` (the scalar
  fallback is exact libm); splat-DIV (scalar, ROW-splat or COL-splat divisor) is
  a multiply-by-reciprocal in SIMD, ~1 ULP off true division even on finite
  normals, while vector-operand DIV uses `svdiv` and matches exactly, with the
  reciprocal guarded so a divisor too small to invert falls back to a real
  divide. MAX/MIN/ReLU are maxNum/minNum on every path.
- **OOM and non-Apple fallback contract.** Every `gemm_sme_*` entry point
  returns `int`: `rc == 0` means it ran and wrote C, and `rc != 0` means an
  allocation failed and C was left untouched, so the caller falls back to the
  reference GEMM after unpacking any SME-packed B (`unpack_b16_sme`,
  `unpack_b_i8_sme`). On non-Apple targets the FFI is not compiled and the
  reference path is unconditional. The pure-store fast path is preserved by
  passing a NULL epilogue descriptor when there is no epilogue work.

### Test suites

| suite                  | count | covers                                                        |
| ---------------------- | ----- | ------------------------------------------------------------- |
| `tests/correctness/`   | 100   | dtypes, shapes, layouts, packed/batched, epilogue graphs      |
| `tests/validation.rs`  | 19    | pre-FFI footprint asserts and panic contracts                 |
| `tests/proptest.rs`    | 17    | randomized shapes/strides/operands against the oracle         |
| `tests/guard_page.rs`  | 13    | reads past the declared footprint fault on a `PROT_NONE` page |
| unit tests             | 11    | in-module invariants (unpackers invert their packers)         |
| doctests               | 5     | the documented API examples                                   |
| `tests/concurrency.rs` | 4     | one GEMM from many threads, bit-identical, across branches    |

169 total. `tests/correctness/main.rs` holds the shared oracles and shape
tables: `SIZES`, and `TAIL_SIZES` for shapes that clear the flop floor and have
ragged M/N/K tails, where tail-predicate handling is exercised.
`PROPTEST_CASES=N` widens the random sweep for a soak run.

The suite should run in debug as well as release: debug enables the flash
scratch poison (§2.7) and Rust's overflow checks, neither live in release.
`SME_GEMM_C_ASAN=1` additionally builds the C kernels under AddressSanitizer.

---

## 7. Benchmarking

Throughput on this machine is sensitive to several things unrelated to the code
under test, so the harnesses are built to control for them.

- **Variants belong in one binary, alternated between timed rounds.** Separate
  builds differ in binary layout as well as thermal state, which on small shapes
  moves results 10–25% on its own. `best_interleaved` in
  `examples/epilogue_bench.rs` and each `examples/vs_*.rs` is the pattern.
- **A sweep should include shapes the change cannot affect.** If those controls
  do not read 1.00×, the harness is measuring something other than the change.
- **Sustained load leaves the machine 5–10% low**, so a headline number follows
  a cooldown. Clock ramp costs the first variant measured ~25%, so a warmup pass
  precedes the first timed round.
- **Below `SME_MIN_FLOPS` the comparison path is the scalar reference**, so a
  ratio measured there reflects the floor rather than the change.
- **A perturbation that does not fail the tests** can mean the code is dead or
  the probe shape falls below a gate, rather than that the change is safe.

Throughput lives in `examples/bench.rs` (per dtype), `epilogue_bench.rs` (fusion
overhead), `batched_bench.rs`, `q4_bench.rs`, `attention.rs`, `roofline.rs` and
`dequant_bench.rs`. Cross-backend comparisons are `vs_accelerate`, `vs_ort`,
`vs_candle` and `vs_burn`, the last three behind their feature flags.

---

## 8. Where to look

- **Add a dtype:** an `Element`/`PackedEpilogue`/`MapElem` impl in
  `src/element.rs` and `src/exec/mod.rs`; a reference GEMM in
  `src/reference.rs`; the C kernel TU in `csrc/` wired into `build.rs` (base TU
  for an M4 instruction, `feat_tu` for an M5 one); the `extern "C"` decls in
  `src/ffi.rs`; the probe flag in `src/probe.rs`; and the entry points in a
  `src/kernels/` module re-exported from `src/lib.rs`.
- **Add an epilogue op:** the opcode in `enum ep_op` (`csrc/epilogue.h`) and the
  mirrored `EpOp` (`src/epilogue/mod.rs`) at the same numeric value; the
  implementation in the per-dtype `ep_apply_nodes_*` and
  `EP_STORE_TILE_ROWMAJOR_*` clusters in `csrc/epilogue.h`; a builder method on
  `Epilogue`/`Dequant` and the forwarding method on `Gemm`
  (`src/exec/builder.rs`); and the scalar evaluators (`ep_apply_cell`,
  `ep_apply_cell_f64`, `dq_apply_cell` in `src/exec/scalar.rs`, and
  `ep_apply_nodes_scalar_*` in `csrc/epilogue.h`) so the fallback matches. For
  an activation, extend `enum ep_act`, the `EP_ACT_*` constants, and the f32/f64
  `act_apply_*` evaluators.
- **A perf change:** the hot loop is `run_streaming` in the per-dtype
  `csrc/gemm_*.c` — start with `gemm_f16f16.c` and read its header comment
  first. The monolithic structure and the node-major store are both
  load-bearing; splitting either costs throughput. Measure per §7.
