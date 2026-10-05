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
  SME matrix unit shared by all its cores. On M5 nearly all the throughput is
  the P unit: one P thread reaches ~4.1 of ~5.0 f16 TF/s and 4-5 threads
  saturate it, while a busy E thread runs ~9x slower than a P thread (§2.1).
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

The kernels parallelize with `dispatch_apply` on the global queue, which runs
work on every core of both clusters. Nearly all SME throughput is the P unit:
with all ten cores busy an E thread runs ~9× slower than a P thread. Two rules
follow everywhere:

- Work is cut into items small enough that an E thread holding one only delays
  the tail. M-chunks are two tiles (`M_CHUNK = 2`, kept even because
  `run_narrow_rowmajor` pairs M-tiles).
- No SME loop carries a dispatch barrier or waits on one specific item. Items
  are claimed off atomic cursors, so a fast worker takes whatever is ready next
  (§2.4, §2.9).

### 2.2 Flat-M parallelism

When M has too few tiles to split, the f16f16 and b16b16 packed paths
parallelize over N instead: A is packed once into a shared buffer and
even-aligned N-tile-pair chunks (`N_CHUNK = 4`) go to both clusters, through the
`[nt_lo, nt_hi)` range `run_streaming` takes. The split needs at least three
chunks: with two, the makespan is the E-cluster's half, which costs more than
running the call on the caller's cluster. The M-chunk path has no such floor,
because its chunks pack their own A slice and a second cluster overlaps that
NEON pack with MOPA issue.

### 2.3 Small- and large-problem paths

Dispatch and the streaming transition cost a few µs each, so the kernels fork by
size:

- **Direct-load B** (f32, f64; B under a footprint budget). B is read straight
  from a row-major `rhs` with no pack. It is re-read once per M-tile, so the
  gate is B's footprint (`n*k*elem`), not the flop count: packing is what makes
  a large B's address range contiguous. f64's budget is smaller, its N-tile
  being 8 lanes.
- **Packed B.** B pre-packed by the caller or built inside the parallel region
  (§2.10); A packed per M-chunk.

Direct-load runs serially below a flop floor and dispatches above it. Its chunks
are claimed off a shared cursor by a pool one worker short of the chunk count,
so a P worker comes back for the spare instead of the call ending on an E
worker; below four chunks the pool is the chunk count. Each worker packs the A
chunk it is about to compute, so the pack overlaps MOPA issue.

The f32 A-pack goes through ZA while A is cache-resident: a vertical ZA load
writes a tile column and a horizontal store reads a row, so 32 instructions
transpose a 16×16 block, where the NEON 4×4 transpose is issue-bound at about
one instruction per float. Once A is DRAM-sized the NEON path's memory-level
parallelism wins, so the ZA pack is gated on A's footprint.

Rust's `sme_worth_it(m,n,k)` requires `has_sme() && k >= 2` and a flop floor:
`1<<12` once `m*n >= 8`, `1<<18` below, since a MOPA pays for a whole 32×32 tile
whatever the output shape. The floor applies only where the fallback is
O(m·n·k). `Packed<T>` and `Q4Weights` would have to unpack their weights first
(O(n·k) regardless of m), so they dispatch on capability alone.

### 2.4 Cache blocking

The packed path uses BLIS-style `jc→ic→mt→nt` blocking, so each packed panel
streams from DRAM once rather than once per opposing tile. The `jc` loop sits
outside the M-chunks, so one B-block serves every M-chunk while it is
L2-resident, and the (N-block, M-chunk) pairs run as one flat `dispatch_apply`
in block-major order (GCD hands out indices in order). A dispatch per block
would wait on its E-thread chunks every block. A reduction epilogue spans all of
N and keeps a single block.

Block sizes come from a per-dtype byte budget over whole 32-wide super-tiles
(one tile in f16f16/b16b16, two packed bands in the others), counting A-block
plus B-block:

| kernels              | budget |
| -------------------- | ------ |
| f16f16, b16b16       | 8 MB   |
| widening 16-bit, int | 16 MB  |
| f32, f64             | 24 MB  |

Budgets are sized to the smaller cluster, since chunks go to both and the M5
E-cluster L2 is 6 MB against the P-cluster's 16. Dense MOPA-bound shapes are
insensitive to the constant. When the problem fits, the outer loops run once.

### 2.5 Streaming regions contain only MOPA

Scalar code inside a streaming region runs at roughly a tenth of its normal
rate, so an `__arm_streaming` or `__arm_locally_streaming` function contains
MOPAs, ZA reads and writes, and the vectorized store epilogue. Elementwise work
that does not map onto ZA runs outside the region: the attention softmax is
parallel NEON in `csrc/attention.c`. Elementwise work that does map onto ZA
stays in: the Q4 unpack is LUTI4, an FMLA into ZA and a MOVA out (§2.9). The
same holds for a trailing gelu, silu, sigmoid or tanh on the 16-bit paths: in a
streaming epilogue they cost ~2.5 ns an output, so the entry points split them
off the op-graph and run them as a NEON pass over C after the kernel
(`csrc/neon_act.h`, row chunks across cores when C is large), computing in f32
on the stored value. Cheap nodes (bias, scale, clamp, relu) stay fused. Per-cell
scalar loops (`ep_dequant_cell*`) survive only for a strided `dst`, since
streaming mode has no scatter store, and even there the dequant and op-graph run
vectorized into a contiguous row first.

Vectorizing such a loop with SVE in place is not an option: with ZA live across
the call, Apple clang 21 aborts with
`Invalid size request on a scalable vector`, and `noinline` does not avoid it.
The same abort fires on an SVE load from `base + (d >> shift) * 32` in a
streaming loop; stepping a separate block counter avoids it.

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
- **Scalar fillers are for edges only** — ragged bands, odd-K slices, and
  layouts with neither stride equal to 1. A fast path gated on the full-band
  case would send every narrow-M call down the scalar path, which is where
  latency matters most.

### 2.7 Flash attention

`flash_attention_{f32,f16,bf16}` run the online-softmax schedule: for each query
block, loop over key blocks computing `S = QKᵀ` with a strided GEMM, fold `S`
into the running per-row max and sum, then accumulate `O += P·V` with a second
GEMM. Only a `block_m × block_n` score tile is live at a time, so traffic is
O(m·d + n·d + n·dv) rather than O(m·n). The softmax passes are parallel NEON
(§2.5); the row-max sweep hides behind the memory-bound exp pass, so it is not
fused into the GEMM.

`FlashParams::auto` keeps the whole score matrix in one tile while it fits a 4M
f32 budget, then uses key tiles of 1024 with as many queries as fit. Query
tiling gives last, since it re-reads K and V once per query block.

The score block and accumulators are thread-local buffers reused across calls,
because allocating up to 16 MB per call costs an mmap and its page faults.
Requests above the cap allocate locally, so a caller cannot pin an arbitrarily
large buffer to a thread. The buffer is taken out of its cell rather than
borrowed, so a re-entrant call allocates instead of panicking.

The buffers are never cleared: every element is written before it is read. Debug
builds poison a reused buffer with all-ones (a NaN in f32, f16 and bf16) to keep
that invariant enforced, since breaking it is otherwise silent — on the first
key block `corr` is 0, so `acc = corr*acc + t` discards finite garbage.

One query row against a KV cache has no GEMM in it: `attention_kv_f16` is NEON
on the calling thread over an f16 cache of row-major
`[position][kv_heads * head_dim]` rows, so appending a key/value row is a row
copy. Scores take FMLAL (f16 × f16 → f32) four key rows at a time with low and
high halves in separate accumulators; the values take FMLAL against f16-rounded
probabilities selected by lane. A transposed key layout would drop the
horizontal sums but cost a strided write per element on every append.
`attention_kv_causal_f16` runs a block of query rows (row i sees keys
`0..=start+i`) as independent rows of the same kernel, spread across cores.

### 2.8 Small m: ZA-vector GEMV

With one live row a MOPA spends 1/32 of its work (1/16 for the 16×16 tiles), so
for m ≤ 4 (m ≤ 2 for f32, m = 1 for f64 and i16, whose MOPA issues twice as
fast) the drivers use SME2 multi-vector ops into ZA vector groups instead, with
A broadcast once per call. The B layout picks the form:

- **Packed B.** One x4 load is four consecutive depths of a band, taken by a
  tuple-by-tuple FMLA (SDOT for the ints) against four broadcast depths of A.
  Group `4r+t` holds row r, band t, one vector per depth phase; the store sums
  the four. When A's rows are contiguous, f16 builds that tuple in registers —
  one 16-byte replicating load, four lane duplicates — instead of
  reading a broadcast copy of A from memory; the values, and so the bits, are
  the same.
- **Row-major B, not pre-packed** (`run_gemv_rm`). B is read in place: one x4
  load is four adjacent bands at one depth, which takes the tuple-by-vector
  FMLA, ~25% faster to issue. Group `(r, g, p)` holds band group g's depth phase
  p, so each band sums the same phases in the same order and the two forms agree
  bit for bit. Loads go depth-major across up to four column groups, because
  walking one group down a power-of-two row stride puts every load in the same
  L2 bank. The row count is a constant in each inlined copy of the loop; a
  runtime count spills.

Packed B runs at the DRAM roof once B is DRAM-sized. Where B is cache-resident
it is bound by the tuple-by-tuple FMLA rate of the P-cluster's one SME unit,
which more threads do not raise, so the in-place form is the faster of the two
there.

The Q4 GEMV reads B band-major: a band is 128 columns, and for each depth it
holds 64 bytes of nibbles, column c at nibble c. One LUTI4 (from a ZT0 table)
turns those 64 bytes into the band's four 32-column tiles at that depth, which
the lane-indexed FMLA multiplies by one A value taken from a 16-byte replicating
load (`LD1RQH`) — so A is never broadcast in memory, and the indexed form issues
at twice the tuple-by-tuple rate. Two depths of a band are one 128-byte line,
loaded as one x2 load: an SME half-line load costs as much as a full one. The
scale factors out of each K-block, so the block sums raw codes and, at the block
edge, `acc*scale` (+`min*sum(A)`) folds into totals. A vg1x4 group w lies in
ZA64 tile w%8, the unit one `ZERO` clears, which bounds what a pass can hold:

- **One f16 row.** Reading a ZA group waits for every FMLA into it to retire, so a
  fold right after a block's last FMLAs stalls the unit. The block sums are
  double-buffered instead: block b sums into set b%2 and folds one 8-depth step
  into block b+1. A pass is either 4 bands with one sum group each, or up to 3
  bands with even and odd depths in separate groups (folded with the same
  scale), because 3 FMLA chains do not cover the accumulate latency. Band
  counts that are a multiple of 4 run in passes of 4, the rest in balanced
  passes of up to 3.
- **Several rows, and one bf16 row.** Rows × bands ≤ 8 sum groups in tiles
  0-3, totals in tiles 4-7, the bands spread evenly over passes; the fold runs
  at the block edge, and a one-row pass of 4 bands or fewer splits depth parity
  as above.

Group indices are register arithmetic, never a table: a core-side load inside
streaming code stalls the unit. A column's bits depend on its pass's layout,
which matters wherever two paths must agree (§2.13). bf16 folds with a bf16 copy
of the scales, built once per weight set. Q4 keeps the GEMV to m ≤ 7, not 4,
because its MOPA path also pays the panel unpack.

A is usually written by the calling core just before the call, and the SME
unit's first read of each line the core stored waits on that core, one line at
a time inside the loop. The kernel loads all of A up front, four vectors per
load with no result used, so those waits overlap; the library's own producers of
SME inputs avoid them entirely (§2.14).

### 2.9 Q4 MOPA path: one shared unpack per B-block

Above the GEMV rows, Q4 unpacks each N-block of B once into a three-slot ring of
f16 (or bf16) panels and multiplies it with the dense `run_streaming`, so the
result is bit for bit eager dequant plus the dense GEMM. The unpack stays in the
streaming region and is all ZA work: LUTI4 turns a 64-byte load into one depth
of a band's four tiles, one FMLA adds `code*scale` onto ZA holding zero or the
min, and MOVA plus four stores write the depth into the four tiles' panels — 8
depths per `ZERO`.
bf16 must round once from f32, so there FMLAL widens `code*scale` exactly into
f32 ZA; its layout splits each row into even and odd columns, and BFCVTN
interleaves them back while rounding. The affine f16 form rounds
`scale*code + min` once, and the eager `dequant_q4_with` computes it exactly in
f64 to match. The unpack costs about one 32-row M-tile of MOPA issue per weight
tile, shared across all of M.

Scheduling lives in `csrc/panel_ring.h` (shared with §2.10). Build units and
GEMM items are two cursors that persistent workers pull from: a build unit is
ready once its slot's previous block is fully multiplied, a GEMM item once its
block is built, and a worker takes whichever is ready (building first while
fewer than two blocks lead the GEMM cursor). Nothing waits on one specific item,
because E-core items run 5–47× slower and any P worker parked behind one stalls
the call. GEMM items are M-chunk × N-subrange, at least 32 per block, so no
single item is long. Block 0's first item per M-chunk packs that chunk of A.

### 2.10 B not pre-packed

A call that does not pre-pack B (`matmul_f16`, `gemm_f32`, …) builds B inside
the parallel region rather than packing it all on the calling thread first. The
f16, bf16, widening and f32 drivers pick one of three schedules:

- **Per-worker columns** (`use_cols`: m ≤ 512, or m ≤ 4096 with ≥ 64 N-tiles and
  A ≤ 8 MB). The first units pack A by M-chunk; each item then takes a narrow
  N-chunk and multiplies it while it is in that worker's cache. At small m a B
  tile meets only a few M-chunks, so a shared panel would cost a cross-core
  (often cross-cluster) round trip per tile for nothing. Each item re-reads all
  of A and spans all of M, which is why a large A, or a narrow N that leaves a
  few long items for E-cores to straggle on, goes to the ring instead.
  - With row-major B and C the item does not pack: its first M-tile multiplies
    straight from B's rows (`run_direct`), so fetching B overlaps the MOPAs
    instead of preceding them on the same core. One load is one 128-byte line —
    two f16 tiles, one f32 super-tile, or (widening) rows 2p and 2p+1 zipped
    into the interleaved bands; a half-line load costs as much as a full one.
    When more M-tiles follow, that pass also writes the line out as packed tiles
    and the rest run the packed kernel on them, because re-walking B's rows per
    M-tile at a stride that is a multiple of 8 KB puts every load in one L2
    bank, and two ZA16 tiles leave no second column to interleave. f16/bf16 pack
    instead at a power-of-two stride from three M-tiles up; f32 and widening
    always go direct, and f32 prefetches B 16 rows ahead on the capturing pass.
  - Otherwise the item NEON-packs its N-chunk into a private panel first.
- **Panel ring** (the rest): §2.9's schedule with a plain copy in place of the
  unpack. f32 ring blocks are 4 MB rather than `run_packed`'s 12 MB, so three
  slots fit L2.
- **In-place GEMV** (m ≤ 4, f32 m ≤ 2, row-major B; §2.8). Other B layouts pack
  per worker and run the packed GEMV.

Every route keeps each output's K-order identical to the pre-packed entry, and
the tests check them against it bit for bit, across transposed, strided and
column-major layouts.

---

### 2.11 Keeping the SME unit awake

A P-cluster's SME unit idles within a few hundred nanoseconds of its last
streaming instruction and the next call pays about a microsecond to wake it, so
a loop of short GEMVs alternating with NEON layernorm, attention and activations
pays it on every call. While an `SmeWarm` guard is alive, a helper thread at
user-interactive QoS issues short bursts of streaming vector FMLAs back to back
(`csrc/sme_warm.c`, `src/warm.rs`). The bursts leave ZA alone, which is what
lets them run beside a call: a call sharing the unit with vector work loses
nothing measurable, but sharing it with ZA work halves a one-row Q4 GEMV. So the
helper never tracks calls in flight; an entry point only sets an activity flag
(a relaxed load and, when clear, a store), since any read-modify-write on a line
the helper polls adds ~0.1 µs to the call. With no call for a few milliseconds
the helper sleeps. macOS has no affinity control, so on a chip with several
P-clusters it helps only when the scheduler shares the caller's cluster.

### 2.12 Spinning workers for short NEON passes

One layer's KV attention over a short cache is a few microseconds on one core,
about what `dispatch_apply` charges to wake its workers. While a `HotPool` is
alive (`src/pool.rs`), its workers spin, each on its own 128-byte mailbox, and a
call posts one job to as many of them as it has items to spare, then claims
items itself. The job lives on the caller's stack: the caller returns only once
every engaged worker has counted itself in the job's `finished` counter, which
sits beside the item cursor so the caller waits on one cache line rather than
missing on each worker's mailbox. A worker counts a job as pending whenever its
posted sequence is ahead of its done one, so a job posted before the worker
started is still taken. Shutdown takes the pool's
in-use flag before stopping the workers, so a job is never posted to one that is
leaving. KV attention splits by KV-head group from 64 cached keys, bit for bit
the one-thread result. Idle workers back off to sleeping after a few
milliseconds. The atomics the workers poll (`LIVE`, each mailbox) sit on cache
lines of their own, apart from those a call writes (`IN_USE`, `SEQ`, `JOB`): a
read-modify-write on a line other cores are spinning on waits for each of them
to give it up. A job can also run *alongside* the caller (`pool::alongside`):
posted to the workers only, while the caller does SME work, the caller draining
whatever is still unclaimed when it gets there (a worker may be asleep).

A pool also holds an `SmeWarm` (§2.11): the loops it serves are the ones whose
SME calls follow NEON work. `HotPool::new` starts one worker per P-core left
after the calling thread and that helper (`hw.perflevel0.physicalcpu`, two on a
4-P-core M5); a worker beyond that would share a core with one of them.

### 2.13 Chained GEMVs: `Mlp` and `SelfAttention`

In a transformer block the SME unit idles while NEON runs the activation
between the MLP's two GEMVs, and the attention between the attention block's
two. For one row with 4-bit weights and a `HotPool` alive, `Mlp` and
`SelfAttention` run that NEON work on pool workers beside the GEMVs instead. The
first GEMV publishes an output counter (`done`) after each pass; the workers
take its outputs as they appear, and the second GEMV, started as soon as the
first returns, waits before each K-block until the workers' counter (`ready`)
covers it (`q4_chain` in `csrc/gemm_f16f16_q4.h`, `src/mlp.rs`,
`src/self_attention.rs`).

- Core-side atomics stall streaming code (on M5 a release store ~60 ns while
  the unit's stores drain, a load ~20 ns), so a pass's columns are published one
  K-block into the next pass, the last pass after the streaming code returns,
  and `ready` is read again only past the depths it last covered. The in-kernel
  publish is a store barrier and a plain store (~40 ns): a consumer needs only
  the pass's stores ordered before it, not the earlier loads a release also
  orders.
- `SelfAttention` keeps its `qkv` projection with outputs grouped by KV head
  (each group's queries, key and value adjacent: an exact column permutation of
  the 4-bit weights, made once), so the first pass already holds whole groups. A
  worker appends its group's key and value to the cache row and attends for it;
  `ready` advances over the leading run of finished groups.
- Results are the same bits with or without a pool. Both paths run the grouped
  projection, and a chain is used only where the plain call would also be one
  call on the calling thread (`gemm_sme_f16f16_q4_single`), so every column
  falls in the same pass either way (§2.8). Larger shapes spread over the cores
  unchained, which is faster for them anyway.
- The activation writes its output to a separate buffer, not back over its
  input (§2.14).

### 2.14 Feeding SME from the cores

The SME unit's first read of a cache line a core has just stored waits on that
core: ~20-30 ns a line, one after another inside a kernel's loop, on M5 0.15 µs
over a 384-wide activation vector and 0.4 µs over 1536. Cleaning the line out
(`DC CVAC`/`CIVAC`) or a prefetch hint does not help. A store that bypasses the
core's cache does: everything the library writes for an SME kernel to read next
— the f32→f16 conversion at a `Linear`'s input, the norms' f16 output, the
activation and GLU passes, the GEMV's broadcast A — is stored with `STNP`
(`na_stnp16` in `csrc/neon_act.h`). That works only for lines the writing core
has not just read, so those passes write out of place. Kernels whose input may
come from user code touch it up front (§2.8).

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
generated code unchanged. `csrc/transpose16.h` and `csrc/panel_ring.h` are the
exceptions: genuinely shared headers, the first included by all four 16-bit
drivers, the second (the §2.9/§2.10 scheduler) by the f16, bf16, widening and
f32 drivers. Every fragment needs an entry in `build.rs`'s `CSRC` list for rerun
tracking, since `cc` never sees it.

### 3.2 The runtime probe

`src/probe.rs` and `csrc/sme_probe.c` detect capabilities via `sysctlbyname` on
the `hw.optional.arm.FEAT_SME*` flags, cached once in a `OnceLock`. `Caps` holds
one bool per group: `sme` (base, M4+), and `sme_f16f16`, `sme_b16b16`,
`sme_i16i64`, `sme_f64f64` (M5+). On non-Apple targets every flag is false and
everything falls back to the scalar reference. Per-dtype entry points consult
the matching flag before dispatching to an M5 kernel. The same probe reads the
P-core count (`hw.perflevel0.physicalcpu`) that sizes a `HotPool` (§2.12).

---

## 4. Rust module layout

`lib.rs` is a thin re-export root: crate docs and `pub use`.

- **`element.rs`** — `Accum` (F32 = widening MOPA, M4+; F16 / Bf16 = native
  16-bit accumulate on M5 `FEAT_SME_F16F16` / `FEAT_SME_B16B16`), the sealed
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
  `attention_half.rs` hold the flash-attention drivers, `kv_attention.rs` the
  KV-cache attention (§2.7).
- **`warm.rs`** — `SmeWarm`, the per-call activity flag (§2.11), and the
  `SME_GEMM_TRACE` per-call report, which hangs off the same per-call hook.
- **`pool.rs`** — `HotPool`, the spinning workers for short NEON passes, and
  jobs that run alongside the caller (§2.12).
- **`mlp.rs`, `self_attention.rs`** — `Mlp` and `SelfAttention`, transformer
  blocks that chain their GEMVs with the NEON work between them (§2.13).
- **`block.rs`** — `Block`, the pre-norm layer: each half normalizes the f32
  residual stream into thread-local f16 scratch, as the next matmul takes it,
  and adds `SelfAttention` or `Mlp` back in. It holds no state of its own
  beyond the parts, so a layer of another shape is the same parts in another
  order.
- **`convert.rs`** — f16↔f32 slice conversion at a model's edges, NEON (the
  `half` crate converts one value per inline-asm `FCVT`).
- **`linear.rs`, `layout.rs`, `nn.rs`** — the model-level layer over the
  kernels. `Linear` holds a layer's weights (Q4 with a lazily built f16 panel,
  or f16) and an optional bias, and routes each call by row count: the Q4 GEMV
  below 8 rows, where reading the weights is the cost, and the dense f16 GEMM on
  the panel from 8, where the Q4 MOPA path would unpack every block again per
  call (the two agree bit for bit). Its bias goes in front of any caller
  epilogue, where it folds into the accumulator; with `rms_norm_input` a per-row
  `1/rms(x)` goes in front of that (the norm's weight is folded into W by
  `WeightLayout::scale_inputs`). `GatedLinear` stores gate and up interleaved in
  whole 32-column tiles of one `Linear`, so each column quantizes and
  accumulates exactly as it would alone, then runs the gated activation as a
  NEON pass (`csrc/neon_ops.c`, with `neon_act.h`'s activations). `ModelFloat`
  lets the helpers take f32 or f16 and convert at their edges in thread-local
  scratch. `WeightLayout` / `prepack` accept weights as checkpoints store them;
  `Q4Weights::quantize` lives with the Q4 kernels (`q4/quantize.rs`). `KvCache`
  (in `kernels/kv_attention.rs`) wraps the KV-cache attention. `nn`'s norms run
  a row in NEON on the calling thread: one row is too little work to thread or
  to stream. `nn::Norm` is the layer form, holding weight, bias and eps.
- **`reference.rs`** — the portable scalar GEMM oracle (§6).
- **`probe.rs`** — runtime capability detection and the P-core count (§3.2).
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

The store is node-major: it reads a 4-row block of the live ZA tile into Z
registers and dispatches each node once per block across all its rows, hoisting
invariant operands (COL vectors, scalars, activation constants) out of the
per-row loop with no memory bounce. Interpreting the graph per output row
instead — an out-of-line node loop with jump-table dispatch and operands
reloaded each row — would cost ~500K dispatches for a 4096² output. The
gelu/silu/tanh rational approximation sits in a `noinline` helper so the common
add/mul/relu nodes still inline. A fused epilogue costs a few percent over the
bare GEMM when compute-bound and up to about a third at the shallowest
store-bound K.

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
| `tests/correctness/`   | 120   | dtypes, shapes, layouts, packed/batched, epilogue graphs      |
| `tests/validation.rs`  | 19    | pre-FFI footprint asserts and panic contracts                 |
| `tests/proptest.rs`    | 17    | randomized shapes/strides/operands against the oracle         |
| `tests/guard_page.rs`  | 13    | reads past the declared footprint fault on a `PROT_NONE` page |
| unit tests             | 11    | in-module invariants (unpackers invert their packers)         |
| doctests               | 5     | the documented API examples                                   |
| `tests/concurrency.rs` | 4     | one GEMM from many threads, bit-identical, across branches    |

189 total. `tests/correctness/main.rs` holds the shared oracles and shape
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
- **Probe a change with the shape it dominates.** A pack change shows up only
  where the packed operand is most of the traffic: the A pack with `m ≫ n`, the
  B pack with `n ≫ m`. A square shape reads 1.00× for a fix worth more
  elsewhere.
- **Confirm microbenchmarks in the library, under load.** A one-core,
  L2-resident kernel can rank layouts in the opposite order to the full machine
  reading DRAM.

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
