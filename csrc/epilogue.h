// Fused GEMM epilogue: an ordered op-graph applied to the accumulator slice
// while it is still live in streaming-SVE registers, between the ZA read and the
// store to C. This deletes the separate post-pass over C that bias/activation
// would otherwise cost (a full extra streaming of the output in the memory-
// bound regimes).
//
// The epilogue is an ORDERED LIST OF NODES (EpNode), the runtime analog of a
// CUTLASS epilogue visitor tree. The running value x starts as the post-beta
// accumulator the store already computes; each node mutates x in sequence:
//   x <- beta*(A@B)
//   for node in nodes: x <- node(x)
//   [x <- alpha*C + x  if read_dst]
//
// The descriptor is passed by const pointer; NULL (or n_nodes==0) means "no
// epilogue" and the kernel takes its original fast path unchanged. For f16/bf16
// the row/col/tensor operands are in T (f16/bf16); for the i8 dequant path
// everything is f32 (post-dequant domain). ACT reuses the activation menu
// (ReLU exact; GELU/SiLU/tanh/sigmoid all via one in-register rational tanh,
// ~4e-2 worst-case vs libm -- see the cross-path divergence note below).
// The bf16 GEMM accumulates in bf16 (B16B16 MOPA) but its fused EPILOGUE runs
// the entire op-graph in f32 (upcast acc + operands, compute, round to bf16),
// so bf16 has the SAME full op set as f16/f32 (incl. div/sqrt/transcendentals).
//
// =============================================================================
// NAVIGABILITY MAP: what lives where across epilogue*.h
// =============================================================================
//
// CONCEPT. This file is the op-graph INTERPRETER that every gemm_*.c kernel
// includes. The Rust side (src/epilogue/mod.rs) builds an array of EpNode and hands
// it across FFI in an ep_desc{16,_f32,_f64}. At the ZA->C store, the kernel reads
// the live accumulator slice into registers and walks the node list, mutating the
// running value x in order, then stores. No separate pass over C.
//
// EpNode FIELDS (one op-graph node; layout mirrored in Rust #[repr(C)]):
//   op     - enum ep_op: which operation (see enum below)
//   aux    - for EP_OP_ACT, the activation kind (enum ep_act); else unused
//   scalar - f32 operand for the *_SCALAR ops; also the alpha for LEAKY_RELU/ELU
//   ptr    - operand base for ROW/COL/TENSOR ops (T* for f16/bf16, f32* for the
//            i8/i16 dequant domain); NULL for scalar/act ops
//   ld     - row stride (in elements) for the TENSOR ops; 0 means "n" but Rust's
//            resolve_nodes already rewrites 0->n before the call
//
// OP-ENCODING per kind (how op/aux/scalar/ptr/ld are read):
//   *_SCALAR (add/mul/sub/div/max/min)  -> uses `scalar`,           ptr/ld unused
//   *_ROW    (per-M operand, length m)  -> reads ptr[i]   (row i),  scalar/ld unused
//   *_COL    (per-N operand, length n)  -> reads ptr[j]   (col j),  scalar/ld unused
//   *_TENSOR (full m*n, row-major)      -> reads ptr[i*ld + j],     scalar unused
//   EP_OP_ACT                           -> act = aux (enum ep_act); scalar = alpha
//                                          for LEAKY_RELU/ELU, else ignored
//
// FUNCTION FAMILIES (per dtype: f16, bf16, f32, f64; + the i8/i16 dequant path):
//
//   ep_apply_nodes_scalar_{f16,bf16,f32,f64,dq}(nodes, n, x, i, j) -> scalar
//       Walk the whole graph for ONE cell, in scalar f32 (f64 for the f64 path).
//       Used by the strided / general / col-major-with-TENSOR store paths where a
//       vectorized slice is not available, and by the dequant cells.
//
//   ep_apply_nodes_{f16,f32,f64}(p, pst, x, ...) -> vector
//   ep_apply_nodes_bf16_f32(p, pst, lo, hi, ...)  (bf16 splits a slice into two
//       f32 halves lo/hi and evaluates the graph in f32)
//       Walk the whole graph for ONE vector slice (a ZA row/col already in a
//       streaming-SVE register). The per-cell vector form.
//
//   ep_block_rowmajor_{f16,bf16,f32,f64}(...) + the EP_STORE_TILE_ROWMAJOR_*
//   macros
//       The NODE-MAJOR register-resident store: read a 4-row block of the live ZA
//       tile into Z registers, then dispatch each op-graph node ONCE across the
//       block (invariant COL/scalar/act operands hoisted), and store. This is the
//       hot store path for the common row-major output. Dispatching per row
//       instead, through an out-of-line node loop, costs ~ms on large outputs.
//       The EP_BLK_*_OP / EP_OPX / EP_OPL / EP_OPH macro families below each
//       ep_block_rowmajor_* are the per-op-kind expansions the dispatch switch
//       expands into.
//
//   ep_store_{f16,bf16,f32,f64}(p, pst, ptr, acc, vb, va, read_dst, nodes, ...)
//       The non-row-major / col-major store helper: apply beta, run the op-graph on
//       one slice, optionally fold read_dst (alpha*C), and store one slice.
//
//   ep_has_tensor(nodes, n)  - true if any node reads a TENSOR operand; the
//       col-major store uses it to route TENSOR graphs to the scalar path (a
//       col-major TENSOR read is a strided gather, not vectorizable).
//
// TRANSCENDENTAL HELPERS (streaming-SVE, NO libm; libm is not callable from a
// streaming region). Each is a vectorized polynomial/range-reduction approx:
//   ep_exp_f32 / ep_log_f32 / ep_erf_f32   (+ _f64 variants, + ep_softplus/ep_tanh)
//       Used to build the Group-B activations (exp/log/elu/selu/softplus/mish/
//       gelu_exact) and GELU/SiLU/sigmoid in-register. The f16 path upcasts to f32
//       and reuses these (forward-declared near the top, defined in the f32
//       section). The exact (libm) versions ep_act_scalar* are used by the
//       strided / col-major-TENSOR fallback store arms; those arms DO run inside
//       the streaming region, so each libm call pays a compiler-inserted
//       SMSTOP/SMSTART round trip (ABI-correct but not free -- see the scalar
//       evaluator section header). They are reserved for those rare paths where
//       accuracy beats speed; never route a hot store through them.
//
// DEQUANT CELL HELPERS (i8->i32 and i16->i64 quantized paths):
//   ep_dequant_cell      - scale an i32 accumulator into f32 (per-tensor or per-N
//                          scale_n), then run the op-graph.
//   ep_dequant_cell_i64  - same for an i64 (i16 SMOPA) accumulator; the i64->f32
//                          narrowing may lose precision past the 24-bit mantissa.
//
// FILES. This header holds the enums, EpNode/ep_desc* structs and the shared
// macros; the evaluators live in the parts it includes at the bottom:
//   epilogue_scalar.h  ep_act_scalar*, ep_apply_nodes_scalar_*, dequant cells
//   epilogue_f16.h     ep_*_f16, EP_STORE_TILE_ROWMAJOR_F16, ep_block_rowmajor_f16
//   epilogue_bf16.h    ep_*_bf16 (the f32-domain bf16 evaluators)
//   epilogue_f32.h     ep_*_f32; the vector transcendentals live here
//   epilogue_f64.h     ep_*_f64
// =============================================================================

#ifndef SME_GEMM_EPILOGUE_H
#define SME_GEMM_EPILOGUE_H

#include <arm_sme.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef __fp16 ep_f16;
typedef __bf16 ep_bf16;

// Saturating m*n*k flop estimate, used by every kernel's parallel/serial and
// pack-strategy thresholds. The Rust side validates that C covers m*n (so
// (uint64_t)m*n cannot overflow a real call), but the *k step still can for
// absurd (unallocatable) shapes -- saturate it to UINT64_MAX rather than wrap,
// so a huge problem can never misclassify as "small" and take a serial path.
static inline uint64_t ep_flops(size_t m, size_t n, size_t k) {
    uint64_t mn = (uint64_t)m * (uint64_t)n;
    if (k != 0 && mn > UINT64_MAX / (uint64_t)k) {
        return UINT64_MAX;
    }
    return mn * (uint64_t)k;
}

// a*b, or SIZE_MAX on overflow. Used for heap-allocation sizes: SIZE_MAX makes
// the following malloc/calloc/apack_scratch/xmalloc return NULL, which every
// alloc site already treats as OOM (the kernel returns its error code and the
// Rust wrapper falls back) -- no extra branch needed. Defense in depth: a
// Rust-validated call cannot reach an overflowing product (the operand slices
// bound m*k / k*n below isize::MAX), but a direct C-FFI caller that bypasses
// validation could. Nest for products of more than two factors.
static inline size_t ep_cmul(size_t a, size_t b) {
    size_t r;
    return __builtin_mul_overflow(a, b, &r) ? SIZE_MAX : r;
}

enum ep_act {
    EP_ACT_NONE = 0,
    EP_ACT_RELU = 1,
    EP_ACT_GELU = 2,    // tanh-approx GELU
    EP_ACT_SILU = 3,    // x * sigmoid(x)
    EP_ACT_TANH = 4,    // tanh(x)
    EP_ACT_SIGMOID = 5, // 1 / (1 + exp(-x))
    // --- Group A: exact / no transcendental. Available on every float dtype
    // (bf16 evaluates its epilogue in f32, so divide-using ops work there too).
    EP_ACT_LEAKY_RELU = 6,  // x>=0 ? x : alpha*x       (alpha in node->scalar)
    EP_ACT_RELU6 = 7,       // clamp(x, 0, 6)
    EP_ACT_HARDSIGMOID = 8, // clamp(x/6 + 0.5, 0, 1)
    EP_ACT_HARDSWISH = 9,   // x * clamp(x/6 + 0.5, 0, 1)
    EP_ACT_ABS = 10,        // |x|
    EP_ACT_NEG = 11,        // -x
    EP_ACT_SQUARE = 12,     // x*x
    EP_ACT_SIGN = 13,       // sign(x) in {-1,0,1}
    EP_ACT_SQRT = 14,       // sqrt(x)
    EP_ACT_SOFTSIGN = 15,   // x / (1 + |x|)
    EP_ACT_RECIP = 16,      // 1 / x
    EP_ACT_RSQRT = 17,      // 1 / sqrt(x)
    // --- Group B: need vectorized exp/log/erf. All float dtypes (bf16 via its
    // f32 epilogue) + the i8/i16 dequant f32 domain.
    EP_ACT_EXP = 18,        // exp(x)
    EP_ACT_LOG = 19,        // log(x)
    EP_ACT_ELU = 20,        // x>=0 ? x : alpha*(exp(x)-1)   (alpha in node->scalar)
    EP_ACT_SELU = 21,       // 1.0507*(x>=0 ? x : 1.6733*(exp(x)-1))
    EP_ACT_SOFTPLUS = 22,   // log1p(exp(x)) (stable form)
    EP_ACT_MISH = 23,       // x * tanh(softplus(x))
    EP_ACT_GELU_EXACT = 24, // 0.5*x*(1+erf(x/sqrt(2)))
};

// SELU constants (standard).
#define EP_SELU_ALPHA 1.6732632423543772f
#define EP_SELU_LAMBDA 1.0507009873554805f

// Op-graph node opcodes. Each node mutates the running accumulator value x.
enum ep_op {
    EP_OP_ADD_SCALAR = 0,  // x += scalar
    EP_OP_MUL_SCALAR = 1,  // x *= scalar
    EP_OP_ADD_ROW = 2,     // x += ptr[i]       (per-M, length m)
    EP_OP_MUL_ROW = 3,     // x *= ptr[i]
    EP_OP_ADD_COL = 4,     // x += ptr[j]       (per-N, length n)
    EP_OP_MUL_COL = 5,     // x *= ptr[j]
    EP_OP_ADD_TENSOR = 6,  // x += ptr[i*ld + j] (m*n row-major)
    EP_OP_MUL_TENSOR = 7,  // x *= ptr[i*ld + j]
    EP_OP_ACT = 8,         // x = act(x), aux = enum ep_act
    EP_OP_MAX_SCALAR = 9,  // x = max(x, scalar)
    EP_OP_MIN_SCALAR = 10, // x = min(x, scalar)
    // --- elementwise binary with scalar/row/col/tensor operands ---------------
    EP_OP_SUB_SCALAR = 11, // x -= scalar
    EP_OP_SUB_ROW = 12,    // x -= ptr[i]
    EP_OP_SUB_COL = 13,    // x -= ptr[j]
    EP_OP_SUB_TENSOR = 14, // x -= ptr[i*ld + j]
    EP_OP_DIV_SCALAR = 15, // x /= scalar
    EP_OP_DIV_ROW = 16,    // x /= ptr[i]
    EP_OP_DIV_COL = 17,    // x /= ptr[j]
    EP_OP_DIV_TENSOR = 18, // x /= ptr[i*ld + j]
    EP_OP_MAX_ROW = 19,    // x = max(x, ptr[i])
    EP_OP_MAX_COL = 20,    // x = max(x, ptr[j])
    EP_OP_MAX_TENSOR = 21, // x = max(x, ptr[i*ld + j])
    EP_OP_MIN_ROW = 22,    // x = min(x, ptr[i])
    EP_OP_MIN_COL = 23,    // x = min(x, ptr[j])
    EP_OP_MIN_TENSOR = 24, // x = min(x, ptr[i*ld + j])
};

// True if an op-graph node reads a TENSOR operand (cannot be vectorized by the
// col-major store, which routes those graphs to the scalar path).
#define EP_OP_IS_TENSOR(op)                                                                    \
    ((op) == EP_OP_ADD_TENSOR || (op) == EP_OP_MUL_TENSOR || (op) == EP_OP_SUB_TENSOR ||       \
     (op) == EP_OP_DIV_TENSOR || (op) == EP_OP_MAX_TENSOR || (op) == EP_OP_MIN_TENSOR)

// One epilogue op-graph node. Layout mirrored exactly in Rust (#[repr(C)]).
//   op:     enum ep_op
//   aux:    activation kind for EP_OP_ACT, else unused
//   scalar: f32 operand for *_SCALAR ops (already in the real/native domain)
//   ptr:    operand base for ROW/COL/TENSOR ops (T* for f16/bf16, f32* for i8)
//   ld:     row stride (elements) for TENSOR ops
typedef struct {
    uint32_t op;
    uint32_t aux;
    float scalar;
    const void *ptr;
    size_t ld;
} EpNode;

// Pin the layout the Rust #[repr(C)] mirror (src/epilogue/mod.rs) asserts against:
// 32 bytes on arm64, with the 4-byte hole before `ptr`. Drift = silent FFI UB.
_Static_assert(sizeof(EpNode) == 32, "EpNode must match the Rust mirror (32 bytes)");
_Static_assert(offsetof(EpNode, op) == 0, "EpNode.op offset must match the Rust mirror");
_Static_assert(offsetof(EpNode, aux) == 4, "EpNode.aux offset must match the Rust mirror");
_Static_assert(offsetof(EpNode, scalar) == 8, "EpNode.scalar offset must match the Rust mirror");
_Static_assert(offsetof(EpNode, ptr) == 16, "EpNode.ptr offset must match the Rust mirror");
_Static_assert(offsetof(EpNode, ld) == 24, "EpNode.ld offset must match the Rust mirror");

// 16-bit epilogue descriptor (f16 and bf16 share the layout): an ordered list of
// op-graph nodes. NULL nodes / n_nodes==0 means identity.
typedef struct {
    uint32_t n_nodes;
    const EpNode *nodes;
} ep_desc16;

// f32 dequant epilogue descriptor: the per-tensor/per-N scale that turns the i32
// accumulator into the real (f32) domain, then an op-graph applied in f32.
typedef struct {
    float scale;          // per-tensor scale (used when scale_n == NULL)
    const float *scale_n; // per-N scale, length n, or NULL
    uint32_t n_nodes;
    const EpNode *nodes;
} ep_dq_f32;

// Pinned against the Rust #[repr(C)] mirror (src/ffi.rs DqF32).
_Static_assert(sizeof(ep_dq_f32) == 32, "ep_dq_f32 must match the Rust mirror");
_Static_assert(offsetof(ep_dq_f32, scale) == 0, "ep_dq_f32.scale offset");
_Static_assert(offsetof(ep_dq_f32, scale_n) == 8, "ep_dq_f32.scale_n offset");
_Static_assert(offsetof(ep_dq_f32, n_nodes) == 16, "ep_dq_f32.n_nodes offset");
_Static_assert(offsetof(ep_dq_f32, nodes) == 24, "ep_dq_f32.nodes offset");

// f32 epilogue descriptor: an ordered op-graph applied in the f32 accumulator
// domain (operands are f32). NULL nodes / n_nodes==0 means identity.
// row_sum / row_max are OPTIONAL per-M reduction outputs (length m, NULL to
// skip), written once per row right after that row's last N-tile store, while
// the row is still L1-hot. See the reduction block in gemm_f32.c for why it is
// there and not fused into the store itself.
typedef struct {
    uint32_t n_nodes;
    const EpNode *nodes;
    float *row_sum;
    float *row_max;
} ep_desc_f32;

// True if this descriptor asks for either per-row reduction.
static inline int ep_has_reduce(const ep_desc_f32 *ep) {
    return ep && (ep->row_sum || ep->row_max);
}

// f64 epilogue descriptor: an ordered op-graph applied in the f64 accumulator
// domain (operands are f64). NULL nodes / n_nodes==0 means identity.
typedef struct {
    uint32_t n_nodes;
    const EpNode *nodes;
} ep_desc_f64;

// activations, so it needs these before the definitions appear.
static svfloat32_t ep_exp_f32(svbool_t p32, svfloat32_t x) __arm_streaming;
static svfloat32_t ep_log_f32(svbool_t p32, svfloat32_t x) __arm_streaming;
static svfloat32_t ep_softplus_f32(svbool_t p32, svfloat32_t x) __arm_streaming;
static svfloat32_t ep_act_rational_f32(svbool_t p32, svfloat32_t x, int act,
                                       float alpha) __arm_streaming;
// Also used by the bf16 epilogue (which computes in f32) before its definition.
static inline svfloat32_t ep_act_apply_f32(svbool_t p32, svfloat32_t x, int act,
                                           float alpha) __arm_streaming;


#include "epilogue_scalar.h"
#include "epilogue_f16.h"
#include "epilogue_bf16.h"
#include "epilogue_f32.h"
#include "epilogue_f64.h"

#endif // SME_GEMM_EPILOGUE_H
