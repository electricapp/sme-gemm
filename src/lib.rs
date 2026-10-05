//! Apple-Silicon-tuned SME GEMM.
//!
//! Streaming-mode SME (Scalable Matrix Extension) matrix-multiply kernels for
//! Apple M4+ -- the fast on-CPU GEMM path that the rest of the Rust ecosystem
//! (gemm, faer, candle, ndarray) does not have. Kernels are hand-written
//! against `arm_sme.h` and tuned for Apple's *per-cluster shared* SME unit
//! (single-threaded, streaming-mode, SVL=512), not the server-SME model.
//!
//! Supported element types: f16 (f32 or f16 accumulate), bf16 (f32 or bf16
//! accumulate), f32, f64, i8->i32, i16->i64. Native 16-bit accumulate needs
//! M5 (`FEAT_SME_F16F16` / `FEAT_SME_B16B16`).
//!
//! On non-Apple targets every entry point transparently falls back to a
//! portable scalar reference, so the crate builds and runs everywhere.
//!
//! `flash_attention_{f32,f16,bf16}` compute `softmax(scale*Q@K^T)@V` a key
//! block at a time, so the `m x n` score matrix is never materialized.
//!
//! Above the kernels sit the pieces a model is built from: [`Linear`] (a
//! layer's 4-bit or f16 weights, quantized from either [`WeightLayout`], with
//! the kernel picked by row count and an optional folded input `RMSNorm`),
//! [`GatedLinear`] (a `SwiGLU` / `GeGLU` MLP's gate and up as one matmul),
//! [`KvCache`] (f16 key/value cache and its attention), [`nn`] (layer and RMS
//! norm), [`SmeWarm`] (keeps the SME unit awake between calls) and [`HotPool`]
//! (spinning workers for short NEON passes). `SME_GEMM_TRACE=1` prints every
//! SME call with its shape, time, and the idle gap before it.
//!
//! Every dtype has a real `prepack_*` that builds the kernel's packed weight
//! panel, so `*_packed` reuse works across f16, bf16, f32, f64, i8 and i16.
//! `*_batched` (and `*_batched_ep`) exist for f16, bf16, i8, f32, and f64. The
//! f64 and i16 paths are M5-only, gated behind the `FEAT_SME_F64F64` /
//! `FEAT_SME_I16I64` probes; without the extension `prepack_*` stores a plain
//! row-major copy and the entry points fall back to the scalar reference.

#[cfg(feature = "burn")]
pub mod burn;
#[cfg(feature = "candle")]
pub mod candle;
pub mod probe;
mod reference;

mod convert;
mod element;
mod epilogue;
mod exec;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod ffi;
mod kernels;
mod layout;
mod linear;
mod mlp;
pub mod nn;
mod pool;
mod warm;

pub use layout::{Prepack, WeightLayout, prepack};
pub use linear::{Gate, GatedLinear, Linear, ModelFloat};
pub use mlp::Mlp;
pub use pool::HotPool;
pub use probe::{Caps, caps, has_sme};
pub use warm::SmeWarm;

pub use element::{Accum, Element, Packed};
pub use epilogue::{Dequant, Epilogue};
pub use exec::{
    Gemm, MapElem, PackedEpilogue, PackedQuant, RowReduce, epilogue_map, softmax_gemm_f32,
    softmax_rows, softmax_rows_with_max,
};

pub use kernels::attention::{FlashParams, flash_attention_f32, flash_attention_f32_with};
pub use kernels::attention_half::{
    flash_attention_bf16, flash_attention_bf16_with, flash_attention_f16, flash_attention_f16_with,
};
pub use kernels::bf16::{
    gemm_bf16, matmul_bf16, matmul_bf16_batched, matmul_bf16_batched_ep, matmul_bf16_packed,
    prepack_bf16,
};
pub use kernels::f16::{
    gemm_f16, matmul_f16, matmul_f16_batched, matmul_f16_batched_ep, matmul_f16_packed, prepack_f16,
};
pub use kernels::f32::{
    gemm_f32, matmul_f32, matmul_f32_batched, matmul_f32_batched_ep, matmul_f32_packed, prepack_f32,
};
pub use kernels::f64::{
    gemm_f64, matmul_f64, matmul_f64_batched, matmul_f64_batched_ep, prepack_f64,
};
pub use kernels::int::{
    gemm_i8, gemm_i16, matmul_i8, matmul_i8_batched, matmul_i8_batched_dequant, matmul_i8_packed,
    matmul_i8_packed_dequant, matmul_i16, matmul_i16_batched, matmul_i16_batched_dequant,
    matmul_i16_dequant, matmul_i16_packed, matmul_i16_packed_dequant, prepack_i8, prepack_i16,
};
pub use kernels::kv_attention::{KvCache, attention_kv_causal_f16, attention_kv_f16};
pub use kernels::q4::{
    Q4_BLOCK, Q4Form, Q4Params, Q4Weights, dequant_q4, dequant_q4_bf16, dequant_q4_bf16_with,
    dequant_q4_with, matmul_q4, matmul_q4_bf16, matmul_q4_bf16_ep, matmul_q4_ep,
};
