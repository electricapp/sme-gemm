//! Per-dtype GEMM entry points: the public `matmul_*` / `gemm_*` / `prepack_*`
//! surface, grouped by element type.

pub(crate) mod attention;
pub(crate) mod attention_half;
pub(crate) mod bf16;
pub(crate) mod f16;
pub(crate) mod f32;
pub(crate) mod f64;
pub(crate) mod int;
pub(crate) mod kv_attention;
pub(crate) mod q4;
