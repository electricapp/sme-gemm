//! Shared execution glue: the [`Gemm`] builder, the pre-FFI guards, the op-graph
//! scalar evaluators, and the per-dtype fused packed/batched paths.

mod builder;
mod guard;
mod macros;
mod packed;
mod scalar;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod unpack;

pub use builder::{Gemm, MapElem, PackedEpilogue, PackedQuant, RowReduce, epilogue_map};
pub(crate) use guard::{
    batched_count, check_strided, checked_dim2, checked_dims, dq_validate, sme_worth_it,
    validate_ep,
};
pub(crate) use macros::{batched_ep_impl, packed_ep_impl};
pub(crate) use packed::{
    batched_ep_f32, batched_ep_f64, f32_packed_ep_impl, f32_packed_ep_reduce, f64_packed_ep_impl,
};
pub use packed::{softmax_gemm_f32, softmax_rows, softmax_rows_with_max};
pub(crate) use scalar::{
    apply_ep_scalar, apply_ep_scalar_f64, dq_apply_cell, is_tensor_op, resolve_nodes,
};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) use unpack::{
    unpack_b_f32_sme, unpack_b_f64_sme, unpack_b_i8_sme, unpack_b_i16_sme, unpack_b16_sme,
};
