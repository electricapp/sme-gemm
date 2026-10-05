//! C-ABI epilogue descriptors and the `unsafe extern "C"` SME kernel block
//! (Apple aarch64 only).

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use crate::epilogue::EpNode;

/// C-ABI mirror of `ep_desc16` in `csrc/epilogue.h` (shared by f16 and bf16):
/// an ordered op-graph node array. Field order must match the C struct exactly.
#[repr(C)]
pub(crate) struct EpDesc16 {
    pub(crate) n_nodes: u32,
    pub(crate) nodes: *const EpNode,
}

/// C-ABI mirror of `ep_desc_f32` in `csrc/epilogue.h`: an ordered op-graph node
/// array applied in the f32 accumulator domain. Field order must match the C
/// struct exactly.
#[repr(C)]
pub(crate) struct EpDescF32 {
    pub(crate) n_nodes: u32,
    pub(crate) nodes: *const EpNode,
    /// Optional per-M reduction outputs (length `m`), written once per row as
    /// each M-tile finishes. Null to skip. See `ep_desc_f32` in the header.
    pub(crate) row_sum: *mut f32,
    pub(crate) row_max: *mut f32,
}

impl EpDescF32 {
    /// Descriptor with no reductions -- the common case.
    pub(crate) const fn nodes_only(n_nodes: u32, nodes: *const EpNode) -> Self {
        Self {
            n_nodes,
            nodes,
            row_sum: core::ptr::null_mut(),
            row_max: core::ptr::null_mut(),
        }
    }
}

/// C-ABI mirror of `ep_desc_f64` in `csrc/epilogue.h`: an ordered op-graph node
/// array applied in the f64 accumulator domain. Field order must match the C
/// struct exactly.
#[repr(C)]
pub(crate) struct EpDescF64 {
    pub(crate) n_nodes: u32,
    pub(crate) nodes: *const EpNode,
}

/// C-ABI mirror of `ep_dq_f32` in `csrc/epilogue.h`: the per-tensor/per-N scale
/// that lifts an integer accumulator into f32, plus the op-graph applied there.
/// Field order must match the C struct exactly.
#[repr(C)]
pub(crate) struct DqF32 {
    pub(crate) scale: f32,
    pub(crate) scale_n: *const f32,
    pub(crate) n_nodes: u32,
    pub(crate) nodes: *const EpNode,
}

// Pin the FFI layout (4-byte holes after `scale` and `n_nodes` on arm64); drift
// would be silent UB. The C side has matching _Static_asserts.
const _: () = {
    assert!(size_of::<DqF32>() == 32, "size_of::<DqF32>() == 32");
    assert!(
        core::mem::offset_of!(DqF32, scale) == 0,
        "core::mem::offset_of!(DqF32, scale) == 0"
    );
    assert!(
        core::mem::offset_of!(DqF32, scale_n) == 8,
        "core::mem::offset_of!(DqF32, scale_n) == 8"
    );
    assert!(
        core::mem::offset_of!(DqF32, n_nodes) == 16,
        "core::mem::offset_of!(DqF32, n_nodes) == 16"
    );
    assert!(
        core::mem::offset_of!(DqF32, nodes) == 24,
        "core::mem::offset_of!(DqF32, nodes) == 24"
    );
};

// ---- C kernels (Apple aarch64 only) ----------------------------------------
unsafe extern "C" {
    /// `out = act(gate) * up` over rows of gate/up interleaved 32 columns at a
    /// time (`neon_ops.c`); `act` is an `EP_ACT_*` kind.
    pub(crate) fn neon_glu_f16(out: *mut u16, input: *const u16, m: usize, n: usize, act: u32);
    /// `dst[i] = src[i]` rounded to f16, `n` values (`neon_ops.c`).
    pub(crate) fn neon_f32_to_f16(dst: *mut u16, src: *const f32, n: usize);
    /// `dst[i] = src[i]` widened to f32, `n` values.
    pub(crate) fn neon_f16_to_f32(dst: *mut f32, src: *const u16, n: usize);
    /// `dst[i] += src[i]`, f16 into f32, `n` values.
    pub(crate) fn neon_add_f16_to_f32(dst: *mut f32, src: *const u16, n: usize);
    /// `dst[i] = act(src[i])` over `n` f16 values, out of place; `act` is a
    /// gelu/silu/sigmoid/tanh `EP_ACT_*` kind.
    pub(crate) fn neon_act_f16(dst: *mut u16, src: *const u16, n: usize, act: u32);
    /// A short burst of streaming vector work for the keep-awake helper
    /// (`warm.rs`); the result only keeps the work from being optimized out.
    pub(crate) fn sme_warm_tick() -> f32;
    pub(crate) fn gemm_sme_f16f16_run(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const u16,
        lhs_cs: isize,
        lhs_rs: isize,
        rhs: *const u16,
        rhs_cs: isize,
        rhs_rs: isize,
        alpha_bits: u16,
        beta_bits: u16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f16f16_batched(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        lhs: *const u16,
        rhs: *const u16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f16f16_batched_ep(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        lhs: *const u16,
        rhs: *const u16,
        ep: *const EpDesc16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f16f16_q4(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        lhs: *const u16,
        nibbles: *const u8,
        scales: *const u16,
        mins: *const u16,
        block: usize,
        ep: *const EpDesc16,
    ) -> core::ffi::c_int;
    /// `gemm_sme_f16f16_q4` as one link of a chain: raises `done` to each
    /// stored output column and waits on `ready` for A's depths (either null).
    pub(crate) fn gemm_sme_f16f16_q4_chained(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        lhs: *const u16,
        nibbles: *const u8,
        scales: *const u16,
        mins: *const u16,
        block: usize,
        ep: *const EpDesc16,
        done: *const core::sync::atomic::AtomicUsize,
        ready: *const core::sync::atomic::AtomicUsize,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_b16b16_q4(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        lhs: *const u16,
        nibbles: *const u8,
        scales: *const u16,
        mins: *const u16,
        scales_bf16: *const u16,
        mins_bf16: *const u16,
        block: usize,
        ep: *const EpDesc16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_b16b16_batched(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        lhs: *const u16,
        rhs: *const u16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_b16b16_batched_ep(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        lhs: *const u16,
        rhs: *const u16,
        ep: *const EpDesc16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i8i32_batched(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut i32,
        lhs: *const i8,
        rhs: *const i8,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i8i32_batched_dequant(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f32,
        lhs: *const i8,
        rhs: *const i8,
        scale: f32,
        scale_n: *const f32,
        n_nodes: u32,
        nodes: *const EpNode,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f16f16_packb(
        b_pack: *mut u16,
        rhs: *const u16,
        n: usize,
        k: usize,
        rhs_rs: isize,
        rhs_cs: isize,
    );
    pub(crate) fn gemm_sme_f16f16_packed_b_elems(n: usize, k: usize) -> usize;
    pub(crate) fn gemm_sme_f16f16_run_packed(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const u16,
        lhs_cs: isize,
        lhs_rs: isize,
        b_pack: *const u16,
        alpha_bits: u16,
        beta_bits: u16,
        ep: *const EpDesc16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f16f32_run(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const u16,
        lhs_cs: isize,
        lhs_rs: isize,
        rhs: *const u16,
        rhs_cs: isize,
        rhs_rs: isize,
        alpha_bits: u16,
        beta_bits: u16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_bf16f32_run(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const u16,
        lhs_cs: isize,
        lhs_rs: isize,
        rhs: *const u16,
        rhs_cs: isize,
        rhs_rs: isize,
        alpha_bits: u16,
        beta_bits: u16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_b16b16_run(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const u16,
        lhs_cs: isize,
        lhs_rs: isize,
        rhs: *const u16,
        rhs_cs: isize,
        rhs_rs: isize,
        alpha_bits: u16,
        beta_bits: u16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_b16b16_packb(
        b_pack: *mut u16,
        rhs: *const u16,
        n: usize,
        k: usize,
        rhs_rs: isize,
        rhs_cs: isize,
    );
    pub(crate) fn gemm_sme_b16b16_packed_b_elems(n: usize, k: usize) -> usize;
    pub(crate) fn gemm_sme_b16b16_run_packed(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut u16,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const u16,
        lhs_cs: isize,
        lhs_rs: isize,
        b_pack: *const u16,
        alpha_bits: u16,
        beta_bits: u16,
        ep: *const EpDesc16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f32_run(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f32,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const f32,
        lhs_cs: isize,
        lhs_rs: isize,
        rhs: *const f32,
        rhs_cs: isize,
        rhs_rs: isize,
        alpha: f32,
        beta: f32,
        ep: *const EpDescF32,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f32_packed_b_elems(n: usize, k: usize) -> usize;
    pub(crate) fn gemm_sme_f32_packb(
        b_pack: *mut f32,
        rhs: *const f32,
        n: usize,
        k: usize,
        rhs_rs: isize,
        rhs_cs: isize,
    );
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_sme_f32_run_packed(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f32,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const f32,
        lhs_cs: isize,
        lhs_rs: isize,
        b_pack: *const f32,
        alpha: f32,
        beta: f32,
        ep: *const EpDescF32,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f32_softmax(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f32,
        lhs: *const f32,
        lhs_cs: isize,
        lhs_rs: isize,
        b_pack: *const f32,
        row_max: *mut f32,
    ) -> core::ffi::c_int;
    pub(crate) fn attn_softmax_rows_f32(
        c: *mut f32,
        m: usize,
        n: usize,
        rs: isize,
        row_max: *const f32,
    );
    pub(crate) fn attn_flash_block_f32(
        s: *mut f32,
        m: usize,
        bj: usize,
        s_rs: isize,
        o: *mut f32,
        dv: usize,
        o_rs: isize,
        row_max: *mut f32,
        row_sum: *mut f32,
    );
    pub(crate) fn attn_flash_finish_f32(
        o: *mut f32,
        m: usize,
        dv: usize,
        o_rs: isize,
        row_sum: *const f32,
    );
    /// A block of query rows against an f16 KV cache, rows across cores.
    pub(crate) fn attn_kv_causal_f16(
        out: *mut f32,
        q: *const f32,
        k: *const u16,
        v: *const u16,
        start: usize,
        rows: usize,
        ld: usize,
        n_heads: usize,
        n_kv_heads: usize,
        hd: usize,
        scale: f32,
    ) -> i32;
    /// One query row against an f16 KV cache (`attention.c`).
    pub(crate) fn attn_kv_f16(
        out: *mut f32,
        q: *const f32,
        k: *const u16,
        v: *const u16,
        len: usize,
        ld: usize,
        n_heads: usize,
        n_kv_heads: usize,
        hd: usize,
        scale: f32,
        scores: *mut f32,
    );
    pub(crate) fn attn_flash_block_f16(
        s: *mut u16,
        m: usize,
        bj: usize,
        s_rs: isize,
        scale: f32,
        row_max: *mut f32,
        row_sum: *mut f32,
        corr: *mut f32,
    );
    pub(crate) fn attn_flash_accum_f16(
        acc: *mut f32,
        t: *const u16,
        m: usize,
        dv: usize,
        t_rs: isize,
        corr: *const f32,
    );
    pub(crate) fn attn_flash_finish_f16(
        out: *mut u16,
        acc: *const f32,
        m: usize,
        dv: usize,
        out_rs: isize,
        row_sum: *const f32,
    );
    pub(crate) fn attn_flash_block_bf16(
        s: *mut u16,
        m: usize,
        bj: usize,
        s_rs: isize,
        scale: f32,
        row_max: *mut f32,
        row_sum: *mut f32,
        corr: *mut f32,
    );
    pub(crate) fn attn_flash_accum_bf16(
        acc: *mut f32,
        t: *const u16,
        m: usize,
        dv: usize,
        t_rs: isize,
        corr: *const f32,
    );
    pub(crate) fn attn_flash_finish_bf16(
        out: *mut u16,
        acc: *const f32,
        m: usize,
        dv: usize,
        out_rs: isize,
        row_sum: *const f32,
    );
    pub(crate) fn gemm_sme_f32_batched(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f32,
        lhs: *const f32,
        rhs: *const f32,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f32_batched_ep(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f32,
        lhs: *const f32,
        rhs: *const f32,
        ep: *const EpDescF32,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f64_run(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f64,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const f64,
        lhs_cs: isize,
        lhs_rs: isize,
        rhs: *const f64,
        rhs_cs: isize,
        rhs_rs: isize,
        alpha: f64,
        beta: f64,
        ep: *const EpDescF64,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f64_packed_b_elems(n: usize, k: usize) -> usize;
    pub(crate) fn gemm_sme_f64_packb(
        b_pack: *mut f64,
        rhs: *const f64,
        n: usize,
        k: usize,
        rhs_rs: isize,
        rhs_cs: isize,
    );
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_sme_f64_run_packed(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f64,
        dst_cs: isize,
        dst_rs: isize,
        read_dst: i32,
        lhs: *const f64,
        lhs_cs: isize,
        lhs_rs: isize,
        b_pack: *const f64,
        alpha: f64,
        beta: f64,
        ep: *const EpDescF64,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f64_batched(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f64,
        lhs: *const f64,
        rhs: *const f64,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_f64_batched_ep(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f64,
        lhs: *const f64,
        rhs: *const f64,
        ep: *const EpDescF64,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i8i32_run(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut i32,
        dst_cs: isize,
        dst_rs: isize,
        lhs: *const i8,
        lhs_cs: isize,
        lhs_rs: isize,
        rhs: *const i8,
        rhs_cs: isize,
        rhs_rs: isize,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i16i64_run(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut i64,
        dst_cs: isize,
        dst_rs: isize,
        lhs: *const i16,
        lhs_cs: isize,
        lhs_rs: isize,
        rhs: *const i16,
        rhs_cs: isize,
        rhs_rs: isize,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i16i64_run_dequant(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f32,
        dst_cs: isize,
        dst_rs: isize,
        lhs: *const i16,
        lhs_cs: isize,
        lhs_rs: isize,
        rhs: *const i16,
        rhs_cs: isize,
        rhs_rs: isize,
        scale: f32,
        scale_n: *const f32,
        n_nodes: u32,
        nodes: *const EpNode,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i16i64_packed_b_elems(n: usize, k: usize) -> usize;
    pub(crate) fn gemm_sme_i16i64_packb(
        b_pack: *mut i16,
        rhs: *const i16,
        n: usize,
        k: usize,
        rhs_rs: isize,
        rhs_cs: isize,
    );
    /// `dst` is `*mut i64` when `dq` is null, `*mut f32` when it is set.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_sme_i16i64_run_packed_impl(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut core::ffi::c_void,
        dst_cs: isize,
        dst_rs: isize,
        lhs: *const i16,
        lhs_cs: isize,
        lhs_rs: isize,
        b_pack: *const i16,
        dq: *const DqF32,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i16i64_batched(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut i64,
        lhs: *const i16,
        rhs: *const i16,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i16i64_batched_dequant(
        count: usize,
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f32,
        lhs: *const i16,
        rhs: *const i16,
        scale: f32,
        scale_n: *const f32,
        n_nodes: u32,
        nodes: *const EpNode,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i8i32_packb(
        b_pack: *mut i8,
        rhs: *const i8,
        n: usize,
        k: usize,
        rhs_rs: isize,
        rhs_cs: isize,
    );
    pub(crate) fn gemm_sme_i8i32_packed_b_elems(n: usize, k: usize) -> usize;
    pub(crate) fn gemm_sme_i8i32_run_packed(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut i32,
        dst_cs: isize,
        dst_rs: isize,
        lhs: *const i8,
        lhs_cs: isize,
        lhs_rs: isize,
        b_pack: *const i8,
    ) -> core::ffi::c_int;
    pub(crate) fn gemm_sme_i8i32_run_packed_dequant(
        m: usize,
        n: usize,
        k: usize,
        dst: *mut f32,
        dst_cs: isize,
        dst_rs: isize,
        lhs: *const i8,
        lhs_cs: isize,
        lhs_rs: isize,
        b_pack: *const i8,
        scale: f32,
        scale_n: *const f32,
        n_nodes: u32,
        nodes: *const EpNode,
    ) -> core::ffi::c_int;
}
