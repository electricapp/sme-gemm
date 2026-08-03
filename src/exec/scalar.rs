//! Scalar op-graph evaluation: node resolution for the FFI, and the per-cell
//! interpreters the non-SME / strided / OOM-fallback stores use.

use std::borrow::Cow;

use super::PackedEpilogue;
use crate::epilogue::{Dequant, EpNode, EpOp, Epilogue, act_apply_f32_a, act_apply_f64_a};

/// True if `op` is a TENSOR-operand op (its `ld == 0` defaults to `n`). Mirrors
/// `EP_OP_IS_TENSOR` in `csrc/epilogue.h`.
pub(crate) const fn is_tensor_op(op: u32) -> bool {
    op == EpOp::AddTensor as u32
        || op == EpOp::MulTensor as u32
        || op == EpOp::SubTensor as u32
        || op == EpOp::DivTensor as u32
        || op == EpOp::MaxTensor as u32
        || op == EpOp::MinTensor as u32
}

/// Build the FFI node array: copy `nodes` (resolving tensor `ld == 0` to `n`,
/// since the kernel interprets `ld` literally) and append the trailing activation
/// node (applied last) when `act` is set.
///
/// Returns a [`Cow`] so the common case -- an in-order graph with no `ld == 0`
/// TENSOR node and no trailing named activation -- borrows the builder's own
/// `nodes` and allocates nothing; only a graph that actually needs rewriting (a
/// defaulted tensor stride, or a trailing act to append) pays for a fresh `Vec`.
pub(crate) fn resolve_nodes(nodes: &[EpNode], act: Option<u32>, n: usize) -> Cow<'_, [EpNode]> {
    let needs_ld = nodes.iter().any(|nd| is_tensor_op(nd.op) && nd.ld == 0);
    if act.is_none() && !needs_ld {
        return Cow::Borrowed(nodes);
    }
    let mut out: Vec<EpNode> = nodes
        .iter()
        .map(|nd| {
            let mut nd = *nd;
            if is_tensor_op(nd.op) && nd.ld == 0 {
                nd.ld = n;
            }
            nd
        })
        .collect();
    if let Some(kind) = act {
        out.push(EpNode {
            op: EpOp::Act as u32,
            aux: kind,
            scalar: 0.0,
            ptr: core::ptr::null(),
            ld: 0,
        });
    }
    Cow::Owned(out)
}

/// Apply the op-graph to a single cell value `x` (in f32) at output `(i, j)`.
/// Operands are read from each node's raw pointer (cast to `*const T`); the
/// pointers were length-validated by [`validate_ep`].
pub(super) fn ep_apply_cell<T: PackedEpilogue>(
    ep: &Epilogue<'_, T>,
    mut x: f32,
    i: usize,
    j: usize,
    n: usize,
) -> f32 {
    for nd in &ep.nodes {
        // SAFETY: ptr (for vector ops) points at a T slice validated by
        // validate_ep to be long enough for this (i, j); reads are in bounds.
        let read = |idx: usize| -> f32 { unsafe { (*nd.ptr.cast::<T>().add(idx)).ep_to_f32() } };
        match nd.op {
            x_op if x_op == EpOp::AddScalar as u32 => x += nd.scalar,
            x_op if x_op == EpOp::MulScalar as u32 => x *= nd.scalar,
            x_op if x_op == EpOp::AddRow as u32 => x += read(i),
            x_op if x_op == EpOp::MulRow as u32 => x *= read(i),
            x_op if x_op == EpOp::AddCol as u32 => x += read(j),
            x_op if x_op == EpOp::MulCol as u32 => x *= read(j),
            x_op if x_op == EpOp::AddTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x += read(i * ld + j);
            }
            x_op if x_op == EpOp::MulTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x *= read(i * ld + j);
            }
            x_op if x_op == EpOp::Act as u32 => x = act_apply_f32_a(nd.aux, x, nd.scalar),
            x_op if x_op == EpOp::MaxScalar as u32 => x = x.max(nd.scalar),
            x_op if x_op == EpOp::MinScalar as u32 => x = x.min(nd.scalar),
            x_op if x_op == EpOp::SubScalar as u32 => x -= nd.scalar,
            x_op if x_op == EpOp::DivScalar as u32 => x /= nd.scalar,
            x_op if x_op == EpOp::SubRow as u32 => x -= read(i),
            x_op if x_op == EpOp::DivRow as u32 => x /= read(i),
            x_op if x_op == EpOp::MaxRow as u32 => x = x.max(read(i)),
            x_op if x_op == EpOp::MinRow as u32 => x = x.min(read(i)),
            x_op if x_op == EpOp::SubCol as u32 => x -= read(j),
            x_op if x_op == EpOp::DivCol as u32 => x /= read(j),
            x_op if x_op == EpOp::MaxCol as u32 => x = x.max(read(j)),
            x_op if x_op == EpOp::MinCol as u32 => x = x.min(read(j)),
            x_op if x_op == EpOp::SubTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x -= read(i * ld + j);
            }
            x_op if x_op == EpOp::DivTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x /= read(i * ld + j);
            }
            x_op if x_op == EpOp::MaxTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x = x.max(read(i * ld + j));
            }
            x_op if x_op == EpOp::MinTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x = x.min(read(i * ld + j));
            }
            _ => {}
        }
    }
    if let Some(kind) = ep.act {
        x = act_apply_f32_a(kind, x, 0.0);
    }
    x
}

/// Scalar fused-epilogue pass over the already-computed product `c` (at strides
/// `rs`/`cs`), applying the op-graph in f32. Used by the non-SME / OOM-fallback
/// path of [`packed_ep_impl`].
pub(crate) fn apply_ep_scalar<T: PackedEpilogue>(
    c: &mut [T],
    m: usize,
    n: usize,
    rs: usize,
    cs: usize,
    ep: &Epilogue<'_, T>,
) {
    for i in 0..m {
        for j in 0..n {
            let idx = i * rs + j * cs;
            let v = ep_apply_cell(ep, c[idx].ep_to_f32(), i, j, n);
            c[idx] = T::ep_from_f32(v);
        }
    }
}

/// Apply the op-graph to a single cell value `x` (in f64) at output `(i, j)`,
/// in double precision. Operands are read as `f64`; the node `scalar` field is
/// f32 and widens to f64. The f64 analog of [`ep_apply_cell`] (which works in
/// f32) -- used by the f64 scalar fallback so it matches the kernel's domain.
fn ep_apply_cell_f64(ep: &Epilogue<'_, f64>, mut x: f64, i: usize, j: usize, n: usize) -> f64 {
    for nd in &ep.nodes {
        // SAFETY: ptr (for vector ops) points at an f64 slice validated by
        // validate_ep to be long enough for this (i, j); reads are in bounds.
        let read = |idx: usize| -> f64 { unsafe { *nd.ptr.cast::<f64>().add(idx) } };
        match nd.op {
            x_op if x_op == EpOp::AddScalar as u32 => x += f64::from(nd.scalar),
            x_op if x_op == EpOp::MulScalar as u32 => x *= f64::from(nd.scalar),
            x_op if x_op == EpOp::AddRow as u32 => x += read(i),
            x_op if x_op == EpOp::MulRow as u32 => x *= read(i),
            x_op if x_op == EpOp::AddCol as u32 => x += read(j),
            x_op if x_op == EpOp::MulCol as u32 => x *= read(j),
            x_op if x_op == EpOp::AddTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x += read(i * ld + j);
            }
            x_op if x_op == EpOp::MulTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x *= read(i * ld + j);
            }
            x_op if x_op == EpOp::Act as u32 => {
                x = act_apply_f64_a(nd.aux, x, f64::from(nd.scalar));
            }
            x_op if x_op == EpOp::MaxScalar as u32 => x = x.max(f64::from(nd.scalar)),
            x_op if x_op == EpOp::MinScalar as u32 => x = x.min(f64::from(nd.scalar)),
            x_op if x_op == EpOp::SubScalar as u32 => x -= f64::from(nd.scalar),
            x_op if x_op == EpOp::DivScalar as u32 => x /= f64::from(nd.scalar),
            x_op if x_op == EpOp::SubRow as u32 => x -= read(i),
            x_op if x_op == EpOp::DivRow as u32 => x /= read(i),
            x_op if x_op == EpOp::MaxRow as u32 => x = x.max(read(i)),
            x_op if x_op == EpOp::MinRow as u32 => x = x.min(read(i)),
            x_op if x_op == EpOp::SubCol as u32 => x -= read(j),
            x_op if x_op == EpOp::DivCol as u32 => x /= read(j),
            x_op if x_op == EpOp::MaxCol as u32 => x = x.max(read(j)),
            x_op if x_op == EpOp::MinCol as u32 => x = x.min(read(j)),
            x_op if x_op == EpOp::SubTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x -= read(i * ld + j);
            }
            x_op if x_op == EpOp::DivTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x /= read(i * ld + j);
            }
            x_op if x_op == EpOp::MaxTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x = x.max(read(i * ld + j));
            }
            x_op if x_op == EpOp::MinTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x = x.min(read(i * ld + j));
            }
            _ => {}
        }
    }
    if let Some(kind) = ep.act {
        x = act_apply_f64_a(kind, x, 0.0);
    }
    x
}

/// f64 scalar fused-epilogue pass over the already-computed product `c` (strides
/// `rs`/`cs`), in double precision. The f64 analog of [`apply_ep_scalar`].
pub(crate) fn apply_ep_scalar_f64(
    c: &mut [f64],
    m: usize,
    n: usize,
    rs: usize,
    cs: usize,
    ep: &Epilogue<'_, f64>,
) {
    for i in 0..m {
        for j in 0..n {
            let idx = i * rs + j * cs;
            c[idx] = ep_apply_cell_f64(ep, c[idx], i, j, n);
        }
    }
}

/// Apply the dequant op-graph to one scalar cell value `x` (f32) at `(i, j)`.
pub(crate) fn dq_apply_cell(dq: &Dequant<'_>, mut x: f32, i: usize, j: usize, n: usize) -> f32 {
    for nd in &dq.nodes {
        // SAFETY: ptr (for vector ops) points at an f32 slice validated by
        // dq_validate to be long enough for this (i, j); reads are in bounds.
        let read = |idx: usize| -> f32 { unsafe { *nd.ptr.cast::<f32>().add(idx) } };
        match nd.op {
            x_op if x_op == EpOp::AddScalar as u32 => x += nd.scalar,
            x_op if x_op == EpOp::MulScalar as u32 => x *= nd.scalar,
            x_op if x_op == EpOp::AddRow as u32 => x += read(i),
            x_op if x_op == EpOp::MulRow as u32 => x *= read(i),
            x_op if x_op == EpOp::AddCol as u32 => x += read(j),
            x_op if x_op == EpOp::MulCol as u32 => x *= read(j),
            x_op if x_op == EpOp::AddTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x += read(i * ld + j);
            }
            x_op if x_op == EpOp::MulTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x *= read(i * ld + j);
            }
            x_op if x_op == EpOp::Act as u32 => x = act_apply_f32_a(nd.aux, x, nd.scalar),
            x_op if x_op == EpOp::MaxScalar as u32 => x = x.max(nd.scalar),
            x_op if x_op == EpOp::MinScalar as u32 => x = x.min(nd.scalar),
            x_op if x_op == EpOp::SubScalar as u32 => x -= nd.scalar,
            x_op if x_op == EpOp::DivScalar as u32 => x /= nd.scalar,
            x_op if x_op == EpOp::SubRow as u32 => x -= read(i),
            x_op if x_op == EpOp::DivRow as u32 => x /= read(i),
            x_op if x_op == EpOp::MaxRow as u32 => x = x.max(read(i)),
            x_op if x_op == EpOp::MinRow as u32 => x = x.min(read(i)),
            x_op if x_op == EpOp::SubCol as u32 => x -= read(j),
            x_op if x_op == EpOp::DivCol as u32 => x /= read(j),
            x_op if x_op == EpOp::MaxCol as u32 => x = x.max(read(j)),
            x_op if x_op == EpOp::MinCol as u32 => x = x.min(read(j)),
            x_op if x_op == EpOp::SubTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x -= read(i * ld + j);
            }
            x_op if x_op == EpOp::DivTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x /= read(i * ld + j);
            }
            x_op if x_op == EpOp::MaxTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x = x.max(read(i * ld + j));
            }
            x_op if x_op == EpOp::MinTensor as u32 => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                x = x.min(read(i * ld + j));
            }
            _ => {}
        }
    }
    if let Some(kind) = dq.act {
        x = act_apply_f32_a(kind, x, 0.0);
    }
    x
}

// The SME-packb inverses `unpack_b16_sme` / `unpack_b_i8_sme` are `pub(crate)`
// (visible only in-crate) and are normally shadowed on every M5 -- the OOM
// fallback that calls them almost never fires -- so they need an in-crate unit
// test. Mirror of `q4::tests::q4_resident_roundtrip_matches_rowmajor`: pack via
// the real prepack path, then assert unpack(pack(B)) == B bit-for-bit. The
// functions are themselves gated to Apple aarch64, so the test is too; and we
// only assert the inverse when the prepack actually produced the SME tile-major
// layout (`packed.sme`), since off-SME `data` is a plain row-major copy for which
