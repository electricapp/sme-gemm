//! Pre-FFI guards: the dimension-product and strided-footprint checks that
//! run before any raw pointer crosses to the C kernels, plus the op-graph
//! operand length validation.

use super::{PackedEpilogue, is_tensor_op};
use crate::epilogue::{Dequant, EpOp, Epilogue};
use crate::probe::has_sme;

/// Below this many flops the streaming-mode setup does not amortize; use the
/// scalar path (a NEON fallback will replace it).
pub(super) const SME_MIN_FLOPS: u128 = 1 << 18;

#[inline]
pub(crate) fn sme_worth_it(m: usize, n: usize, k: usize) -> bool {
    // Saturating so an absurd (unallocatable) shape whose flop product exceeds
    // u128 reads as "huge, definitely worth it" rather than wrapping to a small
    // value (or panicking under overflow-checks).
    has_sme()
        && k >= 2
        && (m as u128)
            .saturating_mul(n as u128)
            .saturating_mul(k as u128)
            >= SME_MIN_FLOPS
}

/// Assert that a strided `d0 x d1` view (row stride `rs`, col stride `cs`, in
/// elements) lies within a slice of `len` elements: its maximum touched index
/// is `(d0-1)*rs + (d1-1)*cs`. Empty views (either dim 0) touch nothing.
/// Checked arithmetic so adversarial strides fail the assert instead of
/// wrapping. This guard runs in every strided entry point before raw pointers
/// cross the FFI -- the C kernels honor strides literally and cannot
/// bounds-check, so it is what makes those entry points safe.
#[track_caller]
pub(crate) fn check_strided(what: &str, len: usize, d0: usize, rs: usize, d1: usize, cs: usize) {
    if d0 == 0 || d1 == 0 {
        return;
    }
    let max_idx = (d0 - 1)
        .checked_mul(rs)
        .and_then(|r| (d1 - 1).checked_mul(cs).and_then(|c| r.checked_add(c)));
    assert!(
        max_idx.is_some_and(|mi| mi < len),
        "{what}: {d0}x{d1} view at strides {rs}/{cs} overflows the slice (len {len})"
    );
}

/// Minimum operand length for a TENSOR-op operand covering `m` rows of `n`
/// elements at row stride `ld`: `(m-1)*ld + n`. Checked arithmetic so an
/// adversarial `ld`/`m` cannot wrap the product down to a value that passes the
/// length assert and lets the C kernel (or the scalar evaluator) read out of
/// bounds at `i*ld + j`. Panics on overflow rather than returning a small value.
#[track_caller]
pub(super) fn tensor_footprint(m: usize, n: usize, ld: usize) -> usize {
    m.saturating_sub(1)
        .checked_mul(ld)
        .and_then(|x| x.checked_add(n))
        .expect("tensor operand footprint overflows usize")
}

/// `a*b*c`, panicking on overflow. Used by the batched/packed length asserts,
/// which cross to the C FFI WITHOUT a `check_strided` backstop -- so a wrapped
/// `count*m*k` product must never be allowed to match a short slice and let the
/// kernel walk past it. (`[profile.release] overflow-checks` covers this crate's
/// own builds, but not downstream consumers, so the check is explicit here.)
#[track_caller]
pub(crate) fn checked_dims(a: usize, b: usize, c: usize) -> usize {
    a.checked_mul(b)
        .and_then(|x| x.checked_mul(c))
        .expect("batch dimension product overflows usize")
}

/// `a*b`, panicking on overflow. Two-factor companion to [`checked_dims`], for
/// the `matmul_*` / `*_packed` / `prepack_*` / [`Gemm`] length asserts -- which
/// hand raw `m`/`n`/`k` to the C kernels with no [`check_strided`] backstop, so a
/// wrapped product that matches a short slice is a silent out-of-bounds walk.
/// (`prepack_i8(&[], 2, 0)` + `matmul_i8_packed(&[], &w, &mut c, 1<<63 | 1)`
/// wraps `m*n` to 2 and the store then writes ~2^63 rows into 2 elements.)
#[track_caller]
pub(crate) const fn checked_dim2(a: usize, b: usize) -> usize {
    a.checked_mul(b).expect("dimension product overflows usize")
}

/// Infer the batch count from the C buffer length: `c_len / (m*n)`, with `m*n`
/// computed via `checked_mul`. Callers guard `m != 0 && n != 0` first, so the
/// divisor is nonzero; the checked product means an overflowing `m*n` panics
/// with a clear message instead of wrapping (to 0 -> divide-by-zero, or to a
/// nonzero value -> a silently-wrong `count` that only the later `checked_dims`
/// asserts would catch).
#[track_caller]
pub(crate) const fn batched_count(c_len: usize, m: usize, n: usize) -> usize {
    c_len / m.checked_mul(n).expect("batched m*n overflows usize")
}

/// Assert each op-graph operand's slice length against `m`/`n`: ROW vectors are
/// per-M (length `m`), COL per-N (length `n`), TENSOR holds `m*n` at stride `ld`
/// (defaulting to `n`, `ld >= n`).
pub(crate) fn validate_ep<T: PackedEpilogue>(ep: &Epilogue<'_, T>, m: usize, n: usize) {
    for (nd, &len) in ep.nodes.iter().zip(&ep.lens) {
        match nd.op {
            x if x == EpOp::AddRow as u32
                || x == EpOp::MulRow as u32
                || x == EpOp::SubRow as u32
                || x == EpOp::DivRow as u32
                || x == EpOp::MaxRow as u32
                || x == EpOp::MinRow as u32 =>
            {
                assert_eq!(len, m, "row operand is per-M (length m)");
            }
            x if x == EpOp::AddCol as u32
                || x == EpOp::MulCol as u32
                || x == EpOp::SubCol as u32
                || x == EpOp::DivCol as u32
                || x == EpOp::MaxCol as u32
                || x == EpOp::MinCol as u32 =>
            {
                assert_eq!(len, n, "col operand is per-N (length n)");
            }
            x if x == EpOp::AddTensor as u32
                || x == EpOp::MulTensor as u32
                || x == EpOp::SubTensor as u32
                || x == EpOp::DivTensor as u32
                || x == EpOp::MaxTensor as u32
                || x == EpOp::MinTensor as u32 =>
            {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                assert!(ld >= n, "tensor row stride must be >= n");
                assert!(
                    m == 0 || len >= tensor_footprint(m, n, ld),
                    "tensor must hold m*n elements (stride ld)"
                );
            }
            _ => {}
        }
    }
}

/// Assert each dequant op-graph operand's slice length against `m`/`n`.
pub(crate) fn dq_validate(dq: &Dequant<'_>, m: usize, n: usize) {
    for (nd, &len) in dq.nodes.iter().zip(&dq.lens) {
        match nd.op {
            x if x == EpOp::AddRow as u32
                || x == EpOp::MulRow as u32
                || x == EpOp::SubRow as u32
                || x == EpOp::DivRow as u32
                || x == EpOp::MaxRow as u32
                || x == EpOp::MinRow as u32 =>
            {
                assert_eq!(len, m, "row operand is per-M (length m)");
            }
            x if x == EpOp::AddCol as u32
                || x == EpOp::MulCol as u32
                || x == EpOp::SubCol as u32
                || x == EpOp::DivCol as u32
                || x == EpOp::MaxCol as u32
                || x == EpOp::MinCol as u32 =>
            {
                assert_eq!(len, n, "col operand is per-N (length n)");
            }
            x if is_tensor_op(x) => {
                let ld = if nd.ld == 0 { n } else { nd.ld };
                assert!(ld >= n, "tensor row stride must be >= n");
                assert!(
                    m == 0 || len >= tensor_footprint(m, n, ld),
                    "tensor must hold m*n elements (stride ld)"
                );
            }
            _ => {}
        }
    }
}
