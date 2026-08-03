//! Portable scalar GEMM reference. Used as the non-Apple fallback and as the
//! correctness oracle in tests. Accumulates in a wider type than the kernel
//! (f32 for the 16-bit floats, f64 for f32 and f64, i32/i64 for the ints) so
//! it is a faithful upper bound on the SME kernels' accuracy.

use half::{bf16, f16};

/// Per-item scalar reference fallback for the batched f32 entry points: runs
/// `count` independent row-major `C_i = A_i @ B_i` (item strides `m*k` / `k*n` /
/// `m*n`, alpha=0/beta=1), exactly as the inlined fallback loop did.
pub(crate) fn batched_gemm_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let count = c.len() / (m * n);
    for i in 0..count {
        gemm_f32(
            m,
            n,
            k,
            &mut c[i * m * n..(i + 1) * m * n],
            n,
            1,
            &a[i * m * k..(i + 1) * m * k],
            k,
            1,
            &b[i * k * n..(i + 1) * k * n],
            n,
            1,
            0.0,
            1.0,
        );
    }
}

/// Per-item scalar reference fallback for the batched f64 entry points. See
/// [`batched_gemm_f32`].
pub(crate) fn batched_gemm_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    let count = c.len() / (m * n);
    for i in 0..count {
        gemm_f64(
            m,
            n,
            k,
            &mut c[i * m * n..(i + 1) * m * n],
            n,
            1,
            &a[i * m * k..(i + 1) * m * k],
            k,
            1,
            &b[i * k * n..(i + 1) * k * n],
            n,
            1,
            0.0,
            1.0,
        );
    }
}

/// Per-item scalar reference fallback for the batched bf16 entry points
/// (f32-accumulate, alpha=0/beta=1). See [`batched_gemm_f32`].
pub(crate) fn batched_gemm_bf16(
    a: &[bf16],
    b: &[bf16],
    c: &mut [bf16],
    m: usize,
    n: usize,
    k: usize,
) {
    let count = c.len() / (m * n);
    for i in 0..count {
        gemm_bf16(
            m,
            n,
            k,
            &mut c[i * m * n..(i + 1) * m * n],
            n,
            1,
            &a[i * m * k..(i + 1) * m * k],
            k,
            1,
            &b[i * k * n..(i + 1) * k * n],
            n,
            1,
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
        );
    }
}

/// Per-item scalar reference fallback for the batched f16 entry points
/// (f32-accumulate, alpha=0/beta=1). See [`batched_gemm_f32`].
pub(crate) fn batched_gemm_f16(a: &[f16], b: &[f16], c: &mut [f16], m: usize, n: usize, k: usize) {
    let count = c.len() / (m * n);
    for i in 0..count {
        gemm_f16(
            m,
            n,
            k,
            &mut c[i * m * n..(i + 1) * m * n],
            n,
            1,
            &a[i * m * k..(i + 1) * m * k],
            k,
            1,
            &b[i * k * n..(i + 1) * k * n],
            n,
            1,
            f16::from_f32(0.0),
            f16::from_f32(1.0),
        );
    }
}

/// Per-item scalar reference fallback for the batched i8 -> i32 entry point. See
/// [`batched_gemm_f32`].
pub(crate) fn batched_gemm_i8(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    let count = c.len() / (m * n);
    for i in 0..count {
        gemm_i8(
            m,
            n,
            k,
            &mut c[i * m * n..(i + 1) * m * n],
            n,
            1,
            &a[i * m * k..(i + 1) * m * k],
            k,
            1,
            &b[i * k * n..(i + 1) * k * n],
            n,
            1,
        );
    }
}

/// Per-item scalar reference fallback for the batched i16 -> i64 entry point. See
/// [`batched_gemm_f32`].
pub(crate) fn batched_gemm_i16(a: &[i16], b: &[i16], c: &mut [i64], m: usize, n: usize, k: usize) {
    let count = c.len() / (m * n);
    for i in 0..count {
        gemm_i16(
            m,
            n,
            k,
            &mut c[i * m * n..(i + 1) * m * n],
            n,
            1,
            &a[i * m * k..(i + 1) * m * k],
            k,
            1,
            &b[i * k * n..(i + 1) * k * n],
            n,
            1,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_i8(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [i32],
    c_rs: usize,
    c_cs: usize,
    a: &[i8],
    a_rs: usize,
    a_cs: usize,
    b: &[i8],
    b_rs: usize,
    b_cs: usize,
) {
    // ikj with a reused per-row i32 accumulator. Bit-identical to ijk: each cell
    // still wraps over l in 0..k order, matching the SMOPA kernel's mod-2^32
    // accumulation (only reachable past k ~ 130,000 at full +-127 magnitude)
    // rather than panicking in debug builds where the SME path silently wraps.
    // For row-major B this streams B contiguously (see gemm_f32).
    let mut acc = vec![0i32; n];
    for i in 0..m {
        acc.fill(0);
        for l in 0..k {
            let av = i32::from(a[i * a_rs + l * a_cs]);
            for j in 0..n {
                acc[j] = acc[j].wrapping_add(av * i32::from(b[l * b_rs + j * b_cs]));
            }
        }
        for j in 0..n {
            c[i * c_rs + j * c_cs] = acc[j];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_i16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [i64],
    c_rs: usize,
    c_cs: usize,
    a: &[i16],
    a_rs: usize,
    a_cs: usize,
    b: &[i16],
    b_rs: usize,
    b_cs: usize,
) {
    // ikj with a reused per-row i64 accumulator. Bit-identical to ijk: each cell
    // still wraps over l in 0..k order, matching the SMOPA kernel's mod-2^64
    // accumulation (unreachable in practice: overflow needs k ~ 2^33). For
    // row-major B this streams B contiguously (see gemm_f32).
    let mut acc = vec![0i64; n];
    for i in 0..m {
        acc.fill(0);
        for l in 0..k {
            let av = i64::from(a[i * a_rs + l * a_cs]);
            for j in 0..n {
                acc[j] = acc[j].wrapping_add(av * i64::from(b[l * b_rs + j * b_cs]));
            }
        }
        for j in 0..n {
            c[i * c_rs + j * c_cs] = acc[j];
        }
    }
}

/// f32 oracle: accumulates in f64 so the reference is a strict accuracy upper
/// bound on the SME kernel (which accumulates in f32), not merely a peer with
/// its own f32 rounding error.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
pub(crate) fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
    c_rs: usize,
    c_cs: usize,
    a: &[f32],
    a_rs: usize,
    a_cs: usize,
    b: &[f32],
    b_rs: usize,
    b_cs: usize,
    alpha: f32,
    beta: f32,
) {
    // ikj order with a reused per-row f64 accumulator: bit-identical to the
    // naive ijk triple loop (each cell still sums over l in 0..k order) but, for
    // the dominant row-major B (b_cs == 1), streams B one contiguous row at a
    // time instead of column-walking it -- 2-5x faster on the non-Apple/OOM
    // fallback at large k. (Col-major B trades places and column-walks here.)
    let mut acc = vec![0.0f64; n];
    for i in 0..m {
        acc.fill(0.0);
        for l in 0..k {
            let av = f64::from(a[i * a_rs + l * a_cs]);
            for j in 0..n {
                acc[j] += av * f64::from(b[l * b_rs + j * b_cs]);
            }
        }
        for j in 0..n {
            let cref = &mut c[i * c_rs + j * c_cs];
            let prev = if alpha == 0.0 {
                0.0
            } else {
                f64::from(alpha) * f64::from(*cref)
            };
            *cref = (prev + f64::from(beta) * acc[j]) as f32;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_f64(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f64],
    c_rs: usize,
    c_cs: usize,
    a: &[f64],
    a_rs: usize,
    a_cs: usize,
    b: &[f64],
    b_rs: usize,
    b_cs: usize,
    alpha: f64,
    beta: f64,
) {
    // ikj with a reused per-row accumulator (bit-identical to ijk; streams
    // row-major B contiguously -- see gemm_f32 for the rationale).
    let mut acc = vec![0.0f64; n];
    for i in 0..m {
        acc.fill(0.0);
        for l in 0..k {
            let av = a[i * a_rs + l * a_cs];
            for j in 0..n {
                acc[j] += av * b[l * b_rs + j * b_cs];
            }
        }
        for j in 0..n {
            let cref = &mut c[i * c_rs + j * c_cs];
            let prev = if alpha == 0.0 { 0.0 } else { alpha * *cref };
            *cref = prev + beta * acc[j];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_bf16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [bf16],
    c_rs: usize,
    c_cs: usize,
    a: &[bf16],
    a_rs: usize,
    a_cs: usize,
    b: &[bf16],
    b_rs: usize,
    b_cs: usize,
    alpha: bf16,
    beta: bf16,
) {
    let (alpha, beta) = (alpha.to_f32(), beta.to_f32());
    // ikj with a reused per-row f32 accumulator (bit-identical to ijk; streams
    // row-major B contiguously -- see gemm_f32 for the rationale).
    let mut acc = vec![0.0f32; n];
    for i in 0..m {
        acc.fill(0.0);
        for l in 0..k {
            let av = a[i * a_rs + l * a_cs].to_f32();
            for j in 0..n {
                acc[j] += av * b[l * b_rs + j * b_cs].to_f32();
            }
        }
        for j in 0..n {
            let cref = &mut c[i * c_rs + j * c_cs];
            let prev = if alpha == 0.0 {
                0.0
            } else {
                alpha * cref.to_f32()
            };
            *cref = bf16::from_f32(prev + beta * acc[j]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_f16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f16],
    c_rs: usize,
    c_cs: usize,
    a: &[f16],
    a_rs: usize,
    a_cs: usize,
    b: &[f16],
    b_rs: usize,
    b_cs: usize,
    alpha: f16,
    beta: f16,
) {
    let (alpha, beta) = (alpha.to_f32(), beta.to_f32());
    // ikj with a reused per-row f32 accumulator (bit-identical to ijk; streams
    // row-major B contiguously -- see gemm_f32 for the rationale).
    let mut acc = vec![0.0f32; n];
    for i in 0..m {
        acc.fill(0.0);
        for l in 0..k {
            let av = a[i * a_rs + l * a_cs].to_f32();
            for j in 0..n {
                acc[j] += av * b[l * b_rs + j * b_cs].to_f32();
            }
        }
        for j in 0..n {
            let cref = &mut c[i * c_rs + j * c_cs];
            let prev = if alpha == 0.0 {
                0.0
            } else {
                alpha * cref.to_f32()
            };
            *cref = f16::from_f32(prev + beta * acc[j]);
        }
    }
}
