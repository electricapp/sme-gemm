//! Differential fuzz: strided `gemm_f32` (C = alpha*C + beta*A@B) against an
//! f64 triple-loop oracle.
//!
//! Two modes, chosen by a fuzz bit:
//! - finite: inputs are small finite floats; the result must match the oracle
//!   within a condition-aware forward-error bound (scaled by sum_l |a||b|, so
//!   fuzzer-found catastrophic cancellation cannot produce false alarms).
//! - wild: raw u32 bit patterns (NaN, Inf, subnormals) are fed straight
//!   through; results are not compared (accumulation-order differences are
//!   legitimate under non-finite math) -- the kernel just must not crash or
//!   trip the sanitizer.
//!
//! Layout bits drive row-/column-major A, B, and C independently to exercise
//! the stride/transpose paths, and dims cross the SME dispatch threshold
//! (m*n*k >= 2^18) so the real streaming kernels run on M4+.

#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use sme_gemm::gemm_f32;

// Caps the f64 oracle's cost per input.
const MAX_MACS: usize = 1 << 21;

// (row_stride, col_stride) for an r x c matrix, row- or column-major.
fn strides(cm: bool, r: usize, c: usize) -> (usize, usize) {
    if cm { (1, r) } else { (c, 1) }
}

// alpha/beta biased toward the special-cased values 0 and +/-1.
fn coef(u: &mut Unstructured<'_>) -> f32 {
    match u.int_in_range(0u8..=5).unwrap_or(0) {
        0 => 0.0,
        1 => 1.0,
        2 => -1.0,
        _ => (f32::from(u.arbitrary::<i8>().unwrap_or(0))) / 32.0, // [-4, 4)
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let m: usize = u.int_in_range(0..=96).unwrap_or(1);
    let n: usize = u.int_in_range(0..=160).unwrap_or(1);
    let k: usize = u.int_in_range(0..=160).unwrap_or(1);
    if m * n.max(1) * k.max(1) > MAX_MACS {
        return;
    }
    let (a_cm, b_cm, c_cm) = (
        u.arbitrary().unwrap_or(false),
        u.arbitrary().unwrap_or(false),
        u.arbitrary().unwrap_or(false),
    );
    let wild = u.arbitrary().unwrap_or(false);
    let (alpha, beta) = (coef(&mut u), coef(&mut u));
    let (a_rs, a_cs) = strides(a_cm, m, k);
    let (b_rs, b_cs) = strides(b_cm, k, n);
    let (c_rs, c_cs) = strides(c_cm, m, n);

    // Fill matrices by cycling the remaining fuzz bytes. Finite mode maps each
    // byte to [-2, 2); wild mode builds raw u32 bit patterns from byte pairs
    // (mirrored to cover sign/exponent bits even from short inputs).
    let bytes = u.take_rest();
    let byte = |idx: usize| -> u8 {
        if bytes.is_empty() {
            0
        } else {
            bytes[idx % bytes.len()]
        }
    };
    let val = |idx: usize| -> f32 {
        if wild {
            let (lo, hi) = (byte(2 * idx), byte(2 * idx + 1));
            f32::from_bits(u32::from_le_bytes([lo, hi, hi, lo]))
        } else {
            (f32::from(byte(idx)) - 127.5) / 64.0
        }
    };
    let a: Vec<f32> = (0..m * k).map(val).collect();
    let b: Vec<f32> = (0..k * n).map(|i| val(i + m * k)).collect();
    let c_init: Vec<f32> = (0..m * n).map(|i| val(i + m * k + k * n)).collect();

    let mut c = c_init.clone();
    gemm_f32(
        m, n, k, &mut c, c_rs, c_cs, &a, a_rs, a_cs, &b, b_rs, b_cs, alpha, beta,
    );
    if wild {
        return; // crash/UB detector only; see module docs.
    }

    // Forward-error bound for f32 accumulation of k products plus the final
    // alpha/beta combine: |err| <= gamma_k * sum|a||b| with gamma_k ~ k*2^-24,
    // doubled for slack (blocked summation, fused epilogue rounding).
    let unit = f64::from(f32::EPSILON);
    for i in 0..m {
        for j in 0..n {
            let (mut acc, mut mag) = (0.0f64, 0.0f64);
            for l in 0..k {
                let p = f64::from(a[i * a_rs + l * a_cs]) * f64::from(b[l * b_rs + j * b_cs]);
                acc += p;
                mag += p.abs();
            }
            let c0 = f64::from(c_init[i * c_rs + j * c_cs]);
            let want = f64::from(alpha) * c0 + f64::from(beta) * acc;
            let scale = (f64::from(alpha) * c0).abs() + f64::from(beta).abs() * mag;
            let tol = scale * (k + 2) as f64 * unit * 2.0 + 1e-9;
            let got = f64::from(c[i * c_rs + j * c_cs]);
            assert!(
                (got - want).abs() <= tol,
                "gemm_f32 {m}x{n}x{k} (a_cm={a_cm} b_cm={b_cm} c_cm={c_cm} \
                 alpha={alpha} beta={beta}) at ({i},{j}): got {got} want {want} tol {tol}"
            );
        }
    }
});
