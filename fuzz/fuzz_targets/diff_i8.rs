//! Differential fuzz: strided `gemm_i8` against an exact i32 triple loop.
//!
//! i8 -> i32 accumulation is exact integer math, so ANY mismatch is a kernel
//! bug -- no tolerance reasoning needed. Fuzz bits choose row-/column-major
//! layouts for A, B, and C independently to drive the stride/transpose paths,
//! and dims reach past the SME dispatch threshold (m*n*k >= 2^18) so the real
//! streaming kernels (not just the scalar fallback) are exercised on M4+.
//! C is seeded with a sentinel so a tile the kernel forgets to write is caught
//! even when the correct value would be 0.

#![no_main]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use sme_gemm::gemm_i8;

// Caps the naive oracle's cost per input; 96*160*160 > 2^21 keeps shapes well
// past the SME threshold while staying fast enough for high exec/s.
const MAX_MACS: usize = 1 << 21;

// (row_stride, col_stride) for an r x c matrix, row- or column-major.
fn strides(cm: bool, r: usize, c: usize) -> (usize, usize) {
    if cm { (1, r) } else { (c, 1) }
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
    let (a_rs, a_cs) = strides(a_cm, m, k);
    let (b_rs, b_cs) = strides(b_cm, k, n);
    let (c_rs, c_cs) = strides(c_cm, m, n);

    // Fill A and B by cycling the remaining fuzz bytes (tiny inputs still make
    // valid matrices; the fuzzer grows entropy where it matters).
    let bytes = u.take_rest();
    let at = |idx: usize| -> i8 {
        if bytes.is_empty() {
            0
        } else {
            bytes[idx % bytes.len()] as i8
        }
    };
    let a: Vec<i8> = (0..m * k).map(at).collect();
    let b: Vec<i8> = (0..k * n).map(|i| at(i.wrapping_add(m * k))).collect();

    let mut c = vec![i32::from_le_bytes([0x5a; 4]); m * n];
    gemm_i8(m, n, k, &mut c, c_rs, c_cs, &a, a_rs, a_cs, &b, b_rs, b_cs);

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for l in 0..k {
                acc += i32::from(a[i * a_rs + l * a_cs]) * i32::from(b[l * b_rs + j * b_cs]);
            }
            let got = c[i * c_rs + j * c_cs];
            assert_eq!(
                got, acc,
                "gemm_i8 {m}x{n}x{k} (a_cm={a_cm} b_cm={b_cm} c_cm={c_cm}) at ({i},{j})"
            );
        }
    }
});
