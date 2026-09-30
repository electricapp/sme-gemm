//! Pre-FFI footprint validation for the strided `gemm_*` entry points: every
//! slice must cover its dims-times-strides footprint BEFORE raw pointers cross
//! to the C kernels (which honor strides literally and cannot bounds-check).
//! Without them a short slice or an oversized stride is silent heap corruption
//! on SME hardware, and a bounds panic off it.

use half::{bf16, f16};
use sme_gemm::{
    Accum, Gemm, gemm_bf16, gemm_f16, gemm_f32, gemm_f64, gemm_i8, gemm_i16, matmul_f16_packed,
    matmul_f32, matmul_i8_packed, prepack_bf16, prepack_f16, prepack_i8,
};

#[test]
#[should_panic(expected = "c: 64x64")]
fn f32_short_c_panics() {
    let a = vec![0.0f32; 64 * 64];
    let b = vec![0.0f32; 64 * 64];
    let mut c = vec![0.0f32; 16]; // far too short for 64x64
    gemm_f32(64, 64, 64, &mut c, 64, 1, &a, 64, 1, &b, 64, 1, 0.0, 1.0);
}

#[test]
#[should_panic(expected = "c: 64x64")]
fn f32_oversized_col_stride_panics() {
    // c holds exactly m*n elements, but a row stride > n (a legal-looking
    // "leading dimension") walks the max index past the end.
    let a = vec![0.0f32; 64 * 64];
    let b = vec![0.0f32; 64 * 64];
    let mut c = vec![0.0f32; 64 * 64];
    gemm_f32(64, 64, 64, &mut c, 128, 1, &a, 64, 1, &b, 64, 1, 0.0, 1.0);
}

#[test]
#[should_panic(expected = "a: 64x64")]
fn f16_short_a_panics() {
    let a = vec![f16::ZERO; 64 * 64 - 1];
    let b = vec![f16::ZERO; 64 * 64];
    let mut c = vec![f16::ZERO; 64 * 64];
    gemm_f16(
        64,
        64,
        64,
        &mut c,
        64,
        1,
        &a,
        64,
        1,
        &b,
        64,
        1,
        f16::ZERO,
        f16::ONE,
        Accum::F32,
    );
}

#[test]
#[should_panic(expected = "b: 64x64")]
fn bf16_short_b_panics() {
    let a = vec![bf16::ZERO; 64 * 64];
    let b = vec![bf16::ZERO; 64 * 64 - 1];
    let mut c = vec![bf16::ZERO; 64 * 64];
    gemm_bf16(
        64,
        64,
        64,
        &mut c,
        64,
        1,
        &a,
        64,
        1,
        &b,
        64,
        1,
        bf16::ZERO,
        bf16::from_f32(1.0),
        Accum::F32,
    );
}

#[test]
#[should_panic(expected = "b: 64x64")]
fn f64_short_b_panics() {
    let a = vec![0.0f64; 64 * 64];
    let b = vec![0.0f64; 64 * 64 - 1];
    let mut c = vec![0.0f64; 64 * 64];
    gemm_f64(64, 64, 64, &mut c, 64, 1, &a, 64, 1, &b, 64, 1, 0.0, 1.0);
}

#[test]
#[should_panic(expected = "c: 64x64")]
fn i8_short_c_panics() {
    let a = vec![0i8; 64 * 64];
    let b = vec![0i8; 64 * 64];
    let mut c = vec![0i32; 64 * 64 - 1];
    gemm_i8(64, 64, 64, &mut c, 64, 1, &a, 64, 1, &b, 64, 1);
}

#[test]
#[should_panic(expected = "a: 64x64")]
fn i16_oversized_a_stride_panics() {
    let a = vec![0i16; 64 * 64];
    let b = vec![0i16; 64 * 64];
    let mut c = vec![0i64; 64 * 64];
    gemm_i16(64, 64, 64, &mut c, 64, 1, &a, 128, 1, &b, 64, 1);
}

#[test]
#[should_panic(expected = "overflows the slice")]
fn stride_overflow_is_caught() {
    // Strides chosen so (m-1)*rs overflows usize: checked arithmetic must
    // report the footprint violation rather than wrapping to a small index.
    let a = vec![0.0f32; 4];
    let b = vec![0.0f32; 4];
    let mut c = vec![0.0f32; 4];
    gemm_f32(2, 2, 2, &mut c, usize::MAX, 1, &a, 2, 1, &b, 2, 1, 0.0, 1.0);
}

#[test]
fn exact_fit_passes() {
    // Tight row-major buffers: max index is exactly len-1; must NOT panic.
    let m = 33;
    let n = 17;
    let k = 5;
    let a = vec![1.0f32; m * k];
    let b = vec![1.0f32; k * n];
    let mut c = vec![0.0f32; m * n];
    gemm_f32(m, n, k, &mut c, n, 1, &a, k, 1, &b, n, 1, 0.0, 1.0);
    assert!(c.iter().all(|&x| (x - k as f32).abs() < 1e-6));
}

#[test]
fn k_zero_is_safe() {
    // k == 0: A/B are empty views (nothing is read from them); C is zeroed.
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];
    let mut c = vec![7.0f32; 4];
    gemm_f32(2, 2, 0, &mut c, 2, 1, &a, 0, 1, &b, 0, 1, 0.0, 1.0);
    assert_eq!(c, vec![0.0; 4]);
}

// m == 0 and n == 0 are no-ops: the kernels short-circuit `if m==0||n==0` BEFORE
// the footprint validation, so C is never written and never bounds-checked. We
// deliberately pass a C that is too short / mis-strided for its (m,n) footprint;
// because the empty-dim short-circuit fires first, it must NOT panic and must
// leave C byte-for-byte untouched. (`matmul_*` asserts c.len()==m*n up front, so
// the empty case there uses an empty C; the strided `gemm_*` is where the
// before-validation short-circuit matters.)
#[test]
fn empty_dims_are_safe_noops() {
    // gemm_f32 with m == 0: C kept at a sentinel, deliberately under-sized for a
    // would-be 0x4 footprint and given a too-large stride. Untouched.
    let a: Vec<f32> = vec![];
    let b = vec![1.0f32; 4]; // k=4, n=4 -> b is k*n... unread anyway.
    let mut c = vec![9.0f32; 3]; // shorter than any m>0 footprint
    gemm_f32(0, 4, 4, &mut c, 999, 1, &a, 4, 1, &b, 4, 1, 0.5, 2.0);
    assert_eq!(c, vec![9.0; 3], "m==0 must not write C");

    // gemm_f32 with n == 0: same -- the n==0 arm of the short-circuit.
    let a2 = vec![1.0f32; 8]; // m=2, k=4
    let b2: Vec<f32> = vec![];
    let mut c2 = vec![5.0f32; 2];
    gemm_f32(2, 0, 4, &mut c2, 999, 1, &a2, 4, 1, &b2, 0, 1, 0.5, 2.0);
    assert_eq!(c2, vec![5.0; 2], "n==0 must not write C");

    // matmul_f32 with m == 0 / n == 0: c.len() must equal m*n (== 0) for the up
    // front length assert; an empty C is the valid empty result and is untouched.
    let mut c3: Vec<f32> = vec![];
    matmul_f32(&[], &[1.0f32; 4], &mut c3, 0, 4, 1);
    assert!(c3.is_empty(), "matmul_f32 m==0 -> empty C");
    let mut c4: Vec<f32> = vec![];
    matmul_f32(&[1.0f32; 4], &[], &mut c4, 4, 0, 1);
    assert!(c4.is_empty(), "matmul_f32 n==0 -> empty C");

    // i8 strided gemm: same empty-dim short-circuit (m==0), too-short C untouched.
    let ai: Vec<i8> = vec![];
    let bi = vec![1i8; 4];
    let mut ci = vec![42i32; 3];
    gemm_i8(0, 4, 4, &mut ci, 999, 1, &ai, 4, 1, &bi, 4, 1);
    assert_eq!(ci, vec![42i32; 3], "i8 m==0 must not write C");
}

// Batched count == 0: an empty C makes the inferred count 0, which the batched
// entry points short-circuit (`m==0 || n==0 || c.is_empty()`) before any FFI or
// length math. Must not panic and must leave the (empty) C untouched.
#[test]
fn batched_count_zero_is_safe() {
    use sme_gemm::{matmul_f32_batched, matmul_i8_batched};
    let mut cf: Vec<f32> = vec![];
    matmul_f32_batched(&[], &[], &mut cf, 4, 4, 4);
    assert!(cf.is_empty(), "f32 batched count==0 -> empty C");

    let mut ci: Vec<i32> = vec![];
    matmul_i8_batched(&[], &[], &mut ci, 4, 4, 4);
    assert!(ci.is_empty(), "i8 batched count==0 -> empty C");
}

// --- dimension-product overflow on the paths with NO `check_strided` backstop --
// They assert slice lengths against a dimension PRODUCT, then hand raw m/n/k to
// the C kernels. A product that wraps mod 2^64 can match a short slice, and the
// kernel then walks far past it -- so those asserts need checked arithmetic.

#[test]
#[should_panic(expected = "dimension product overflows usize")]
fn packed_i8_wrapped_mn_panics() {
    // The sharpest case: k == 0 makes `m*k` legitimately 0 for ANY m, so the A
    // assert cannot catch a bogus m; only the `m*n` product can. With n == 2 and
    // m == 2^63+1, `m*n` wraps to 2, so an unchecked product accepts a 2-element
    // C and the packed i8 store writes ~2^63 rows into it. (k == 0 also keeps
    // every internal allocation zero-sized, so no OOM check catches it either.)
    let w = prepack_i8(&[], 2, 0);
    let mut c = vec![0i32; 2];
    matmul_i8_packed(&[], &w, &mut c, (1usize << 63) | 1);
}

#[test]
#[should_panic(expected = "dimension product overflows usize")]
fn packed_f16_wrapped_mk_panics() {
    let b = vec![f16::ZERO; 4]; // k=2, n=2
    let w = prepack_f16(&b, 2, 2);
    let a = vec![f16::ZERO; 2]; // == (2^63+1)*2 mod 2^64
    let mut c = vec![f16::ZERO; 2];
    matmul_f16_packed(&a, &w, &mut c, (1usize << 63) | 1);
}

#[test]
#[should_panic(expected = "dimension product overflows usize")]
fn gemm_builder_wrapped_mk_panics() {
    // Same wrap through the `Gemm` fused-epilogue builder (packed_ep_impl).
    // bf16 rather than f16 so this stays runnable under Miri on aarch64 hosts
    // (`Gemm::new` seeds beta via `T::one()`, and half's f16::from_f32 is inline
    // fcvt asm there, which Miri cannot interpret; bf16's is pure Rust).
    let b = vec![bf16::ZERO; 4];
    let w = prepack_bf16(&b, 2, 2);
    let a = vec![bf16::ZERO; 2];
    let mut c = vec![bf16::ZERO; 2];
    Gemm::new(&a, &w, (1usize << 63) | 1).run(&mut c);
}

#[test]
#[should_panic(expected = "dimension product overflows usize")]
fn prepack_wrapped_kn_panics() {
    // `k*n` wraps to 0, so an unchecked length assert accepts an empty B and
    // reaches the C packer with k == 2^63.
    drop(prepack_f16(&[], 2, 1usize << 63));
}

#[test]
#[should_panic(expected = "dimension product overflows usize")]
fn matmul_f32_wrapped_mn_panics() {
    matmul_f32(&[], &[], &mut [0.0f32; 2], (1usize << 63) | 1, 2, 0);
}

#[test]
fn dim_products_at_the_limit_still_work() {
    // The guard must reject only genuine overflow, not large-but-exact products.
    // usize::MAX/2 * 2 does not overflow, so this reaches the ordinary "wrong
    // length" assert instead -- i.e. no false positive from the checked mul.
    let r = std::panic::catch_unwind(|| {
        matmul_f32(&[], &[], &mut [], usize::MAX / 2, 2, 0);
    });
    assert!(r.is_err(), "c.len() != m*n must still panic");
    // ...and an ordinary shape is unaffected.
    let mut c = vec![0.0f32; 6];
    matmul_f32(&[1.0; 6], &[1.0; 4], &mut c, 3, 2, 2);
    assert_eq!(c, vec![2.0; 6]);
}

#[test]
fn col_major_views_pass() {
    // Column-major A/B/C (row stride 1, col stride = rows): footprint math
    // must accept transposed layouts, and the result must match row-major.
    let (m, n, k) = (40, 24, 8);
    let a_rm: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.25).collect();
    let b_rm: Vec<f32> = (0..k * n).map(|i| (i % 7) as f32 * 0.5).collect();

    // Transpose into col-major buffers.
    let mut a_cm = vec![0.0f32; m * k];
    for i in 0..m {
        for l in 0..k {
            a_cm[l * m + i] = a_rm[i * k + l];
        }
    }
    let mut b_cm = vec![0.0f32; k * n];
    for l in 0..k {
        for j in 0..n {
            b_cm[j * k + l] = b_rm[l * n + j];
        }
    }

    let mut c_rm = vec![0.0f32; m * n];
    gemm_f32(m, n, k, &mut c_rm, n, 1, &a_rm, k, 1, &b_rm, n, 1, 0.0, 1.0);
    let mut c_cm = vec![0.0f32; m * n];
    gemm_f32(m, n, k, &mut c_cm, 1, m, &a_cm, 1, m, &b_cm, 1, k, 0.0, 1.0);

    for i in 0..m {
        for j in 0..n {
            assert!((c_rm[i * n + j] - c_cm[j * m + i]).abs() < 1e-4);
        }
    }
}
