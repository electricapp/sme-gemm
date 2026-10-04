//! Guard-page overrun/underrun tests for the kernel operands.
//!
//! `AddressSanitizer` cannot see out-of-bounds accesses issued by SME/SVE
//! intrinsics in the C kernels (no shadow checks are emitted for them), so
//! these tests place every operand hard against an inaccessible page instead:
//! buffers either END at a `PROT_NONE` guard page (overrun traps) or START
//! right after one (underrun traps). An unpredicated load/store even one
//! element past a buffer edge faults the process immediately rather than
//! silently corrupting the heap. Results are not checked here -- correctness
//! is tests/correctness.rs' job; this suite only has to survive.
//!
//! On non-SME hosts the same calls run the bounds-checked Rust reference, so
//! the suite is cheap there and the guard pages are exercised for real on M4+.

// mmap/mprotect and the FFI kernels: nothing here is Miri-runnable.
#![cfg(not(miri))]

use half::{bf16, f16};
use sme_gemm::{
    Accum, Dequant, caps, gemm_f32, matmul_bf16, matmul_f16, matmul_f32, matmul_f64, matmul_i8,
    matmul_i8_packed, matmul_i8_packed_dequant, matmul_i16, prepack_i8,
};

/// Where the inaccessible page sits relative to the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Guard {
    /// Data ends exactly at the guard page: catches overruns.
    End,
    /// Data starts exactly after the guard page: catches underruns.
    Start,
}

/// A `len`-element buffer of `T` flush against a `PROT_NONE` page.
struct GuardedBuf<T> {
    base: *mut libc::c_void,
    map_len: usize,
    ptr: *mut T,
    len: usize,
}

impl<T: Copy> GuardedBuf<T> {
    fn new(len: usize, fill: T, guard: Guard) -> Self {
        // SAFETY: sysconf with a valid _SC_ name; no memory involved.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let bytes = len * size_of::<T>();
        let data_pages = bytes.div_ceil(page).max(1);
        let map_len = (data_pages + 1) * page;
        // SAFETY: plain anonymous mapping; checked against MAP_FAILED below.
        let base = unsafe {
            libc::mmap(
                core::ptr::null_mut(),
                map_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANON | libc::MAP_PRIVATE,
                -1,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED, "mmap failed");
        let (guard_off, data_off) = match guard {
            // [data pages][guard]: data ends at the guard page boundary.
            Guard::End => (data_pages * page, data_pages * page - bytes),
            // [guard][data pages]: data starts right after the guard page.
            Guard::Start => (0, page),
        };
        // SAFETY: guard_off is page-aligned and inside the mapping.
        let rc = unsafe { libc::mprotect(base.add(guard_off), page, libc::PROT_NONE) };
        assert_eq!(rc, 0, "mprotect failed");
        // SAFETY: data_off < map_len, so the offset stays inside the mapping.
        // data_off is a multiple of size_of::<T>() (bytes is, and page is a
        // power of two >= 8), so ptr is sufficiently aligned for T.
        let ptr = unsafe { base.add(data_off) }.cast::<T>();
        for i in 0..len {
            // SAFETY: i < len elements all lie in the accessible data range.
            unsafe { ptr.add(i).write(fill) };
        }
        Self {
            base,
            map_len,
            ptr,
            len,
        }
    }

    const fn as_slice(&self) -> &[T] {
        // SAFETY: ptr..ptr+len is initialized, accessible, and uniquely owned.
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above, through &mut self.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for GuardedBuf<T> {
    fn drop(&mut self) {
        // SAFETY: exactly the mapping created in `new`.
        unsafe { libc::munmap(self.base, self.map_len) };
    }
}

/// Tail-stressing shapes: SVL=512 tiles are 32-wide (f32/i32), so every
/// non-multiple-of-32 edge forces predicated tails; products straddle the SME
/// dispatch threshold (2^18) so both the streaming kernels and the small-path
/// code run against the guards.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 33, 257),
    (31, 32, 33),
    (32, 32, 32),
    (65, 65, 97),    // > 2^18, ragged everywhere
    (65, 97, 129),   // > 2^18, ragged everywhere
    (64, 16, 2048),  // narrow-N path
    (16, 1024, 129), // N-heavy, ragged K
];

/// The flat-M N-parallel `dispatch_apply` branch; only worth the cycles when
/// the SME kernels actually run.
const PARALLEL_SHAPE: (usize, usize, usize) = (64, 1024, 1024);

fn shapes() -> Vec<(usize, usize, usize)> {
    let mut v = SHAPES.to_vec();
    if caps().sme {
        v.push(PARALLEL_SHAPE);
    }
    v
}

/// Run `f(a, b, c, m, n, k)` with A, B, and C all flush against guard pages,
/// for every shape, in both guard directions.
fn run_guarded<TA: Copy, TC: Copy>(
    fill_a: TA,
    fill_c: TC,
    f: impl Fn(&[TA], &[TA], &mut [TC], usize, usize, usize),
) {
    for &(m, n, k) in &shapes() {
        for guard in [Guard::End, Guard::Start] {
            let a = GuardedBuf::new(m * k, fill_a, guard);
            let b = GuardedBuf::new(k * n, fill_a, guard);
            let mut c = GuardedBuf::new(m * n, fill_c, guard);
            f(a.as_slice(), b.as_slice(), c.as_mut_slice(), m, n, k);
        }
    }
}

#[test]
fn f32_guarded() {
    run_guarded(1.5f32, 0.0f32, |a, b, c, m, n, k| {
        matmul_f32(a, b, c, m, n, k);
    });
}

#[test]
fn f32_strided_col_major_c_guarded() {
    // Column-major C exercises the strided store path.
    run_guarded(1.5f32, 0.0f32, |a, b, c, m, n, k| {
        gemm_f32(m, n, k, c, 1, m, a, k, 1, b, n, 1, 0.0, 1.0);
    });
}

#[test]
fn f64_guarded() {
    run_guarded(1.5f64, 0.0f64, |a, b, c, m, n, k| {
        matmul_f64(a, b, c, m, n, k);
    });
}

#[test]
fn f16_guarded_both_accums() {
    for acc in [Accum::F32, Accum::F16] {
        run_guarded(f16::from_f32(1.5), f16::ZERO, |a, b, c, m, n, k| {
            matmul_f16(a, b, c, m, n, k, acc);
        });
    }
}

#[test]
fn bf16_guarded_both_accums() {
    for acc in [Accum::F32, Accum::Bf16] {
        run_guarded(bf16::from_f32(1.5), bf16::ZERO, |a, b, c, m, n, k| {
            matmul_bf16(a, b, c, m, n, k, acc);
        });
    }
}

#[test]
fn i8_guarded() {
    run_guarded(3i8, 0i32, |a, b, c, m, n, k| {
        matmul_i8(a, b, c, m, n, k);
    });
}

#[test]
fn i16_guarded() {
    run_guarded(3i16, 0i64, |a, b, c, m, n, k| {
        matmul_i16(a, b, c, m, n, k);
    });
}

// The integer drivers' col-major store arms were the only ones still forming an
// unclamped hi-band base (`col + 16` / `col + 8`) when that band is empty.
#[test]
fn i8_strided_col_major_c_guarded() {
    use sme_gemm::gemm_i8;
    run_guarded(3i8, 0i32, |a, b, c, m, n, k| {
        gemm_i8(m, n, k, c, 1, m, a, k, 1, b, n, 1);
    });
}

#[test]
fn i16_strided_col_major_c_guarded() {
    use sme_gemm::gemm_i16;
    run_guarded(3i16, 0i64, |a, b, c, m, n, k| {
        gemm_i16(m, n, k, c, 1, m, a, k, 1, b, n, 1);
    });
}

// The i16 fused-dequant store plus its per-N scale/bias operands, which the
// vectorized dequant reads with predicated vectors right up to their edge.
#[test]
fn i16_dequant_guarded() {
    use sme_gemm::matmul_i16_dequant;
    for &(m, n, k) in &shapes() {
        for guard in [Guard::End, Guard::Start] {
            let a = GuardedBuf::new(m * k, 3i16, guard);
            let b = GuardedBuf::new(k * n, 2i16, guard);
            let scale = GuardedBuf::new(n, 0.001f32, guard);
            let bias = GuardedBuf::new(n, 0.25f32, guard);
            let mut c = GuardedBuf::new(m * n, 0.0f32, guard);
            let dq = Dequant::new(0.001)
                .scale_per_n(scale.as_slice())
                .add_col(bias.as_slice())
                .relu();
            matmul_i16_dequant(a.as_slice(), b.as_slice(), c.as_mut_slice(), m, n, k, &dq);
        }
    }
}

#[test]
fn i16_packed_and_dequant_guarded() {
    use sme_gemm::{matmul_i16_packed, matmul_i16_packed_dequant, prepack_i16};
    for &(m, n, k) in &shapes() {
        let b_host: Vec<i16> = (0..k * n).map(|i| (i % 11) as i16 - 5).collect();
        let packed = prepack_i16(&b_host, n, k);
        for guard in [Guard::End, Guard::Start] {
            let a = GuardedBuf::new(m * k, 3i16, guard);
            let mut c = GuardedBuf::new(m * n, 0i64, guard);
            matmul_i16_packed(a.as_slice(), &packed, c.as_mut_slice(), m);

            let bias = GuardedBuf::new(n, 0.25f32, guard);
            let dq = Dequant::new(0.001).add_col(bias.as_slice()).relu();
            let mut cd = GuardedBuf::new(m * n, 0.0f32, guard);
            matmul_i16_packed_dequant(a.as_slice(), &packed, cd.as_mut_slice(), m, &dq);
        }
    }
}

// The 4-bit-resident path: A and C guarded, plus a fused epilogue's per-column
// bias. The nibbles and scales live in the Q4Weights' own tile-major buffers, so
// what the guards can reach here is the A-pack read, the predicated ZA store,
// and the epilogue's operand loads. Q4 tile dequant also reads its scales with
// fixed-width NEON loads, which the ragged-n shapes push against the tile edge.
#[test]
fn q4_resident_guarded() {
    use sme_gemm::{Epilogue, Q4Weights, matmul_q4, matmul_q4_bf16, matmul_q4_ep};
    for &(m, n, k) in &shapes() {
        let quants: Vec<u8> = (0..(k * n).div_ceil(2)).map(|i| (i % 251) as u8).collect();
        let scales: Vec<f16> = (0..n * k.div_ceil(32))
            .map(|i| f16::from_f32(0.01 + 0.001 * (i % 5) as f32))
            .collect();
        let w = Q4Weights::new(&quants, &scales, n, k);
        for guard in [Guard::End, Guard::Start] {
            let a = GuardedBuf::new(m * k, f16::from_f32(0.5), guard);
            let mut c = GuardedBuf::new(m * n, f16::ZERO, guard);
            matmul_q4(a.as_slice(), &w, c.as_mut_slice(), m);

            let bias = GuardedBuf::new(n, f16::from_f32(0.25), guard);
            let ep = Epilogue::new().add_col(bias.as_slice()).relu();
            matmul_q4_ep(a.as_slice(), &w, c.as_mut_slice(), m, &ep);

            let ab = GuardedBuf::new(m * k, bf16::from_f32(0.5), guard);
            let mut cb = GuardedBuf::new(m * n, bf16::ZERO, guard);
            matmul_q4_bf16(ab.as_slice(), &w, cb.as_mut_slice(), m);
        }
    }
}

#[test]
fn i8_packed_and_dequant_guarded() {
    // Packed-B path: A and C guarded (B lives in the library-owned pack), and
    // the dequant epilogue's per-column bias operand guarded too -- the C
    // epilogue reads it with predicated vectors right up to its edge.
    for &(m, n, k) in &shapes() {
        let b_host: Vec<i8> = (0..k * n).map(|i| (i % 5) as i8 - 2).collect();
        let packed = prepack_i8(&b_host, n, k);
        for guard in [Guard::End, Guard::Start] {
            let a = GuardedBuf::new(m * k, 3i8, guard);
            let mut c = GuardedBuf::new(m * n, 0i32, guard);
            matmul_i8_packed(a.as_slice(), &packed, c.as_mut_slice(), m);

            let bias = GuardedBuf::new(n, 0.25f32, guard);
            let dq = Dequant::new(0.001).add_col(bias.as_slice()).relu();
            let mut cd = GuardedBuf::new(m * n, 0.0f32, guard);
            matmul_i8_packed_dequant(a.as_slice(), &packed, cd.as_mut_slice(), m, &dq);
        }
    }
}
