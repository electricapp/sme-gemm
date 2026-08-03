//! Quantized integer GEMM entry points: `i8 -> i32` (M4+) and `i16 -> i64`
//! (M5+), with the fused-dequant variants that scale into f32.
//!
//! Split by width; the strided-dequant tests below cover both, since the arm
//! they exercise is shared C.

mod i16;
mod i8;

pub use i8::{
    gemm_i8, matmul_i8, matmul_i8_batched, matmul_i8_batched_dequant, matmul_i8_packed,
    matmul_i8_packed_dequant, prepack_i8,
};
pub use i16::{
    gemm_i16, matmul_i16, matmul_i16_batched, matmul_i16_batched_dequant, matmul_i16_dequant,
    matmul_i16_packed, matmul_i16_packed_dequant, prepack_i16,
};

// The kernels' fused-dequant store has a strided arm for a column-major or
// otherwise non-unit-column-stride `dst`. No safe entry point reaches it -- every
// Rust dequant call passes (dst_cs=1, dst_rs=n) -- but it is live C ABI, so these
// drive it directly. Without them the arm is unreachable and therefore untested.
// `not(miri)` for the same reason as `packed_size_tests` in kernels/f16.rs.
#[cfg(all(test, target_os = "macos", target_arch = "aarch64", not(miri)))]
mod strided_dequant_tests {
    use crate::epilogue::Dequant;
    use crate::exec::resolve_nodes;
    use crate::ffi::{
        DqF32, gemm_sme_i16i64_packb, gemm_sme_i16i64_packed_b_elems,
        gemm_sme_i16i64_run_packed_impl,
    };
    use crate::probe::caps;

    const fn rnd(s: &mut u64) -> i32 {
        *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (*s >> 48) as i32 % 17 - 8
    }

    /// `want[i][j]` for `dq = scale*acc + bias[j]`, then relu, from an exact
    /// i64 accumulation.
    fn oracle(
        a: &[i32],
        b: &[i32],
        m: usize,
        n: usize,
        k: usize,
        scale: f32,
        bias: &[f32],
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i64;
                for l in 0..k {
                    acc += i64::from(a[i * k + l]) * i64::from(b[l * n + j]);
                }
                let v = f64::from(scale) * acc as f64 + f64::from(bias[j]);
                out[i * n + j] = v.max(0.0);
            }
        }
        out
    }

    #[test]
    fn i8_strided_dequant_store_matches_oracle() {
        if !caps().sme {
            return;
        }
        let (m, n, k) = (48usize, 40usize, 64usize);
        let mut s = 0x51de_0001u64;
        let ai: Vec<i32> = (0..m * k).map(|_| rnd(&mut s)).collect();
        let bi: Vec<i32> = (0..k * n).map(|_| rnd(&mut s)).collect();
        let a: Vec<i8> = ai.iter().map(|&v| v as i8).collect();
        let b: Vec<i8> = bi.iter().map(|&v| v as i8).collect();
        let scale = 0.003_f32;
        let bias: Vec<f32> = (0..n).map(|j| 0.05 * (j as f32) - 0.4).collect();
        let dq = Dequant::new(scale).add_col(&bias).relu();
        let fnodes = resolve_nodes(&dq.nodes, dq.act, n);
        let desc = DqF32 {
            scale,
            scale_n: core::ptr::null(),
            n_nodes: u32::try_from(fnodes.len()).expect("nodes fit u32"),
            nodes: fnodes.as_ptr(),
        };

        // Column-major output: dst_cs = m (stride between columns), dst_rs = 1.
        let mut c = vec![0.0f32; m * n];
        // SAFETY: b_pack is sized by the kernel's own packed_b_elems and filled
        // by its packb; a is row-major m*k; c is m*n and the (m, 1) strides
        // describe it column-major, which is exactly the arm under test.
        let rc = unsafe {
            let elems = crate::ffi::gemm_sme_i8i32_packed_b_elems(n, k);
            let mut bp = vec![0i8; elems];
            crate::ffi::gemm_sme_i8i32_packb(bp.as_mut_ptr(), b.as_ptr(), n, k, n as isize, 1);
            crate::ffi::gemm_sme_i8i32_run_packed_dequant(
                m,
                n,
                k,
                c.as_mut_ptr(),
                m as isize,
                1,
                a.as_ptr(),
                1,
                k as isize,
                bp.as_ptr(),
                scale,
                core::ptr::null(),
                desc.n_nodes,
                desc.nodes,
            )
        };
        assert_eq!(rc, 0, "strided i8 dequant kernel must succeed");

        let want = oracle(&ai, &bi, m, n, k, scale, &bias);
        for i in 0..m {
            for j in 0..n {
                let got = f64::from(c[j * m + i]);
                let w = want[i * n + j];
                assert!(
                    (got - w).abs() <= 1e-4 * (1.0 + w.abs()),
                    "i8 strided dequant ({i},{j}): {got} vs {w}"
                );
            }
        }
    }

    #[test]
    fn i16_strided_dequant_store_matches_oracle() {
        if !caps().sme_i16i64 {
            return;
        }
        let (m, n, k) = (24usize, 20usize, 64usize);
        let mut s = 0x9c4e_0001u64;
        let ai: Vec<i32> = (0..m * k).map(|_| rnd(&mut s)).collect();
        let bi: Vec<i32> = (0..k * n).map(|_| rnd(&mut s)).collect();
        let a: Vec<i16> = ai.iter().map(|&v| v as i16).collect();
        let b: Vec<i16> = bi.iter().map(|&v| v as i16).collect();
        let scale = 0.002_f32;
        let bias: Vec<f32> = (0..n).map(|j| 0.03 * (j as f32) - 0.2).collect();
        let dq = Dequant::new(scale).add_col(&bias).relu();
        let fnodes = resolve_nodes(&dq.nodes, dq.act, n);
        let desc = DqF32 {
            scale,
            scale_n: core::ptr::null(),
            n_nodes: u32::try_from(fnodes.len()).expect("nodes fit u32"),
            nodes: fnodes.as_ptr(),
        };

        let mut c = vec![0.0f32; m * n];
        // SAFETY: as the i8 case, with the i16 packb and its void-typed dst.
        let rc = unsafe {
            let elems = gemm_sme_i16i64_packed_b_elems(n, k);
            let mut bp = vec![0i16; elems];
            gemm_sme_i16i64_packb(bp.as_mut_ptr(), b.as_ptr(), n, k, n as isize, 1);
            gemm_sme_i16i64_run_packed_impl(
                m,
                n,
                k,
                c.as_mut_ptr().cast::<core::ffi::c_void>(),
                m as isize,
                1,
                a.as_ptr(),
                1,
                k as isize,
                bp.as_ptr(),
                &raw const desc,
            )
        };
        assert_eq!(rc, 0, "strided i16 dequant kernel must succeed");

        let want = oracle(&ai, &bi, m, n, k, scale, &bias);
        for i in 0..m {
            for j in 0..n {
                let got = f64::from(c[j * m + i]);
                let w = want[i * n + j];
                assert!(
                    (got - w).abs() <= 1e-4 * (1.0 + w.abs()),
                    "i16 strided dequant ({i},{j}): {got} vs {w}"
                );
            }
        }
    }
}
