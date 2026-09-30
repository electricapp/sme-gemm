//! candle integration (enabled by the `candle` feature): a `CustomOp2` that
//! runs `A @ B` on the SME kernels for CPU f32 / f16 / bf16 tensors.

use candle_core::{CpuStorage, CustomOp2, Error, Layout, Result, Shape, Tensor};
use half::{bf16, f16};

use crate::{Accum, matmul_bf16, matmul_f16, matmul_f32};

/// A candle [`CustomOp2`] computing `A @ B` (2D, contiguous) on the SME kernels.
#[derive(Debug, Clone, Copy)]
pub struct SmeMatmul;

impl CustomOp2 for SmeMatmul {
    fn name(&self) -> &'static str {
        "sme-matmul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (m, k) = l1.shape().dims2()?;
        let (k2, n) = l2.shape().dims2()?;
        if k != k2 {
            return Err(Error::Msg(format!(
                "sme-matmul: inner dims differ ({k} vs {k2})"
            )));
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            return Err(Error::Msg("sme-matmul: inputs must be contiguous".into()));
        }
        let (o1, o2) = (l1.start_offset(), l2.start_offset());
        let shape = Shape::from((m, n));
        let short = || Error::Msg("sme-matmul: storage shorter than expected".into());
        match (s1, s2) {
            (CpuStorage::F32(a), CpuStorage::F32(b)) => {
                let a = a.get(o1..o1 + m * k).ok_or_else(short)?;
                let b = b.get(o2..o2 + k * n).ok_or_else(short)?;
                let mut c = vec![0f32; m * n];
                matmul_f32(a, b, &mut c, m, n, k);
                Ok((CpuStorage::F32(c), shape))
            }
            (CpuStorage::F16(a), CpuStorage::F16(b)) => {
                let a = a.get(o1..o1 + m * k).ok_or_else(short)?;
                let b = b.get(o2..o2 + k * n).ok_or_else(short)?;
                let mut c = vec![f16::ZERO; m * n];
                matmul_f16(a, b, &mut c, m, n, k, Accum::F16);
                Ok((CpuStorage::F16(c), shape))
            }
            (CpuStorage::BF16(a), CpuStorage::BF16(b)) => {
                let a = a.get(o1..o1 + m * k).ok_or_else(short)?;
                let b = b.get(o2..o2 + k * n).ok_or_else(short)?;
                let mut c = vec![bf16::ZERO; m * n];
                matmul_bf16(a, b, &mut c, m, n, k, Accum::Bf16);
                Ok((CpuStorage::BF16(c), shape))
            }
            _ => Err(Error::Msg(
                "sme-matmul: only f32/f16/bf16 CPU tensors".into(),
            )),
        }
    }
}

/// Run `a @ b` on the SME kernels via candle (2D, contiguous, f32/f16/bf16).
///
/// # Errors
/// Returns an error if the tensors are not 2D contiguous CPU f32/f16/bf16 with
/// matching inner dimensions.
pub fn sme_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.apply_op2(b, SmeMatmul)
}

#[cfg(test)]
mod tests {
    // Tests assert on results directly; unwrap IS the assertion.
    #![allow(clippy::unwrap_used)]

    use super::sme_matmul;
    use candle_core::{DType, Device, Tensor};

    /// Max over elements of `|got - want| / (1 + |want|)`, both cast to f32.
    fn max_rel(got: &Tensor, want: &Tensor) -> f32 {
        let got = got.to_dtype(DType::F32).unwrap();
        let want = want.to_dtype(DType::F32).unwrap();
        let num = (&got - &want).unwrap().abs().unwrap();
        let den = (want.abs().unwrap() + 1f64).unwrap();
        (num / den)
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn matches_candle_matmul() {
        let dev = Device::Cpu;
        let (m, k, n) = (33usize, 17usize, 24usize);
        let a = Tensor::randn(0f32, 1f32, (m, k), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, (k, n), &dev).unwrap();
        let got = sme_matmul(&a, &b).unwrap();
        let want = a.matmul(&b).unwrap();
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-5, "max rel diff {rel}");
    }

    #[test]
    fn matches_candle_matmul_f16() {
        let dev = Device::Cpu;
        let (m, k, n) = (33usize, 17usize, 24usize);
        let a = Tensor::randn(0f32, 1f32, (m, k), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let b = Tensor::randn(0f32, 1f32, (k, n), &dev)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let got = sme_matmul(&a, &b).unwrap();
        let want = a.matmul(&b).unwrap();
        let tol = 5e-2 * (k as f32).sqrt();
        let rel = max_rel(&got, &want);
        assert!(rel < tol, "max rel diff {rel} (tol {tol})");
    }

    // A sliced (`narrow`ed) input has `start_offset > 0` but stays contiguous, so
    // the op must add the offset to the storage range and still compute the right
    // product. Slice off the first 2 rows of A, then matmul; compare against a
    // plain contiguous matmul of the same logical (sliced) data.
    #[test]
    fn matmul_sliced_start_offset() {
        let dev = Device::Cpu;
        let (m_full, k, n) = (35usize, 17usize, 24usize);
        let a_full = Tensor::randn(0f32, 1f32, (m_full, k), &dev).unwrap();
        let b = Tensor::randn(0f32, 1f32, (k, n), &dev).unwrap();
        // narrow along dim 0: rows 2..2+33 -> start_offset = 2*k > 0, contiguous.
        let a = a_full.narrow(0, 2, 33).unwrap();
        assert!(a.layout().start_offset() > 0, "expected a non-zero offset");
        assert!(a.is_contiguous(), "narrowed row-slice must stay contiguous");
        let got = sme_matmul(&a, &b).unwrap();
        // Reference: a freshly built offset-0 tensor holding the same logical data
        // (round-trip through host data resets the start offset to 0). If the op
        // ignored start_offset it would read the wrong rows and diverge here.
        let a0 = Tensor::from_vec(
            a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            (33usize, k),
            &dev,
        )
        .unwrap();
        assert_eq!(a0.layout().start_offset(), 0, "fresh tensor has offset 0");
        let want = sme_matmul(&a0, &b).unwrap();
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-6, "sliced offset matmul diverges: max rel {rel}");
    }

    // A non-contiguous (transposed) input must be REJECTED with `Err`, not produce
    // a silently-wrong result (the op reads the raw storage and would otherwise
    // index it row-major regardless of the transposed layout).
    #[test]
    fn non_contiguous_input_errors() {
        let dev = Device::Cpu;
        let (k, n) = (17usize, 24usize);
        // a_t is the transpose of a (k x m) tensor -> a is m x k but non-contiguous.
        let a_src = Tensor::randn(0f32, 1f32, (k, 20usize), &dev).unwrap();
        let a = a_src.t().unwrap(); // 20 x k, transposed (non-contiguous)
        assert!(!a.is_contiguous(), "transpose must be non-contiguous");
        let b = Tensor::randn(0f32, 1f32, (k, n), &dev).unwrap();
        assert!(
            sme_matmul(&a, &b).is_err(),
            "non-contiguous input must return Err"
        );
    }

    // Mismatched inner dimensions (k != k2) must return `Err`.
    #[test]
    fn k_mismatch_errors() {
        let dev = Device::Cpu;
        let a = Tensor::randn(0f32, 1f32, (8usize, 17usize), &dev).unwrap();
        // b has inner dim 16 != 17.
        let b = Tensor::randn(0f32, 1f32, (16usize, 24usize), &dev).unwrap();
        assert!(sme_matmul(&a, &b).is_err(), "k-mismatch must return Err");
    }

    #[test]
    fn matches_candle_matmul_bf16() {
        let dev = Device::Cpu;
        let (m, k, n) = (33usize, 17usize, 24usize);
        let a = Tensor::randn(0f32, 1f32, (m, k), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let b = Tensor::randn(0f32, 1f32, (k, n), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let got = sme_matmul(&a, &b).unwrap();
        // candle has no CPU bf16 matmul; reference in f32 then cast back.
        let want = a
            .to_dtype(DType::F32)
            .unwrap()
            .matmul(&b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let tol = 5e-2 * (k as f32).sqrt();
        let rel = max_rel(&got, &want);
        assert!(rel < tol, "max rel diff {rel} (tol {tol})");
    }
}
