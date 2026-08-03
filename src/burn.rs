//! burn integration (enabled by the `burn` feature).
//!
//! An adapter that runs `A @ B` on the SME kernels for burn CPU f32 / f16
//! tensors. burn has no per-op custom-op hook (unlike candle's `CustomOp2`), so
//! this is a plain helper, not a
//! [`Backend`] impl.

use ::burn::tensor::backend::Backend;
use ::burn::tensor::{DType, Tensor, TensorData};
use half::f16;

/// Run `a @ b` (2D) on the SME kernels via burn, returning a burn tensor.
///
/// `a` is `m x k`, `b` is `k x n`; the result is `m x n`. f32 and f16 host
/// data dispatch to [`crate::matmul_f32`] / [`crate::matmul_f16`]; any other
/// dtype falls back to burn's own [`Tensor::matmul`].
///
/// # Panics
/// Panics if the inner dimensions differ (`k != k2`), or if reading the host
/// data for a dispatched dtype fails.
#[must_use]
pub fn sme_matmul<B: Backend>(a: Tensor<B, 2>, b: Tensor<B, 2>) -> Tensor<B, 2> {
    let [m, k] = a.dims();
    let [k2, n] = b.dims();
    assert_eq!(k, k2, "sme-matmul: inner dims differ ({k} vs {k2})");
    let device = a.device();
    let da = a.into_data();
    let db = b.into_data();
    match da.dtype {
        DType::F32 => {
            let av = da.as_slice::<f32>().expect("f32 host data");
            let bv = db.as_slice::<f32>().expect("f32 host data");
            let mut c = vec![0f32; m * n];
            crate::matmul_f32(av, bv, &mut c, m, n, k);
            Tensor::from_data(TensorData::new(c, [m, n]), &device)
        }
        // Reachable only for burn backends whose float element is f16 (e.g. tch /
        // wgpu / candle). The bundled `NdArray` backend has no f16 float storage
        // (`FloatNdArrayElement` is implemented for f32/f64 only), so an NdArray
        // tensor never reports `DType::F16` and this arm is dead for it -- hence no
        // NdArray unit test exercises it. f16 GEMM correctness is covered directly
        // by `matmul_f16` (correctness/proptest) and through the candle adapter.
        DType::F16 => {
            let av = da.as_slice::<f16>().expect("f16 host data");
            let bv = db.as_slice::<f16>().expect("f16 host data");
            let mut c = vec![f16::ZERO; m * n];
            crate::matmul_f16(av, bv, &mut c, m, n, k, crate::Accuracy::Fast);
            Tensor::from_data(TensorData::new(c, [m, n]), &device)
        }
        _ => {
            // Unsupported dtype: rebuild the consumed inputs and defer to burn.
            let a = Tensor::<B, 2>::from_data(da, &device);
            let b = Tensor::<B, 2>::from_data(db, &device);
            a.matmul(b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sme_matmul;
    use ::burn::backend::NdArray;
    use ::burn::backend::ndarray::NdArrayDevice;
    use ::burn::tensor::{Distribution, Tensor};

    /// Max over elements of `|got - want| / (1 + |want|)`.
    fn max_rel(got: &Tensor<NdArray<f32>, 2>, want: &Tensor<NdArray<f32>, 2>) -> f32 {
        let num = (got.clone() - want.clone()).abs();
        let den = want.clone().abs() + 1.0;
        (num / den).max().into_scalar()
    }

    #[test]
    fn matches_burn_matmul() {
        let dev = NdArrayDevice::default();
        let (m, k, n) = (33usize, 17usize, 24usize);
        let a = Tensor::<NdArray<f32>, 2>::random([m, k], Distribution::Default, &dev);
        let b = Tensor::<NdArray<f32>, 2>::random([k, n], Distribution::Default, &dev);
        let got = sme_matmul(a.clone(), b.clone());
        let want = a.matmul(b);
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-4, "max rel diff {rel}");
    }

    // Unsupported-dtype fallback: the adapter dispatches f32/f16 to the SME
    // kernels and defers every other dtype to burn's own `Tensor::matmul`. An
    // f64 backend (`NdArray<f64>` reports `DType::F64`) takes that fallback arm;
    // assert it produces exactly burn's native matmul (i.e. it rebuilds the
    // consumed inputs and dispatches correctly rather than crashing or
    // mis-routing into an f32/f16 branch).
    #[test]
    fn unsupported_dtype_falls_back_to_burn() {
        let dev = NdArrayDevice::default();
        let (m, k, n) = (33usize, 17usize, 24usize);
        let a = Tensor::<NdArray<f64>, 2>::random([m, k], Distribution::Default, &dev);
        let b = Tensor::<NdArray<f64>, 2>::random([k, n], Distribution::Default, &dev);
        let got = sme_matmul(a.clone(), b.clone());
        let want = a.matmul(b);
        // The fallback rebuilds the consumed inputs and defers to burn's own f64
        // matmul, so it must match burn's native matmul of the same data. Compare
        // to a tight tolerance (the into_data/from_data round-trip is not bitwise).
        let num = (got - want.clone()).abs();
        let den = want.abs() + 1.0;
        let mr: f64 = (num / den).max().into_scalar();
        assert!(
            mr < 1e-5,
            "f64 fallback must match burn's native matmul: max rel {mr}"
        );
    }
}
