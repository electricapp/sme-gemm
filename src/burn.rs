//! burn integration (enabled by the `burn` feature).
//!
//! An adapter that runs `A @ B` on the SME kernels for burn f32 / f16 tensors.
//! burn has no per-op custom-op hook (unlike candle's `CustomOp2`), so this is
//! a plain helper over [`Tensor`], not a backend.

use ::burn::tensor::{DType, Tensor, TensorData};
use half::f16;

/// Run `a @ b` (2D) on the SME kernels via burn, returning a burn tensor.
///
/// `a` is `m x k`, `b` is `k x n`; the result is `m x n` on `a`'s device. When
/// both are f32 or both f16, the host data dispatch to [`crate::matmul_f32`] /
/// [`crate::matmul_f16`]; any other dtype pair falls back to burn's own
/// [`Tensor::matmul`].
///
/// # Panics
/// Panics if the inner dimensions differ (`k != k2`), or if reading the host
/// data for a dispatched dtype fails.
#[must_use]
pub fn sme_matmul(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    let [m, k] = a.dims();
    let [k2, n] = b.dims();
    assert_eq!(k, k2, "sme-matmul: inner dims differ ({k} vs {k2})");
    let device = a.device();
    let da = a.into_data();
    let db = b.into_data();
    match (da.dtype(), db.dtype()) {
        (DType::F32, DType::F32) => {
            let av = da.as_slice::<f32>().expect("f32 host data");
            let bv = db.as_slice::<f32>().expect("f32 host data");
            let mut c = vec![0f32; m * n];
            crate::matmul_f32(av, bv, &mut c, m, n, k);
            Tensor::from_data(TensorData::new(c, [m, n]), (&device, DType::F32))
        }
        (DType::F16, DType::F16) => {
            let av = da.as_slice::<f16>().expect("f16 host data");
            let bv = db.as_slice::<f16>().expect("f16 host data");
            let mut c = vec![f16::ZERO; m * n];
            crate::matmul_f16(av, bv, &mut c, m, n, k, crate::Accum::F16);
            Tensor::from_data(TensorData::new(c, [m, n]), (&device, DType::F16))
        }
        (ta, tb) => {
            // Unsupported dtypes: rebuild the consumed inputs and defer to burn.
            let a = Tensor::<2>::from_data(da, (&device, ta));
            let b = Tensor::<2>::from_data(db, (&device, tb));
            a.matmul(b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::sme_matmul;
    use ::burn::tensor::{DType, Device, Distribution, Tensor};

    /// Max over elements of `|got - want| / (1 + |want|)`, in f64.
    fn max_rel(got: Tensor<2>, want: Tensor<2>) -> f64 {
        let num = (got - want.clone()).abs();
        let den = want.abs() + 1.0;
        (num / den).max().into_scalar()
    }

    /// `[m, k]` and `[k, n]` uniform random tensors of `dtype` on the flex
    /// (pure-Rust CPU) backend.
    fn operands(dtype: DType) -> (Tensor<2>, Tensor<2>) {
        let dev = Device::flex();
        let (m, k, n) = (33usize, 17usize, 24usize);
        let a = Tensor::<2>::random([m, k], Distribution::Default, (&dev, dtype));
        let b = Tensor::<2>::random([k, n], Distribution::Default, (&dev, dtype));
        (a, b)
    }

    #[test]
    fn matches_burn_matmul() {
        let (a, b) = operands(DType::F32);
        let got = sme_matmul(a.clone(), b.clone());
        assert_eq!(got.dtype(), DType::F32, "f32 in, f32 out");
        let rel = max_rel(got, a.matmul(b));
        assert!(rel < 1e-4, "max rel diff {rel}");
    }

    // f16 inputs take the matmul_f16 arm (f16 accumulation): compare against
    // burn's f32 matmul of the same f16 values.
    #[test]
    fn f16_matches_burn_matmul() {
        let (a, b) = operands(DType::F16);
        let got = sme_matmul(a.clone(), b.clone());
        assert_eq!(got.dtype(), DType::F16, "f16 in, f16 out");
        let want = a.cast(DType::F32).matmul(b.cast(DType::F32));
        let rel = max_rel(got.cast(DType::F32), want);
        assert!(rel < 1e-2, "max rel diff {rel}");
    }

    // Unsupported-dtype fallback: the adapter dispatches f32/f16 to the SME
    // kernels and defers every other dtype to burn's own `Tensor::matmul`. f64
    // tensors take that fallback arm; assert it produces exactly burn's native
    // matmul (i.e. it rebuilds the consumed inputs at their own dtype and
    // dispatches correctly rather than crashing or mis-routing into an f32/f16
    // branch).
    #[test]
    fn unsupported_dtype_falls_back_to_burn() {
        let (a, b) = operands(DType::F64);
        let got = sme_matmul(a.clone(), b.clone());
        assert_eq!(got.dtype(), DType::F64, "the fallback keeps f64");
        // The fallback rebuilds the consumed inputs and defers to burn's own f64
        // matmul, so it must match burn's native matmul of the same data. Compare
        // to a tight tolerance (the into_data/from_data round-trip is not bitwise).
        let mr = max_rel(got, a.matmul(b));
        assert!(
            mr < 1e-5,
            "f64 fallback must match burn's native matmul: max rel {mr}"
        );
    }
}
