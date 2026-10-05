//! f16 <-> f32 slice conversion at a model's edges.
//!
//! `half` converts one value per inline-asm `FCVT` (~0.3 ns a value on M5),
//! which a projection's input and its residual add each pay per call; the NEON
//! passes here do 16 values a step and round the same way.

use half::f16;
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
use half::slice::HalfFloatSliceExt;

/// `dst = src` rounded to f16.
pub(crate) fn to_f16(src: &[f32], dst: &mut [f16]) {
    assert_eq!(src.len(), dst.len(), "conversion is elementwise");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    // SAFETY: both slices hold `src.len()` values; f16 is repr(transparent)
    // over the u16 the pass writes.
    unsafe {
        crate::ffi::neon_f32_to_f16(dst.as_mut_ptr().cast::<u16>(), src.as_ptr(), src.len());
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    dst.convert_from_f32_slice(src);
}

/// `dst = src` widened to f32.
pub(crate) fn to_f32(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "conversion is elementwise");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    // SAFETY: as `to_f16`.
    unsafe {
        crate::ffi::neon_f16_to_f32(dst.as_mut_ptr(), src.as_ptr().cast::<u16>(), src.len());
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    src.convert_to_f32_slice(dst);
}

/// `dst += src`, f16 into f32.
pub(crate) fn add_f16(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "the add is elementwise");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    // SAFETY: as `to_f16`.
    unsafe {
        crate::ffi::neon_add_f16_to_f32(dst.as_mut_ptr(), src.as_ptr().cast::<u16>(), src.len());
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s.to_f32();
    }
}
