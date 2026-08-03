//! Runtime SME capability detection (cached). Backed by `sysctlbyname` on
//! Apple; everywhere else all capabilities report `false`.

use std::sync::OnceLock;

/// Which SME instruction groups the running CPU supports.
// Distinct CPU feature flags: a bool per capability is the natural shape.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Caps {
    /// Base `FEAT_SME`: f16->f32 / bf16->f32 widening MOPA, i8->i32 MOPA,
    /// f32->f32 MOPA. Present on Apple M4 and later.
    pub sme: bool,
    /// `FEAT_SME_F16F16`: non-widening 32x32 fp16 MOPA (~2x f16 throughput). M5+.
    pub sme_f16f16: bool,
    /// `FEAT_SME_B16B16`: non-widening 32x32 bf16 MOPA. M5+.
    pub sme_b16b16: bool,
    /// `FEAT_SME_I16I64`: i16->i64 MOPA. M5+.
    pub sme_i16i64: bool,
    /// `FEAT_SME_F64F64`: f64->f64 MOPA. M5+.
    pub sme_f64f64: bool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", not(miri)))]
unsafe extern "C" {
    fn sme_sysctl_flag(name: *const core::ffi::c_char) -> i32;
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", not(miri)))]
fn flag(name: &[u8]) -> bool {
    debug_assert_eq!(name.last().copied(), Some(0), "name must be nul-terminated");
    // SAFETY: `name` is a nul-terminated byte string; the C side only reads it.
    unsafe { sme_sysctl_flag(name.as_ptr().cast::<core::ffi::c_char>()) != 0 }
}

fn detect() -> Caps {
    // Miri cannot execute FFI (neither the sysctl probe nor the C kernels
    // behind it): report no SME so every entry point takes the pure-Rust
    // reference path -- exactly the code Miri can check for UB.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", not(miri)))]
    {
        Caps {
            sme: flag(b"hw.optional.arm.FEAT_SME\0"),
            sme_f16f16: flag(b"hw.optional.arm.FEAT_SME_F16F16\0"),
            sme_b16b16: flag(b"hw.optional.arm.FEAT_SME_B16B16\0"),
            sme_i16i64: flag(b"hw.optional.arm.FEAT_SME_I16I64\0"),
            sme_f64f64: flag(b"hw.optional.arm.FEAT_SME_F64F64\0"),
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", not(miri))))]
    {
        Caps::default()
    }
}

/// Detected SME capabilities for this process (probed once, cached).
#[must_use]
pub fn caps() -> Caps {
    static CAPS: OnceLock<Caps> = OnceLock::new();
    *CAPS.get_or_init(detect)
}

/// Whether base `FEAT_SME` is available (Apple M4+).
#[inline]
#[must_use]
pub fn has_sme() -> bool {
    caps().sme
}
