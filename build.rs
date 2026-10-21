//! Compiles the SME kernels (Apple aarch64 only; a no-op stub elsewhere).
//! Base `FEAT_SME` (M4+) TUs build together; the `FEAT_SME_F16F16` and
//! `FEAT_SME_B16B16` (M5+) TUs add their target features and are only called
//! behind a runtime probe, so building them keeps the M4 floor.
#![allow(missing_docs)]

fn main() {
    // Gate on the TARGET being built, not the host this script runs on (a
    // host `cfg` breaks cross builds and `cargo miri test --target ...`,
    // which must skip the C kernels entirely).
    let target = |var: &str| std::env::var(var).unwrap_or_default();
    if target("CARGO_CFG_TARGET_OS") == "macos" && target("CARGO_CFG_TARGET_ARCH") == "aarch64" {
        compile_sme_kernels();
        if std::env::var_os("CARGO_FEATURE_DISPATCH_CMP").is_some() {
            compile_dispatch_cmp();
        }
    }
}

/// Every kernel source and included fragment under `csrc/`. Fragments are `.h` files
/// `#include`d into the TU that owns their statics (see `csrc/epilogue.h` and
/// the `gemm_*_{batched,small,q4}.h` splits), so they need rerun tracking even
/// though they are never compiled on their own.
const CSRC: &[&str] = &[
    "pack.h",
    "panel_ring.h",
    "epilogue.h",
    "epilogue_scalar.h",
    "epilogue_f16.h",
    "epilogue_f32.h",
    "epilogue_f64.h",
    "epilogue_tile.h",
    "neon_act.h",
    "sme_gemm_api.h",
    "sme_tile.h",
    "sme_probe.cc",
    "sme_runtime_shims.cc",
    "sme_warm.cc",
    "neon_ops.cc",
    "attention.cc",
    "attention_kv_sme.h",
    "gemm_widen.cc",
    "gemm_i8i32.cc",
    "gemm_f32.cc",
    "gemm_float.h",
    "gemm_int.h",
    "gemm_f16f16.cc",
    "gemm_half.h",
    "gemm_f16f16_q4.h",
    "gemm_f16f16_lowbit.h",
    "gemm_b16b16.cc",
    "gemm_b16b16_q4.h",
    "gemm_f64.cc",
    "gemm_i16i64.cc",
    "gemm_q2.cc",
];

fn compile_sme_kernels() {
    use std::path::Path;

    // Opt-in AddressSanitizer for the C kernels: SME_GEMM_C_ASAN=1.
    // ASan's shadow checks are inline loads/compares (streaming-mode legal;
    // only the cold report paths are calls), so instrumenting the kernels is
    // viable -- but it is experimental and slow, hence the env knob rather
    // than a feature. Don't combine with `cargo fuzz` (rustc links its own
    // static ASan runtime; this knob links Apple clang's dynamic one).
    println!("cargo:rerun-if-env-changed=SME_GEMM_C_ASAN");
    let c_asan = std::env::var("SME_GEMM_C_ASAN").is_ok_and(|v| v == "1");

    for f in CSRC {
        println!("cargo:rerun-if-changed=csrc/{f}");
    }

    require_apple_m4();

    // C++26 without exceptions or RTTI, and nothing from libc++ that needs
    // linking: `std::atomic` and the templates are header-only.
    let base = move |b: &mut cc::Build| {
        b.cpp(true)
            .std("c++2c")
            .cpp_link_stdlib(None)
            .flag("-fno-exceptions")
            .flag("-fno-rtti")
            .flag("-mcpu=apple-m4")
            .flag("-ffp-contract=fast")
            .opt_level(3);
        if c_asan {
            b.flag("-fsanitize=address");
        }
        for w in [
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wshadow",
            "-Wpointer-arith",
            "-Wcast-qual",
            "-Wwrite-strings",
            "-Wvla",
            "-Wdouble-promotion",
            "-Wredundant-decls",
            "-Wundef",
            "-Wformat=2",
            "-Wcast-align",
            "-Wunreachable-code-aggressive",
            "-Wconditional-uninitialized",
        ] {
            b.flag(w);
        }
    };
    let have = |f: &str| Path::new(&format!("csrc/{f}")).exists();

    // One M5+ extension TU compiled with its target features (each is only
    // *called* behind a runtime probe, so it does not raise the M4 floor).
    // Every referenced kernel is required: its FFI symbols are declared
    // unconditionally on macOS/aarch64, so a missing source file would otherwise
    // surface only as an opaque link error (or, worse, link against a stale
    // archive) -- assert it exists up front with a clear message instead.
    let feat_tu = |file: &str, lib: &str, feats: &[&str]| {
        assert!(have(file), "required SME kernel csrc/{file} is missing");
        let mut b = cc::Build::new();
        base(&mut b);
        b.file(format!("csrc/{file}"));
        for f in feats {
            b.flag("-Xclang")
                .flag("-target-feature")
                .flag("-Xclang")
                .flag(f);
        }
        b.compile(lib);
    };

    // Base FEAT_SME translation units + the streaming-ABI shims.
    let mut sme = cc::Build::new();
    base(&mut sme);
    sme.file("csrc/sme_runtime_shims.cc")
        .file("csrc/sme_probe.cc")
        .file("csrc/sme_warm.cc")
        .file("csrc/neon_ops.cc");
    for f in [
        "gemm_widen.cc",
        "gemm_i8i32.cc",
        "gemm_f32.cc",
        "attention.cc",
    ] {
        assert!(have(f), "required SME kernel csrc/{f} is missing");
        sme.file(format!("csrc/{f}"));
    }
    sme.compile("sme_gemm_base");

    // b16b16 needs +sve-b16b16 too (svmul_bf16/svmla_bf16 in the epilogue).
    feat_tu("gemm_f16f16.cc", "sme_gemm_f16f16", &["+sme-f16f16"]);
    feat_tu(
        "gemm_b16b16.cc",
        "sme_gemm_b16b16",
        &["+sme-b16b16", "+sve-b16b16"],
    );
    feat_tu("gemm_f64.cc", "sme_gemm_f64", &["+sme-f64f64"]);
    feat_tu("gemm_i16i64.cc", "sme_gemm_i16i64", &["+sme-i16i64"]);
    feat_tu("gemm_q2.cc", "sme_gemm_q2", &["+sme2p1"]);

    link_compiler_rt(c_asan);
}

/// Metal MPS + `MLCompute` ANE backends for `examples/dispatch`. Feature-gated so
/// the default rlib does not link those frameworks.
fn compile_dispatch_cmp() {
    println!("cargo:rerun-if-changed=csrc/dispatch_cmp.m");
    println!("cargo:rerun-if-changed=csrc/dispatch_cmp.h");
    cc::Build::new()
        .file("csrc/dispatch_cmp.m")
        .include("csrc")
        .flag("-fobjc-arc")
        .flag("-Wno-deprecated-declarations")
        .opt_level(3)
        .compile("sme_gemm_dispatch_cmp");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MLCompute");
}

// Every TU is built `-mcpu=apple-m4 -std=c++2c`; the CPU name is what enables
// SME in the first place. Clang gained it in 16; older ones reject the flag
// with a bare "unsupported argument", once per source file and with no
// indication of what to do about it. Check once, up front, and say so.
fn require_apple_m4() {
    let cc = cc::Build::new().cpp(true).get_compiler();
    let path = cc.path().to_string_lossy().into_owned();
    let ok = std::process::Command::new(&path)
        .args(["-mcpu=apple-m4", "-std=c++2c", "-x", "c++", "-c", "-o"])
        .arg(std::env::temp_dir().join("sme_gemm_mcpu_probe.o"))
        .arg("-")
        .stdin(std::process::Stdio::null())
        .output()
        .is_ok_and(|o| o.status.success());
    assert!(
        ok,
        "`{path}` does not accept -mcpu=apple-m4 -std=c++2c, so it cannot build the SME \
         kernels. Apple clang 16 (Xcode 16) or a matching upstream clang is the \
         minimum; select one via $CC or `xcode-select`."
    );
}

// Streaming functions call __arm_tpidr2_{save,restore} from compiler-rt. Resolve
// the runtime dir from the SAME compiler the cc crate used (honors $CC / cc-rs
// target selection) rather than a bare `clang` on PATH, which can be a different
// toolchain than the one that built the objects.
fn link_compiler_rt(c_asan: bool) {
    let cc_path = cc::Build::new()
        .cpp(true)
        .get_compiler()
        .path()
        .to_string_lossy()
        .into_owned();
    let out = std::process::Command::new(&cc_path)
        .arg("--print-runtime-dir")
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "failed to run `{cc_path} --print-runtime-dir` (needed for the \
                 compiler-rt __arm_tpidr2 lazy-save routines the SME streaming \
                 kernels call): {e}. A clang with SME/arm_sme.h support (Xcode \
                 15+/Apple clang 16+) is required to build this crate."
            )
        });
    let rt = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(
        !rt.is_empty(),
        "`{cc_path} --print-runtime-dir` returned nothing; cannot locate \
         libclang_rt.osx (the SME kernels need its __arm_tpidr2 routines). \
         Ensure a compiler-rt-providing clang is selected (Xcode / $CC)."
    );
    println!("cargo:rustc-link-search=native={rt}");
    println!("cargo:rustc-link-lib=static=clang_rt.osx");

    // The instrumented objects need the ASan runtime; rustc drives the final
    // link and knows nothing about it, so pull in clang's dynamic runtime
    // (the standard macOS ASan link mode) from the same runtime dir.
    if c_asan {
        println!("cargo:rustc-link-lib=dylib=clang_rt.asan_osx_dynamic");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{rt}");
    }
}
