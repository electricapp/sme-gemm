//! Dispatch overhead: P-core Accelerate vs SME vs ANE vs Metal MPS, µs/call.
//! Graph compile / buffer alloc / SME `prepack_f32` sit outside the timer; the
//! timed region is submit + wait-for-result. Accelerate is `cblas_sgemm` at
//! `USER_INTERACTIVE` (Apple's CPU GEMM; M4/M5 may use AMX internally). Backends
//! are timed separately so a Metal wait does not disturb the CPU numbers.
//!   cargo run --release --example dispatch --features dispatch-cmp

use std::time::Instant;

use sme_gemm::{matmul_f32_packed, prepack_f32};

const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[link(name = "pthread")]
unsafe extern "C" {
    fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32;
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe extern "C" {
    fn sme_cmp_gpu_init() -> i32;
    fn sme_cmp_gpu_prepare(m: usize, n: usize, k: usize, a: *const f32, b: *const f32) -> i32;
    fn sme_cmp_gpu_gemm() -> i32;
    fn sme_cmp_gpu_drop();
    fn sme_cmp_ane_init() -> i32;
    fn sme_cmp_ane_prepare(
        m: usize,
        n: usize,
        k: usize,
        a: *mut f32,
        b: *mut f32,
        c: *mut f32,
    ) -> i32;
    fn sme_cmp_ane_layer_device() -> i32;
    fn sme_cmp_ane_gemm() -> i32;
    fn sme_cmp_ane_drop();
}

const SIZES: &[(usize, usize, usize)] = &[
    (8, 8, 8),
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (128, 128, 128),
    (256, 256, 256),
    (1, 64, 64),
    (1, 256, 256),
    (1, 1024, 1024),
    (1, 4096, 4096),
];

fn shape_label(m: usize, n: usize, k: usize) -> String {
    if m == n && n == k {
        format!("{m}³")
    } else {
        format!("{m}×{n}×{k}")
    }
}

const fn iters(m: usize, n: usize, k: usize) -> u32 {
    match m.saturating_mul(n).saturating_mul(k) {
        0..=4_096 => 400,
        4_097..=65_536 => 80,
        65_537..=2_097_152 => 20,
        _ => 6,
    }
}

/// Best-of-N for one backend. Warmup is inside so a cold MPS pipeline compile
/// does not land in the timed window.
fn best_us(n_iters: u32, f: &mut dyn FnMut()) -> f64 {
    for _ in 0..8 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..12 {
        let t = Instant::now();
        for _ in 0..n_iters {
            f();
        }
        best = best.min(t.elapsed().as_secs_f64() / f64::from(n_iters));
    }
    best * 1e6
}

const fn ane_label(dev: i32) -> &'static str {
    match dev {
        0 => "CPU",
        1 => "GPU",
        3 => "ANE",
        _ => "?",
    }
}

#[derive(Debug, Clone, Copy)]
struct ShapeTimes {
    pcore: f64,
    sme: f64,
    gpu: Option<f64>,
    ane: Option<f64>,
    ane_dev: i32,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn time_shape(m: usize, n: usize, k: usize, gpu_ok: bool, ane_ok: bool) -> ShapeTimes {
    let a = vec![0.01f32; m * k];
    let b = vec![0.02f32; k * n];
    let packed = prepack_f32(&b, n, k);
    let mut c_sme = vec![0.0f32; m * n];
    let mut c_p = vec![0.0f32; m * n];
    let mut c_ane = vec![0.0f32; m * n];
    let mut a_ane = a.clone();
    let mut b_ane = b.clone();

    // SAFETY: a/b live for this shape; GPU copies them into shared MTLBuffers.
    let gpu_ready = gpu_ok && unsafe { sme_cmp_gpu_prepare(m, n, k, a.as_ptr(), b.as_ptr()) } != 0;
    // SAFETY: host buffers outlive the compiled graph (dropped at next shape
    // or sme_cmp_ane_drop).
    let ane_ready = ane_ok
        && unsafe {
            sme_cmp_ane_prepare(
                m,
                n,
                k,
                a_ane.as_mut_ptr(),
                b_ane.as_mut_ptr(),
                c_ane.as_mut_ptr(),
            )
        } != 0;
    let ane_dev = if ane_ready {
        // SAFETY: prepare succeeded, so the layer device is populated.
        unsafe { sme_cmp_ane_layer_device() }
    } else {
        -1
    };

    // SAFETY: row-major M×K, K×N, M×N with matching leading dims; cblas
    // only reads a/b and writes c_p within bounds.
    let mut pcore_fn = || unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            NO_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            0.0,
            c_p.as_mut_ptr(),
            n as i32,
        );
    };
    let mut sme_fn = || matmul_f32_packed(&a, &packed, &mut c_sme, m);
    let mut gpu_fn = || {
        if gpu_ready {
            // SAFETY: prepare bound the MPS kernel and shared buffers.
            assert_eq!(unsafe { sme_cmp_gpu_gemm() }, 1, "MPS command buffer error");
        }
    };
    let mut ane_fn = || {
        if ane_ready {
            // SAFETY: prepare compiled the graph against a_ane/b_ane/c_ane.
            assert_eq!(unsafe { sme_cmp_ane_gemm() }, 1, "ANE/MLC execute failed");
        }
    };

    // Sequential, not interleaved: a Metal waitUntilCompleted in the same
    // round disturbs the CPU backends, and the 1×K×K numbers would
    // not match the throughput tables.
    let n_iters = iters(m, n, k);
    let pcore = best_us(n_iters, &mut pcore_fn);
    let sme = best_us(n_iters, &mut sme_fn);
    let gpu = gpu_ready.then(|| best_us(n_iters, &mut gpu_fn));
    let ane = ane_ready.then(|| best_us(n_iters, &mut ane_fn));
    ShapeTimes {
        pcore,
        sme,
        gpu,
        ane,
        ane_dev,
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("dispatch example is macOS aarch64 only");
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() {
    // SAFETY: this thread only; 0 relative priority is the documented default.
    let qos = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
    assert!(qos == 0, "pthread_set_qos_class_self_np USER_INTERACTIVE");

    // SAFETY: ObjC inits return 0 on nil device; no other shared state yet.
    let gpu_ok = unsafe { sme_cmp_gpu_init() } != 0;
    // SAFETY: as above, ANE device probe.
    let ane_ok = unsafe { sme_cmp_ane_init() } != 0;
    println!(
        "QoS USER_INTERACTIVE; GPU {}; ANE device {}",
        if gpu_ok { "yes" } else { "no" },
        if ane_ok { "yes" } else { "no" }
    );

    print!("{:<16}", "");
    for &(m, n, k) in SIZES {
        print!(" {:>12}", shape_label(m, n, k));
    }
    println!();

    let times: Vec<ShapeTimes> = SIZES
        .iter()
        .map(|&(m, n, k)| {
            eprint!("  timing {} ...\r", shape_label(m, n, k));
            time_shape(m, n, k, gpu_ok, ane_ok)
        })
        .collect();
    eprintln!();

    // SAFETY: pair of the inits; no further FFI after this.
    unsafe {
        sme_cmp_gpu_drop();
        sme_cmp_ane_drop();
    }

    print_row(
        "P-core (cblas)",
        &times.iter().map(|t| Some(t.pcore)).collect::<Vec<_>>(),
    );
    print_row(
        "SME (packed)",
        &times.iter().map(|t| Some(t.sme)).collect::<Vec<_>>(),
    );
    print_row(
        "GPU (MPS)",
        &times.iter().map(|t| t.gpu).collect::<Vec<_>>(),
    );
    print_ane_row(
        &times.iter().map(|t| t.ane).collect::<Vec<_>>(),
        &times.iter().map(|t| t.ane_dev).collect::<Vec<_>>(),
    );
    println!("µs/call, f32, best-of-N per backend. SME B is pre-packed (untimed).");
    println!("ANE is MLCompute MatMul on aneDevice; `/CPU` or `/GPU` means the");
    println!("MatMul layer did not stay on the ANE after compile. GPU times include");
    println!("command-buffer create + commit + waitUntilCompleted; buffers shared.");
}

fn print_row(name: &str, cells: &[Option<f64>]) {
    print!("{name:<16}");
    for cell in cells {
        match cell {
            Some(v) => print!(" {v:>12.2}"),
            None => print!(" {:>12}", "--"),
        }
    }
    println!();
}

fn print_ane_row(ane: &[Option<f64>], dev: &[i32]) {
    print!("{:<16}", "ANE (MLC)");
    for (cell, &d) in ane.iter().zip(dev) {
        match cell {
            Some(v) if d == 3 => print!(" {v:>12.2}"),
            Some(v) => print!(" {v:>8.2}/{}", ane_label(d)),
            None => print!(" {:>12}", "--"),
        }
    }
    println!();
}
