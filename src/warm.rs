//! Keeping the SME unit awake between calls.
//!
//! A P-cluster's SME unit idles within a few hundred nanoseconds of its last
//! streaming instruction, and the next call pays about a microsecond to wake
//! it. A loop that alternates short SME calls with NEON work (norms, attention,
//! activations) pays it on every call: measured on M5, a 1152x384 m=1 Q4 GEMV
//! takes 2.2 us back to back and 3.0 us after 1 us of NEON work.
//!
//! While an [`SmeWarm`] is alive, a helper thread issues a ~0.1 us burst of SME
//! work whenever no library call is in flight, so the unit never idles. Calls
//! mark themselves busy (one atomic increment, skipped entirely when no guard
//! exists) and the helper steps aside, so they do not share the unit with it.
//! When no call has started for a few milliseconds the helper sleeps, so a
//! forgotten guard costs a wakeup every 0.2 ms rather than a core.
//!
//! `SME_GEMM_TRACE=1` prints every library SME call to stderr with its shape,
//! time, and the idle gap before it, and flags calls that follow a gap long
//! enough for the unit to fall asleep while no [`SmeWarm`] is alive. Printing
//! costs microseconds a line, so traced timings run slow; the gaps exclude it.
//!
//! The unit is per cluster and macOS offers no affinity: the helper runs at
//! user-interactive quality of service, which places it on a P-core, so on a chip with one
//! P-cluster it shares the caller's unit. With several P-clusters it helps only
//! when the scheduler puts both threads in the same one.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Live [`SmeWarm`] guards.
static REFS: AtomicU32 = AtomicU32::new(0);
/// Library SME calls in flight (counted only while a guard is alive).
static BUSY: AtomicU32 = AtomicU32::new(0);
/// Library SME calls started, so the helper can tell activity from idleness.
static CALLS: AtomicU64 = AtomicU64::new(0);
static HELPER: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);

/// No call for this long and the helper backs off to sleeping.
const IDLE_AFTER: Duration = Duration::from_millis(3);
const IDLE_SLEEP: Duration = Duration::from_micros(200);

/// Keeps the calling P-cluster's SME unit awake between library calls while
/// alive; see the [module docs](self). Guards nest and may be dropped on any
/// thread; the helper stops with the last one.
///
/// ```
/// let _warm = sme_gemm::SmeWarm::new();
/// // ... a loop of small matmul_q4 calls and NEON glue ...
/// ```
#[derive(Debug)]
pub struct SmeWarm(());

impl SmeWarm {
    /// Starts the helper if this is the first live guard. A no-op without SME.
    #[must_use]
    pub fn new() -> Self {
        let mut helper = HELPER
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if REFS.fetch_add(1, Ordering::AcqRel) == 0 && crate::caps().sme {
            // A helper from an earlier guard may still be on its way out.
            if let Some(h) = helper.take() {
                drop(h.join());
            }
            *helper = std::thread::Builder::new()
                .name("sme-gemm-warm".into())
                .spawn(run_helper)
                .ok();
        }
        drop(helper);
        Self(())
    }
}

impl Default for SmeWarm {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SmeWarm {
    fn drop(&mut self) {
        let mut helper = HELPER
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if REFS.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(h) = helper.take()
        {
            drop(h.join());
        }
    }
}

/// Marks one library SME call in flight for the helper, and times it for the
/// trace; held across the call.
pub(crate) struct Busy {
    warm: bool,
    trace: Option<Trace>,
}

struct Trace {
    name: &'static str,
    shape: (usize, usize, usize),
    start: Instant,
    idle: Option<Duration>,
}

/// Gaps from this long the SME unit has gone to sleep (measured on M5: 100 ns
/// costs nothing, 300 ns the full wake).
const COLD_AFTER: Duration = Duration::from_nanos(300);

fn trace_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("SME_GEMM_TRACE").is_some_and(|v| !v.is_empty() && v != "0")
    })
}

/// Nanoseconds since the first traced call, so the gap can live in an atomic.
fn trace_clock() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// End of the last traced call (0: none yet).
static LAST_END: AtomicU64 = AtomicU64::new(0);

/// The busy mark (and trace record) for the call `name` of shape `m x n x k`,
/// about to enter a streaming region.
#[inline]
pub(crate) fn busy(name: &'static str, m: usize, n: usize, k: usize) -> Busy {
    let warm = REFS.load(Ordering::Relaxed) > 0;
    if warm {
        BUSY.fetch_add(1, Ordering::AcqRel);
        CALLS.fetch_add(1, Ordering::Relaxed);
    }
    let trace = trace_on().then(|| {
        let (now, last) = (trace_clock(), LAST_END.load(Ordering::Relaxed));
        Trace {
            name,
            shape: (m, n, k),
            start: Instant::now(),
            idle: (last > 0).then(|| Duration::from_nanos(now.saturating_sub(last))),
        }
    });
    Busy { warm, trace }
}

impl Drop for Busy {
    #[inline]
    fn drop(&mut self) {
        if self.warm {
            BUSY.fetch_sub(1, Ordering::Release);
        }
        if let Some(t) = self.trace.take() {
            report(&t, self.warm);
            LAST_END.store(trace_clock(), Ordering::Relaxed);
        }
    }
}

#[cold]
fn report(t: &Trace, warm: bool) {
    let (m, n, k) = t.shape;
    let us = t.start.elapsed().as_secs_f64() * 1e6;
    let note = match t.idle {
        Some(gap) if !warm && gap >= COLD_AFTER && gap < Duration::from_secs(1) => format!(
            "  [{:.1} us idle: the SME unit was asleep and this call paid ~1 us to wake it; \
             an SmeWarm keeps it awake]",
            gap.as_secs_f64() * 1e6
        ),
        Some(gap) => format!("  [{:.1} us idle]", gap.as_secs_f64() * 1e6),
        None => String::new(),
    };
    eprintln!(
        "sme-gemm: {} {m}x{n}x{k} {us:.2} us{note}",
        t.name.trim_start_matches("gemm_sme_")
    );
}

fn run_helper() {
    set_user_interactive();
    let mut seen = CALLS.load(Ordering::Relaxed);
    let mut last = Instant::now();
    let mut n: u32 = 0;
    while REFS.load(Ordering::Acquire) > 0 {
        if BUSY.load(Ordering::Acquire) <= neon_phase() {
            tick();
        } else {
            std::hint::spin_loop();
        }
        n = n.wrapping_add(1);
        if n.is_multiple_of(64) {
            let c = CALLS.load(Ordering::Relaxed);
            if c == seen {
                if last.elapsed() > IDLE_AFTER {
                    std::thread::sleep(IDLE_SLEEP);
                }
            } else {
                seen = c;
                last = Instant::now();
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn set_user_interactive() {
    // QOS_CLASS_USER_INTERACTIVE: the class GCD places on P-cores.
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos: u32, relpri: i32) -> i32;
    }
    // SAFETY: sets the calling thread's own QoS class; no memory is involved.
    let _ = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) };
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(crate) const fn set_user_interactive() {}

/// Busy calls that are in a NEON phase (`csrc/neon_act.h`), so idle for SME.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn neon_phase() -> u32 {
    unsafe extern "C" {
        safe static sme_warm_neon: AtomicU32;
    }
    sme_warm_neon.load(Ordering::Acquire)
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
const fn neon_phase() -> u32 {
    0
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn tick() {
    // SAFETY: a self-contained streaming region with its own ZA state; it
    // touches no memory.
    unsafe { crate::ffi::sme_warm_tick() };
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn tick() {
    std::thread::sleep(IDLE_SLEEP);
}
