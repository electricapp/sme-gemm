//! [`HotPool`]: worker threads kept spinning, so a pass of a few microseconds
//! can split across cores.
//!
//! GCD's `dispatch_apply` costs microseconds to wake its workers, which is as
//! long as one layer's attention over a short KV cache takes on one core. While
//! a `HotPool` is alive its workers spin on their own cache line each, and the
//! library's row-parallel NEON passes (KV attention) hand them items through it
//! in a few hundred nanoseconds. The caller takes items too and returns only
//! once every worker it engaged has reported back, so a job can live on the
//! caller's stack. Workers that see no job for a few milliseconds back off to
//! sleeping, so a forgotten pool costs a wakeup every 0.1 ms rather than cores.
//!
//! Workers run at user-interactive priority, which places them on P-cores. Leave
//! room for the calling thread and, with an [`SmeWarm`](crate::SmeWarm), its
//! helper: on a 4-P-core M5 that is two workers.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const MAX_WORKERS: usize = HotPool::MAX_WORKERS;
const IDLE_AFTER: Duration = Duration::from_millis(3);
const IDLE_SLEEP: Duration = Duration::from_micros(100);

/// One worker's mailbox: the job sequence it was handed and the last it
/// finished. Each on its own cache line, so posting to one does not disturb
/// the others' spinning.
#[repr(align(128))]
struct Slot {
    posted: AtomicU64,
    done: AtomicU64,
}

static SLOTS: [Slot; MAX_WORKERS] = [const {
    Slot {
        posted: AtomicU64::new(0),
        done: AtomicU64::new(0),
    }
}; MAX_WORKERS];
/// A cache line of its own. The workers poll `LIVE` on every spin, so the
/// caller-side atomics (`IN_USE`, `SEQ`, `JOB`) must not share its line: a
/// read-modify-write on a line other cores are spinning on waits for each of
/// them to give it up, which cost a post ~0.1 us.
#[repr(align(128))]
struct Line<T>(T);

impl<T> core::ops::Deref for Line<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

/// Running workers; 0 while no pool is alive.
static LIVE: Line<AtomicUsize> = Line(AtomicUsize::new(0));
/// Held by the one call using the pool, and by shutdown.
static IN_USE: Line<AtomicBool> = Line(AtomicBool::new(false));
static SEQ: Line<AtomicU64> = Line(AtomicU64::new(0));
static JOB: Line<AtomicPtr<Job<'static>>> = Line(AtomicPtr::new(core::ptr::null_mut()));
static REFS: AtomicUsize = AtomicUsize::new(0);
static HANDLES: Mutex<Vec<JoinHandle<()>>> = Mutex::new(Vec::new());

struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync),
    items: usize,
    next: AtomicUsize,
    panicked: AtomicBool,
}

impl Job<'_> {
    /// Claims and runs items until none are left.
    fn drain(&self) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            if i >= self.items {
                return;
            }
            if catch_unwind(AssertUnwindSafe(|| (self.f)(i))).is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
        }
    }
}

/// Keeps `workers` threads spinning for the library's row-parallel passes while
/// alive; see the [module docs](self). Guards nest; the first one's worker
/// count holds until the last is dropped.
///
/// ```
/// let _pool = sme_gemm::HotPool::new(2);
/// // ... KvCache::attend now splits its heads across the caller and 2 workers
/// ```
#[derive(Debug)]
pub struct HotPool(());

impl HotPool {
    /// Workers a pool can hold.
    pub const MAX_WORKERS: usize = 8;

    /// Starts `workers` threads (at most [`HotPool::MAX_WORKERS`]) if no pool
    /// is alive.
    #[must_use]
    pub fn new(workers: usize) -> Self {
        let mut handles = HANDLES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if REFS.fetch_add(1, Ordering::AcqRel) == 0 {
            let n = workers.min(MAX_WORKERS);
            LIVE.store(n, Ordering::Release);
            for w in 0..n {
                if let Ok(h) = std::thread::Builder::new()
                    .name(format!("sme-gemm-pool-{w}"))
                    .spawn(move || run_worker(w))
                {
                    handles.push(h);
                }
            }
        }
        drop(handles);
        Self(())
    }
}

impl Drop for HotPool {
    fn drop(&mut self) {
        let mut handles = HANDLES
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if REFS.fetch_sub(1, Ordering::AcqRel) == 1 {
            // Take the pool from any call using it, so no job is posted to a
            // worker that is leaving.
            while IN_USE.swap(true, Ordering::Acquire) {
                std::hint::spin_loop();
            }
            LIVE.store(0, Ordering::Release);
            IN_USE.store(false, Ordering::Release);
            for h in handles.drain(..) {
                drop(h.join());
            }
        }
    }
}

fn run_worker(w: usize) {
    crate::warm::set_user_interactive();
    let slot = &SLOTS[w];
    let mut last = Instant::now();
    let mut spins: u32 = 0;
    while LIVE.load(Ordering::Acquire) > w {
        // Pending whenever posted is ahead of done, so a job posted before this
        // thread got here is still taken.
        let p = slot.posted.load(Ordering::Acquire);
        if p == slot.done.load(Ordering::Relaxed) {
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(256) && last.elapsed() > IDLE_AFTER {
                std::thread::sleep(IDLE_SLEEP);
            } else {
                std::hint::spin_loop();
            }
            continue;
        }
        // SAFETY: the caller stored JOB before posting this sequence (Release,
        // read here with Acquire) and keeps the job alive until this slot's
        // `done` reaches it, which happens only after the drain below.
        let job = unsafe { &*JOB.load(Ordering::Acquire) };
        job.drain();
        slot.done.store(p, Ordering::Release);
        last = Instant::now();
    }
}

/// Whether a pool is alive (a call may still find it busy).
pub(crate) fn live() -> bool {
    LIVE.load(Ordering::Relaxed) > 0
}

/// A raw pointer the pool's items may share: each writes a disjoint range.
#[derive(Clone, Copy)]
pub(crate) struct Shared<T>(pub(crate) *mut T);
impl<T> Shared<T> {
    /// The pointer (a method, so closures capture the `Sync` wrapper).
    pub(crate) const fn ptr(self) -> *mut T {
        self.0
    }
}
// SAFETY: every user has its items write disjoint ranges of the one buffer,
// and the pool returns only after all of them finish.
unsafe impl<T> Sync for Shared<T> {}
// SAFETY: as for Sync.
unsafe impl<T> Send for Shared<T> {}

/// A job posted by [`alongside`], as the calling thread sees it meanwhile.
pub(crate) struct Posted<'a>(&'a Job<'a>);

impl Posted<'_> {
    /// Runs on the calling thread every item no worker has claimed yet (a
    /// worker may be asleep after an idle spell), returning once those are done.
    pub(crate) fn drain(&self) {
        if self.0.next.load(Ordering::Relaxed) < self.0.items {
            self.0.drain();
        }
    }
}

/// The end of an [`alongside`] job: runs whatever no worker claimed, waits for
/// the engaged workers, and releases the pool. A drop guard, so it also runs
/// when the caller's work unwinds (the job lives on the caller's frame).
struct Finish<'a, 'j> {
    job: &'a Job<'j>,
    engaged: &'a [Slot],
    seq: u64,
}

impl Drop for Finish<'_, '_> {
    fn drop(&mut self) {
        // A load first: the line is usually a worker's, and taking it for a
        // read-modify-write when every item is claimed costs more.
        if self.job.next.load(Ordering::Relaxed) < self.job.items {
            self.job.drain();
        }
        for s in self.engaged {
            while s.done.load(Ordering::Acquire) != self.seq {
                std::hint::spin_loop();
            }
        }
        JOB.store(core::ptr::null_mut(), Ordering::Relaxed);
        IN_USE.store(false, Ordering::Release);
    }
}

/// Posts `f(i)` for every `i < items` to the pool's workers and runs `main` on
/// the calling thread meanwhile -- typically SME work that the items feed or
/// follow, synchronized through atomics of the caller's own. Returns `main`'s
/// result once every item is done (draining any still unclaimed); `None`,
/// running nothing, when no pool is alive or another call holds it.
///
/// # Panics
/// Re-raises (as a new panic) if any `f(i)` panicked, after every worker has
/// finished.
pub(crate) fn alongside<R>(
    items: usize,
    f: &(dyn Fn(usize) + Sync),
    main: impl FnOnce(&Posted<'_>) -> R,
) -> Option<R> {
    if items == 0 || LIVE.load(Ordering::Relaxed) == 0 || IN_USE.swap(true, Ordering::Acquire) {
        return None;
    }
    let live = LIVE.load(Ordering::Acquire);
    if live == 0 {
        IN_USE.store(false, Ordering::Release);
        return None;
    }
    let job = Job {
        f,
        items,
        next: AtomicUsize::new(0),
        panicked: AtomicBool::new(false),
    };
    JOB.store(
        (&raw const job).cast::<Job<'static>>().cast_mut(),
        Ordering::Release,
    );
    let seq = SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    let engaged = &SLOTS[..live.min(items)];
    for s in engaged {
        s.posted.store(seq, Ordering::Release);
    }
    // Wait for the workers even if `main` unwinds: the job lives on this frame.
    let finish = Finish {
        job: &job,
        engaged,
        seq,
    };
    let r = main(&Posted(&job));
    drop(finish);
    assert!(
        !job.panicked.load(Ordering::Relaxed),
        "a HotPool task panicked"
    );
    Some(r)
}

/// Runs `f(i)` for every `i < items` on the calling thread and the pool's
/// workers. Returns `false` without running anything when no pool is alive,
/// there is a single item, or another call holds the pool, so the caller can
/// fall back to doing the work itself.
///
/// # Panics
/// Re-raises (as a new panic) if any `f(i)` panicked, after every worker has
/// finished.
pub(crate) fn parallel(items: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
    if items < 2 || LIVE.load(Ordering::Relaxed) == 0 || IN_USE.swap(true, Ordering::Acquire) {
        return false;
    }
    let live = LIVE.load(Ordering::Acquire);
    if live == 0 {
        IN_USE.store(false, Ordering::Release);
        return false;
    }
    let job = Job {
        f,
        items,
        next: AtomicUsize::new(0),
        panicked: AtomicBool::new(false),
    };
    JOB.store(
        (&raw const job).cast::<Job<'static>>().cast_mut(),
        Ordering::Release,
    );
    let seq = SEQ.fetch_add(1, Ordering::Relaxed) + 1;
    let engaged = &SLOTS[..live.min(items - 1)];
    for s in engaged {
        s.posted.store(seq, Ordering::Release);
    }
    job.drain();
    for s in engaged {
        while s.done.load(Ordering::Acquire) != seq {
            std::hint::spin_loop();
        }
    }
    JOB.store(core::ptr::null_mut(), Ordering::Relaxed);
    IN_USE.store(false, Ordering::Release);
    assert!(
        !job.panicked.load(Ordering::Relaxed),
        "a HotPool task panicked"
    );
    true
}
