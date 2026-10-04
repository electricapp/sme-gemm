// Panel ring: B is built block by block into a small ring of slots while the
// GEMM consumes earlier blocks, so building B overlaps the multiply instead of
// preceding it on one thread. Workers pull from two cursors -- build units (ready
// once their slot's previous block is consumed) and GEMM items (ready once their
// block is built) -- and take whichever is ready, so no worker sits behind one
// slow (E-core) item. Included by each GEMM TU; the slot layout is the caller's.
#ifndef SME_GEMM_PANEL_RING_H
#define SME_GEMM_PANEL_RING_H

#include <dispatch/dispatch.h>
#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>

#define RING_SLOTS 3  // ring depth
#define RING_ITEMS 32 // min GEMM items per block
#define RING_AHEAD 2  // blocks built ahead of the GEMM cursor before GEMM takes priority

// Per-thread reusable pool (counters, then panels); a malloc per call shows up
// directly in latency. Grows monotonically; only the calling thread resizes it.
static _Thread_local void *g_ring_pool = NULL;
static _Thread_local size_t g_ring_pool_cap = 0;

// `n_ctr` zeroed counters followed, 64-byte aligned, by `panel_bytes` of panels.
// NULL on overflow or allocation failure.
__attribute__((unused)) static void *ring_pool(size_t n_ctr, size_t panel_bytes,
                                               _Atomic size_t **ctr) {
    if (n_ctr > SIZE_MAX / sizeof(_Atomic size_t)) return NULL;
    size_t ctr_pad = (n_ctr * sizeof(_Atomic size_t) + 63) & ~(size_t)63;
    if (panel_bytes > SIZE_MAX - ctr_pad) return NULL;
    size_t bytes = ctr_pad + panel_bytes;
    if (g_ring_pool_cap < bytes) {
        free(g_ring_pool);
        g_ring_pool = malloc(bytes);
        g_ring_pool_cap = g_ring_pool ? bytes : 0;
        if (!g_ring_pool) return NULL;
    }
    *ctr = (_Atomic size_t *)g_ring_pool;
    for (size_t i = 0; i < n_ctr; i++)
        atomic_init(&(*ctr)[i], 0);
    return (char *)g_ring_pool + ctr_pad;
}

// B built per call: per-worker column items (each re-reads all of A) or the
// shared ring. Columns win at small m, and while A is cache-sized with enough
// N-columns to balance; narrow N at large m wants the ring's M x N items
// (4096x512x512 f16: 0.89x of packed through columns, 0.98x through the ring).
static inline int use_cols(size_t m, size_t n, size_t a_bytes) {
    return m <= 512 || (m <= 4096 && (n + 31) / 32 >= 64 && a_bytes <= ((size_t)8 << 20));
}

// Runs n_blocks blocks of n_d build units and n_g GEMM items each; block b lives
// in slot b % RING_SLOTS. `ctr` holds 2*n_blocks + 2 zeroed counters.
__attribute__((unused)) static void ring_run(size_t n_blocks, size_t n_d, size_t n_g,
                                             _Atomic size_t *ctr,
                                             void (^build)(size_t blk, size_t unit),
                                             void (^gemm)(size_t blk, size_t item)) {
    _Atomic size_t *d_done = ctr, *g_done = d_done + n_blocks;
    _Atomic size_t *d_cur = g_done + n_blocks, *g_cur = d_cur + 1;
    size_t n_dt = n_blocks * n_d, n_gt = n_blocks * n_g;
    // Extra workers are harmless: a late one finds both cursors exhausted.
    dispatch_apply(
        n_gt < 64 ? n_gt : 64, dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^(size_t w) {
          (void)w;
          for (unsigned spins = 0;;) {
              size_t di = atomic_load_explicit(d_cur, memory_order_relaxed);
              size_t gi = atomic_load_explicit(g_cur, memory_order_relaxed);
              if (gi >= n_gt) return;
              size_t sd = di / n_d, sg = gi / n_g;
              int d_ok = di < n_dt &&
                         (sd < RING_SLOTS || atomic_load_explicit(&g_done[sd - RING_SLOTS],
                                                                  memory_order_acquire) == n_g);
              int g_ok = atomic_load_explicit(&d_done[sg], memory_order_acquire) == n_d;
              if (d_ok && (!g_ok || sd <= sg + RING_AHEAD)) {
                  if (!atomic_compare_exchange_weak_explicit(
                          d_cur, &di, di + 1, memory_order_relaxed, memory_order_relaxed))
                      continue;
                  build(sd, di % n_d);
                  atomic_fetch_add_explicit(&d_done[sd], 1, memory_order_release);
                  spins = 0;
                  continue;
              }
              if (g_ok) {
                  if (!atomic_compare_exchange_weak_explicit(
                          g_cur, &gi, gi + 1, memory_order_relaxed, memory_order_relaxed))
                      continue;
                  gemm(sg, gi % n_g);
                  atomic_fetch_add_explicit(&g_done[sg], 1, memory_order_release);
                  spins = 0;
                  continue;
              }
              if (++spins & 1023)
                  __builtin_arm_yield();
              else
                  sched_yield();
          }
        });
}

#endif
