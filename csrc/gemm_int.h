// The integer MOPA kernels as one template over the element type: i8 x i8 -> i32
// (16x16 ZA32 tiles) and i16 x i16 -> i64 (8x8 ZA64 tiles), each SMOPA folding
// four K values into one accumulator, so panels are [ceil(K/4), 64 bytes] 4-way
// interleaved. The output is the raw accumulator or, through a dequant
// descriptor, f32 = nodes(scale * acc) computed in-register at the store.
// Included by gemm_i8i32.cc and gemm_i16i64.cc, which instantiate it under their
// own target features and keep the GEMV and dtype-specific paths. Packing is
// pack.h's, over the same atom.
#ifndef SME_GEMM_INT_H
#define SME_GEMM_INT_H

#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <type_traits>

#include "epilogue.h"
#include "pack.h"
#include "sme_tile.h"

// i8 accumulates into i32, i16 into i64.
template <class T> using IntMopa = Mopa<T, std::conditional_t<sizeof(T) == 1, int32_t, int64_t>>;

// Resident-bytes budget (A-block plus B-block) for run_streaming's cache
// blocking, and the N-block width the drivers' outer loop shares with it.
#define INT_CACHE_BUDGET_BYTES ((size_t)16 << 20)
static inline size_t int_nc_blk(size_t kp4, size_t n_tiles) {
    return quad_nc_blk(quad_budget(kp4, INT_CACHE_BUDGET_BYTES), n_tiles);
}

// An integer kernel's output: the raw accumulator, or f32 through a dequant
// descriptor. The kernels are instantiated per output type, so a raw call has
// no descriptor to forget and a dequant call no integer buffer to misread.
template <class Acc> struct RawOut {
    Acc *c;
};
struct DequantOut {
    float *c;
    const ep_dq_f32 *dq;
};
template <class Acc> static inline RawOut<Acc> out_at(RawOut<Acc> o, size_t i) {
    return {o.c + i};
}
static inline DequantOut out_at(DequantOut o, size_t i) {
    return {o.c + i, o.dq};
}

// Rows [r0, r1) of a dequantized super-tile, read from ZA tiles (ZL, ZH): i8's
// tiles are the 16-lane lo and hi halves of a 32-wide row, i16's two 8-lane
// tiles pack (uzp1) into one 16-lane row. Strided rows go through a buffer and
// out cell by cell (no scatter store in streaming mode). The values arrive as
// parameters, since the node walk may be an out-of-line call.
template <class A, int ZL, int ZH, bool Strided>
SME_INLINE void dq_rows(float *c, long dst_cs, long dst_rs, size_t m0, size_t n0, size_t r0,
                        size_t r1, size_t nc, svbool_t plo, svbool_t phi, svfloat32_t sclo,
                        svfloat32_t schi, const EpNode *nodes,
                        uint32_t n_nodes) __arm_streaming __arm_in("za") {
    using Acc = typename A::Acc;
    svbool_t p32 = svptrue_b32();
    for (size_t r = r0; r < r1; r++) {
        uint32_t s = (uint32_t)(r - r0);
        float row[32];
        float *rp = Strided ? row : c + (long)(m0 + r) * dst_rs + (long)n0;
        if constexpr (sizeof(Acc) == 4) {
            svfloat32_t lo = svmul_x(p32, svcvt_f32_x(p32, za_row<ZL, Acc>(s)), sclo);
            lo = ep_apply_nodes_f32(p32, plo, lo, nodes, n_nodes, 1, m0 + r, n0, m0);
            svst1(plo, rp, lo);
            if (nc > 16) {
                svfloat32_t hi = svmul_x(p32, svcvt_f32_x(p32, za_row<ZH, Acc>(s)), schi);
                hi = ep_apply_nodes_f32(p32, phi, hi, nodes, n_nodes, 1, m0 + r, n0 + 16, m0);
                svst1(phi, rp + 16, hi);
            }
        } else {
            // i64 -> f32 may lose precision past the 24-bit mantissa.
            svbool_t p64 = svptrue_b64();
            svfloat32_t lo = svcvt_f32_x(p64, za_row<ZL, Acc>(s));
            svfloat32_t hi = svcvt_f32_x(p64, za_row<ZH, Acc>(s));
            svfloat32_t v = svmul_x(p32, svuzp1(lo, hi), sclo);
            v = ep_apply_nodes_f32(p32, plo, v, nodes, n_nodes, 1, m0 + r, n0, m0);
            svst1(plo, rp, v);
        }
        if constexpr (Strided)
            for (size_t j = 0; j < nc; j++)
                c[(long)(m0 + r) * dst_rs + (long)(n0 + j) * dst_cs] = row[j];
    }
}

// The dequant store of one super-tile: scale (per-tensor splat or per-N
// vector), op-graph and f32 store, vectorized whatever C's layout.
template <class A, bool RowMajor>
SME_INLINE void dq_store(float *c, long dst_cs, long dst_rs, size_t m0, size_t n0, size_t mr,
                         size_t nc, const ep_dq_f32 *dq) __arm_streaming __arm_in("za") {
    constexpr size_t W = A::kTile;
    // f32 lanes of the row's first and second vector (i16 rows are one vector).
    size_t nlo = W == 16 && nc > 16 ? 16 : nc, nhi = W == 16 && nc > 16 ? nc - 16 : 0;
    svbool_t plo = svwhilelt_b32((uint64_t)0, (uint64_t)nlo);
    svbool_t phi = svwhilelt_b32((uint64_t)0, (uint64_t)nhi);
    svfloat32_t sclo = dq->scale_n ? svld1(plo, dq->scale_n + n0) : svdup_n_f32(dq->scale);
    svfloat32_t schi = dq->scale_n && nhi ? svld1(phi, dq->scale_n + n0 + 16) : sclo;
    size_t mr_lo = mr < W ? mr : W;
    const EpNode *nodes = dq->nodes;
    uint32_t n_nodes = dq->n_nodes;
    if (RowMajor || dst_cs == 1) {
        dq_rows<A, 0, 1, false>(c, dst_cs, dst_rs, m0, n0, 0, mr_lo, nc, plo, phi, sclo, schi,
                                nodes, n_nodes);
        dq_rows<A, 2, 3, false>(c, dst_cs, dst_rs, m0, n0, W, mr, nc, plo, phi, sclo, schi, nodes,
                                n_nodes);
    } else {
        dq_rows<A, 0, 1, true>(c, dst_cs, dst_rs, m0, n0, 0, mr_lo, nc, plo, phi, sclo, schi, nodes,
                               n_nodes);
        dq_rows<A, 2, 3, true>(c, dst_cs, dst_rs, m0, n0, W, mr, nc, plo, phi, sclo, schi, nodes,
                               n_nodes);
    }
}

template <class A, bool RowMajor = false>
SME_INLINE void int_store(RawOut<typename A::Acc> o, long dst_cs, long dst_rs, size_t m0, size_t n0,
                          size_t mr, size_t nc) __arm_streaming __arm_in("za") {
    quad_store_raw<RowMajor>(o.c, dst_cs, dst_rs, m0, n0, mr, nc);
}
template <class A, bool RowMajor = false>
SME_INLINE void int_store(DequantOut o, long dst_cs, long dst_rs, size_t m0, size_t n0, size_t mr,
                          size_t nc) __arm_streaming __arm_in("za") {
    dq_store<A, RowMajor>(o.c, dst_cs, dst_rs, m0, n0, mr, nc, o.dq);
}

// The packed kernel: [nt_lo, nt_hi) is the N-tile sub-range this invocation
// owns -- (0, n_tiles) for the whole problem, or a slice for the N-parallel
// paths -- and b_pack points at super-tile nt_lo's bands. Super-tiles are
// independent (zero, accumulate, store), so any chunk granularity is safe; the
// dequant store writes disjoint columns.
template <class T, class Out>
__arm_locally_streaming __arm_new("za") static void run_streaming(Out out, long dst_cs, long dst_rs,
                                                                  const T *a_pack, const T *b_pack,
                                                                  size_t m, size_t n, size_t mt_lo,
                                                                  size_t mt_hi, size_t nt_lo,
                                                                  size_t nt_hi, size_t kp4) {
    using A = IntMopa<T>;
    constexpr size_t W = A::kTile;
    size_t per_tile = ep_cmul(kp4, A::kRow);
    // Cache blocking (BLIS jc->ic) in super-tiles; see gemm_float.h.
    size_t budget = quad_budget(kp4, INT_CACHE_BUDGET_BYTES);
    size_t nc_blk = quad_nc_blk(budget, nt_hi - nt_lo);
    size_t mc_blk = budget / 2;
    if (mc_blk < 1) mc_blk = 1;
    if (mc_blk > mt_hi - mt_lo) mc_blk = mt_hi - mt_lo;

    for (size_t jc = nt_lo; jc < nt_hi; jc += nc_blk) {
        size_t jc_end = jc + nc_blk < nt_hi ? jc + nc_blk : nt_hi;
        for (size_t ic = mt_lo; ic < mt_hi; ic += mc_blk) {
            size_t ic_end = ic + mc_blk < mt_hi ? ic + mc_blk : mt_hi;
            for (size_t mt = ic; mt < ic_end; mt++) {
                const T *a_lo = a_pack + 2 * mt * per_tile, *a_hi = a_lo + per_tile;
                size_t m0 = mt * 2 * W, mr = m - m0 < 2 * W ? m - m0 : 2 * W;
                for (size_t nt = jc; nt < jc_end; nt++) {
                    const T *b_lo = b_pack + 2 * (nt - nt_lo) * per_tile, *b_hi = b_lo + per_tile;
                    size_t n0 = nt * 2 * W, nc = n - n0 < 2 * W ? n - n0 : 2 * W;
                    svzero_za();
                    quad_accumulate<A, KSteps<2>>(kp4, a_lo, a_hi, b_lo, b_hi, mr > W, nc > W);
                    int_store<A>(out, dst_cs, dst_rs, m0, n0, mr, nc);
                }
            }
        }
    }
}

// One row-major batch item, already in streaming mode, from its packed panels.
template <class T, class Out>
static void batch_one(Out out, const T *a_pack, const T *b_pack, size_t m, size_t n, size_t kp4,
                      size_t per_tile) __arm_streaming __arm_inout("za") {
    using A = IntMopa<T>;
    constexpr size_t W = A::kTile;
    size_t m_tiles = (m + 2 * W - 1) / (2 * W), n_tiles = (n + 2 * W - 1) / (2 * W);
    for (size_t mt = 0; mt < m_tiles; mt++) {
        const T *a_lo = a_pack + 2 * mt * per_tile, *a_hi = a_lo + per_tile;
        size_t m0 = mt * 2 * W, mr = m - m0 < 2 * W ? m - m0 : 2 * W;
        for (size_t nt = 0; nt < n_tiles; nt++) {
            const T *b_lo = b_pack + 2 * nt * per_tile, *b_hi = b_lo + per_tile;
            size_t n0 = nt * 2 * W, nc = n - n0 < 2 * W ? n - n0 : 2 * W;
            svzero_za();
            quad_accumulate<A, KSteps<2>>(kp4, a_lo, a_hi, b_lo, b_hi, mr > W, nc > W);
            int_store<A, true>(out, 1, (long)n, m0, n0, mr, nc);
        }
    }
}

template <class T, class Out>
__arm_locally_streaming __arm_new("za") static void run_batched(size_t count, Out out,
                                                                const T *a_packs, const T *b_packs,
                                                                size_t m, size_t n, size_t kp4,
                                                                size_t per_tile, size_t a_per,
                                                                size_t b_per) {
    for (size_t bi = 0; bi < count; bi++)
        batch_one(out_at(out, bi * m * n), a_packs + bi * a_per, b_packs + bi * b_per, m, n, kp4,
                  per_tile);
}

// `count` independent row-major C_i = A_i @ B_i of one shape in ONE streaming
// session, each item dequantized alike when `out` carries a descriptor. Items
// are packed up front, outside streaming mode.
template <class T, class Out>
static int run_batched_impl(size_t count, size_t m, size_t n, size_t k, Out out, const T *lhs,
                            const T *rhs) {
    using A = IntMopa<T>;
    constexpr size_t W = A::kTile;
    if (count == 0 || m == 0 || n == 0) return 0;
    size_t kp4 = (k + 3) / 4;
    size_t m_tiles = (m + 2 * W - 1) / (2 * W), n_tiles = (n + 2 * W - 1) / (2 * W);
    size_t per_tile = ep_cmul(kp4, A::kRow);
    size_t a_per = 2 * m_tiles * per_tile, b_per = 2 * n_tiles * per_tile;

    // pack_bands fully writes every band (zero-filling empty and partial bands
    // and partial p-groups), so the buffers need no zeroing. `count` is the one
    // factor no C-side size invariant bounds, so the byte products are checked
    // (-1 makes the Rust side fall back).
    size_t a_bytes, b_bytes;
    if (__builtin_mul_overflow(count, ep_cmul(a_per, sizeof(T)), &a_bytes) ||
        __builtin_mul_overflow(count, ep_cmul(b_per, sizeof(T)), &b_bytes))
        return -1;
    T *a_packs = (T *)malloc(a_bytes);
    T *b_packs = (T *)malloc(b_bytes);
    if (!a_packs || !b_packs) {
        free(a_packs);
        free(b_packs);
        return -1;
    }
    for (size_t bi = 0; bi < count; bi++) {
        const T *li = lhs + bi * (m * k), *ri = rhs + bi * (k * n);
        pack_bands<A>(a_packs + bi * a_per, li, (long)k, 1, m, k, 0, 2 * m_tiles);
        pack_bands<A>(b_packs + bi * b_per, ri, 1, (long)n, n, k, 0, 2 * n_tiles);
    }
    run_batched(count, out, a_packs, b_packs, m, n, kp4, per_tile, a_per, b_per);
    free(a_packs);
    free(b_packs);
    return 0;
}

#endif // SME_GEMM_INT_H
