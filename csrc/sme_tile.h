// Compile-time building blocks for the SME kernels: per-dtype MOPA atoms, the
// 2x2 super-tile K-loop, and walkers over a super-tile's ZA slices. Everything
// here is a template or always_inline, so a kernel built from it is still one
// streaming function: ZA tile numbers reach their instructions as template
// arguments (they must be immediates), and no helper becomes a call with ZA live.
//
// Kernels pass per-slice work to the walkers as lambdas, which must inline too:
//   [&] [[gnu::always_inline]] (auto zl, auto zh, uint32_t s, size_t r)
//       __arm_streaming __arm_in("za") { ... za_row<zl, float>(s) ... }
// zl/zh are Tile<Z> values, usable wherever a tile number is.
#ifndef SME_GEMM_SME_TILE_H
#define SME_GEMM_SME_TILE_H

#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <type_traits>

#define SME_INLINE __attribute__((always_inline)) inline

// A ZA tile number as a type, so it survives being passed to a lambda.
template <int Z> using Tile = std::integral_constant<int, Z>;

// The SVE vector of each element type, a broadcast into it, and its zero.
template <class T> struct Sve;
#define SME_SVE_TYPE(T, V, DUP)                                                                    \
    template <> struct Sve<T> {                                                                    \
        using Vec = V;                                                                             \
        static SME_INLINE Vec dup(T x) __arm_streaming { return DUP(x); }                          \
        static SME_INLINE Vec zero() __arm_streaming { return DUP((T)0); }                         \
    }
SME_SVE_TYPE(int8_t, svint8_t, svdup_n_s8);
SME_SVE_TYPE(int16_t, svint16_t, svdup_n_s16);
SME_SVE_TYPE(int32_t, svint32_t, svdup_n_s32);
SME_SVE_TYPE(int64_t, svint64_t, svdup_n_s64);
SME_SVE_TYPE(__fp16, svfloat16_t, svdup_n_f16);
SME_SVE_TYPE(__bf16, svbfloat16_t, svdup_n_bf16);
SME_SVE_TYPE(float, svfloat32_t, svdup_n_f32);
SME_SVE_TYPE(double, svfloat64_t, svdup_n_f64);
#undef SME_SVE_TYPE
template <class T> using Vec = typename Sve<T>::Vec;

// Predicates by element size in bytes.
template <size_t B> SME_INLINE svbool_t ptrue() __arm_streaming {
    if constexpr (B == 1)
        return svptrue_b8();
    else if constexpr (B == 2)
        return svptrue_b16();
    else if constexpr (B == 4)
        return svptrue_b32();
    else
        return svptrue_b64();
}
template <size_t B> SME_INLINE svcount_t ptrue_c() __arm_streaming {
    if constexpr (B == 1)
        return svptrue_c8();
    else if constexpr (B == 2)
        return svptrue_c16();
    else if constexpr (B == 4)
        return svptrue_c32();
    else
        return svptrue_c64();
}
template <size_t B> SME_INLINE svbool_t whilelt(uint64_t i, uint64_t n) __arm_streaming {
    if constexpr (B == 1)
        return svwhilelt_b8(i, n);
    else if constexpr (B == 2)
        return svwhilelt_b16(i, n);
    else if constexpr (B == 4)
        return svwhilelt_b32(i, n);
    else
        return svwhilelt_b64(i, n);
}

// One MOPA dtype: packed element T accumulating into Acc. At the 512-bit
// streaming vector length a vector holds kRow packed elements, a ZA tile is
// kTile x kTile accumulators, and one MOPA folds kGroup K values into each
// (2 for the widening 16-bit MOPAs, 4 for the integer ones).
template <class T_, class Acc_> struct Mopa {
    using T = T_;
    using Acc = Acc_;
    static constexpr size_t kRow = 64 / sizeof(T);
    static constexpr size_t kTile = 64 / sizeof(Acc);
    static constexpr size_t kGroup = sizeof(Acc) / sizeof(T);

    // ZA tile Z += a (rows, under pn) x b (columns, under pm).
    template <int Z>
    static SME_INLINE void mopa(svbool_t pn, svbool_t pm, Vec<T> a,
                                Vec<T> b) __arm_streaming __arm_inout("za") {
        if constexpr (sizeof(Acc) == 2)
            svmopa_za16_m(Z, pn, pm, a, b);
        else if constexpr (sizeof(Acc) == 4)
            svmopa_za32_m(Z, pn, pm, a, b);
        else
            svmopa_za64_m(Z, pn, pm, a, b);
    }
    template <int Z>
    static SME_INLINE void mopa(svbool_t p, Vec<T> a, Vec<T> b) __arm_streaming __arm_inout("za") {
        mopa<Z>(p, p, a, b);
    }
};

// ZA slices by accumulator type: a row (horizontal slice) or column (vertical)
// of tile Z, read into a vector or stored straight to memory.
template <int Z, class Acc> SME_INLINE Vec<Acc> za_row(uint32_t s) __arm_streaming __arm_in("za") {
    svbool_t p = ptrue<sizeof(Acc)>();
    if constexpr (sizeof(Acc) == 4)
        return svread_hor_za32_m(Sve<Acc>::zero(), p, Z, s);
    else
        return svread_hor_za64_m(Sve<Acc>::zero(), p, Z, s);
}
template <int Z, class Acc> SME_INLINE Vec<Acc> za_col(uint32_t s) __arm_streaming __arm_in("za") {
    svbool_t p = ptrue<sizeof(Acc)>();
    if constexpr (sizeof(Acc) == 4)
        return svread_ver_za32_m(Sve<Acc>::zero(), p, Z, s);
    else
        return svread_ver_za64_m(Sve<Acc>::zero(), p, Z, s);
}
template <int Z, class Acc>
SME_INLINE void za_st_row(uint32_t s, svbool_t p, Acc *x) __arm_streaming __arm_in("za") {
    if constexpr (sizeof(Acc) == 4)
        svst1_hor_za32(Z, s, p, x);
    else
        svst1_hor_za64(Z, s, p, x);
}
template <int Z, class Acc>
SME_INLINE void za_st_col(uint32_t s, svbool_t p, Acc *x) __arm_streaming __arm_in("za") {
    if constexpr (sizeof(Acc) == 4)
        svst1_ver_za32(Z, s, p, x);
    else
        svst1_ver_za64(Z, s, p, x);
}
// Row s of tiles ZL / ZH as the lo / hi halves of one output row, and column s
// as those of one output column: the slice readers of the tile epilogue
// (epilogue_tile.h).
template <int ZL, int ZH, class Acc> struct ZaRowPair {
    SME_INLINE auto operator()(uint32_t s) const __arm_streaming __arm_in("za") {
        return svcreate2(za_row<ZL, Acc>(s), za_row<ZH, Acc>(s));
    }
};
template <int ZL, int ZH, class Acc> struct ZaColPair {
    SME_INLINE auto operator()(uint32_t s) const __arm_streaming __arm_in("za") {
        return svcreate2(za_col<ZL, Acc>(s), za_col<ZH, Acc>(s));
    }
};
template <int Z, class Acc>
SME_INLINE void za_ld_row(uint32_t s, svbool_t p, const Acc *x) __arm_streaming __arm_inout("za") {
    if constexpr (sizeof(Acc) == 4)
        svld1_hor_za32(Z, s, p, x);
    else
        svld1_hor_za64(Z, s, p, x);
}

// --- the 2x2 super-tile ------------------------------------------------------
//
// ZA tile 2*i + j accumulates A band i against B band j, so a super-tile is
// 2*kTile square. HiM / HiN say whether the hi-M / hi-N band holds data: when
// one is all pad (a narrow edge) its two tiles are dead and their MOPAs are
// dropped at compile time. Stores read ZA through the mr/nc predicates, so they
// never see the dead tiles.
template <class A, bool HiM, bool HiN>
SME_INLINE void quad_mopa(svbool_t p, Vec<typename A::T> al, Vec<typename A::T> ah,
                          Vec<typename A::T> bl,
                          Vec<typename A::T> bh) __arm_streaming __arm_inout("za") {
    A::template mopa<0>(p, al, bl);
    if constexpr (HiN) A::template mopa<1>(p, al, bh);
    if constexpr (HiM) A::template mopa<2>(p, ah, bl);
    if constexpr (HiM && HiN) A::template mopa<3>(p, ah, bh);
}

// U consecutive depth vectors of a packed band in one load (U = 1, 2 or 4),
// and the I-th of them.
template <int U, class T> SME_INLINE auto ld_depths(const T *x) __arm_streaming {
    if constexpr (U == 1)
        return svld1(ptrue<sizeof(T)>(), x);
    else if constexpr (U == 2)
        return svld1_x2(ptrue_c<sizeof(T)>(), x);
    else
        return svld1_x4(ptrue_c<sizeof(T)>(), x);
}
template <int U, int I, class W> SME_INLINE auto depth(W w) __arm_streaming {
    if constexpr (U == 1)
        return w;
    else if constexpr (U == 2)
        return svget2(w, I);
    else
        return svget4(w, I);
}

// Accumulate depth groups [d, kd) of one super-tile from packed bands (kRow
// elements per group), U groups per load, and the remainder at U/2, ..., 1.
template <class A, bool HiM, bool HiN, int U, class T>
SME_INLINE void quad_kloop(size_t &d, size_t kd, const T *al, const T *ah, const T *bl,
                           const T *bh) __arm_streaming __arm_inout("za") {
    constexpr size_t R = A::kRow;
    svbool_t p = ptrue<sizeof(T)>();
    for (; d + U <= kd; d += U) {
        auto AL = ld_depths<U>(al + d * R);
        auto AH = HiM ? ld_depths<U>(ah + d * R) : AL;
        auto BL = ld_depths<U>(bl + d * R);
        auto BH = HiN ? ld_depths<U>(bh + d * R) : BL;
        quad_mopa<A, HiM, HiN>(p, depth<U, 0>(AL), depth<U, 0>(AH), depth<U, 0>(BL),
                               depth<U, 0>(BH));
        if constexpr (U > 1)
            quad_mopa<A, HiM, HiN>(p, depth<U, 1>(AL), depth<U, 1>(AH), depth<U, 1>(BL),
                                   depth<U, 1>(BH));
        if constexpr (U > 2) {
            quad_mopa<A, HiM, HiN>(p, depth<U, 2>(AL), depth<U, 2>(AH), depth<U, 2>(BL),
                                   depth<U, 2>(BH));
            quad_mopa<A, HiM, HiN>(p, depth<U, 3>(AL), depth<U, 3>(AH), depth<U, 3>(BL),
                                   depth<U, 3>(BH));
        }
    }
    if constexpr (U > 1) quad_kloop<A, HiM, HiN, U / 2>(d, kd, al, ah, bl, bh);
}

// Depth groups per load: Full for a whole super-tile, Narrow for one with a
// dead band (fewer live loads, so less to amortize).
template <int Full, int Narrow = Full> struct KSteps {
    static constexpr int kFull = Full, kNarrow = Narrow;
};

// One super-tile across all kd depth groups, from packed bands. The live-band
// dispatch happens once here, so each K-loop stays branch-free.
template <class A, class K, class T>
SME_INLINE void quad_accumulate(size_t kd, const T *al, const T *ah, const T *bl, const T *bh,
                                bool hi_m, bool hi_n) __arm_streaming __arm_inout("za") {
    size_t d = 0;
    if (!hi_n && !hi_m)
        quad_kloop<A, false, false, K::kNarrow>(d, kd, al, ah, bl, bh);
    else if (!hi_n)
        quad_kloop<A, true, false, K::kNarrow>(d, kd, al, ah, bl, bh);
    else if (!hi_m)
        quad_kloop<A, false, true, K::kNarrow>(d, kd, al, ah, bl, bh);
    else
        quad_kloop<A, true, true, K::kFull>(d, kd, al, ah, bl, bh);
}

// The same with B read in place from row-major B (one K value per MOPA only):
// depth d of B's lo / hi band is bl + d*b_rs / bh + d*b_rs under pbl / pbh, so
// pad columns load as zero. U depths per iteration, then the remainder.
template <class A, bool HiM, bool HiN, int U, class T>
SME_INLINE void quad_kloop_rowb(size_t &d, size_t kd, const T *al, const T *ah, const T *bl,
                                const T *bh, long b_rs, svbool_t pbl,
                                svbool_t pbh) __arm_streaming __arm_inout("za") {
    static_assert(A::kGroup == 1, "B rows are only MOPA operands without a K interleave");
    constexpr size_t R = A::kRow;
    svbool_t p = ptrue<sizeof(T)>();
    for (; d + U <= kd; d += U) {
        Vec<T> al0 = svld1(p, al + d * R), ah0 = HiM ? svld1(p, ah + d * R) : al0;
        Vec<T> bl0 = svld1(pbl, bl + (long)d * b_rs),
               bh0 = HiN ? svld1(pbh, bh + (long)d * b_rs) : bl0;
        if constexpr (U == 1) {
            quad_mopa<A, HiM, HiN>(p, al0, ah0, bl0, bh0);
        } else {
            size_t e = d + 1;
            Vec<T> al1 = svld1(p, al + e * R), ah1 = HiM ? svld1(p, ah + e * R) : al1;
            Vec<T> bl1 = svld1(pbl, bl + (long)e * b_rs);
            Vec<T> bh1 = HiN ? svld1(pbh, bh + (long)e * b_rs) : bl1;
            quad_mopa<A, HiM, HiN>(p, al0, ah0, bl0, bh0);
            quad_mopa<A, HiM, HiN>(p, al1, ah1, bl1, bh1);
        }
    }
    if constexpr (U > 1) quad_kloop_rowb<A, HiM, HiN, 1>(d, kd, al, ah, bl, bh, b_rs, pbl, pbh);
}
template <class A, int U, class T>
SME_INLINE void quad_accumulate_rowb(size_t kd, const T *al, const T *ah, const T *bl, const T *bh,
                                     long b_rs, svbool_t pbl, svbool_t pbh, bool hi_m,
                                     bool hi_n) __arm_streaming __arm_inout("za") {
    size_t d = 0;
    if (!hi_n && !hi_m)
        quad_kloop_rowb<A, false, false, U>(d, kd, al, ah, bl, bh, b_rs, pbl, pbh);
    else if (!hi_n)
        quad_kloop_rowb<A, true, false, U>(d, kd, al, ah, bl, bh, b_rs, pbl, pbh);
    else if (!hi_m)
        quad_kloop_rowb<A, false, true, U>(d, kd, al, ah, bl, bh, b_rs, pbl, pbh);
    else
        quad_kloop_rowb<A, true, true, U>(d, kd, al, ah, bl, bh, b_rs, pbl, pbh);
}

// L2 blocking (BLIS jc -> ic) in super-tiles: how many fit in `bytes` of A-block
// plus B-block, when a super-tile is two packed bands of 64 bytes per depth group.
// A band-sized unit would double the working set. Never below 2.
static inline size_t quad_budget(size_t kd, size_t bytes) {
    size_t tile;
    if (__builtin_mul_overflow(kd, (size_t)128, &tile)) tile = SIZE_MAX;
    size_t b = bytes / (tile ? tile : 1);
    return b < 2 ? 2 : b;
}

// The N-block width for a budget: half of it, rounded down to an even split of
// n_tiles. The budget is a ceiling, not a target, and a ragged last block (64
// tiles at width 48 -> 48+16) amortizes the A-block re-read over too little N.
static inline size_t quad_nc_blk(size_t budget, size_t n_tiles) {
    size_t nc = budget / 2;
    if (nc < 1) nc = 1;
    if (nc >= n_tiles) return n_tiles;
    size_t blocks = (n_tiles + nc - 1) / nc;
    return (n_tiles + blocks - 1) / blocks;
}

// Walk the live slices of a super-tile with T-row tiles. quad_rows visits output
// rows [0, mr): f(lo-N tile, hi-N tile, slice, row), rows below T in tiles 0/1
// and the rest in 2/3. quad_cols visits columns [0, nc): f(lo-M tile, hi-M tile,
// slice, col), columns below T in tiles 0/2 and the rest in 1/3.
template <size_t T, class F>
SME_INLINE void quad_rows(size_t mr, F &&f) __arm_streaming __arm_in("za") {
    for (size_t r = 0; r < (mr < T ? mr : T); r++)
        f(Tile<0>{}, Tile<1>{}, (uint32_t)r, r);
    for (size_t r = T; r < mr; r++)
        f(Tile<2>{}, Tile<3>{}, (uint32_t)(r - T), r);
}
template <size_t T, class F>
SME_INLINE void quad_cols(size_t nc, F &&f) __arm_streaming __arm_in("za") {
    for (size_t c = 0; c < (nc < T ? nc : T); c++)
        f(Tile<0>{}, Tile<2>{}, (uint32_t)c, c);
    for (size_t c = T; c < nc; c++)
        f(Tile<1>{}, Tile<3>{}, (uint32_t)(c - T), c);
}

// The whole super-tile out of ZA into a row-major [2T][2T] scratch, for the
// strided-C stores (no scatter store in streaming mode).
template <class Acc> SME_INLINE void quad_to_scratch(Acc *scratch) __arm_streaming __arm_in("za") {
    constexpr size_t T = 64 / sizeof(Acc);
    svbool_t p = ptrue<sizeof(Acc)>();
    for (uint32_t r = 0; r < T; r++) {
        za_st_row<0>(r, p, scratch + r * 2 * T);
        za_st_row<1>(r, p, scratch + r * 2 * T + T);
        za_st_row<2>(r, p, scratch + (T + r) * 2 * T);
        za_st_row<3>(r, p, scratch + (T + r) * 2 * T + T);
    }
}

// The raw store of a super-tile's live mr x nc corner at (m0, n0): straight from
// ZA to C in column slices for col-major C and row slices for row-major C, or
// through a scratch for any other strides. RowMajor skips the layout test. The
// hi band's base is clamped when the band is empty, so no pointer past C is
// formed.
template <bool RowMajor = false, class Acc>
SME_INLINE void quad_store_raw(Acc *dst, long dst_cs, long dst_rs, size_t m0, size_t n0, size_t mr,
                               size_t nc) __arm_streaming __arm_in("za") {
    constexpr size_t W = 64 / sizeof(Acc);
    size_t mr_lo = mr < W ? mr : W, mr_hi = mr > W ? mr - W : 0;
    size_t nc_lo = nc < W ? nc : W, nc_hi = nc > W ? nc - W : 0;
    if (!RowMajor && dst_rs == 1) {
        svbool_t plo = whilelt<sizeof(Acc)>(0, mr_lo), phi = whilelt<sizeof(Acc)>(0, mr_hi);
        size_t hw = mr_hi ? W : 0;
        quad_cols<W>(
            nc, [&] [[gnu::always_inline]] (auto zl, auto zh, uint32_t s,
                                            size_t c) __arm_streaming __arm_in("za") {
                Acc *col = dst + (long)(n0 + c) * dst_cs + (long)m0;
                za_st_col<zl>(s, plo, col);
                za_st_col<zh>(s, phi, col + hw);
            });
    } else if (RowMajor || dst_cs == 1) {
        svbool_t plo = whilelt<sizeof(Acc)>(0, nc_lo), phi = whilelt<sizeof(Acc)>(0, nc_hi);
        size_t hw = nc_hi ? W : 0;
        quad_rows<W>(
            mr, [&] [[gnu::always_inline]] (auto zl, auto zh, uint32_t s,
                                            size_t r) __arm_streaming __arm_in("za") {
                Acc *row = dst + (long)(m0 + r) * dst_rs + (long)n0;
                za_st_row<zl>(s, plo, row);
                za_st_row<zh>(s, phi, row + hw);
            });
    } else {
        Acc scratch[4 * W * W];
        quad_to_scratch(scratch);
        for (size_t r = 0; r < mr; r++)
            for (size_t c = 0; c < nc; c++)
                dst[(long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs] = scratch[r * 2 * W + c];
    }
}

// Per-thread reusable A-pack scratch (no per-call malloc/free on the serial
// small/medium paths), one per element type per TU. Grows monotonically, lives
// for the thread's lifetime.
template <class T> static T *apack_scratch(size_t bytes) {
    static thread_local T *buf = NULL;
    static thread_local size_t cap = 0;
    if (bytes == 0)
        bytes = 1; // k==0: return a real (tiny) buffer, not a
                   // history-dependent NULL that looks like OOM
    if (cap < bytes) {
        free(buf);
        buf = (T *)malloc(bytes);
        cap = buf ? bytes : 0;
    }
    return buf;
}

#endif // SME_GEMM_SME_TILE_H
