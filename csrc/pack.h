// Operand packing for the MOPA kernels, one template over the MOPA atom A
// (Mopa<T, Acc> in sme_tile.h). A band is A::kTile lanes -- rows of A or columns
// of B -- by k depths, stored as ceil(k / A::kGroup) depth groups of A::kRow
// elements, one 64-byte vector each:
//
//   band[p * kRow + kGroup * i + s] = src[i, kGroup * p + s],
//
// zero past the operand's lane and depth edges. One lane's group is a unit of
// kGroup * sizeof(T) bytes (2, 4 or 8), so a lane-contiguous-in-depth operand
// (row-major A, column-major B) packs by transposing units, 16 / unit lanes at a
// time, four lane groups to a depth group; a lanes-contiguous operand (row-major
// B, column-major A) packs by copying each depth row (kGroup 1) or interleaving
// kGroup of them. Packing moves bit patterns: only the sizes matter.
#ifndef SME_GEMM_PACK_H
#define SME_GEMM_PACK_H

#include <arm_neon.h>
#include <stddef.h>
#include <string.h>

#include "epilogue.h"
#include "sme_tile.h"

// N x N transpose of U-byte units, N = 16 / U: r[t] becomes unit t of each of
// the N input vectors. A vtrn ladder, log2(N) stages of trn at doubling width.
template <size_t U> static inline void pack_transpose(uint8x16_t *r) {
    if constexpr (U == 2) {
        uint16x8_t t[8];
        for (int i = 0; i < 8; i++)
            t[i] = vreinterpretq_u16_u8(r[i]);
        uint16x8x2_t t01 = vtrnq_u16(t[0], t[1]), t23 = vtrnq_u16(t[2], t[3]);
        uint16x8x2_t t45 = vtrnq_u16(t[4], t[5]), t67 = vtrnq_u16(t[6], t[7]);
        uint32x4x2_t q02 =
            vtrnq_u32(vreinterpretq_u32_u16(t01.val[0]), vreinterpretq_u32_u16(t23.val[0]));
        uint32x4x2_t q13 =
            vtrnq_u32(vreinterpretq_u32_u16(t01.val[1]), vreinterpretq_u32_u16(t23.val[1]));
        uint32x4x2_t q46 =
            vtrnq_u32(vreinterpretq_u32_u16(t45.val[0]), vreinterpretq_u32_u16(t67.val[0]));
        uint32x4x2_t q57 =
            vtrnq_u32(vreinterpretq_u32_u16(t45.val[1]), vreinterpretq_u32_u16(t67.val[1]));
        uint32x4_t lo[4] = {q02.val[0], q13.val[0], q02.val[1], q13.val[1]};
        uint32x4_t hi[4] = {q46.val[0], q57.val[0], q46.val[1], q57.val[1]};
        for (int i = 0; i < 4; i++) {
            uint64x2_t a = vreinterpretq_u64_u32(lo[i]), b = vreinterpretq_u64_u32(hi[i]);
            r[i] = vreinterpretq_u8_u64(vtrn1q_u64(a, b));
            r[i + 4] = vreinterpretq_u8_u64(vtrn2q_u64(a, b));
        }
    } else if constexpr (U == 4) {
        uint32x4x2_t t01 = vtrnq_u32(vreinterpretq_u32_u8(r[0]), vreinterpretq_u32_u8(r[1]));
        uint32x4x2_t t23 = vtrnq_u32(vreinterpretq_u32_u8(r[2]), vreinterpretq_u32_u8(r[3]));
        r[0] =
            vreinterpretq_u8_u32(vcombine_u32(vget_low_u32(t01.val[0]), vget_low_u32(t23.val[0])));
        r[1] =
            vreinterpretq_u8_u32(vcombine_u32(vget_low_u32(t01.val[1]), vget_low_u32(t23.val[1])));
        r[2] = vreinterpretq_u8_u32(
            vcombine_u32(vget_high_u32(t01.val[0]), vget_high_u32(t23.val[0])));
        r[3] = vreinterpretq_u8_u32(
            vcombine_u32(vget_high_u32(t01.val[1]), vget_high_u32(t23.val[1])));
    } else {
        static_assert(U == 8);
        uint64x2_t a = vreinterpretq_u64_u8(r[0]), b = vreinterpretq_u64_u8(r[1]);
        r[0] = vreinterpretq_u8_u64(vtrn1q_u64(a, b));
        r[1] = vreinterpretq_u8_u64(vtrn2q_u64(a, b));
    }
}

// One depth group of a full band from kGroup depth rows of contiguous lanes
// (row s at row + s * depth_stride): interleave them lane by lane.
template <class A>
static inline void pack_interleave(typename A::T *dst, const typename A::T *row,
                                   long depth_stride) {
    using T = typename A::T;
    constexpr size_t G = A::kGroup, C = 16 / sizeof(T); // lanes per 16-byte chunk
    long ds = depth_stride;
    for (size_t c = 0; c < A::kTile; c += C) {
        if constexpr (sizeof(T) == 2 && G == 2) {
            const uint16_t *s = (const uint16_t *)(row + c);
            uint16x8x2_t v = {{vld1q_u16(s), vld1q_u16(s + ds)}};
            vst2q_u16((uint16_t *)(dst + G * c), v);
        } else if constexpr (sizeof(T) == 2 && G == 4) {
            const uint16_t *s = (const uint16_t *)(row + c);
            uint16x8x4_t v = {
                {vld1q_u16(s), vld1q_u16(s + ds), vld1q_u16(s + 2 * ds), vld1q_u16(s + 3 * ds)}};
            vst4q_u16((uint16_t *)(dst + G * c), v);
        } else {
            static_assert(sizeof(T) == 1 && G == 4);
            const uint8_t *s = (const uint8_t *)(row + c);
            uint8x16x4_t v = {
                {vld1q_u8(s), vld1q_u8(s + ds), vld1q_u8(s + 2 * ds), vld1q_u8(s + 3 * ds)}};
            vst4q_u8((uint8_t *)(dst + G * c), v);
        }
    }
}

// Depth groups [p0, groups) of one band, element by element, zero past `valid`
// lanes and k depths: the edges and the strided-both-ways layouts.
template <class A>
static void pack_scalar(typename A::T *dst, const typename A::T *src, long lane_stride,
                        long depth_stride, size_t k, size_t valid, size_t p0) {
    constexpr size_t L = A::kTile, G = A::kGroup, R = A::kRow;
    for (size_t p = p0; p < (k + G - 1) / G; p++)
        for (size_t i = 0; i < L; i++)
            for (size_t s = 0; s < G; s++) {
                size_t d = G * p + s;
                dst[p * R + G * i + s] = i < valid && d < k
                                             ? src[(long)i * lane_stride + (long)d * depth_stride]
                                             : (typename A::T)0;
            }
}

// One whole depth group of a band whose lanes are contiguous, from kGroup rows
// (row s at row + s*depth_stride): all kTile lanes, or lanes [0, valid) and
// zero past them at the edge.
template <class A>
static inline void pack_lanes_full(typename A::T *out, const typename A::T *row,
                                   long depth_stride) {
    if constexpr (A::kGroup == 1)
        memcpy(out, row, A::kRow * sizeof(typename A::T));
    else
        pack_interleave<A>(out, row, depth_stride);
}
template <class A>
static void pack_lanes_edge(typename A::T *out, const typename A::T *row, long depth_stride,
                            size_t valid) {
    // Zero the whole group (a fixed 64 bytes, so no library call), then the lanes.
    memset(out, 0, A::kRow * sizeof(typename A::T));
    for (size_t i = 0; i < valid; i++)
        for (size_t g = 0; g < A::kGroup; g++)
            out[A::kGroup * i + g] = row[(long)i + (long)g * depth_stride];
}

// The whole depth groups [0, full) of a band whose lanes are each contiguous in
// depth. DEPTH-OUTER: the four lane groups of a depth group fill its 64-byte
// line back to back, so the panel is walked once. Lanes in whole N-lane groups
// transpose, two blocks a step: each lane's 32 contiguous bytes load together
// and the two vtrn ladders overlap (one alone runs latency-bound over its
// strided loads). The rest copy unit by unit. Out of line: the transposes'
// strided addresses are set up once per band here, not in every band's
// prologue (a 64x64 pack ran 1.16-1.36x with them inlined).
template <class A>
__attribute__((noinline)) static void pack_depth_groups(typename A::T *dst,
                                                        const typename A::T *src, long lane_stride,
                                                        size_t full, size_t valid) {
    using T = typename A::T;
    constexpr size_t L = A::kTile, G = A::kGroup, R = A::kRow, U = G * sizeof(T), N = 16 / U;
    if (valid < L) memset(dst, 0, full * R * sizeof(T)); // an edge band: lanes past valid
    size_t vt = valid & ~(N - 1), pt = full & ~(N - 1), p0 = 0;
    for (; p0 + 2 * N <= pt; p0 += 2 * N)
        for (size_t rg = 0; rg < vt; rg += N) {
            uint8x16_t r[N], q[N];
            for (size_t j = 0; j < N; j++) {
                const uint8_t *l = (const uint8_t *)(src + (long)(rg + j) * lane_stride + p0 * G);
                r[j] = vld1q_u8(l);
                q[j] = vld1q_u8(l + 16);
            }
            pack_transpose<U>(r);
            pack_transpose<U>(q);
            for (size_t t = 0; t < N; t++) {
                vst1q_u8((uint8_t *)(dst + (p0 + t) * R + rg * G), r[t]);
                vst1q_u8((uint8_t *)(dst + (p0 + N + t) * R + rg * G), q[t]);
            }
        }
    for (; p0 < pt; p0 += N)
        for (size_t rg = 0; rg < vt; rg += N) {
            uint8x16_t r[N];
            for (size_t j = 0; j < N; j++)
                r[j] = vld1q_u8((const uint8_t *)(src + (long)(rg + j) * lane_stride + p0 * G));
            pack_transpose<U>(r);
            for (size_t t = 0; t < N; t++)
                vst1q_u8((uint8_t *)(dst + (p0 + t) * R + rg * G), r[t]);
        }
    // Lanes [i0, valid) of group p unit by unit.
    auto rest = [&](size_t p, size_t i0) {
        T *o = dst + p * R;
        for (size_t i = i0; i < valid; i++)
            memcpy(o + G * i, src + (long)i * lane_stride + p * G, U);
    };
    if (vt < valid) // valid lanes past the transposed ones, in the transposed groups
        for (size_t p = 0; p < pt; p++)
            rest(p, vt);
    for (size_t p = pt; p < full; p++) // the groups past them, every lane
        rest(p, 0);
}

// One band: lanes [0, valid) of src, lane i at src + i*lane_stride, depth d at
// + d*depth_stride.
template <class A>
static void pack_band(typename A::T *dst, const typename A::T *src, long lane_stride,
                      long depth_stride, size_t k, size_t valid) {
    constexpr size_t L = A::kTile, G = A::kGroup, R = A::kRow;
    size_t groups = (k + G - 1) / G, full = k / G; // all / with all kGroup depths
    if (valid == 0) {                              // a band past the edge
        memset(dst, 0, groups * R * sizeof(typename A::T));
        return;
    }
    if (lane_stride == 1 && valid == L) {
        for (size_t p = 0; p < full; p++)
            pack_lanes_full<A>(dst + p * R, src + (long)(p * G) * depth_stride, depth_stride);
    } else if (lane_stride == 1) {
        for (size_t p = 0; p < full; p++)
            pack_lanes_edge<A>(dst + p * R, src + (long)(p * G) * depth_stride, depth_stride,
                               valid);
    } else if (depth_stride == 1) {
        pack_depth_groups<A>(dst, src, lane_stride, full, valid);
    } else {
        full = 0;
    }
    if (full < groups) pack_scalar<A>(dst, src, lane_stride, depth_stride, k, valid, full);
}

// Whole blocks of B full bands of a lanes-contiguous operand advancing through
// depth together, from band b0 on; returns the first band not packed. Band j of
// a block reads lanes j*kTile on of each source row.
template <class A, size_t B>
__attribute__((noinline)) static size_t
pack_band_blocks(typename A::T *dst, const typename A::T *src, long depth_stride, size_t lanes,
                 size_t k, size_t b0, size_t b1) {
    constexpr size_t L = A::kTile, G = A::kGroup, R = A::kRow;
    size_t per_band = ep_cmul((k + G - 1) / G, R), full = k / G, b = b0;
    for (; b + B <= b1 && (b + B) * L <= lanes; b += B) {
        const typename A::T *s = src + b * L;
        typename A::T *o = dst + (b - b0) * per_band;
        for (size_t p = 0; p < full; p++)
            for (size_t j = 0; j < B; j++)
                pack_lanes_full<A>(o + j * per_band + p * R,
                                   s + j * L + (long)(p * G) * depth_stride, depth_stride);
        if (full * G < k) // the partial last depth group
            for (size_t j = 0; j < B; j++)
                pack_scalar<A>(o + j * per_band, s + j * L, 1, depth_stride, k, L, full);
    }
    return b;
}

// Bands [b0, b1) of an operand with `lanes` lanes (band b is lanes [b*kTile,
// (b+1)*kTile)) into consecutive bands at dst, which is band b0. Bands past the
// edge zero-fill without reading src: the base is clamped, so no out-of-bounds
// pointer is even formed.
//
// With lanes contiguous, a band covers 64 / kGroup bytes of each source row, so
// kGroup bands share each row's 64-byte line and advance through depth
// together. Past an 8 KB row pitch the rows sit pages apart and four bands go
// together, one walk per page for four lines (G = 1: f32 4096^2 prepack 0.90x,
// f64 0.81x); below it, extra bands only add write streams (1024^2: 1.05x).
template <class A>
static void pack_bands(typename A::T *dst, const typename A::T *src, long lane_stride,
                       long depth_stride, size_t lanes, size_t k, size_t b0, size_t b1) {
    constexpr size_t L = A::kTile, G = A::kGroup, R = A::kRow;
    size_t per_band = ep_cmul((k + G - 1) / G, R), b = b0;
    if (lane_stride == 1) {
        if ((size_t)depth_stride * sizeof(typename A::T) > 8192)
            b = pack_band_blocks<A, 4>(dst, src, depth_stride, lanes, k, b0, b1);
        else if constexpr (G > 1)
            b = pack_band_blocks<A, G>(dst, src, depth_stride, lanes, k, b0, b1);
    }
    for (; b < b1; b++) {
        size_t l0 = b * L, v = l0 < lanes ? (lanes - l0 < L ? lanes - l0 : L) : 0;
        pack_band<A>(dst + (b - b0) * per_band, v ? src + (long)l0 * lane_stride : src, lane_stride,
                     depth_stride, k, v);
    }
}

#endif // SME_GEMM_PACK_H
