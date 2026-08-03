// NEON 8x8 u16 transpose (vtrn ladder), shared by every 16-bit pack path: a
// scalar transpose is ~8x slower and was the dominant pack cost. Pure NEON, no
// SME and no epilogue.h, so the standalone widening TUs can include it too.
#ifndef SME_GEMM_TRANSPOSE16_H
#define SME_GEMM_TRANSPOSE16_H

#include <arm_neon.h>
#include <stdint.h>

static inline void transpose_8x8_u16(uint16x8_t r[8]) {
    uint16x8x2_t t01 = vtrnq_u16(r[0], r[1]);
    uint16x8x2_t t23 = vtrnq_u16(r[2], r[3]);
    uint16x8x2_t t45 = vtrnq_u16(r[4], r[5]);
    uint16x8x2_t t67 = vtrnq_u16(r[6], r[7]);
    uint32x4x2_t q02 =
        vtrnq_u32(vreinterpretq_u32_u16(t01.val[0]), vreinterpretq_u32_u16(t23.val[0]));
    uint32x4x2_t q13 =
        vtrnq_u32(vreinterpretq_u32_u16(t01.val[1]), vreinterpretq_u32_u16(t23.val[1]));
    uint32x4x2_t q46 =
        vtrnq_u32(vreinterpretq_u32_u16(t45.val[0]), vreinterpretq_u32_u16(t67.val[0]));
    uint32x4x2_t q57 =
        vtrnq_u32(vreinterpretq_u32_u16(t45.val[1]), vreinterpretq_u32_u16(t67.val[1]));
    r[0] = vreinterpretq_u16_u64(
        vtrn1q_u64(vreinterpretq_u64_u32(q02.val[0]), vreinterpretq_u64_u32(q46.val[0])));
    r[4] = vreinterpretq_u16_u64(
        vtrn2q_u64(vreinterpretq_u64_u32(q02.val[0]), vreinterpretq_u64_u32(q46.val[0])));
    r[1] = vreinterpretq_u16_u64(
        vtrn1q_u64(vreinterpretq_u64_u32(q13.val[0]), vreinterpretq_u64_u32(q57.val[0])));
    r[5] = vreinterpretq_u16_u64(
        vtrn2q_u64(vreinterpretq_u64_u32(q13.val[0]), vreinterpretq_u64_u32(q57.val[0])));
    r[2] = vreinterpretq_u16_u64(
        vtrn1q_u64(vreinterpretq_u64_u32(q02.val[1]), vreinterpretq_u64_u32(q46.val[1])));
    r[6] = vreinterpretq_u16_u64(
        vtrn2q_u64(vreinterpretq_u64_u32(q02.val[1]), vreinterpretq_u64_u32(q46.val[1])));
    r[3] = vreinterpretq_u16_u64(
        vtrn1q_u64(vreinterpretq_u64_u32(q13.val[1]), vreinterpretq_u64_u32(q57.val[1])));
    r[7] = vreinterpretq_u16_u64(
        vtrn2q_u64(vreinterpretq_u64_u32(q13.val[1]), vreinterpretq_u64_u32(q57.val[1])));
}

#endif // SME_GEMM_TRANSPOSE16_H
