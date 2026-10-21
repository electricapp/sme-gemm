// Causal attention over the f16 KV cache on the SME unit, for blocks of query
// rows (attn_kv_causal_sme_f16). Included at the end of attention.cc for its
// NEON exp and row passes.
//
// The same arithmetic as attn_kv_f16 -- q*scale rounded to f16, f32 scores from
// f16 products, f32 exp, probabilities rounded to f16, f32 sums of p*v -- with
// both products on the widening f16 MOPA (Mopa<__fp16, float>, sme_tile.h) and
// the softmax in NEON between them. Only the order of the sums differs.
//
// Each KV head's keys and values are packed once as MOPA bands. A unit is one KV
// head and a tile of R query positions times the head's g query heads (M = g*R
// rows, a 32-row M-tile when g divides 32), so the group's heads share every K
// and V load. Per chunk of ATTN_KC keys a unit computes its scores into an f32
// row buffer, takes the masked softmax with running max and sum (flash
// attention's rescale, exact when one chunk holds all the keys), and adds P@V
// into an f32 output it normalizes at the end. Units go to workers longest
// first, by a shared cursor.
#ifndef SME_GEMM_ATTENTION_KV_SME_H
#define SME_GEMM_ATTENTION_KV_SME_H

#include "pack.h"
#include "sme_tile.h"
#include <arm_sme.h>

#define ATTN_KC 1024 // keys per chunk (a multiple of 32)

typedef Mopa<__fp16, float> AttnMopa;

// s[r][c] = (q K^T)[r][c] for M rows against kc keys: super-tiles of 32 keys,
// stored row-major (row stride ld_s). qp holds M's bands, kp the chunk's first
// key band; a band is kd depth pairs.
__arm_locally_streaming __arm_new("za") static void attn_scores_sme(float *s, size_t ld_s,
                                                                    const __fp16 *qp, size_t m,
                                                                    const __fp16 *kp, size_t kc,
                                                                    size_t kd) {
    size_t band = kd * 32;
    for (size_t m0 = 0; m0 < m; m0 += 32) {
        const __fp16 *al = qp + (m0 / 16) * band, *ah = al + band;
        size_t mr = m - m0 < 32 ? m - m0 : 32;
        for (size_t n0 = 0; n0 < kc; n0 += 32) {
            const __fp16 *bl = kp + (n0 / 16) * band, *bh = bl + band;
            size_t nc = kc - n0 < 32 ? kc - n0 : 32;
            svzero_za();
            quad_accumulate<AttnMopa, KSteps<2>>(kd, al, ah, bl, bh, mr > 16, nc > 16);
            quad_store_raw<true>(s, 1, (long)ld_s, m0, n0, mr, nc);
        }
    }
}

// o[r][d] = (P V)[r][d] for M rows over kd depth pairs: pp holds P's bands, vp
// V's first band at the chunk's depth, its bands v_band apart.
__arm_locally_streaming __arm_new("za") static void attn_pv_sme(float *o, const __fp16 *pp,
                                                                size_t m, size_t kd,
                                                                const __fp16 *vp, size_t v_band,
                                                                size_t hd) {
    size_t band = kd * 32;
    for (size_t m0 = 0; m0 < m; m0 += 32) {
        const __fp16 *al = pp + (m0 / 16) * band, *ah = al + band;
        size_t mr = m - m0 < 32 ? m - m0 : 32;
        for (size_t d0 = 0; d0 < hd; d0 += 32) {
            const __fp16 *bl = vp + (d0 / 16) * v_band, *bh = bl + v_band;
            size_t nc = hd - d0 < 32 ? hd - d0 : 32;
            svzero_za();
            quad_accumulate<AttnMopa, KSteps<2>>(kd, al, ah, bl, bh, mr > 16, nc > 16);
            quad_store_raw<true>(o, 1, (long)hd, m0, d0, mr, nc);
        }
    }
}

// dst[0..n) = f16(src[0..n) * s), n a multiple of 8.
static void attn_to_f16(__fp16 *dst, const float *src, size_t n, float s) {
    for (size_t i = 0; i < n; i += 8) {
        float32x4_t lo = vmulq_n_f32(vld1q_f32(src + i), s);
        float32x4_t hi = vmulq_n_f32(vld1q_f32(src + i + 4), s);
        vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
    }
}

// One worker's buffers, sized for M rows, head_dim hd and ATTN_KC keys.
typedef struct {
    __fp16 *qrow, *qpack, *prow, *ppack;
    float *s, *ob, *o, *mx, *sum, *corr;
} attn_scratch;

static void *attn_scratch_alloc(attn_scratch *b, size_t m, size_t m_pad, size_t hd) {
    size_t h16 = ep_cmul(m_pad, hd), p16 = ep_cmul(m_pad, ATTN_KC);
    size_t f32s = ep_cmul(m, ATTN_KC) + 2 * ep_cmul(m, hd) + 3 * m;
    char *p = (char *)malloc(ep_cmul(2 * h16 + 2 * p16, 2) + ep_cmul(f32s, 4));
    if (!p) return NULL;
    b->s = (float *)(void *)p;
    b->ob = b->s + m * ATTN_KC;
    b->o = b->ob + m * hd;
    b->mx = b->o + m * hd;
    b->sum = b->mx + m;
    b->corr = b->sum + m;
    b->qrow = (__fp16 *)(void *)(b->corr + m);
    b->qpack = b->qrow + h16;
    b->prow = b->qpack + h16;
    b->ppack = b->prow + p16;
    return p;
}

// Rows at positions [p0, p0 + np) of KV head j's g query heads: row r is head
// j*g + r/R at position p0 + r%R, rows with r%R >= np pad.
static void attn_unit(const attn_scratch *b, float *out, const float *q, const __fp16 *kp,
                      const __fp16 *vp, size_t v_band, size_t start, size_t p0, size_t np, size_t j,
                      size_t g, size_t R, size_t qd, size_t hd, float scale) {
    size_t m = g * R, m_bands = 2 * ((m + 31) / 32), kd = hd / 2;
    size_t kend = start + p0 + np, k_band = kd * 32;
    for (size_t r = 0; r < m; r++) {
        __fp16 *row = b->qrow + r * hd;
        if (r % R < np)
            attn_to_f16(row, q + (p0 + r % R) * qd + (j * g + r / R) * hd, hd, scale);
        else
            memset(row, 0, hd * sizeof(__fp16));
        b->mx[r] = -INFINITY;
        b->sum[r] = 0.0f;
    }
    pack_bands<AttnMopa>(b->qpack, b->qrow, (long)hd, 1, m, hd, 0, m_bands);
    memset(b->o, 0, m * hd * sizeof(float));
    for (size_t c0 = 0; c0 < kend; c0 += ATTN_KC) {
        size_t kc = kend - c0 < ATTN_KC ? kend - c0 : ATTN_KC, kcp = (kc + 31) & ~(size_t)31;
        attn_scores_sme(b->s, ATTN_KC, b->qpack, m, kp + (c0 / 16) * k_band, kc, kd);
        for (size_t r = 0; r < m; r++) {
            size_t valid = r % R < np ? start + p0 + r % R + 1 : 0;
            size_t vr = valid > c0 ? (valid < c0 + kc ? valid : c0 + kc) - c0 : 0;
            float *sr = b->s + r * ATTN_KC;
            __fp16 *pr = b->prow + r * kcp;
            b->corr[r] = 1.0f;
            if (vr) {
                float cm = row_maxv(sr, vr), nm = cm > b->mx[r] ? cm : b->mx[r];
                float c = b->mx[r] == -INFINITY ? 0.0f : expf(b->mx[r] - nm);
                float e = row_exp_sum(sr, vr, nm);
                b->sum[r] = b->sum[r] * c + e;
                b->mx[r] = nm;
                b->corr[r] = c;
                size_t t = 0;
                for (; t + 8 <= vr; t += 8)
                    vst1q_f16(pr + t, vcombine_f16(vcvt_f16_f32(vld1q_f32(sr + t)),
                                                   vcvt_f16_f32(vld1q_f32(sr + t + 4))));
                for (; t < vr; t++)
                    pr[t] = (__fp16)sr[t];
            }
            memset(pr + vr, 0, (kcp - vr) * sizeof(__fp16));
        }
        pack_bands<AttnMopa>(b->ppack, b->prow, (long)kcp, 1, m, kcp, 0, m_bands);
        attn_pv_sme(b->ob, b->ppack, m, kcp / 2, vp + (c0 / 2) * 32, v_band, hd);
        for (size_t r = 0; r < m; r++) {
            float *orow = b->o + r * hd, *brow = b->ob + r * hd;
            float32x4_t c = vdupq_n_f32(b->corr[r]);
            for (size_t d = 0; d < hd; d += 4)
                vst1q_f32(orow + d, vfmaq_f32(vld1q_f32(brow + d), vld1q_f32(orow + d), c));
        }
    }
    for (size_t r = 0; r < m; r++) {
        if (r % R >= np) continue;
        float *dst = out + (p0 + r % R) * qd + (j * g + r / R) * hd;
        float inv = 1.0f / b->sum[r];
        for (size_t d = 0; d < hd; d += 4)
            vst1q_f32(dst + d, vmulq_n_f32(vld1q_f32(b->o + r * hd + d), inv));
    }
}

// attn_kv_causal_f16 on the SME unit: q and out are [rows][n_heads*hd], k and v
// hold start + rows rows of ld. hd is a multiple of 8. Returns -1 if an
// allocation fails.
int attn_kv_causal_sme_f16(float *out, const float *q, const __fp16 *k, const __fp16 *v,
                           size_t start, size_t rows, size_t ld, size_t n_heads, size_t n_kv_heads,
                           size_t hd, float scale) {
    size_t g = n_heads / n_kv_heads, qd = n_heads * hd, keys = start + rows;
    size_t R = g >= 32 ? 1 : 32 / g, m = g * R, m_pad = (m + 31) & ~(size_t)31;
    size_t n_qt = (rows + R - 1) / R, units = n_kv_heads * n_qt;
    size_t kd = hd / 2, kb = 2 * ((keys + 31) / 32), k_band = kd * 32;
    size_t vb = 2 * ((hd + 31) / 32), v_band = ((keys + 31) & ~(size_t)31) / 2 * 32;
    size_t kj = ep_cmul(kb, k_band), per_head = kj + ep_cmul(vb, v_band);
    __fp16 *pk = (__fp16 *)malloc(ep_cmul(ep_cmul(per_head, n_kv_heads), sizeof(__fp16)));
    if (!pk) return -1;
    // Keys as lanes against head_dim depths; values as head_dim lanes against key
    // depths, each band zero past the keys up to the 32-key padding.
    dispatch_apply(2 * n_kv_heads, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t i) {
                     __fp16 *dst = pk + (i / 2) * per_head;
                     const __fp16 *src = (i % 2 ? v : k) + (i / 2) * hd;
                     if (i % 2 == 0) {
                         pack_bands<AttnMopa>(dst, src, (long)ld, 1, keys, hd, 0, kb);
                         return;
                     }
                     size_t used = (keys + 1) / 2 * 32;
                     for (size_t bi = 0; bi < vb; bi++) {
                         __fp16 *band = dst + kj + bi * v_band;
                         pack_bands<AttnMopa>(band, src, 1, (long)ld, hd, keys, bi, bi + 1);
                         memset(band + used, 0, (v_band - used) * sizeof(__fp16));
                     }
                   });
    std::atomic<size_t> cur_v{0}, *cur = &cur_v;
    std::atomic<int> failed_v{0}, *failed = &failed_v;
    size_t W = units < ATTN_MAX_CHUNKS ? units : ATTN_MAX_CHUNKS;
    dispatch_apply(W, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t) {
      attn_scratch b;
      void *mem = NULL;
      for (size_t u; (u = cur->fetch_add(1, std::memory_order_relaxed)) < units;) {
          if (!mem && !(mem = attn_scratch_alloc(&b, m, m_pad, hd))) {
              failed->store(1, std::memory_order_relaxed);
              return;
          }
          // Later positions see more keys: hand those out first.
          size_t qt = n_qt - 1 - u / n_kv_heads, j = u % n_kv_heads, p0 = qt * R;
          size_t np = rows - p0 < R ? rows - p0 : R;
          const __fp16 *kp = pk + j * per_head;
          attn_unit(&b, out, q, kp, kp + kj, v_band, start, p0, np, j, g, R, qd, hd, scale);
      }
      free(mem);
    });
    free(pk);
    return failed->load(std::memory_order_relaxed) ? -1 : 0;
}

#endif // SME_GEMM_ATTENTION_KV_SME_H
