// The small/medium f32 path: B is packed per N-tile into a small scratch instead
// of a full packed-B buffer, and the dispatch is lighter. Split out of gemm_f32.c
// and included in place -- it closes over that file's packing helpers.
#ifndef SME_GEMM_F32_SMALL_H
#define SME_GEMM_F32_SMALL_H

// Small-GEMM fast path: serial (one streaming region, no GCD), A pre-packed, B
// loaded straight from row-major rhs -- B[d, n0..] is already contiguous, so no
// B pack. Targets the small regime where the parallel path's dispatch + full
// A/B packing dwarf the ~16us of compute.
__arm_locally_streaming __arm_new("za") static void run_small(
    float *dst, long dst_cs, long dst_rs, const float *a_pack, const float *rhs, long rhs_rs,
    size_t m, size_t n, size_t k, size_t mt_lo, size_t mt_hi, float alpha, float beta,
    int read_dst, const ep_desc_f32 *ep) {
    svbool_t p32 = svptrue_b32();
    const svfloat32_t z32 = svdup_n_f32(0.0f);
    const svfloat32_t va = svdup_n_f32(alpha);
    const svfloat32_t vb = svdup_n_f32(beta);
    size_t per_tile = ep_cmul(k, 16);
    size_t n_tiles = (n + 31) / 32;
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);
    // Bias fold (see run_streaming): rank-1 FMOPA bias-init, bias-only -> free.
    int fold_bias = ep && ep->n_nodes == 1 && beta == 1.0f && !read_dst &&
                    ep->nodes[0].op == EP_OP_ADD_COL;
    const float *fold_bias_ptr = fold_bias ? (const float *)ep->nodes[0].ptr : NULL;
    const ep_desc_f32 *ep_eff = fold_bias ? NULL : ep;
    int has_ep = ep_eff && ep_eff->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep_eff->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep_eff->n_nodes : 0;
    int has_tensor = has_ep && ep_has_tensor(nodes, n_nodes);
    int pure = (beta == 1.0f && !read_dst && !has_ep);

    for (size_t mt = mt_lo; mt < mt_hi; mt++) {
        const float *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
        const float *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
        size_t m0 = mt * 32;
        size_t mr = m - m0 < 32 ? m - m0 : 32;
        size_t mr_lo = mr < 16 ? mr : 16;
        size_t mr_hi = mr > 16 ? mr - 16 : 0;
        for (size_t nt = 0; nt < n_tiles; nt++) {
            size_t n0 = nt * 32;
            size_t nc = n - n0 < 32 ? n - n0 : 32;
            size_t vc0 = nc < 16 ? nc : 16;
            size_t vc1 = nc > 16 ? nc - 16 : 0;
            // Predicated loads zero the partial N-band tail (and stop the last
            // band reading past the rhs buffer).
            svbool_t pb0 = svwhilelt_b32((uint32_t)0, (uint32_t)vc0);
            svbool_t pb1 = svwhilelt_b32((uint32_t)0, (uint32_t)vc1);
            const float *b0 = rhs + (long)n0;
            // hi-N band: only read when vc1>0 (narrow-M / full). Clamp to b0 for
            // the narrow-N case so no one-past-end pointer is formed (b1 is then
            // unused, and pb1 is all-false anyway).
            const float *b1 = vc1 ? rhs + (long)n0 + 16 : b0;

            if (fold_bias) {
                bias_init_za(p32, fold_bias_ptr, n0, n, nc, mr);
            } else {
                svzero_za();
            }
            if (nc <= 16) { // narrow-N: hi-N band is pad, za1/za3 dead
                for (size_t d = 0; d < k; d++) {
                    svfloat32_t al = svld1_f32(p32, a_lo + d * 16);
                    svfloat32_t ah = svld1_f32(p32, a_hi + d * 16);
                    svfloat32_t bl = svld1_f32(pb0, b0 + (long)d * rhs_rs);
                    F32_STEP_NARROW_N(al, ah, bl);
                }
            } else if (mr <= 16) { // narrow-M: hi-M band is pad, za2/za3 dead
                for (size_t d = 0; d < k; d++) {
                    svfloat32_t al = svld1_f32(p32, a_lo + d * 16);
                    svfloat32_t bl = svld1_f32(pb0, b0 + (long)d * rhs_rs);
                    svfloat32_t bh = svld1_f32(pb1, b1 + (long)d * rhs_rs);
                    F32_STEP_NARROW_M(al, bl, bh);
                }
            } else {
                for (size_t d = 0; d < k; d++) {
                    svfloat32_t al = svld1_f32(p32, a_lo + d * 16);
                    svfloat32_t ah = svld1_f32(p32, a_hi + d * 16);
                    svfloat32_t bl = svld1_f32(pb0, b0 + (long)d * rhs_rs);
                    svfloat32_t bh = svld1_f32(pb1, b1 + (long)d * rhs_rs);
                    F32_STEP_FULL(al, ah, bl, bh);
                }
            }

            if (col_major) {
                svbool_t plo = svwhilelt_b32((uint32_t)0, (uint32_t)mr_lo);
                svbool_t phi = svwhilelt_b32((uint32_t)0, (uint32_t)mr_hi);
                for (size_t c = 0; c < vc0; c++) {
                    float *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                    // clamp so no one-past-end pointer is formed; the hi-M band is
                    // empty (phi all-false) when narrow-M (mr<=16).
                    float *colh = mr_hi ? col + 16 : col;
                    if (pure) {
                        svst1_ver_za32(0, (uint32_t)c, plo, col);
                        svst1_ver_za32(2, (uint32_t)c, phi, colh);
                    } else {
                        svfloat32_t accl = svread_ver_za32_m(z32, p32, 0, (uint32_t)c);
                        svfloat32_t acch = svread_ver_za32_m(z32, p32, 2, (uint32_t)c);
                        if (has_ep && !has_tensor) {
                            ep_store_f32(p32, plo, col, accl, vb, va, read_dst, nodes, n_nodes, 0,
                                         n0 + c, n0, m0);
                            ep_store_f32(p32, phi, colh, acch, vb, va, read_dst, nodes, n_nodes,
                                         0, n0 + c, n0, m0 + 16);
                        } else if (has_ep) {
                            float tmp[32];
                            svst1_f32(plo, tmp, accl);
                            svst1_f32(phi, tmp + 16, acch);
                            for (size_t r = 0; r < mr; r++) {
                                float v = tmp[r] * beta;
                                v = ep_apply_nodes_scalar_f32(nodes, n_nodes, v, m0 + r, n0 + c);
                                float *cell = col + (long)r * dst_rs;
                                *cell = read_dst ? alpha * (*cell) + v : v;
                            }
                        } else {
                            svfloat32_t lo = svmul_x(p32, accl, vb);
                            svfloat32_t hi = svmul_x(p32, acch, vb);
                            if (read_dst) {
                                lo = svmla_x(p32, lo, svld1_f32(plo, col), va);
                                hi = svmla_x(p32, hi, svld1_f32(phi, colh), va);
                            }
                            svst1_f32(plo, col, lo);
                            svst1_f32(phi, colh, hi);
                        }
                    }
                }
                for (size_t c = 16; c < nc; c++) {
                    uint32_t cc = (uint32_t)(c - 16);
                    float *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                    // clamp so no one-past-end pointer is formed; the hi-M band is
                    // empty (phi all-false) when narrow-M (mr<=16).
                    float *colh = mr_hi ? col + 16 : col;
                    if (pure) {
                        svst1_ver_za32(1, cc, plo, col);
                        svst1_ver_za32(3, cc, phi, colh);
                    } else {
                        svfloat32_t accl = svread_ver_za32_m(z32, p32, 1, cc);
                        svfloat32_t acch = svread_ver_za32_m(z32, p32, 3, cc);
                        if (has_ep && !has_tensor) {
                            ep_store_f32(p32, plo, col, accl, vb, va, read_dst, nodes, n_nodes, 0,
                                         n0 + c, n0, m0);
                            ep_store_f32(p32, phi, colh, acch, vb, va, read_dst, nodes, n_nodes,
                                         0, n0 + c, n0, m0 + 16);
                        } else if (has_ep) {
                            float tmp[32];
                            svst1_f32(plo, tmp, accl);
                            svst1_f32(phi, tmp + 16, acch);
                            for (size_t r = 0; r < mr; r++) {
                                float v = tmp[r] * beta;
                                v = ep_apply_nodes_scalar_f32(nodes, n_nodes, v, m0 + r, n0 + c);
                                float *cell = col + (long)r * dst_rs;
                                *cell = read_dst ? alpha * (*cell) + v : v;
                            }
                        } else {
                            svfloat32_t lo = svmul_x(p32, accl, vb);
                            svfloat32_t hi = svmul_x(p32, acch, vb);
                            if (read_dst) {
                                lo = svmla_x(p32, lo, svld1_f32(plo, col), va);
                                hi = svmla_x(p32, hi, svld1_f32(phi, colh), va);
                            }
                            svst1_f32(plo, col, lo);
                            svst1_f32(phi, colh, hi);
                        }
                    }
                }
            } else if (row_major) {
                svbool_t plo = svwhilelt_b32((uint32_t)0, (uint32_t)vc0);
                svbool_t phi = svwhilelt_b32((uint32_t)0, (uint32_t)vc1);
                float *rbase = dst + (long)m0 * dst_rs + (long)n0 * dst_cs;
                if (has_ep) {
#define EP_RDL0(s) svread_hor_za32_m(z32, p32, 0, (s))
#define EP_RDH0(s) svread_hor_za32_m(z32, p32, 1, (s))
                    EP_STORE_TILE_ROWMAJOR_F32(p32, plo, phi, rbase, dst_rs, EP_RDL0, EP_RDH0, 0,
                                               mr_lo, vb, va, read_dst, nodes, n_nodes, m0, n0);
#undef EP_RDL0
#undef EP_RDH0
#define EP_RDL2(s) svread_hor_za32_m(z32, p32, 2, (s))
#define EP_RDH2(s) svread_hor_za32_m(z32, p32, 3, (s))
                    EP_STORE_TILE_ROWMAJOR_F32(p32, plo, phi, rbase, dst_rs, EP_RDL2, EP_RDH2, 16,
                                               mr, vb, va, read_dst, nodes, n_nodes, m0, n0);
#undef EP_RDL2
#undef EP_RDH2
                } else
                for (size_t r = 0; r < mr_lo; r++) {
                    float *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                    // clamp so no one-past-end pointer is formed; the hi-N band is
                    // empty (phi all-false) when narrow-N (vc1==0).
                    float *rph = vc1 ? rp + 16 : rp;
                    if (pure) {
                        svst1_hor_za32(0, (uint32_t)r, plo, rp);
                        svst1_hor_za32(1, (uint32_t)r, phi, rph);
                    } else {
                        svfloat32_t accl = svread_hor_za32_m(z32, p32, 0, (uint32_t)r);
                        svfloat32_t acch = svread_hor_za32_m(z32, p32, 1, (uint32_t)r);
                        svfloat32_t lo = svmul_x(p32, accl, vb);
                        svfloat32_t hi = svmul_x(p32, acch, vb);
                        if (read_dst) {
                            lo = svmla_x(p32, lo, svld1_f32(plo, rp), va);
                            hi = svmla_x(p32, hi, svld1_f32(phi, rph), va);
                        }
                        svst1_f32(plo, rp, lo);
                        svst1_f32(phi, rph, hi);
                    }
                }
                if (!has_ep)
                for (size_t r = 16; r < mr; r++) {
                    uint32_t rr = (uint32_t)(r - 16);
                    float *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                    // clamp so no one-past-end pointer is formed; the hi-N band is
                    // empty (phi all-false) when narrow-N (vc1==0).
                    float *rph = vc1 ? rp + 16 : rp;
                    if (pure) {
                        svst1_hor_za32(2, rr, plo, rp);
                        svst1_hor_za32(3, rr, phi, rph);
                    } else {
                        svfloat32_t accl = svread_hor_za32_m(z32, p32, 2, rr);
                        svfloat32_t acch = svread_hor_za32_m(z32, p32, 3, rr);
                        {
                            svfloat32_t lo = svmul_x(p32, accl, vb);
                            svfloat32_t hi = svmul_x(p32, acch, vb);
                            if (read_dst) {
                                lo = svmla_x(p32, lo, svld1_f32(plo, rp), va);
                                hi = svmla_x(p32, hi, svld1_f32(phi, rph), va);
                            }
                            svst1_f32(plo, rp, lo);
                            svst1_f32(phi, rph, hi);
                        }
                    }
                }
            } else {
                float scratch[32 * 32];
                svbool_t pg = svptrue_b32();
                for (uint32_t r = 0; r < 16; r++) {
                    svst1_f32(pg, scratch + (size_t)r * 32, svread_hor_za32_m(z32, p32, 0, r));
                    svst1_f32(pg, scratch + (size_t)r * 32 + 16, svread_hor_za32_m(z32, p32, 1, r));
                    svst1_f32(pg, scratch + (size_t)(16 + r) * 32, svread_hor_za32_m(z32, p32, 2, r));
                    svst1_f32(pg, scratch + (size_t)(16 + r) * 32 + 16,
                              svread_hor_za32_m(z32, p32, 3, r));
                }
                for (size_t r = 0; r < mr; r++) {
                    for (size_t c = 0; c < nc; c++) {
                        float ab = scratch[r * 32 + c] * beta;
                        if (has_ep)
                            ab = ep_apply_nodes_scalar_f32(nodes, n_nodes, ab, m0 + r, n0 + c);
                        float *cell = dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                        *cell = read_dst ? alpha * (*cell) + ab : ab;
                    }
                }
            }
        }
    }
}

#endif // SME_GEMM_F32_SMALL_H
