// The small/medium f64 path: B is packed per N-tile into a small scratch instead
// of a full packed-B buffer, and the dispatch is lighter. Split out of gemm_f64.c
// and included in place -- it closes over that file's packing helpers.
#ifndef SME_GEMM_F64_SMALL_H
#define SME_GEMM_F64_SMALL_H

// Small-GEMM fast path: A pre-packed, B loaded straight from row-major rhs (no B
// pack -- B[d,n0..] is contiguous). Serial when one chunk, else a light dispatch.
__arm_locally_streaming __arm_new("za") static void run_small(
    double *dst, long dst_cs, long dst_rs, const double *a_pack, const double *rhs, long rhs_rs,
    size_t m, size_t n, size_t k, size_t mt_lo, size_t mt_hi, double alpha, double beta,
    int read_dst, const ep_desc_f64 *ep) {
    svbool_t p64 = svptrue_b64();
    const svfloat64_t z64 = svdup_n_f64(0.0);
    const svfloat64_t va = svdup_n_f64(alpha);
    const svfloat64_t vb = svdup_n_f64(beta);
    size_t per_tile = ep_cmul(k, 8);
    size_t n_tiles = (n + 15) / 16;
    int col_major = (dst_rs == 1);
    int row_major = (dst_cs == 1);
    // Bias fold (see run_streaming): rank-1 FMOPA bias-init, bias-only -> free.
    int fold_bias = ep && ep->n_nodes == 1 && beta == 1.0 && !read_dst &&
                    ep->nodes[0].op == EP_OP_ADD_COL;
    const double *fold_bias_ptr = fold_bias ? (const double *)ep->nodes[0].ptr : NULL;
    const ep_desc_f64 *ep_eff = fold_bias ? NULL : ep;
    int has_ep = ep_eff && ep_eff->n_nodes > 0;
    const EpNode *nodes = has_ep ? ep_eff->nodes : NULL;
    uint32_t n_nodes = has_ep ? ep_eff->n_nodes : 0;
    int has_tensor = has_ep && ep_has_tensor(nodes, n_nodes);
    int pure = (beta == 1.0 && !read_dst && !has_ep);

    for (size_t mt = mt_lo; mt < mt_hi; mt++) {
        const double *a_lo = a_pack + (size_t)(2 * mt) * per_tile;
        const double *a_hi = a_pack + (size_t)(2 * mt + 1) * per_tile;
        size_t m0 = mt * 16;
        size_t mr = m - m0 < 16 ? m - m0 : 16;
        size_t mr_lo = mr < 8 ? mr : 8;
        size_t mr_hi = mr > 8 ? mr - 8 : 0;
        for (size_t nt = 0; nt < n_tiles; nt++) {
            size_t n0 = nt * 16;
            size_t nc = n - n0 < 16 ? n - n0 : 16;
            size_t vc0 = nc < 8 ? nc : 8;
            size_t vc1 = nc > 8 ? nc - 8 : 0;
            svbool_t pb0 = svwhilelt_b64((uint64_t)0, (uint64_t)vc0);
            svbool_t pb1 = svwhilelt_b64((uint64_t)0, (uint64_t)vc1);
            const double *b0 = rhs + (long)n0;
            // hi-N band: only read when vc1>0 (narrow-M / full). Clamp to b0 for
            // the narrow-N case so no one-past-end pointer is formed (b1 is then
            // unused, and pb1 is all-false anyway).
            const double *b1 = vc1 ? rhs + (long)n0 + 8 : b0;

            if (fold_bias) {
                bias_init_za(p64, fold_bias_ptr, n0, n, nc, mr);
            } else {
                svzero_za();
            }
            if (nc <= 8) { // narrow-N: hi-N band is pad, za1/za3 dead
                for (size_t d = 0; d < k; d++) {
                    svfloat64_t al = svld1_f64(p64, a_lo + d * 8);
                    svfloat64_t ah = svld1_f64(p64, a_hi + d * 8);
                    svfloat64_t bl = svld1_f64(pb0, b0 + (long)d * rhs_rs);
                    F64_STEP_NARROW_N(al, ah, bl);
                }
            } else if (mr <= 8) { // narrow-M: hi-M band is pad, za2/za3 dead
                for (size_t d = 0; d < k; d++) {
                    svfloat64_t al = svld1_f64(p64, a_lo + d * 8);
                    svfloat64_t bl = svld1_f64(pb0, b0 + (long)d * rhs_rs);
                    svfloat64_t bh = svld1_f64(pb1, b1 + (long)d * rhs_rs);
                    F64_STEP_NARROW_M(al, bl, bh);
                }
            } else {
                for (size_t d = 0; d < k; d++) {
                    svfloat64_t al = svld1_f64(p64, a_lo + d * 8);
                    svfloat64_t ah = svld1_f64(p64, a_hi + d * 8);
                    svfloat64_t bl = svld1_f64(pb0, b0 + (long)d * rhs_rs);
                    svfloat64_t bh = svld1_f64(pb1, b1 + (long)d * rhs_rs);
                    F64_STEP_FULL(al, ah, bl, bh);
                }
            }

            if (col_major) {
                svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)mr_lo);
                svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)mr_hi);
                for (size_t c = 0; c < vc0; c++) {
                    double *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                    // clamp so no one-past-end pointer is formed; the hi-M band is
                    // empty (phi all-false) when narrow-M (mr<=8).
                    double *colh = mr_hi ? col + 8 : col;
                    if (pure) {
                        svst1_ver_za64(0, (uint32_t)c, plo, col);
                        svst1_ver_za64(2, (uint32_t)c, phi, colh);
                    } else {
                        svfloat64_t accl = svread_ver_za64_f64_m(z64, p64, 0, (uint32_t)c);
                        svfloat64_t acch = svread_ver_za64_f64_m(z64, p64, 2, (uint32_t)c);
                        if (has_ep && !has_tensor) {
                            ep_store_f64(p64, plo, col, accl, vb, va, read_dst, nodes, n_nodes, 0,
                                         n0 + c, n0, m0);
                            ep_store_f64(p64, phi, colh, acch, vb, va, read_dst, nodes, n_nodes,
                                         0, n0 + c, n0, m0 + 8);
                        } else if (has_ep) {
                            double tmp[16];
                            svst1_f64(plo, tmp, accl);
                            svst1_f64(phi, tmp + 8, acch);
                            for (size_t r = 0; r < mr; r++) {
                                double v = tmp[r] * beta;
                                v = ep_apply_nodes_scalar_f64(nodes, n_nodes, v, m0 + r, n0 + c);
                                double *cell = col + (long)r * dst_rs;
                                *cell = read_dst ? alpha * (*cell) + v : v;
                            }
                        } else {
                            svfloat64_t lo = svmul_x(p64, accl, vb);
                            svfloat64_t hi = svmul_x(p64, acch, vb);
                            if (read_dst) {
                                lo = svmla_x(p64, lo, svld1_f64(plo, col), va);
                                hi = svmla_x(p64, hi, svld1_f64(phi, colh), va);
                            }
                            svst1_f64(plo, col, lo);
                            svst1_f64(phi, colh, hi);
                        }
                    }
                }
                for (size_t c = 8; c < nc; c++) {
                    uint32_t cc = (uint32_t)(c - 8);
                    double *col = dst + (long)(n0 + c) * dst_cs + (long)m0 * dst_rs;
                    // clamp so no one-past-end pointer is formed; the hi-M band is
                    // empty (phi all-false) when narrow-M (mr<=8).
                    double *colh = mr_hi ? col + 8 : col;
                    if (pure) {
                        svst1_ver_za64(1, cc, plo, col);
                        svst1_ver_za64(3, cc, phi, colh);
                    } else {
                        svfloat64_t accl = svread_ver_za64_f64_m(z64, p64, 1, cc);
                        svfloat64_t acch = svread_ver_za64_f64_m(z64, p64, 3, cc);
                        if (has_ep && !has_tensor) {
                            ep_store_f64(p64, plo, col, accl, vb, va, read_dst, nodes, n_nodes, 0,
                                         n0 + c, n0, m0);
                            ep_store_f64(p64, phi, colh, acch, vb, va, read_dst, nodes, n_nodes,
                                         0, n0 + c, n0, m0 + 8);
                        } else if (has_ep) {
                            double tmp[16];
                            svst1_f64(plo, tmp, accl);
                            svst1_f64(phi, tmp + 8, acch);
                            for (size_t r = 0; r < mr; r++) {
                                double v = tmp[r] * beta;
                                v = ep_apply_nodes_scalar_f64(nodes, n_nodes, v, m0 + r, n0 + c);
                                double *cell = col + (long)r * dst_rs;
                                *cell = read_dst ? alpha * (*cell) + v : v;
                            }
                        } else {
                            svfloat64_t lo = svmul_x(p64, accl, vb);
                            svfloat64_t hi = svmul_x(p64, acch, vb);
                            if (read_dst) {
                                lo = svmla_x(p64, lo, svld1_f64(plo, col), va);
                                hi = svmla_x(p64, hi, svld1_f64(phi, colh), va);
                            }
                            svst1_f64(plo, col, lo);
                            svst1_f64(phi, colh, hi);
                        }
                    }
                }
            } else if (row_major) {
                svbool_t plo = svwhilelt_b64((uint64_t)0, (uint64_t)vc0);
                svbool_t phi = svwhilelt_b64((uint64_t)0, (uint64_t)vc1);
                double *rbase = dst + (long)m0 * dst_rs + (long)n0 * dst_cs;
                if (has_ep) {
#define EP_RDL0(s) svread_hor_za64_f64_m(z64, p64, 0, (s))
#define EP_RDH0(s) svread_hor_za64_f64_m(z64, p64, 1, (s))
                    EP_STORE_TILE_ROWMAJOR_F64(p64, plo, phi, rbase, dst_rs, EP_RDL0, EP_RDH0, 0,
                                               mr_lo, vb, va, read_dst, nodes, n_nodes, m0, n0);
#undef EP_RDL0
#undef EP_RDH0
#define EP_RDL2(s) svread_hor_za64_f64_m(z64, p64, 2, (s))
#define EP_RDH2(s) svread_hor_za64_f64_m(z64, p64, 3, (s))
                    EP_STORE_TILE_ROWMAJOR_F64(p64, plo, phi, rbase, dst_rs, EP_RDL2, EP_RDH2, 8, mr,
                                               vb, va, read_dst, nodes, n_nodes, m0, n0);
#undef EP_RDL2
#undef EP_RDH2
                } else
                for (size_t r = 0; r < mr_lo; r++) {
                    double *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                    // clamp so no one-past-end pointer is formed; the hi-N band is
                    // empty (phi all-false) when narrow-N (vc1==0).
                    double *rph = vc1 ? rp + 8 : rp;
                    if (pure) {
                        svst1_hor_za64(0, (uint32_t)r, plo, rp);
                        svst1_hor_za64(1, (uint32_t)r, phi, rph);
                    } else {
                        svfloat64_t accl = svread_hor_za64_f64_m(z64, p64, 0, (uint32_t)r);
                        svfloat64_t acch = svread_hor_za64_f64_m(z64, p64, 1, (uint32_t)r);
                        svfloat64_t lo = svmul_x(p64, accl, vb);
                        svfloat64_t hi = svmul_x(p64, acch, vb);
                        if (read_dst) {
                            lo = svmla_x(p64, lo, svld1_f64(plo, rp), va);
                            hi = svmla_x(p64, hi, svld1_f64(phi, rph), va);
                        }
                        svst1_f64(plo, rp, lo);
                        svst1_f64(phi, rph, hi);
                    }
                }
                if (!has_ep)
                for (size_t r = 8; r < mr; r++) {
                    uint32_t rr = (uint32_t)(r - 8);
                    double *rp = dst + (long)(m0 + r) * dst_rs + (long)n0 * dst_cs;
                    // clamp so no one-past-end pointer is formed; the hi-N band is
                    // empty (phi all-false) when narrow-N (vc1==0).
                    double *rph = vc1 ? rp + 8 : rp;
                    if (pure) {
                        svst1_hor_za64(2, rr, plo, rp);
                        svst1_hor_za64(3, rr, phi, rph);
                    } else {
                        svfloat64_t accl = svread_hor_za64_f64_m(z64, p64, 2, rr);
                        svfloat64_t acch = svread_hor_za64_f64_m(z64, p64, 3, rr);
                        svfloat64_t lo = svmul_x(p64, accl, vb);
                        svfloat64_t hi = svmul_x(p64, acch, vb);
                        if (read_dst) {
                            lo = svmla_x(p64, lo, svld1_f64(plo, rp), va);
                            hi = svmla_x(p64, hi, svld1_f64(phi, rph), va);
                        }
                        svst1_f64(plo, rp, lo);
                        svst1_f64(phi, rph, hi);
                    }
                }
            } else {
                double scratch[16 * 16];
                for (uint32_t r = 0; r < 8; r++) {
                    svst1_f64(p64, scratch + (size_t)r * 16, svread_hor_za64_f64_m(z64, p64, 0, r));
                    svst1_f64(p64, scratch + (size_t)r * 16 + 8,
                              svread_hor_za64_f64_m(z64, p64, 1, r));
                    svst1_f64(p64, scratch + (size_t)(8 + r) * 16,
                              svread_hor_za64_f64_m(z64, p64, 2, r));
                    svst1_f64(p64, scratch + (size_t)(8 + r) * 16 + 8,
                              svread_hor_za64_f64_m(z64, p64, 3, r));
                }
                for (size_t r = 0; r < mr; r++)
                    for (size_t c = 0; c < nc; c++) {
                        double ab = scratch[r * 16 + c] * beta;
                        if (has_ep)
                            ab = ep_apply_nodes_scalar_f64(nodes, n_nodes, ab, m0 + r, n0 + c);
                        double *cell = dst + (long)(m0 + r) * dst_rs + (long)(n0 + c) * dst_cs;
                        *cell = read_dst ? alpha * (*cell) + ab : ab;
                    }
            }
        }
    }
}

#endif // SME_GEMM_F64_SMALL_H
