// A short burst of streaming-mode work for the keep-awake helper (src/warm.rs):
// enough to stop the cluster's SME unit idling between a caller's calls.
// Vector FMLAs only, no ZA: they share the unit with a call in flight without
// slowing it, so the helper never has to stand aside. (A ZA burst does slow
// it, by 2x on a one-row Q4 GEMV, measured on M5.)
#include <arm_sme.h>

float sme_warm_tick(void);

__arm_locally_streaming float sme_warm_tick(void) {
    svbool_t pg = svptrue_b32();
    svfloat32_t h = svdup_n_f32(0.5f), a = svdup_n_f32(1.0f), b = a, c = a, d = a;
    for (int i = 0; i < 8; i++) {
        a = svmla_f32_x(pg, a, a, h);
        b = svmla_f32_x(pg, b, b, h);
        c = svmla_f32_x(pg, c, c, h);
        d = svmla_f32_x(pg, d, d, h);
    }
    return svaddv_f32(pg, svadd_f32_x(pg, svadd_f32_x(pg, a, b), svadd_f32_x(pg, c, d)));
}
