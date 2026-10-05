// A short burst of SME work for the keep-awake helper (src/warm.rs): enough to
// stop the cluster's SME unit idling between a caller's calls, short enough
// that a call arriving mid-burst waits at most ~0.1 us for the unit.
#include <arm_sme.h>
#include <stdatomic.h>
#include <stdint.h>

#include "neon_act.h"

void sme_warm_tick(void);

// Library calls currently in a NEON phase (e.g. neon_act.h's post-pass) while
// their busy mark is held; the helper counts those calls as idle.
_Atomic uint32_t sme_warm_neon;

__arm_locally_streaming __arm_new("za") void sme_warm_tick(void) {
    svfloat32_t x = svdup_n_f32(1.0f);
    svfloat32x4_t x4 = svcreate4(x, x, x, x);
    for (uint32_t i = 0; i < 32; i++)
        svmla_single_za32_f32_vg1x4(i & 7, x4, x);
}
