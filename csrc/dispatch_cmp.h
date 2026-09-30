// Comparison backends for examples/dispatch.rs: Metal MPS (GPU) and
// MLCompute ANE. Not part of the GEMM library; compiled only with the
// `dispatch-cmp` feature. C ABI so the example can call it without ObjC.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1 on success. GPU uses the default Metal device, shared storage.
int sme_cmp_gpu_init(void);
// Allocate shared Metal buffers, copy A/B in (untimed), rebuild the MPS kernel.
int sme_cmp_gpu_prepare(size_t m, size_t n, size_t k, const float *a, const float *b);
// Encode + commit + waitUntilCompleted. 1 if the command buffer succeeded.
int sme_cmp_gpu_gemm(void);
void sme_cmp_gpu_drop(void);

// 1 if an ANE device object exists (not proof a MatMul will stay on it).
int sme_cmp_ane_init(void);
// Compile an MLCompute MatMul graph for this shape against host buffers.
int sme_cmp_ane_prepare(size_t m, size_t n, size_t k, float *a, float *b, float *c);
// MLCDeviceType of the MatMul layer after compile: 0 CPU, 1 GPU, 3 ANE, -1 none.
int sme_cmp_ane_layer_device(void);
int sme_cmp_ane_gemm(void);
void sme_cmp_ane_drop(void);

#ifdef __cplusplus
}
#endif
