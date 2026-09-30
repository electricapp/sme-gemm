// GPU (Metal MPS) and ANE (MLCompute) GEMM for the dispatch-overhead example.
// Compiled only with `--features dispatch-cmp`. MLCompute is deprecated as of
// macOS 14 but still the public C/ObjC way to submit a MatMul to the ANE
// without a CoreML model file.
#include "dispatch_cmp.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MLCompute/MLCompute.h>

#include <string.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

static id<MTLDevice> g_mtl;
static id<MTLCommandQueue> g_queue;
static id<MTLBuffer> g_buf_a, g_buf_b, g_buf_c;
static MPSMatrix *g_mat_a, *g_mat_b, *g_mat_c;
static MPSMatrixMultiplication *g_mps;
static NSUInteger g_ba, g_bb, g_bc;

static MLCDevice *g_ane_dev;
static MLCInferenceGraph *g_ane_graph;
static MLCMatMulLayer *g_ane_mm;
static MLCTensorData *g_ane_da, *g_ane_db, *g_ane_dc;
static int g_ane_layer_device = -1;

int sme_cmp_gpu_init(void) {
    g_mtl = MTLCreateSystemDefaultDevice();
    if (g_mtl == nil) {
        return 0;
    }
    g_queue = [g_mtl newCommandQueue];
    return g_queue != nil;
}

int sme_cmp_gpu_prepare(size_t m, size_t n, size_t k, const float *a, const float *b) {
    if (g_mtl == nil || m == 0 || n == 0 || k == 0) {
        return 0;
    }
    g_ba = m * k * sizeof(float);
    g_bb = k * n * sizeof(float);
    g_bc = m * n * sizeof(float);
    MTLResourceOptions opt = MTLResourceStorageModeShared;
    g_buf_a = [g_mtl newBufferWithLength:g_ba options:opt];
    g_buf_b = [g_mtl newBufferWithLength:g_bb options:opt];
    g_buf_c = [g_mtl newBufferWithLength:g_bc options:opt];
    if (g_buf_a == nil || g_buf_b == nil || g_buf_c == nil) {
        return 0;
    }
    memcpy(g_buf_a.contents, a, g_ba);
    memcpy(g_buf_b.contents, b, g_bb);
    memset(g_buf_c.contents, 0, g_bc);
    MPSMatrixDescriptor *da = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                    columns:k
                                                                   rowBytes:k * sizeof(float)
                                                                   dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *db = [MPSMatrixDescriptor matrixDescriptorWithRows:k
                                                                    columns:n
                                                                   rowBytes:n * sizeof(float)
                                                                   dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dc = [MPSMatrixDescriptor matrixDescriptorWithRows:m
                                                                    columns:n
                                                                   rowBytes:n * sizeof(float)
                                                                   dataType:MPSDataTypeFloat32];
    g_mat_a = [[MPSMatrix alloc] initWithBuffer:g_buf_a descriptor:da];
    g_mat_b = [[MPSMatrix alloc] initWithBuffer:g_buf_b descriptor:db];
    g_mat_c = [[MPSMatrix alloc] initWithBuffer:g_buf_c descriptor:dc];
    g_mps = [[MPSMatrixMultiplication alloc] initWithDevice:g_mtl
                                              transposeLeft:NO
                                             transposeRight:NO
                                                 resultRows:m
                                              resultColumns:n
                                            interiorColumns:k
                                                      alpha:1.0
                                                       beta:0.0];
    return g_mps != nil;
}

int sme_cmp_gpu_gemm(void) {
    __block int ok = 0;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        if (cmd == nil) {
            return 0;
        }
        [g_mps encodeToCommandBuffer:cmd
                          leftMatrix:g_mat_a
                         rightMatrix:g_mat_b
                        resultMatrix:g_mat_c];
        [cmd commit];
        [cmd waitUntilCompleted];
        ok = cmd.error == nil;
    }
    return ok;
}

void sme_cmp_gpu_drop(void) {
    g_mps = nil;
    g_mat_a = nil;
    g_mat_b = nil;
    g_mat_c = nil;
    g_buf_a = nil;
    g_buf_b = nil;
    g_buf_c = nil;
    g_queue = nil;
    g_mtl = nil;
}

int sme_cmp_ane_init(void) {
    g_ane_dev = [MLCDevice aneDevice];
    return g_ane_dev != nil;
}

int sme_cmp_ane_prepare(size_t m, size_t n, size_t k, float *a, float *b, float *c) {
    g_ane_graph = nil;
    g_ane_mm = nil;
    g_ane_da = nil;
    g_ane_db = nil;
    g_ane_dc = nil;
    g_ane_layer_device = -1;
    if (g_ane_dev == nil || m == 0 || n == 0 || k == 0) {
        return 0;
    }
    // Rank-3 so MatMul sees a batch dim; ANE is pickier about 2-D GEMM.
    MLCTensorDescriptor *da = [MLCTensorDescriptor descriptorWithShape:@[ @1, @(m), @(k) ]
                                                              dataType:MLCDataTypeFloat32];
    MLCTensorDescriptor *db = [MLCTensorDescriptor descriptorWithShape:@[ @1, @(k), @(n) ]
                                                              dataType:MLCDataTypeFloat32];
    if (da == nil || db == nil) {
        return 0;
    }
    MLCTensor *ta = [MLCTensor tensorWithDescriptor:da];
    MLCTensor *tb = [MLCTensor tensorWithDescriptor:db];
    MLCGraph *g = [MLCGraph graph];
    g_ane_mm = [MLCMatMulLayer layerWithDescriptor:[MLCMatMulDescriptor descriptor]];
    if (g_ane_mm == nil) {
        return 0;
    }
    MLCTensor *tc = [g nodeWithLayer:g_ane_mm sources:@[ ta, tb ]];
    if (tc == nil) {
        return 0;
    }
    MLCInferenceGraph *ig = [MLCInferenceGraph graphWithGraphObjects:@[ g ]];
    if (![ig addInputs:@{@"A" : ta, @"B" : tb}]) {
        return 0;
    }
    if (![ig addOutputs:@{@"C" : tc}]) {
        return 0;
    }
    if (![ig compileWithOptions:MLCGraphCompilationOptionsNone device:g_ane_dev]) {
        return 0;
    }
    g_ane_layer_device = (int)g_ane_mm.deviceType;
    g_ane_da = [MLCTensorData dataWithBytesNoCopy:a length:m * k * sizeof(float)];
    g_ane_db = [MLCTensorData dataWithBytesNoCopy:b length:k * n * sizeof(float)];
    g_ane_dc = [MLCTensorData dataWithBytesNoCopy:c length:m * n * sizeof(float)];
    g_ane_graph = ig;
    return 1;
}

int sme_cmp_ane_layer_device(void) { return g_ane_layer_device; }

int sme_cmp_ane_gemm(void) {
    if (g_ane_graph == nil) {
        return 0;
    }
    BOOL ok = [g_ane_graph executeWithInputsData:@{@"A" : g_ane_da, @"B" : g_ane_db}
                                     outputsData:@{@"C" : g_ane_dc}
                                       batchSize:1
                                         options:MLCExecutionOptionsSynchronous
                               completionHandler:nil];
    return ok ? 1 : 0;
}

void sme_cmp_ane_drop(void) {
    g_ane_graph = nil;
    g_ane_mm = nil;
    g_ane_da = nil;
    g_ane_db = nil;
    g_ane_dc = nil;
    g_ane_dev = nil;
    g_ane_layer_device = -1;
}

#pragma clang diagnostic pop
