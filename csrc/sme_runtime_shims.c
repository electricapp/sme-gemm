// Streaming-mode SME ABI runtime shims.
//
// Clang lowers stack init / by-value struct copies inside `__arm_streaming`
// (and `__arm_streaming_compatible`) functions to calls against
// `__arm_sc_memset` / `__arm_sc_memcpy` / `__arm_sc_memmove`. The darwin
// compiler-rt build doesn't ship the streaming-safe variants yet (LLVM 21
// / Xcode 26), so this TU provides scalar fallbacks. Scalar `stp`/`str` is
// streaming-mode-safe, and clang only emits these for bookkeeping (not
// the FMOPA hot path). See https://github.com/llvm/llvm-project/issues/80009.
//
// These live in their own TU (always linked) so the F16F32 kernel TUs
// can be cfg-gated for an M5-only slim build without taking the shims
// down with them — every SME TU, F16F16 and F16F32 alike, links against
// these symbols.

#include <arm_sme.h>
#include <stddef.h>
#include <string.h>

__attribute__((used, visibility("default"))) void *
__arm_sc_memset(void *dest, int ch, size_t count) __arm_streaming_compatible {
    return memset(dest, ch, count);
}

__attribute__((used, visibility("default"))) void *
__arm_sc_memcpy(void *dest, const void *src, size_t count) __arm_streaming_compatible {
    return memcpy(dest, src, count);
}

__attribute__((used, visibility("default"))) void *
__arm_sc_memmove(void *dest, const void *src, size_t count) __arm_streaming_compatible {
    return memmove(dest, src, count);
}
