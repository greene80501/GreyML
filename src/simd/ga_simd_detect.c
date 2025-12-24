/*
 * GreyML backend: ga simd detect.
 *
 * SIMD feature detection and specialized code paths for accelerated math.
 */

#include "greyarea/ga_simd.h"

#ifdef _WIN32
#include <intrin.h>

bool ga_simd_has_avx2(void) {
    int info[4];
    __cpuidex(info, 0, 0);
    int nIds = info[0];
    
    if (nIds >= 7) {
        __cpuidex(info, 7, 0);
        return (info[1] & (1 << 5)) != 0;  // Check AVX2 bit (ECX bit 5)
    }
    return false;
}

bool ga_simd_has_avx512(void) {
    int info[4];
    __cpuidex(info, 0, 0);
    int nIds = info[0];
    
    if (nIds >= 7) {
        __cpuidex(info, 7, 0);
        // Check AVX-512F bit (EBX bit 16) and other AVX-512 extensions
        return (info[1] & (1 << 16)) != 0;
    }
    return false;
}

#else
// Linux / GCC / Clang implementation
#include <cpuid.h>

bool ga_simd_has_avx2(void) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;
    }
    return false;
#endif
}

bool ga_simd_has_avx512(void) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f");
#else
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0;
    }
    return false;
#endif
}

#endif
