/*
 * GreyML backend: ga simd detect.
 *
 * SIMD feature detection and specialized code paths for accelerated math.
 */

#include "greyarea/ga_simd.h"
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