/*
 * GreyML backend: ga simd avx.
 *
 * SIMD feature detection and specialized code paths for accelerated math.
 */

#include "greyarea/ga_simd.h"

// Placeholder for AVX-specific kernels. Keeping a symbol here prevents
// empty-translation-unit issues and documents the intended extension point.
int ga_simd_avx_placeholder(void) {
    return (int)(ga_simd_has_avx512() || ga_simd_has_avx2());
}
