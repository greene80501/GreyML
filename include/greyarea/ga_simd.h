/*
 * GreyML C API header: ga simd.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include <stdbool.h>

// Runtime CPU feature detection functions
bool ga_simd_has_avx2(void);
bool ga_simd_has_avx512(void);