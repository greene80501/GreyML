/*
 * GreyML C API header: ga simd.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include <stdint.h>
#include <stdbool.h>

// Runtime CPU feature detection functions
bool ga_simd_has_avx2(void);
bool ga_simd_has_avx512(void);

// AVX2-accelerated kernels (float32 only).
void ga_simd_add_f32(const float* a, const float* b, float* out, int64_t n);
void ga_simd_mul_f32(const float* a, const float* b, float* out, int64_t n);
void ga_simd_relu_f32(const float* src, float* dst, int64_t n);
void ga_simd_sigmoid_f32(const float* src, float* dst, int64_t n);
void ga_simd_tanh_f32(const float* src, float* dst, int64_t n);
float ga_simd_sum_f32(const float* src, int64_t n);
float ga_simd_mean_f32(const float* src, int64_t n);
