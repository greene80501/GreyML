/*
 * GreyML backend: ga simd avx.
 *
 * SIMD feature detection and specialized code paths for accelerated math.
 */

#include "greyarea/ga_simd.h"
#include <math.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
static inline __m256 ga_tanh_approx_ps(__m256 x) {
    const __m256 max_v = _mm256_set1_ps(3.0f);
    const __m256 min_v = _mm256_set1_ps(-3.0f);
    __m256 clamped = _mm256_max_ps(min_v, _mm256_min_ps(x, max_v));
    __m256 x2 = _mm256_mul_ps(clamped, clamped);
    __m256 num = _mm256_mul_ps(clamped, _mm256_add_ps(_mm256_set1_ps(27.0f), x2));
    __m256 den = _mm256_add_ps(_mm256_set1_ps(27.0f), _mm256_mul_ps(_mm256_set1_ps(9.0f), x2));
    return _mm256_div_ps(num, den);
}
#endif

void ga_simd_add_f32(const float* a, const float* b, float* out, int64_t n) {
#if defined(__AVX2__)
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
#else
    for (int64_t i = 0; i < n; i++) out[i] = a[i] + b[i];
#endif
}

void ga_simd_mul_f32(const float* a, const float* b, float* out, int64_t n) {
#if defined(__AVX2__)
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] * b[i];
#else
    for (int64_t i = 0; i < n; i++) out[i] = a[i] * b[i];
#endif
}

void ga_simd_relu_f32(const float* src, float* dst, int64_t n) {
#if defined(__AVX2__)
    int64_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_max_ps(v, zero));
    }
    for (; i < n; i++) dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
#else
    for (int64_t i = 0; i < n; i++) dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
#endif
}

void ga_simd_sigmoid_f32(const float* src, float* dst, int64_t n) {
#if defined(__AVX2__)
    int64_t i = 0;
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 t = ga_tanh_approx_ps(_mm256_mul_ps(x, half));
        __m256 y = _mm256_mul_ps(half, _mm256_add_ps(t, one));
        _mm256_storeu_ps(dst + i, y);
    }
    for (; i < n; i++) dst[i] = 1.0f / (1.0f + expf(-src[i]));
#else
    for (int64_t i = 0; i < n; i++) dst[i] = 1.0f / (1.0f + expf(-src[i]));
#endif
}

void ga_simd_tanh_f32(const float* src, float* dst, int64_t n) {
#if defined(__AVX2__)
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(src + i);
        __m256 y = ga_tanh_approx_ps(x);
        _mm256_storeu_ps(dst + i, y);
    }
    for (; i < n; i++) dst[i] = tanhf(src[i]);
#else
    for (int64_t i = 0; i < n; i++) dst[i] = tanhf(src[i]);
#endif
}

float ga_simd_sum_f32(const float* src, int64_t n) {
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 7 < n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(src + i));
    }
    __m128 low = _mm256_castps256_ps128(acc);
    __m128 high = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float total = _mm_cvtss_f32(sum);
    for (; i < n; i++) total += src[i];
    return total;
#else
    float total = 0.0f;
    for (int64_t i = 0; i < n; i++) total += src[i];
    return total;
#endif
}

float ga_simd_mean_f32(const float* src, int64_t n) {
    if (n <= 0) return 0.0f;
    return ga_simd_sum_f32(src, n) / (float)n;
}
