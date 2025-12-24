/*
 * GreyML backend: ga ops matmul.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "greyarea/ga_ops.h"
#include "greyarea/ga_tensor.h"
#include "greyarea/ga_autograd.h"

#define BLOCK_SIZE 64

#if defined(_MSC_VER)
#define GA_RESTRICT __restrict
#else
#define GA_RESTRICT restrict
#endif

#if defined(__AVX2__)
#if defined(__FMA__)
#define GA_FMADD_PS(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#else
#define GA_FMADD_PS(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#endif

static inline void kernel_6x16_avx2(
    int64_t k,
    const float* GA_RESTRICT ptr_a, int64_t lda,
    const float* GA_RESTRICT ptr_b, int64_t ldb,
    float* GA_RESTRICT ptr_c, int64_t ldc) {
    __m256 c0_0 = _mm256_setzero_ps(); __m256 c0_1 = _mm256_setzero_ps();
    __m256 c1_0 = _mm256_setzero_ps(); __m256 c1_1 = _mm256_setzero_ps();
    __m256 c2_0 = _mm256_setzero_ps(); __m256 c2_1 = _mm256_setzero_ps();
    __m256 c3_0 = _mm256_setzero_ps(); __m256 c3_1 = _mm256_setzero_ps();
    __m256 c4_0 = _mm256_setzero_ps(); __m256 c4_1 = _mm256_setzero_ps();
    __m256 c5_0 = _mm256_setzero_ps(); __m256 c5_1 = _mm256_setzero_ps();

    for (int64_t p = 0; p < k; p++) {
        __m256 a0 = _mm256_set1_ps(ptr_a[0 * lda + p]);
        __m256 a1 = _mm256_set1_ps(ptr_a[1 * lda + p]);
        __m256 a2 = _mm256_set1_ps(ptr_a[2 * lda + p]);
        __m256 a3 = _mm256_set1_ps(ptr_a[3 * lda + p]);
        __m256 a4 = _mm256_set1_ps(ptr_a[4 * lda + p]);
        __m256 a5 = _mm256_set1_ps(ptr_a[5 * lda + p]);

        __m256 b0 = _mm256_loadu_ps(ptr_b + p * ldb + 0);
        __m256 b1 = _mm256_loadu_ps(ptr_b + p * ldb + 8);

        c0_0 = GA_FMADD_PS(a0, b0, c0_0); c0_1 = GA_FMADD_PS(a0, b1, c0_1);
        c1_0 = GA_FMADD_PS(a1, b0, c1_0); c1_1 = GA_FMADD_PS(a1, b1, c1_1);
        c2_0 = GA_FMADD_PS(a2, b0, c2_0); c2_1 = GA_FMADD_PS(a2, b1, c2_1);
        c3_0 = GA_FMADD_PS(a3, b0, c3_0); c3_1 = GA_FMADD_PS(a3, b1, c3_1);
        c4_0 = GA_FMADD_PS(a4, b0, c4_0); c4_1 = GA_FMADD_PS(a4, b1, c4_1);
        c5_0 = GA_FMADD_PS(a5, b0, c5_0); c5_1 = GA_FMADD_PS(a5, b1, c5_1);
    }

    _mm256_storeu_ps(ptr_c + 0 * ldc + 0, _mm256_add_ps(c0_0, _mm256_loadu_ps(ptr_c + 0 * ldc + 0)));
    _mm256_storeu_ps(ptr_c + 0 * ldc + 8, _mm256_add_ps(c0_1, _mm256_loadu_ps(ptr_c + 0 * ldc + 8)));
    _mm256_storeu_ps(ptr_c + 1 * ldc + 0, _mm256_add_ps(c1_0, _mm256_loadu_ps(ptr_c + 1 * ldc + 0)));
    _mm256_storeu_ps(ptr_c + 1 * ldc + 8, _mm256_add_ps(c1_1, _mm256_loadu_ps(ptr_c + 1 * ldc + 8)));
    _mm256_storeu_ps(ptr_c + 2 * ldc + 0, _mm256_add_ps(c2_0, _mm256_loadu_ps(ptr_c + 2 * ldc + 0)));
    _mm256_storeu_ps(ptr_c + 2 * ldc + 8, _mm256_add_ps(c2_1, _mm256_loadu_ps(ptr_c + 2 * ldc + 8)));
    _mm256_storeu_ps(ptr_c + 3 * ldc + 0, _mm256_add_ps(c3_0, _mm256_loadu_ps(ptr_c + 3 * ldc + 0)));
    _mm256_storeu_ps(ptr_c + 3 * ldc + 8, _mm256_add_ps(c3_1, _mm256_loadu_ps(ptr_c + 3 * ldc + 8)));
    _mm256_storeu_ps(ptr_c + 4 * ldc + 0, _mm256_add_ps(c4_0, _mm256_loadu_ps(ptr_c + 4 * ldc + 0)));
    _mm256_storeu_ps(ptr_c + 4 * ldc + 8, _mm256_add_ps(c4_1, _mm256_loadu_ps(ptr_c + 4 * ldc + 8)));
    _mm256_storeu_ps(ptr_c + 5 * ldc + 0, _mm256_add_ps(c5_0, _mm256_loadu_ps(ptr_c + 5 * ldc + 0)));
    _mm256_storeu_ps(ptr_c + 5 * ldc + 8, _mm256_add_ps(c5_1, _mm256_loadu_ps(ptr_c + 5 * ldc + 8)));
}
#endif
#ifdef GA_USE_BLAS
#include "cblas.h"

static GATensor* ga_matmul_blas(GATensor* a, GATensor* b) {
    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];
    int64_t out_shape[2] = {m, n};
    GATensor* out = ga_tensor_empty(2, out_shape, GA_FLOAT32);
    if (!out) return NULL;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)n, (int)k,
                1.0f,
                (float*)a->data, (int)k,
                (float*)b->data, (int)n,
                0.0f,
                (float*)out->data, (int)n);
    return out;
}
#endif

static inline void matmul_block(
    const float* A,
    const float* B,
    float* C,
    int64_t n,
    int64_t k,
    int64_t i,
    int64_t j,
    int64_t i_max,
    int64_t j_max) {
    float packA[BLOCK_SIZE * BLOCK_SIZE];
    float packB[BLOCK_SIZE * BLOCK_SIZE];
    int64_t i_block = i_max - i;
    int64_t j_block = j_max - j;
    for (int64_t p = 0; p < k; p += BLOCK_SIZE) {
        int64_t p_max = (p + BLOCK_SIZE < k) ? p + BLOCK_SIZE : k;
        int64_t k_block = p_max - p;
        for (int64_t ii = 0; ii < i_block; ii++) {
            memcpy(packA + ii * k_block, A + (i + ii) * k + p, (size_t)k_block * sizeof(float));
        }
        for (int64_t pp = 0; pp < k_block; pp++) {
            memcpy(packB + pp * j_block, B + (p + pp) * n + j, (size_t)j_block * sizeof(float));
        }
        for (int64_t ii = 0; ii < i_block; ii += 6) {
            int64_t ii_max = (ii + 6 < i_block) ? ii + 6 : i_block;
            for (int64_t jj = 0; jj < j_block; jj += 16) {
                int64_t jj_max = (jj + 16 < j_block) ? jj + 16 : j_block;
#if defined(__AVX2__)
                if (ii + 6 <= i_block && jj + 16 <= j_block) {
                    kernel_6x16_avx2(
                        k_block,
                        packA + ii * k_block, k_block,
                        packB + jj, j_block,
                        C + (i + ii) * n + (j + jj), n);
                    continue;
                }
#endif
                for (int64_t ii2 = ii; ii2 < ii_max; ii2++) {
                    float* c_row = C + (i + ii2) * n + j;
                    float* a_row = packA + ii2 * k_block;
                    for (int64_t pp = 0; pp < k_block; pp++) {
                        float val_a = a_row[pp];
                        float* b_row = packB + pp * j_block;
                        for (int64_t jj2 = jj; jj2 < jj_max; jj2++) {
                            c_row[jj2] += val_a * b_row[jj2];
                        }
                    }
                }
            }
        }
    }
}

GATensor* ga_matmul(GATensor* a, GATensor* b) {
    assert(a && b);
    assert(a->ndim == 2 && b->ndim == 2);
    assert(a->shape[1] == b->shape[0]);
    assert(a->dtype == GA_FLOAT32 && b->dtype == GA_FLOAT32);

    GATensor* a_contig = ga_tensor_is_contiguous(a) ? (ga_tensor_retain(a), a) : ga_tensor_contiguous(a);
    GATensor* b_contig = ga_tensor_is_contiguous(b) ? (ga_tensor_retain(b), b) : ga_tensor_contiguous(b);
    if (!a_contig || !b_contig) {
        if (a_contig) ga_tensor_release(a_contig);
        if (b_contig) ga_tensor_release(b_contig);
        return NULL;
    }

#ifdef GA_USE_BLAS
    GATensor* blas = ga_matmul_blas(a_contig, b_contig);
    if (blas) {
        ga_tensor_release(a_contig);
        ga_tensor_release(b_contig);
        return blas;
    }
#endif

    int64_t m = a_contig->shape[0];
    int64_t k = a_contig->shape[1];
    int64_t n = b_contig->shape[1];
    int64_t out_shape[2] = {m, n};

    GATensor* out = ga_tensor_empty(2, out_shape, GA_FLOAT32);
    if (!out) {
        ga_tensor_release(a_contig);
        ga_tensor_release(b_contig);
        return NULL;
    }
    float* A = (float*)a_contig->data;
    float* B = (float*)b_contig->data;
    float* C = (float*)out->data;

    memset(C, 0, (size_t)(m * n) * sizeof(float));

#if defined(_OPENMP) && defined(_MSC_VER)
    if (m <= INT_MAX && n <= INT_MAX) {
        int i_block;
        int j_block;
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (i_block = 0; i_block < (int)m; i_block += BLOCK_SIZE) {
            for (j_block = 0; j_block < (int)n; j_block += BLOCK_SIZE) {
                int64_t i = (int64_t)i_block;
                int64_t j = (int64_t)j_block;
                int64_t i_max = (i + BLOCK_SIZE < m) ? i + BLOCK_SIZE : m;
                int64_t j_max = (j + BLOCK_SIZE < n) ? j + BLOCK_SIZE : n;

                matmul_block(A, B, C, n, k, i, j, i_max, j_max);
            }
        }
    } else
#endif
    {
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
        for (int64_t i = 0; i < m; i += BLOCK_SIZE) {
            for (int64_t j = 0; j < n; j += BLOCK_SIZE) {
                int64_t i_max = (i + BLOCK_SIZE < m) ? i + BLOCK_SIZE : m;
                int64_t j_max = (j + BLOCK_SIZE < n) ? j + BLOCK_SIZE : n;

                matmul_block(A, B, C, n, k, i, j, i_max, j_max);
            }
        }
    }
    if (ga_is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        GANode* node = ga_node_create("matmul", 2, 0, 0, 0);
        if (node) {
            node->inputs[0] = a;
            node->inputs[1] = b;
            ga_tensor_retain(a);
            ga_tensor_retain(b);
            node->backward_fn = ga_autograd_matmul;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a_contig);
    ga_tensor_release(b_contig);
    return out;
}

GATensor* ga_dot(GATensor* a, GATensor* b) {
    assert(a && b);
    assert(a->ndim == 1 && b->ndim == 1 && a->shape[0] == b->shape[0]);
    int64_t shape[1] = {1};
    GATensor* out = ga_tensor_empty(1, shape, GA_FLOAT32);
    float sum = 0.0f;
    float* av = (float*)a->data;
    float* bv = (float*)b->data;
    for (int64_t i = 0; i < a->size; i++) sum += av[i] * bv[i];
    ((float*)out->data)[0] = sum;
    return out;
}

GATensor* ga_outer(GATensor* a, GATensor* b) {
    assert(a && b);
    assert(a->ndim == 1 && b->ndim == 1);
    int64_t shape[2] = {a->shape[0], b->shape[0]};
    GATensor* out = ga_tensor_empty(2, shape, GA_FLOAT32);
    float* av = (float*)a->data;
    float* bv = (float*)b->data;
    float* ov = (float*)out->data;
    for (int64_t i = 0; i < shape[0]; i++) {
        for (int64_t j = 0; j < shape[1]; j++) {
            ov[i * shape[1] + j] = av[i] * bv[j];
        }
    }
    return out;
}

GATensor* ga_bmm(GATensor* a, GATensor* b) {
    assert(a && b);
    assert(a->ndim == 3 && b->ndim == 3);
    assert(a->shape[0] == b->shape[0] && a->shape[2] == b->shape[1]);
    int64_t batch = a->shape[0];
    int64_t m = a->shape[1];
    int64_t k = a->shape[2];
    int64_t n = b->shape[2];
    int64_t out_shape[3] = {batch, m, n};
    GATensor* out = ga_tensor_empty(3, out_shape, GA_FLOAT32);
    float* A = (float*)a->data;
    float* B = (float*)b->data;
    float* C = (float*)out->data;
    for (int64_t bch = 0; bch < batch; bch++) {
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int64_t p = 0; p < k; p++) {
                    sum += A[bch * m * k + i * k + p] * B[bch * k * n + p * n + j];
                }
                C[bch * m * n + i * n + j] = sum;
            }
        }
    }
    return out;
}
