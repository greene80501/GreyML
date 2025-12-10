/*
 * GreyML backend: ga ops matmul.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>  // ADD THIS LINE
#include "greyarea/ga_ops.h"
#include "greyarea/ga_tensor.h"
#include "greyarea/ga_autograd.h"

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

GATensor* ga_matmul(GATensor* a, GATensor* b) {
    assert(a && b);  // Now assert will link correctly
    assert(a->ndim == 2 && b->ndim == 2);
    assert(a->shape[1] == b->shape[0]);
    assert(a->dtype == GA_FLOAT32 && b->dtype == GA_FLOAT32);
    
#ifdef GA_USE_BLAS
    GATensor* blas = ga_matmul_blas(a, b);
    if (blas) return blas;
#endif
    
    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];
    int64_t out_shape[2] = {m, n};
    
    GATensor* out = ga_tensor_empty(2, out_shape, GA_FLOAT32);
    float* A = (float*)a->data;
    float* B = (float*)b->data;
    float* C = (float*)out->data;
    
    // Naive O(n³) implementation
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
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
