/*
 * GreyML backend: ga ops blas.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#include "greyarea/ga_ops.h"

// If GA_USE_BLAS is defined, use cblas_sgemm
#ifdef GA_USE_BLAS
#include <cblas.h>

GATensor* ga_matmul_blas(GATensor* a, GATensor* b) {
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
