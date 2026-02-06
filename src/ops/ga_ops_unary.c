/*
 * GreyML backend: ga ops unary.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#include "greyarea/ga_ops.h"
#include "greyarea/ga_common.h"
#include "greyarea/ga_autograd.h"
#include "greyarea/ga_simd.h"
#include <math.h>
#include <limits.h>

// Helper dispatch for float32/float64 unary ops
static void for_each_float(const GATensor* a, GATensor* out, void (*fn32)(float*, const float*, int64_t), void (*fn64)(double*, const double*, int64_t)) {
    if (a->dtype == GA_FLOAT64) {
        fn64((double*)out->data, (const double*)a->data, a->size);
    } else {
        fn32((float*)out->data, (const float*)a->data, a->size);
    }
}

#if defined(_OPENMP)
#define GA_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#else
#define GA_OMP_PARALLEL_FOR
#endif

#if defined(_OPENMP) && defined(_MSC_VER)
#define GA_OMP_FOR_I64(n, idx, ...) \
    do { \
        if ((n) <= INT_MAX) { \
            int _i; \
            GA_OMP_PARALLEL_FOR \
            for (_i = 0; _i < (int)(n); _i++) { \
                int64_t idx = (int64_t)_i; \
                __VA_ARGS__ \
            } \
        } else { \
            for (int64_t idx = 0; idx < (n); idx++) { \
                __VA_ARGS__ \
            } \
        } \
    } while (0)
#else
#define GA_OMP_FOR_I64(n, idx, ...) \
    do { \
        GA_OMP_PARALLEL_FOR \
        for (int64_t idx = 0; idx < (n); idx++) { \
            __VA_ARGS__ \
        } \
    } while (0)
#endif

static void op_neg_f32(float* dst, const float* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = -src[i]; });
}
static void op_neg_f64(double* dst, const double* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = -src[i]; });
}
static void op_relu_f32(float* dst, const float* src, int64_t n) {
    if (ga_simd_has_avx2()) {
        ga_simd_relu_f32(src, dst, n);
        return;
    }
    GA_OMP_FOR_I64(n, i, { dst[i] = src[i] > 0 ? src[i] : 0.0f; });
}
static void op_relu_f64(double* dst, const double* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = src[i] > 0 ? src[i] : 0.0; });
}
static void op_sigmoid_f32(float* dst, const float* src, int64_t n) {
    if (ga_simd_has_avx2()) {
        ga_simd_sigmoid_f32(src, dst, n);
        return;
    }
    GA_OMP_FOR_I64(n, i, { dst[i] = 1.0f / (1.0f + expf(-src[i])); });
}
static void op_sigmoid_f64(double* dst, const double* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = 1.0 / (1.0 + exp(-src[i])); });
}
static void op_exp_f32(float* dst, const float* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = expf(src[i]); });
}
static void op_exp_f64(double* dst, const double* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = exp(src[i]); });
}
static void op_log_f32(float* dst, const float* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = logf(src[i]); });
}
static void op_log_f64(double* dst, const double* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = log(src[i]); });
}
static void op_sqrt_f32(float* dst, const float* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = sqrtf(src[i]); });
}
static void op_sqrt_f64(double* dst, const double* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = sqrt(src[i]); });
}
static void op_abs_f32(float* dst, const float* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = fabsf(src[i]); });
}
static void op_abs_f64(double* dst, const double* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = fabs(src[i]); });
}
static void op_square_f32(float* dst, const float* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = src[i] * src[i]; });
}
static void op_square_f64(double* dst, const double* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = src[i] * src[i]; });
}
static void op_tanh_f32(float* dst, const float* src, int64_t n) {
    if (ga_simd_has_avx2()) {
        ga_simd_tanh_f32(src, dst, n);
        return;
    }
    GA_OMP_FOR_I64(n, i, { dst[i] = tanhf(src[i]); });
}
static void op_tanh_f64(double* dst, const double* src, int64_t n) {
    GA_OMP_FOR_I64(n, i, { dst[i] = tanh(src[i]); });
}

GATensor* ga_neg(GATensor* a) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    for_each_float(a, out, op_neg_f32, op_neg_f64);
    ga_tensor_release(a);
    return out;
}

GATensor* ga_relu(GATensor* a) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    for_each_float(a, out, op_relu_f32, op_relu_f64);
    if (ga_is_grad_enabled() && a->requires_grad) {
        GANode* node = ga_node_create("relu", 1, 0, 0, 0);
        if (node) {
            node->inputs[0] = a;
            ga_tensor_retain(a);
            node->backward_fn = ga_autograd_relu;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    return out;
}

GATensor* ga_sigmoid(GATensor* a) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    for_each_float(a, out, op_sigmoid_f32, op_sigmoid_f64);
    if (ga_is_grad_enabled() && a->requires_grad) {
        GANode* node = ga_node_create("sigmoid", 1, 1, 0, 0);
        if (node) {
            node->inputs[0] = a;
            ga_tensor_retain(a);
            node->saved_tensors[0] = out;
            ga_tensor_retain(out);
            node->backward_fn = ga_autograd_sigmoid;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    return out;
}

GATensor* ga_exp(GATensor* a) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    for_each_float(a, out, op_exp_f32, op_exp_f64);
    ga_tensor_release(a);
    return out;
}

GATensor* ga_log(GATensor* a) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    for_each_float(a, out, op_log_f32, op_log_f64);
    if (ga_is_grad_enabled() && a->requires_grad) {
        GANode* node = ga_node_create("log", 1, 0, 0, 0);
        if (node) {
            node->inputs[0] = a;
            ga_tensor_retain(a);
            node->backward_fn = ga_autograd_log;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    return out;
}

GATensor* ga_sqrt(GATensor* a) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    for_each_float(a, out, op_sqrt_f32, op_sqrt_f64);
    ga_tensor_release(a);
    return out;
}

GATensor* ga_abs(GATensor* a) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    for_each_float(a, out, op_abs_f32, op_abs_f64);
    ga_tensor_release(a);
    return out;
}

GATensor* ga_square(GATensor* a) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    for_each_float(a, out, op_square_f32, op_square_f64);
    ga_tensor_release(a);
    return out;
}

GATensor* ga_tanh(GATensor* a) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    for_each_float(a, out, op_tanh_f32, op_tanh_f64);
    ga_tensor_release(a);
    return out;
}
