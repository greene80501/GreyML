/*
 * GreyML backend: ga ops binary.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#include <assert.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include "greyarea/ga_ops.h"
#include "greyarea/ga_tensor.h"
#include "greyarea/ga_autograd.h"
#include "greyarea/ga_simd.h"

#if defined(_OPENMP)
#define GA_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define GA_OMP_CRITICAL _Pragma("omp critical")
#else
#define GA_OMP_PARALLEL_FOR
#define GA_OMP_CRITICAL
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

static void compute_output_shape(const GATensor* a, const GATensor* b, int* out_ndim, int64_t* out_shape) {
    int ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    *out_ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        int64_t da = (i < ndim - a->ndim) ? 1 : a->shape[i - (ndim - a->ndim)];
        int64_t db = (i < ndim - b->ndim) ? 1 : b->shape[i - (ndim - b->ndim)];
        out_shape[i] = da > db ? da : db;
    }
}

static int shapes_match(const GATensor* a, const GATensor* b) {
    if (a->ndim != b->ndim) return 0;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

static void get_broadcast_strides(const GATensor* t, const int64_t* out_shape, int ndim, int64_t* out_strides) {
    int offset = ndim - t->ndim;
    for (int i = 0; i < ndim; i++) {
        if (i < offset) {
            out_strides[i] = 0;
        } else {
            int64_t t_dim = t->shape[i - offset];
            int64_t out_dim = out_shape[i];
            out_strides[i] = (t_dim == 1 && out_dim > 1) ? 0 : t->strides[i - offset];
        }
    }
}

#define GA_BROADCAST_INDICES(linear, idx_a, idx_b, str_a, str_b, str_out, ndim) \
    do { \
        int64_t rem = (linear); \
        (idx_a) = 0; \
        (idx_b) = 0; \
        for (int d = 0; d < (ndim); d++) { \
            int64_t coord = rem / (str_out)[d]; \
            rem %= (str_out)[d]; \
            (idx_a) += coord * (str_a)[d]; \
            (idx_b) += coord * (str_b)[d]; \
        } \
    } while (0)

GATensor* ga_add(GATensor* a, GATensor* b) {
    assert(a && b);
    if (a->dtype != b->dtype) { ga_errno = GA_ERR_INVALID_DTYPE; return NULL; }
    if (!ga_tensor_broadcastable(a, b)) { ga_errno = GA_ERR_INVALID_SHAPE; return NULL; }

    if (shapes_match(a, b) && ga_tensor_is_contiguous(a) && ga_tensor_is_contiguous(b)) {
        GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
        if (!out) return NULL;
        float* out_data = (float*)out->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;

        if (a->dtype == GA_FLOAT32 && ga_simd_has_avx2()) {
            ga_simd_add_f32(a_data, b_data, out_data, out->size);
        } else {
            GA_OMP_FOR_I64(out->size, i, { out_data[i] = a_data[i] + b_data[i]; });
        }

        bool requires = ga_is_grad_enabled() && (a->requires_grad || b->requires_grad);
        if (requires) {
            GANode* node = ga_node_create("add", 2, 0, 0, 0);
            if (node) {
                node->inputs[0] = a;
                node->inputs[1] = b;
                ga_tensor_retain(a);
                ga_tensor_retain(b);
                node->backward_fn = ga_autograd_add;
                out->requires_grad = true;
                out->grad_fn = node;
            }
        }
        return out;
    }

    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);
    if (!a || !b) {
        if (a) ga_tensor_release(a);
        if (b) ga_tensor_release(b);
        return NULL;
    }

    int ndim;
    int64_t shape[GA_MAX_DIMS];
    compute_output_shape(a, b, &ndim, shape);

    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    if (!out) { ga_tensor_release(a); ga_tensor_release(b); return NULL; }

    int64_t str_a[GA_MAX_DIMS], str_b[GA_MAX_DIMS], str_out[GA_MAX_DIMS];
    get_broadcast_strides(a, shape, ndim, str_a);
    get_broadcast_strides(b, shape, ndim, str_b);
    memcpy(str_out, out->strides, (size_t)ndim * sizeof(int64_t));

    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;

    GA_OMP_FOR_I64(out->size, i, {
        int64_t idx_a, idx_b;
        GA_BROADCAST_INDICES(i, idx_a, idx_b, str_a, str_b, str_out, ndim);
        out_data[i] = a_data[idx_a] + b_data[idx_b];
    });

    bool requires = ga_is_grad_enabled() && (a->requires_grad || b->requires_grad);
    if (requires) {
        GANode* node = ga_node_create("add", 2, 0, 0, 0);
        if (node) {
            node->inputs[0] = a;
            node->inputs[1] = b;
            ga_tensor_retain(a);
            ga_tensor_retain(b);
            node->backward_fn = ga_autograd_add;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    ga_tensor_release(b);
    return out;
}

GATensor* ga_sub(GATensor* a, GATensor* b) {
    assert(a && b);
    if (a->dtype != b->dtype) { ga_errno = GA_ERR_INVALID_DTYPE; return NULL; }
    if (!ga_tensor_broadcastable(a, b)) { ga_errno = GA_ERR_INVALID_SHAPE; return NULL; }

    if (shapes_match(a, b) && ga_tensor_is_contiguous(a) && ga_tensor_is_contiguous(b)) {
        GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
        if (!out) return NULL;
        float* out_data = (float*)out->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;

        GA_OMP_FOR_I64(out->size, i, { out_data[i] = a_data[i] - b_data[i]; });

        bool requires = ga_is_grad_enabled() && (a->requires_grad || b->requires_grad);
        if (requires) {
            GANode* node = ga_node_create("sub", 2, 0, 0, 0);
            if (node) {
                node->inputs[0] = a;
                node->inputs[1] = b;
                ga_tensor_retain(a);
                ga_tensor_retain(b);
                node->backward_fn = ga_autograd_sub;
                out->requires_grad = true;
                out->grad_fn = node;
            }
        }
        return out;
    }

    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);
    if (!a || !b) {
        if (a) ga_tensor_release(a);
        if (b) ga_tensor_release(b);
        return NULL;
    }

    int ndim;
    int64_t shape[GA_MAX_DIMS];
    compute_output_shape(a, b, &ndim, shape);

    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    if (!out) { ga_tensor_release(a); ga_tensor_release(b); return NULL; }

    int64_t str_a[GA_MAX_DIMS], str_b[GA_MAX_DIMS], str_out[GA_MAX_DIMS];
    get_broadcast_strides(a, shape, ndim, str_a);
    get_broadcast_strides(b, shape, ndim, str_b);
    memcpy(str_out, out->strides, (size_t)ndim * sizeof(int64_t));

    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;

    GA_OMP_FOR_I64(out->size, i, {
        int64_t idx_a, idx_b;
        GA_BROADCAST_INDICES(i, idx_a, idx_b, str_a, str_b, str_out, ndim);
        out_data[i] = a_data[idx_a] - b_data[idx_b];
    });

    bool requires = ga_is_grad_enabled() && (a->requires_grad || b->requires_grad);
    if (requires) {
        GANode* node = ga_node_create("sub", 2, 0, 0, 0);
        if (node) {
            node->inputs[0] = a;
            node->inputs[1] = b;
            ga_tensor_retain(a);
            ga_tensor_retain(b);
            node->backward_fn = ga_autograd_sub;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    ga_tensor_release(b);
    return out;
}

GATensor* ga_mul(GATensor* a, GATensor* b) {
    assert(a && b);
    if (a->dtype != b->dtype) { ga_errno = GA_ERR_INVALID_DTYPE; return NULL; }
    if (!ga_tensor_broadcastable(a, b)) { ga_errno = GA_ERR_INVALID_SHAPE; return NULL; }

    if (shapes_match(a, b) && ga_tensor_is_contiguous(a) && ga_tensor_is_contiguous(b)) {
        GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
        if (!out) return NULL;
        float* out_data = (float*)out->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;

        if (a->dtype == GA_FLOAT32 && ga_simd_has_avx2()) {
            ga_simd_mul_f32(a_data, b_data, out_data, out->size);
        } else {
            GA_OMP_FOR_I64(out->size, i, { out_data[i] = a_data[i] * b_data[i]; });
        }

        bool requires = ga_is_grad_enabled() && (a->requires_grad || b->requires_grad);
        if (requires) {
            GANode* node = ga_node_create("mul", 2, 0, 0, 0);
            if (node) {
                node->inputs[0] = a;
                node->inputs[1] = b;
                ga_tensor_retain(a);
                ga_tensor_retain(b);
                node->backward_fn = ga_autograd_mul;
                out->requires_grad = true;
                out->grad_fn = node;
            }
        }
        return out;
    }

    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);
    if (!a || !b) {
        if (a) ga_tensor_release(a);
        if (b) ga_tensor_release(b);
        return NULL;
    }

    int ndim;
    int64_t shape[GA_MAX_DIMS];
    compute_output_shape(a, b, &ndim, shape);

    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    if (!out) { ga_tensor_release(a); ga_tensor_release(b); return NULL; }

    int64_t str_a[GA_MAX_DIMS], str_b[GA_MAX_DIMS], str_out[GA_MAX_DIMS];
    get_broadcast_strides(a, shape, ndim, str_a);
    get_broadcast_strides(b, shape, ndim, str_b);
    memcpy(str_out, out->strides, (size_t)ndim * sizeof(int64_t));

    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;

    GA_OMP_FOR_I64(out->size, i, {
        int64_t idx_a, idx_b;
        GA_BROADCAST_INDICES(i, idx_a, idx_b, str_a, str_b, str_out, ndim);
        out_data[i] = a_data[idx_a] * b_data[idx_b];
    });

    bool requires = ga_is_grad_enabled() && (a->requires_grad || b->requires_grad);
    if (requires) {
        GANode* node = ga_node_create("mul", 2, 0, 0, 0);
        if (node) {
            node->inputs[0] = a;
            node->inputs[1] = b;
            ga_tensor_retain(a);
            ga_tensor_retain(b);
            node->backward_fn = ga_autograd_mul;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    ga_tensor_release(b);
    return out;
}

GATensor* ga_div(GATensor* a, GATensor* b) {
    assert(a && b);
    if (a->dtype != b->dtype) { ga_errno = GA_ERR_INVALID_DTYPE; return NULL; }
    if (!ga_tensor_broadcastable(a, b)) { ga_errno = GA_ERR_INVALID_SHAPE; return NULL; }

    if (shapes_match(a, b) && ga_tensor_is_contiguous(a) && ga_tensor_is_contiguous(b)) {
        GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
        if (!out) return NULL;
        float* out_data = (float*)out->data;
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        int div_zero = 0;

        GA_OMP_FOR_I64(out->size, i, {
            float denom = b_data[i];
            if (denom == 0.0f) {
                GA_OMP_CRITICAL
                {
                    div_zero = 1;
                }
                out_data[i] = 0.0f;
            } else {
                out_data[i] = a_data[i] / denom;
            }
        });

        if (div_zero) {
            ga_errno = GA_ERR_DIV_BY_ZERO;
            ga_tensor_release(out);
            return NULL;
        }

        bool requires = ga_is_grad_enabled() && (a->requires_grad || b->requires_grad);
        if (requires) {
            GANode* node = ga_node_create("div", 2, 0, 0, 0);
            if (node) {
                node->inputs[0] = a;
                node->inputs[1] = b;
                ga_tensor_retain(a);
                ga_tensor_retain(b);
                node->backward_fn = ga_autograd_div;
                out->requires_grad = true;
                out->grad_fn = node;
            }
        }
        return out;
    }

    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);
    if (!a || !b) {
        if (a) ga_tensor_release(a);
        if (b) ga_tensor_release(b);
        return NULL;
    }

    int ndim;
    int64_t shape[GA_MAX_DIMS];
    compute_output_shape(a, b, &ndim, shape);

    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    if (!out) { ga_tensor_release(a); ga_tensor_release(b); return NULL; }

    int64_t str_a[GA_MAX_DIMS], str_b[GA_MAX_DIMS], str_out[GA_MAX_DIMS];
    get_broadcast_strides(a, shape, ndim, str_a);
    get_broadcast_strides(b, shape, ndim, str_b);
    memcpy(str_out, out->strides, (size_t)ndim * sizeof(int64_t));

    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    int div_zero = 0;

    GA_OMP_FOR_I64(out->size, i, {
        int64_t idx_a, idx_b;
        GA_BROADCAST_INDICES(i, idx_a, idx_b, str_a, str_b, str_out, ndim);
        float denom = b_data[idx_b];
        if (denom == 0.0f) {
            GA_OMP_CRITICAL
            {
                div_zero = 1;
            }
            out_data[i] = 0.0f;
            continue;
        }
        out_data[i] = a_data[idx_a] / denom;
    });

    if (div_zero) {
        ga_errno = GA_ERR_DIV_BY_ZERO;
        ga_tensor_release(out);
        ga_tensor_release(a);
        ga_tensor_release(b);
        return NULL;
    }

    bool requires = ga_is_grad_enabled() && (a->requires_grad || b->requires_grad);
    if (requires) {
        GANode* node = ga_node_create("div", 2, 0, 0, 0);
        if (node) {
            node->inputs[0] = a;
            node->inputs[1] = b;
            ga_tensor_retain(a);
            ga_tensor_retain(b);
            node->backward_fn = ga_autograd_div;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    ga_tensor_release(b);
    return out;
}

GATensor* ga_pow(GATensor* a, GATensor* b) {
    assert(a && b);
    if (a->dtype != b->dtype) { ga_errno = GA_ERR_INVALID_DTYPE; return NULL; }
    if (!ga_tensor_broadcastable(a, b)) { ga_errno = GA_ERR_INVALID_SHAPE; return NULL; }
    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);

    int ndim;
    int64_t shape[GA_MAX_DIMS];
    compute_output_shape(a, b, &ndim, shape);

    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    if (!out) { ga_tensor_release(a); ga_tensor_release(b); return NULL; }

    int64_t str_a[GA_MAX_DIMS], str_b[GA_MAX_DIMS], str_out[GA_MAX_DIMS];
    get_broadcast_strides(a, shape, ndim, str_a);
    get_broadcast_strides(b, shape, ndim, str_b);
    memcpy(str_out, out->strides, (size_t)ndim * sizeof(int64_t));

    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;

    GA_OMP_FOR_I64(out->size, i, {
        int64_t idx_a, idx_b;
        GA_BROADCAST_INDICES(i, idx_a, idx_b, str_a, str_b, str_out, ndim);
        out_data[i] = powf(a_data[idx_a], b_data[idx_b]);
    });

    if (ga_is_grad_enabled() && (a->requires_grad || b->requires_grad)) {
        GANode* node = ga_node_create("pow", 2, 1, 0, 0);
        if (node) {
            node->inputs[0] = a;
            node->inputs[1] = b;
            ga_tensor_retain(a);
            ga_tensor_retain(b);
            node->saved_tensors[0] = out;
            ga_tensor_retain(out);
            node->backward_fn = ga_autograd_pow;
            out->requires_grad = true;
            out->grad_fn = node;
        } else {
            out->requires_grad = (a->requires_grad || b->requires_grad);
        }
    }
    ga_tensor_release(a);
    ga_tensor_release(b);
    return out;
}

GATensor* ga_add_scalar(GATensor* a, float scalar) {
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    GA_OMP_FOR_I64(a->size, i, { out_data[i] = a_data[i] + scalar; });
    return out;
}

GATensor* ga_mul_scalar(GATensor* a, float scalar) {
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    GA_OMP_FOR_I64(a->size, i, { out_data[i] = a_data[i] * scalar; });
    return out;
}
