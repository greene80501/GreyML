/*
 * GreyML backend: ga ops binary.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#include <assert.h>
#include <math.h>
#include "greyarea/ga_ops.h"
#include "greyarea/ga_tensor.h"
#include "greyarea/ga_autograd.h"

static void broadcast_shapes(const GATensor* a, const GATensor* b, int* out_ndim, int64_t* out_shape) {
    int ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    *out_ndim = ndim;
    
    for (int i = 0; i < ndim; i++) {
        int64_t dim_a = (i < a->ndim) ? a->shape[a->ndim - 1 - i] : 1;
        int64_t dim_b = (i < b->ndim) ? b->shape[b->ndim - 1 - i] : 1;
        assert(dim_a == dim_b || dim_a == 1 || dim_b == 1);
        out_shape[ndim - 1 - i] = dim_a > dim_b ? dim_a : dim_b;
    }
}

GATensor* ga_add(GATensor* a, GATensor* b) {
    assert(a && b);
    assert(a->dtype == b->dtype);
    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);
    
    int ndim;
    int64_t shape[GA_MAX_DIMS];
    broadcast_shapes(a, b, &ndim, shape);
    
    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    if (!out) {
        ga_tensor_release(a);
        ga_tensor_release(b);
        return NULL;
    }
    
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    for (int64_t i = 0; i < out->size; i++) {
        int64_t a_idx = (a->size == 1) ? 0 : i % a->size;
        int64_t b_idx = (b->size == 1) ? 0 : i % b->size;
        out_data[i] = a_data[a_idx] + b_data[b_idx];
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
    ga_tensor_release(a);
    ga_tensor_release(b);
    return out;
}

GATensor* ga_mul(GATensor* a, GATensor* b) {
    assert(a && b);
    assert(a->dtype == b->dtype);
    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);
    
    int ndim;
    int64_t shape[GA_MAX_DIMS];
    broadcast_shapes(a, b, &ndim, shape);
    
    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    for (int64_t i = 0; i < out->size; i++) {
        int64_t a_idx = (a->size == 1) ? 0 : i % a->size;
        int64_t b_idx = (b->size == 1) ? 0 : i % b->size;
        out_data[i] = a_data[a_idx] * b_data[b_idx];
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
    ga_tensor_release(a);
    ga_tensor_release(b);
    return out;
}

GATensor* ga_pow(GATensor* a, GATensor* b) {
    assert(a && b);
    assert(a->dtype == b->dtype);
    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);
    
    int ndim;
    int64_t shape[GA_MAX_DIMS];
    broadcast_shapes(a, b, &ndim, shape);
    
    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    if (!out) {
        ga_tensor_release(a);
        ga_tensor_release(b);
        return NULL;
    }
    
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    for (int64_t i = 0; i < out->size; i++) {
        int64_t a_idx = (a->size == 1) ? 0 : i % a->size;
        int64_t b_idx = (b->size == 1) ? 0 : i % b->size;
        out_data[i] = powf(a_data[a_idx], b_data[b_idx]);
    }
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
    for (int64_t i = 0; i < a->size; i++) {
        out_data[i] = a_data[i] + scalar;
    }
    return out;
}

GATensor* ga_mul_scalar(GATensor* a, float scalar) {
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    for (int64_t i = 0; i < a->size; i++) {
        out_data[i] = a_data[i] * scalar;
    }
    return out;
}

// NEW: Added implementations

GATensor* ga_sub(GATensor* a, GATensor* b) {
    assert(a && b);
    assert(a->dtype == b->dtype);
    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);
    
    int ndim;
    int64_t shape[GA_MAX_DIMS];
    broadcast_shapes(a, b, &ndim, shape);
    
    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    if (!out) {
        ga_tensor_release(a);
        ga_tensor_release(b);
        return NULL;
    }
    
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    for (int64_t i = 0; i < out->size; i++) {
        int64_t a_idx = (a->size == 1) ? 0 : i % a->size;
        int64_t b_idx = (b->size == 1) ? 0 : i % b->size;
        out_data[i] = a_data[a_idx] - b_data[b_idx];
    }

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

GATensor* ga_div(GATensor* a, GATensor* b) {
    assert(a && b);
    assert(a->dtype == b->dtype);
    a = ga_tensor_contiguous(a);
    b = ga_tensor_contiguous(b);
    
    int ndim;
    int64_t shape[GA_MAX_DIMS];
    broadcast_shapes(a, b, &ndim, shape);
    
    GATensor* out = ga_tensor_empty(ndim, shape, a->dtype);
    if (!out) {
        ga_tensor_release(a);
        ga_tensor_release(b);
        return NULL;
    }
    
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    float* b_data = (float*)b->data;
    
    for (int64_t i = 0; i < out->size; i++) {
        int64_t a_idx = (a->size == 1) ? 0 : i % a->size;
        int64_t b_idx = (b->size == 1) ? 0 : i % b->size;
        if (b_data[b_idx] == 0.0f) {
            ga_errno = GA_ERR_DIV_BY_ZERO;
            ga_tensor_release(out);
            ga_tensor_release(a);
            ga_tensor_release(b);
            return NULL;
        }
        out_data[i] = a_data[a_idx] / b_data[b_idx];
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
