/*
 * GreyML backend: ga ops reduce.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#include <math.h>
#include <string.h>
#include "greyarea/ga_ops.h"
#include "greyarea/ga_tensor.h"
#include "greyarea/ga_autograd.h"

GATensor* ga_sum(GATensor* a, int dim, bool keepdim) {
    a = ga_tensor_contiguous(a);
    if (dim < 0) dim = a->ndim + dim;
    
    int out_ndim = keepdim ? a->ndim : a->ndim - 1;
    int64_t out_shape[GA_MAX_DIMS];
    for (int i = 0, j = 0; i < a->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = a->shape[i];
        }
    }
    
    GATensor* out = ga_tensor_empty(out_ndim, out_shape, a->dtype);
    if (!out) return NULL;
    
    float* out_data = (float*)out->data;
    memset(out_data, 0, out->size * sizeof(float));
    
    float* a_data = (float*)a->data;
    int64_t stride = (dim == -1) ? 1 : a->strides[dim];
    
    for (int64_t i = 0; i < a->size; i++) {
        int64_t out_idx = i / (stride * a->shape[dim]) * stride + i % stride;
        out_data[out_idx] += a_data[i];
    }
    if (ga_is_grad_enabled() && a->requires_grad) {
        GANode* node = ga_node_create("sum", 1, 0, 0, 1);
        if (node) {
            node->inputs[0] = a;
            ga_tensor_retain(a);
            node->saved_ints[0] = dim;
            node->backward_fn = ga_autograd_sum;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    return out;
}

GATensor* ga_min(GATensor* a, int dim, bool keepdim) {
    a = ga_tensor_contiguous(a);
    dim = (dim < 0) ? a->ndim + dim : dim;
    
    int out_ndim = keepdim ? a->ndim : a->ndim - 1;
    int64_t out_shape[GA_MAX_DIMS];
    for (int i = 0, j = 0; i < a->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = a->shape[i];
        }
    }
    
    GATensor* out = ga_tensor_empty(out_ndim, out_shape, a->dtype);
    if (!out) return NULL;
    
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    int64_t stride = (dim == -1) ? 1 : a->strides[dim];
    
    for (int64_t i = 0; i < out->size; i++) {
        out_data[i] = INFINITY;
    }
    for (int64_t i = 0; i < a->size; i++) {
        int64_t out_idx = i / (stride * a->shape[dim]) * stride + i % stride;
        if (a_data[i] < out_data[out_idx]) {
            out_data[out_idx] = a_data[i];
        }
    }
    ga_tensor_release(a);
    return out;
}

GATensor* ga_argmax(GATensor* a, int dim) {
    a = ga_tensor_contiguous(a);
    dim = (dim < 0) ? a->ndim + dim : dim;
    
    int out_ndim = a->ndim - 1;
    int64_t out_shape[GA_MAX_DIMS];
    for (int i = 0, j = 0; i < a->ndim; i++) {
        if (i != dim) out_shape[j++] = a->shape[i];
    }
    
    GATensor* out = ga_tensor_empty(out_ndim, out_shape, GA_INT64);
    if (!out) return NULL;
    
    int64_t inner = a->shape[dim];
    int64_t outer = a->size / inner;
    float* a_data = (float*)a->data;
    int64_t* out_data = (int64_t*)out->data;
    
    for (int64_t o = 0; o < outer; o++) {
        float best = -INFINITY;
        int64_t best_idx = 0;
        for (int64_t i = 0; i < inner; i++) {
            float v = a_data[o * inner + i];
            if (v > best) {
                best = v;
                best_idx = i;
            }
        }
        out_data[o] = best_idx;
    }
    ga_tensor_release(a);
    return out;
}

// NEW: Added implementations

GATensor* ga_mean(GATensor* a, int dim, bool keepdim) {
    GATensor* sum_tensor = ga_sum(a, dim, keepdim);
    if (!sum_tensor) return NULL;
    
    // Calculate the number of elements in the reduced dimension
    int64_t dim_size = (dim == -1) ? a->size : a->shape[dim];
    float scale = 1.0f / (float)dim_size;
    
    GATensor* result = ga_mul_scalar(sum_tensor, scale);
    if (ga_is_grad_enabled() && a->requires_grad) {
        GANode* node = ga_node_create("mean", 1, 0, 1, 0);
        if (node) {
            node->inputs[0] = a;
            ga_tensor_retain(a);
            node->saved_scalars[0] = scale;
            node->backward_fn = ga_autograd_mean;
            result->requires_grad = true;
            result->grad_fn = node;
        }
    }
    ga_tensor_release(sum_tensor);
    return result;
}

GATensor* ga_var(GATensor* a, int dim, bool keepdim, bool unbiased) {
    GATensor* mean_tensor = ga_mean(a, dim, keepdim);
    if (!mean_tensor) return NULL;
    
    // (a - mean)^2
    GATensor* diff = ga_sub(a, mean_tensor);
    GATensor* sq_diff = ga_mul(diff, diff);
    
    GATensor* var = ga_mean(sq_diff, dim, keepdim);
    ga_tensor_release(mean_tensor);
    ga_tensor_release(diff);
    ga_tensor_release(sq_diff);
    
    if (unbiased && dim >= 0 && dim < a->ndim) {
        // Apply Bessel's correction: multiply by n/(n-1)
        float n = (float)a->shape[dim];
        if (n > 1.0f) {
            float correction = n / (n - 1.0f);
            GATensor* corrected = ga_mul_scalar(var, correction);
            ga_tensor_release(var);
            return corrected;
        }
    }
    
    return var;
}

GATensor* ga_max(GATensor* a, int dim, bool keepdim) {
    a = ga_tensor_contiguous(a);
    dim = (dim < 0) ? a->ndim + dim : dim;
    
    int out_ndim = keepdim ? a->ndim : a->ndim - 1;
    int64_t out_shape[GA_MAX_DIMS];
    for (int i = 0, j = 0; i < a->ndim; i++) {
        if (i == dim) {
            if (keepdim) out_shape[j++] = 1;
        } else {
            out_shape[j++] = a->shape[i];
        }
    }
    
    GATensor* out = ga_tensor_empty(out_ndim, out_shape, a->dtype);
    if (!out) return NULL;
    
    float* out_data = (float*)out->data;
    float* a_data = (float*)a->data;
    int64_t stride = (dim == -1) ? 1 : a->strides[dim];
    
    // Initialize to -inf
    for (int64_t i = 0; i < out->size; i++) {
        out_data[i] = -INFINITY;
    }
    
    // Find maximum values along dimension
    for (int64_t i = 0; i < a->size; i++) {
        int64_t out_idx = i / (stride * a->shape[dim]) * stride + i % stride;
        if (a_data[i] > out_data[out_idx]) {
            out_data[out_idx] = a_data[i];
        }
    }
    
    ga_tensor_release(a);
    return out;
}

GATensor* ga_flatten(GATensor* a) {
    return ga_tensor_flatten(a, 0, a->ndim - 1);
}
