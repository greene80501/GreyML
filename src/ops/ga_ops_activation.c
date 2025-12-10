/*
 * GreyML backend: ga ops activation.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include "greyarea/ga_ops.h"
#include "greyarea/ga_common.h"
#include "greyarea/ga_autograd.h"

// Softmax along the last dimension (dim must be last for v0.1)
GATensor* ga_softmax(GATensor* a, int dim) {
    a = ga_tensor_contiguous(a);
    if (dim < 0) dim = a->ndim + dim;
    if (dim != a->ndim - 1) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        ga_tensor_release(a);
        return NULL;
    }
    
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    if (!out) {
        ga_tensor_release(a);
        return NULL;
    }
    
    int64_t inner = a->shape[a->ndim - 1];
    int64_t outer = a->size / inner;
    
    for (int64_t o = 0; o < outer; o++) {
        float* src = (float*)a->data + o * inner;
        float* dst = (float*)out->data + o * inner;
        float max_val = -INFINITY;
        for (int64_t i = 0; i < inner; i++) if (src[i] > max_val) max_val = src[i];
        float sum = 0.0f;
        for (int64_t i = 0; i < inner; i++) {
            dst[i] = expf(src[i] - max_val);
            sum += dst[i];
        }
        float inv = 1.0f / sum;
        for (int64_t i = 0; i < inner; i++) dst[i] *= inv;
    }
    if (ga_is_grad_enabled() && a->requires_grad) {
        GANode* node = ga_node_create("softmax", 1, 1, 0, 1);
        if (node) {
            node->inputs[0] = a;
            ga_tensor_retain(a);
            node->saved_tensors[0] = out;
            ga_tensor_retain(out);
            node->saved_ints[0] = dim;
            node->backward_fn = ga_autograd_softmax;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    return out;
}

GATensor* ga_log_softmax(GATensor* a, int dim) {
    GATensor* sm = ga_softmax(a, dim);
    if (!sm) return NULL;
    GATensor* out = ga_log(sm);
    ga_tensor_release(sm);
    return out;
}

GATensor* ga_leaky_relu(GATensor* a, float slope) {
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    float* src = (float*)a->data;
    float* dst = (float*)out->data;
    for (int64_t i = 0; i < a->size; i++) {
        dst[i] = src[i] > 0 ? src[i] : slope * src[i];
    }
    ga_tensor_release(a);
    return out;
}

GATensor* ga_silu(GATensor* a) {
    // x * sigmoid(x)
    GATensor* sig = ga_sigmoid(ga_tensor_clone(a));
    GATensor* out = ga_mul(a, sig);
    return out;
}

GATensor* ga_gelu(GATensor* a) {
    // Approximate GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    a = ga_tensor_contiguous(a);
    GATensor* out = ga_tensor_empty(a->ndim, a->shape, a->dtype);
    float* src = (float*)a->data;
    float* dst = (float*)out->data;
    const float c0 = 0.7978845608f; // sqrt(2/pi)
    const float c1 = 0.044715f;
    for (int64_t i = 0; i < a->size; i++) {
        float x = src[i];
        float inner = c0 * (x + c1 * x * x * x);
        float t = tanhf(inner);
        dst[i] = 0.5f * x * (1.0f + t);
    }
    ga_tensor_release(a);
    return out;
}
GATensor* ga_log_softmax(GATensor* a, int dim) {
    GATensor* sm = ga_softmax(a, dim);
    if (!sm) return NULL;
    GATensor* out = ga_log(sm);
    ga_tensor_release(sm);
    return out;
}
