/*
 * GreyML backend: ga ops transform.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <string.h>
#include "greyarea/ga_ops.h"
#include "greyarea/ga_tensor.h"
#include "greyarea/ga_common.h"
#include "greyarea/ga_autograd.h"

static bool same_shape_except_dim(GATensor* a, GATensor* b, int dim) {
    if (a->ndim != b->ndim) return false;
    for (int i = 0; i < a->ndim; i++) {
        if (i == dim) continue;
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

GATensor* ga_reshape(GATensor* a, int ndim, const int64_t* shape) {
    GATensor* out = ga_tensor_reshape(a, ndim, shape);
    if (!out) return NULL;
    if (ga_is_grad_enabled() && a && a->requires_grad) {
        GANode* node = ga_node_create("reshape", 1, 0, 0, GA_MAX_DIMS + 1);
        if (node) {
            node->inputs[0] = a;
            ga_tensor_retain(a);
            node->saved_ints[0] = a->ndim;
            for (int i = 0; i < a->ndim; i++) node->saved_ints[i + 1] = (int)a->shape[i];
            node->backward_fn = ga_autograd_reshape;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    return out;
}

GATensor* ga_where(GATensor* cond, GATensor* x, GATensor* y) {
    if (!cond || !x || !y) return NULL;
    if (!ga_tensor_same_shape(x, y) || !ga_tensor_same_shape(x, cond)) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    GATensor* out = ga_tensor_empty(x->ndim, x->shape, x->dtype);
    float* dst = (float*)out->data;
    float* xv = (float*)x->data;
    float* yv = (float*)y->data;
    bool* cv = (bool*)cond->data;
    for (int64_t i = 0; i < out->size; i++) {
        dst[i] = cv[i] ? xv[i] : yv[i];
    }
    return out;
}

GATensor* ga_cat(GATensor** tensors, int num_tensors, int dim) {
    if (num_tensors == 0) return NULL;
    dim = (dim < 0) ? tensors[0]->ndim + dim : dim;
    
    int64_t total_dim_size = 0;
    for (int i = 0; i < num_tensors; i++) {
        total_dim_size += tensors[i]->shape[dim];
    }
    
    int64_t out_shape[GA_MAX_DIMS];
    memcpy(out_shape, tensors[0]->shape, sizeof(int64_t) * tensors[0]->ndim);
    out_shape[dim] = total_dim_size;
    
    GATensor* out = ga_tensor_empty(tensors[0]->ndim, out_shape, tensors[0]->dtype);
    size_t elem_size = ga_dtype_size(tensors[0]->dtype);
    uint8_t* out_data = (uint8_t*)out->data;
    
    size_t offset_bytes = 0;
    for (int i = 0; i < num_tensors; i++) {
        GATensor* t = ga_tensor_contiguous(tensors[i]);
        size_t copy_bytes = (size_t)t->size * elem_size;
        memcpy(out_data + offset_bytes, t->data, copy_bytes);
        offset_bytes += copy_bytes;
        ga_tensor_release(t);
    }
    return out;
}

GATensor* ga_stack(GATensor** tensors, int num_tensors, int dim) {
    if (num_tensors == 0) return NULL;
    dim = (dim < 0) ? tensors[0]->ndim + dim + 1 : dim;
    int64_t out_shape[GA_MAX_DIMS];
    int out_ndim = tensors[0]->ndim + 1;
    for (int i = 0, o = 0; i < out_ndim; i++) {
        if (i == dim) {
            out_shape[o++] = num_tensors;
        } else {
            out_shape[o++] = tensors[0]->shape[i - (i > dim ? 1 : 0)];
        }
    }
    // Unsqueeze each tensor then cat
    GATensor* tmp[GA_MAX_DIMS];
    for (int i = 0; i < num_tensors; i++) {
        tmp[i] = ga_tensor_unsqueeze(tensors[i], dim);
    }
    GATensor* result = ga_cat(tmp, num_tensors, dim);
    for (int i = 0; i < num_tensors; i++) ga_tensor_release(tmp[i]);
    return result;
}

void ga_split(GATensor* tensor, int chunks, int dim, GATensor** outputs) {
    dim = (dim < 0) ? tensor->ndim + dim : dim;
    int64_t size = tensor->shape[dim];
    int64_t chunk_size = size / chunks;
    int64_t offset_elems = 0;
    size_t elem_size = ga_dtype_size(tensor->dtype);
    for (int c = 0; c < chunks; c++) {
        int64_t this_size = (c == chunks - 1) ? (size - chunk_size * (chunks - 1)) : chunk_size;
        int64_t new_shape[GA_MAX_DIMS];
        memcpy(new_shape, tensor->shape, tensor->ndim * sizeof(int64_t));
        new_shape[dim] = this_size;
        GATensor* out = ga_tensor_empty(tensor->ndim, new_shape, tensor->dtype);
        uint8_t* dst = (uint8_t*)out->data;
        uint8_t* src = (uint8_t*)tensor->data + offset_elems * elem_size;
        memcpy(dst, src, out->size * elem_size);
        outputs[c] = out;
        offset_elems += out->size;
    }
}

GATensor* ga_gather(GATensor* tensor, int dim, GATensor* index) {
    dim = (dim < 0) ? tensor->ndim + dim : dim;
    assert(index->dtype == GA_INT64 || index->dtype == GA_INT32);
    if (!same_shape_except_dim(tensor, index, dim)) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    int64_t out_shape[GA_MAX_DIMS];
    memcpy(out_shape, index->shape, sizeof(int64_t) * index->ndim);
    GATensor* out = ga_tensor_empty(index->ndim, out_shape, tensor->dtype);
    float* dst = (float*)out->data;
    float* src = (float*)tensor->data;
    int64_t inner = tensor->shape[dim];
    for (int64_t i = 0; i < index->size; i++) {
        int64_t idx = (index->dtype == GA_INT64) ? ((int64_t*)index->data)[i] : ((int32_t*)index->data)[i];
        if (idx < 0 || idx >= inner) { ga_errno = GA_ERR_OUT_OF_BOUNDS; ga_tensor_release(out); return NULL; }
        // Compute base offset: flatten assumption contiguous
        int64_t stride = tensor->strides[dim];
        dst[i] = src[(i / stride) * stride * inner + idx * stride + (i % stride)];
    }
    return out;
}

GATensor* ga_scatter(GATensor* tensor, int dim, GATensor* index, GATensor* src) {
    dim = (dim < 0) ? tensor->ndim + dim : dim;
    if (!same_shape_except_dim(tensor, index, dim) || !same_shape_except_dim(src, index, dim)) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    GATensor* out = ga_tensor_clone(tensor);
    float* dst = (float*)out->data;
    float* s = (float*)src->data;
    int64_t inner = tensor->shape[dim];
    for (int64_t i = 0; i < index->size; i++) {
        int64_t idx = (index->dtype == GA_INT64) ? ((int64_t*)index->data)[i] : ((int32_t*)index->data)[i];
        if (idx < 0 || idx >= inner) { ga_errno = GA_ERR_OUT_OF_BOUNDS; ga_tensor_release(out); return NULL; }
        int64_t stride = tensor->strides[dim];
        dst[(i / stride) * stride * inner + idx * stride + (i % stride)] = s[i];
    }
    return out;
}

GATensor* ga_transpose(GATensor* a) {
    if (!a) return NULL;
    a = ga_tensor_contiguous(a);
    if (!a) return NULL;
    if (a->ndim < 2) return a;

    int64_t old0 = a->shape[a->ndim - 2];
    int64_t old1 = a->shape[a->ndim - 1];
    int64_t out_shape[GA_MAX_DIMS];
    memcpy(out_shape, a->shape, sizeof(int64_t) * a->ndim);
    out_shape[a->ndim - 2] = old1;
    out_shape[a->ndim - 1] = old0;

    GATensor* out = ga_tensor_empty(a->ndim, out_shape, a->dtype);
    float* src = (float*)a->data;
    float* dst = (float*)out->data;
    int64_t outer = a->size / (old0 * old1);
    for (int64_t o = 0; o < outer; o++) {
        for (int64_t i = 0; i < old0; i++) {
            for (int64_t j = 0; j < old1; j++) {
                int64_t src_idx = o * old0 * old1 + i * old1 + j;
                int64_t dst_idx = o * old0 * old1 + j * old0 + i;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
    if (ga_is_grad_enabled() && a->requires_grad) {
        GANode* node = ga_node_create("transpose", 1, 0, 0, 0);
        if (node) {
            node->inputs[0] = a;
            ga_tensor_retain(a);
            node->backward_fn = ga_autograd_transpose;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(a);
    return out;
}

GATensor* ga_flatten(GATensor* a) {
    if (!a) return NULL;
    int64_t flat_shape[1] = {a->size};
    GATensor* out = ga_tensor_reshape(a, 1, flat_shape);
    if (!out) return NULL;
    if (ga_is_grad_enabled() && a->requires_grad) {
        GANode* node = ga_node_create("flatten", 1, 0, 0, GA_MAX_DIMS + 1);
        if (node) {
            node->inputs[0] = a;
            ga_tensor_retain(a);
            node->saved_ints[0] = a->ndim;
            for (int i = 0; i < a->ndim; i++) node->saved_ints[i + 1] = (int)a->shape[i];
            node->backward_fn = ga_autograd_flatten;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    return out;
}
