/*
 * GreyML backend: ga tensor.
 *
 * Foundational runtime utilities including error handling, memory management, random utilities, and tensor lifecycle helpers.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "greyarea/ga_tensor.h"
#include "greyarea/ga_mem.h"
#include "greyarea/ga_random.h"

// Compute strides (made non-static for use in other files)
void compute_strides(GATensor* t) {
    int64_t stride = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        t->strides[i] = stride;
        stride *= t->shape[i];
    }
}

GATensor* ga_tensor_empty(int ndim, const int64_t* shape, GADtype dtype) {
    if (ndim > GA_MAX_DIMS) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    
    GATensor* t = (GATensor*)malloc(sizeof(GATensor));
    if (!t) {
        ga_errno = GA_ERR_ALLOC_FAILED;
        return NULL;
    }
    
    t->ndim = ndim;
    t->dtype = dtype;
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    compute_strides(t);
    
    size_t data_size = t->size * ga_dtype_size(dtype);
    t->data = ga_cached_alloc(data_size);
    if (!t->data) {
        free(t);
        ga_errno = GA_ERR_ALLOC_FAILED;
        return NULL;
    }
    
    t->owns_data = true;
    t->requires_grad = false;
    t->grad = NULL;
    t->grad_fn = NULL;
    t->refcount = 1;
    t->is_view = false;
    t->base = NULL;
    return t;
}

GATensor* ga_tensor_zeros(int ndim, const int64_t* shape, GADtype dtype) {
    GATensor* t = ga_tensor_empty(ndim, shape, dtype);
    if (t) memset(t->data, 0, t->size * ga_dtype_size(dtype));
    return t;
}

GATensor* ga_tensor_ones(int ndim, const int64_t* shape, GADtype dtype) {
    GATensor* t = ga_tensor_empty(ndim, shape, dtype);
    if (t && dtype == GA_FLOAT32) {
        float* data = (float*)t->data;
        for (int64_t i = 0; i < t->size; i++) data[i] = 1.0f;
    }
    return t;
}

GATensor* ga_tensor_full(int ndim, const int64_t* shape, GADtype dtype, const void* fill_value) {
    GATensor* t = ga_tensor_empty(ndim, shape, dtype);
    if (t && fill_value) {
        ga_tensor_fill(t, fill_value);
    }
    return t;
}

static inline float ga_rand_float01(void) {
    return ga_random_float();
}

GATensor* ga_tensor_rand(int ndim, const int64_t* shape) {
    GATensor* t = ga_tensor_empty(ndim, shape, GA_FLOAT32);
    if (t) {
        float* data = (float*)t->data;
        for (int64_t i = 0; i < t->size; i++) data[i] = ga_rand_float01();
    }
    return t;
}

GATensor* ga_tensor_randn(int ndim, const int64_t* shape) {
    // Box-Muller for normal distribution
    GATensor* t = ga_tensor_empty(ndim, shape, GA_FLOAT32);
    if (!t) return NULL;
    float* data = (float*)t->data;
    for (int64_t i = 0; i < t->size; i += 2) {
        float u1 = ga_rand_float01();
        float u2 = ga_rand_float01();
        float r = sqrtf(-2.0f * logf(u1 + 1e-7f));
        float theta = 2.0f * 3.1415926535f * u2;
        data[i] = r * cosf(theta);
        if (i + 1 < t->size) data[i + 1] = r * sinf(theta);
    }
    return t;
}

void ga_tensor_retain(GATensor* tensor) {
    if (tensor) tensor->refcount++;
}

void ga_tensor_release(GATensor* tensor) {
    if (!tensor) return;
    if (--tensor->refcount == 0) {
        if (tensor->owns_data && !tensor->is_view) {
            size_t data_size = tensor->size * ga_dtype_size(tensor->dtype);
            ga_cached_free(tensor->data, data_size);
        }
        if (tensor->grad) ga_tensor_release(tensor->grad);
        if (tensor->base) ga_tensor_release(tensor->base);
        free(tensor);
    }
}

bool ga_tensor_is_contiguous(const GATensor* t) {
    int64_t expected = 1;
    for (int i = t->ndim - 1; i >= 0; i--) {
        if (t->strides[i] != expected) return false;
        expected *= t->shape[i];
    }
    return true;
}

GATensor* ga_tensor_contiguous(const GATensor* tensor) {
    if (ga_tensor_is_contiguous(tensor)) {
        ga_tensor_retain((GATensor*)tensor);
        return (GATensor*)tensor;
    }
    
    GATensor* out = ga_tensor_empty(tensor->ndim, tensor->shape, tensor->dtype);
    if (!out) return NULL;
    
    size_t elem_size = ga_dtype_size(tensor->dtype);
    uint8_t* dst = (uint8_t*)out->data;
    uint8_t* src = (uint8_t*)tensor->data;
    int ndim = tensor->ndim;
    for (int64_t linear = 0; linear < tensor->size; linear++) {
        int64_t rem = linear;
        int64_t offset = 0;
        for (int d = ndim - 1; d >= 0; d--) {
            int64_t idx = rem % tensor->shape[d];
            rem /= tensor->shape[d];
            offset += idx * tensor->strides[d];
        }
        memcpy(dst + linear * elem_size, src + offset * elem_size, elem_size);
    }
    return out;
}

GATensor* ga_tensor_reshape(GATensor* tensor, int ndim, const int64_t* new_shape) {
    int64_t new_size = 1;
    for (int i = 0; i < ndim; i++) new_size *= new_shape[i];
    if (new_size != tensor->size) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    
    GATensor* view = (GATensor*)malloc(sizeof(GATensor));
    memcpy(view, tensor, sizeof(GATensor));
    view->ndim = ndim;
    for (int i = 0; i < ndim; i++) view->shape[i] = new_shape[i];
    view->is_view = true;
    view->owns_data = false;
    view->refcount = 1;
    ga_tensor_retain(tensor);
    view->base = tensor;
    compute_strides(view);
    return view;
}

// NEW: Implements ga_tensor_from_data for zero-copy numpy interop
GATensor* ga_tensor_from_data(int ndim, const int64_t* shape, GADtype dtype, void* data) {
    if (!data) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    if (ndim > GA_MAX_DIMS) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    
    GATensor* t = (GATensor*)malloc(sizeof(GATensor));
    if (!t) {
        ga_errno = GA_ERR_ALLOC_FAILED;
        return NULL;
    }
    
    t->ndim = ndim;
    t->dtype = dtype;
    t->size = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    compute_strides(t);
    
    t->data = data;              // Use external data pointer
    t->owns_data = false;        // Critical: doesn't own the data
    t->is_view = true;           // Essentially a view into external data
    t->requires_grad = false;
    t->grad = NULL;
    t->grad_fn = NULL;
    t->refcount = 1;
    t->base = NULL;
    return t;
}

GATensor* ga_tensor_arange(int64_t start, int64_t stop, int64_t step, GADtype dtype) {
    if (step == 0) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    int64_t len = (stop - start + step - (step > 0 ? 1 : -1)) / step;
    if (len < 0) len = 0;
    int64_t shape[1] = {len};
    GATensor* t = ga_tensor_empty(1, shape, dtype);
    if (!t) return NULL;
    if (dtype == GA_FLOAT32) {
        float* d = (float*)t->data;
        int64_t v = start;
        for (int64_t i = 0; i < len; i++, v += step) d[i] = (float)v;
    } else if (dtype == GA_INT64) {
        int64_t* d = (int64_t*)t->data;
        int64_t v = start;
        for (int64_t i = 0; i < len; i++, v += step) d[i] = v;
    }
    return t;
}

GATensor* ga_tensor_linspace(float start, float stop, int64_t steps) {
    if (steps <= 0) return NULL;
    int64_t shape[1] = {steps};
    GATensor* t = ga_tensor_empty(1, shape, GA_FLOAT32);
    if (!t) return NULL;
    float* d = (float*)t->data;
    if (steps == 1) {
        d[0] = start;
        return t;
    }
    float step = (stop - start) / (float)(steps - 1);
    for (int64_t i = 0; i < steps; i++) {
        d[i] = start + step * (float)i;
    }
    return t;
}

GATensor* ga_tensor_eye(int64_t n, int64_t m, GADtype dtype) {
    int64_t shape[2] = {n, m};
    GATensor* t = ga_tensor_zeros(2, shape, dtype);
    if (!t) return NULL;
    int64_t diag = n < m ? n : m;
    if (dtype == GA_FLOAT32) {
        float* d = (float*)t->data;
        for (int64_t i = 0; i < diag; i++) d[i * m + i] = 1.0f;
    } else if (dtype == GA_FLOAT64) {
        double* d = (double*)t->data;
        for (int64_t i = 0; i < diag; i++) d[i * m + i] = 1.0;
    }
    return t;
}

// NEW: Required by Python bindings
int ga_tensor_ndim(GATensor* tensor) {
    if (!tensor) return 0;
    return tensor->ndim;
}

void ga_tensor_get_shape(GATensor* tensor, int64_t* shape) {
    if (!tensor || !shape) return;
    for (int i = 0; i < tensor->ndim; i++) {
        shape[i] = tensor->shape[i];
    }
}

void ga_tensor_set_requires_grad(GATensor* tensor, bool value) {
    if (!tensor) return;
    tensor->requires_grad = value;
}

void ga_tensor_fill(GATensor* tensor, const void* value) {
    if (!tensor || !value) return;
    size_t elem_size = ga_dtype_size(tensor->dtype);
    for (int64_t i = 0; i < tensor->size; i++) {
        memcpy((uint8_t*)tensor->data + i * elem_size, value, elem_size);
    }
}

void ga_tensor_copy_from(GATensor* tensor, const void* data) {
    if (!tensor || !data) return;
    size_t data_size = tensor->size * ga_dtype_size(tensor->dtype);
    memcpy(tensor->data, data, data_size);
}

void ga_tensor_copy_to(const GATensor* tensor, void* buffer) {
    if (!tensor || !buffer) return;
    size_t data_size = tensor->size * ga_dtype_size(tensor->dtype);
    memcpy(buffer, tensor->data, data_size);
}

// NEW: Added implementations (functions that were only declared in header)

GATensor* ga_tensor_clone(const GATensor* tensor) {
    if (!tensor) return NULL;
    
    GATensor* clone = ga_tensor_empty(tensor->ndim, tensor->shape, tensor->dtype);
    if (!clone) return NULL;
    
    // Copy data
    size_t data_size = tensor->size * ga_dtype_size(tensor->dtype);
    memcpy(clone->data, tensor->data, data_size);
    
    // Copy gradient info but don't retain grad (create new)
    if (tensor->grad) {
        clone->grad = ga_tensor_clone(tensor->grad);
    }
    
    clone->requires_grad = tensor->requires_grad;
    return clone;
}

GATensor* ga_tensor_detach(GATensor* tensor) {
    if (!tensor) return NULL;
    
    GATensor* detached = ga_tensor_clone(tensor);
    if (detached) {
        detached->requires_grad = false;
        detached->grad = NULL;
        detached->grad_fn = NULL;
    }
    return detached;
}

GATensor* ga_tensor_unsqueeze(GATensor* tensor, int dim) {
    if (!tensor || tensor->ndim >= GA_MAX_DIMS) return NULL;
    if (dim < 0) dim = tensor->ndim + dim + 1;
    if (dim < 0 || dim > tensor->ndim) return NULL;
    
    // Create new shape with extra dimension of size 1
    int new_ndim = tensor->ndim + 1;
    int64_t new_shape[GA_MAX_DIMS];
    
    for (int i = 0; i < dim; i++) {
        new_shape[i] = tensor->shape[i];
    }
    new_shape[dim] = 1;
    for (int i = dim; i < tensor->ndim; i++) {
        new_shape[i + 1] = tensor->shape[i];
    }
    
    return ga_tensor_reshape(tensor, new_ndim, new_shape);
}

GATensor* ga_tensor_squeeze(GATensor* tensor, int dim) {
    if (!tensor) return NULL;
    
    if (dim < 0) {
        // Squeeze all dimensions of size 1
        int new_ndim = 0;
        int64_t new_shape[GA_MAX_DIMS];
        
        for (int i = 0; i < tensor->ndim; i++) {
            if (tensor->shape[i] != 1) {
                new_shape[new_ndim++] = tensor->shape[i];
            }
        }
        
        if (new_ndim == 0) {
            // Keep at least one dimension
            new_shape[0] = 1;
            new_ndim = 1;
        }
        
        return ga_tensor_reshape(tensor, new_ndim, new_shape);
    } else {
        // Squeeze specific dimension
        if (dim < 0) dim = tensor->ndim + dim;
        if (dim < 0 || dim >= tensor->ndim || tensor->shape[dim] != 1) {
            ga_tensor_retain(tensor);
            return tensor;
        }
        
        int new_ndim = tensor->ndim - 1;
        int64_t new_shape[GA_MAX_DIMS];
        
        for (int i = 0, j = 0; i < tensor->ndim; i++) {
            if (i != dim) {
                new_shape[j++] = tensor->shape[i];
            }
        }
        
        return ga_tensor_reshape(tensor, new_ndim, new_shape);
    }
}

GATensor* ga_tensor_flatten(GATensor* tensor, int start_dim, int end_dim) {
    if (!tensor) return NULL;
    if (start_dim < 0) start_dim = tensor->ndim + start_dim;
    if (end_dim < 0) end_dim = tensor->ndim + end_dim;
    if (start_dim < 0 || end_dim >= tensor->ndim || start_dim > end_dim) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        ga_tensor_retain(tensor);
        return tensor;
    }
    
    int64_t new_shape[GA_MAX_DIMS];
    int new_ndim = 0;
    for (int i = 0; i < start_dim; i++) {
        new_shape[new_ndim++] = tensor->shape[i];
    }
    int64_t flat = 1;
    for (int i = start_dim; i <= end_dim; i++) {
        flat *= tensor->shape[i];
    }
    new_shape[new_ndim++] = flat;
    for (int i = end_dim + 1; i < tensor->ndim; i++) {
        new_shape[new_ndim++] = tensor->shape[i];
    }
    return ga_tensor_reshape(tensor, new_ndim, new_shape);
}


GATensor* ga_tensor_get_grad(GATensor* tensor) {
    if (!tensor) return NULL;
    return tensor->grad;
}

void ga_tensor_zero_grad(GATensor* tensor) {
    if (!tensor || !tensor->grad) return;
    ga_tensor_release(tensor->grad);
    tensor->grad = NULL;
}

GATensor* ga_tensor_transpose(GATensor* tensor, int dim0, int dim1) {
    if (!tensor) return NULL;
    if (dim0 < 0) dim0 = tensor->ndim + dim0;
    if (dim1 < 0) dim1 = tensor->ndim + dim1;
    if (dim0 < 0 || dim1 < 0 || dim0 >= tensor->ndim || dim1 >= tensor->ndim) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    int64_t new_shape[GA_MAX_DIMS];
    memcpy(new_shape, tensor->shape, sizeof(int64_t) * tensor->ndim);
    int64_t tmp = new_shape[dim0];
    new_shape[dim0] = new_shape[dim1];
    new_shape[dim1] = tmp;
    return ga_tensor_reshape(tensor, tensor->ndim, new_shape);
}

// ------------------------------------------------------------------
// Additional tensor helpers (views, indexing, broadcast checks)
// ------------------------------------------------------------------

bool ga_tensor_is_view(const GATensor* tensor) { return tensor ? tensor->is_view : false; }

bool ga_tensor_same_shape(const GATensor* a, const GATensor* b) {
    if (!a || !b || a->ndim != b->ndim) return false;
    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    return true;
}

bool ga_tensor_broadcastable(const GATensor* a, const GATensor* b) {
    if (!a || !b) return false;
    int ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    for (int i = 0; i < ndim; i++) {
        int64_t da = (i < a->ndim) ? a->shape[a->ndim - 1 - i] : 1;
        int64_t db = (i < b->ndim) ? b->shape[b->ndim - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) return false;
    }
    return true;
}

void* ga_tensor_data(GATensor* tensor) { return tensor ? tensor->data : NULL; }
float* ga_tensor_data_f32(GATensor* tensor) { return tensor ? (float*)tensor->data : NULL; }
double* ga_tensor_data_f64(GATensor* tensor) { return tensor ? (double*)tensor->data : NULL; }

GATensor* ga_tensor_permute(GATensor* tensor, const int* dims) {
    if (!tensor || !dims) return NULL;
    int ndim = tensor->ndim;
    int64_t new_shape[GA_MAX_DIMS];
    int64_t new_strides[GA_MAX_DIMS];
    bool seen[GA_MAX_DIMS] = {0};
    for (int i = 0; i < ndim; i++) {
        int d = dims[i];
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim || seen[d]) {
            ga_errno = GA_ERR_INVALID_SHAPE;
            return NULL;
        }
        seen[d] = true;
        new_shape[i] = tensor->shape[d];
        new_strides[i] = tensor->strides[d];
    }
    GATensor* view = (GATensor*)malloc(sizeof(GATensor));
    memcpy(view, tensor, sizeof(GATensor));
    view->ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        view->shape[i] = new_shape[i];
        view->strides[i] = new_strides[i];
    }
    view->is_view = true;
    view->owns_data = false;
    view->refcount = 1;
    ga_tensor_retain(tensor);
    view->base = tensor;
    return view;
}

GATensor* ga_tensor_expand(GATensor* tensor, int ndim, const int64_t* shape) {
    if (!tensor || ndim < tensor->ndim || ndim > GA_MAX_DIMS) return NULL;
    int64_t new_shape[GA_MAX_DIMS];
    int64_t new_strides[GA_MAX_DIMS];
    int offset = ndim - tensor->ndim;
    for (int i = 0; i < ndim; i++) {
        int64_t target = shape[i];
        int src_idx = i - offset;
        int64_t src_dim = (src_idx >= 0) ? tensor->shape[src_idx] : 1;
        int64_t src_stride = (src_idx >= 0) ? tensor->strides[src_idx] : 0;
        if (src_dim != target && src_dim != 1) {
            ga_errno = GA_ERR_INVALID_SHAPE;
            return NULL;
        }
        new_shape[i] = target;
        new_strides[i] = (src_dim == 1) ? 0 : src_stride;
    }
    GATensor* view = (GATensor*)malloc(sizeof(GATensor));
    memcpy(view, tensor, sizeof(GATensor));
    view->ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        view->shape[i] = new_shape[i];
        view->strides[i] = new_strides[i];
    }
    view->is_view = true;
    view->owns_data = false;
    view->refcount = 1;
    ga_tensor_retain(tensor);
    view->base = tensor;
    return view;
}

static int64_t ga_tensor_offset(const GATensor* t, const int64_t* indices) {
    int64_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        offset += indices[i] * t->strides[i];
    }
    return offset;
}

GATensor* ga_tensor_get(GATensor* tensor, const int64_t* indices) {
    if (!tensor || !indices) return NULL;
    int64_t offset = ga_tensor_offset(tensor, indices);
    GATensor* out = ga_tensor_empty(0, NULL, tensor->dtype);
    size_t elem_size = ga_dtype_size(tensor->dtype);
    memcpy(out->data, (uint8_t*)tensor->data + offset * elem_size, elem_size);
    out->ndim = 0;
    out->size = 1;
    return out;
}

void ga_tensor_set(GATensor* tensor, const int64_t* indices, const void* value) {
    if (!tensor || !indices || !value) return;
    int64_t offset = ga_tensor_offset(tensor, indices);
    size_t elem_size = ga_dtype_size(tensor->dtype);
    memcpy((uint8_t*)tensor->data + offset * elem_size, value, elem_size);
}

GATensor* ga_tensor_slice(GATensor* tensor, const int64_t* start, const int64_t* end, const int64_t* step) {
    if (!tensor || !start || !end || !step) return NULL;
    int64_t new_shape[GA_MAX_DIMS];
    int64_t new_strides[GA_MAX_DIMS];
    int ndim = tensor->ndim;
    for (int i = 0; i < ndim; i++) {
        int64_t s = start[i];
        int64_t e = end[i];
        int64_t st = step[i];
        if (st == 0 || s < 0 || e > tensor->shape[i] || s > e) {
            ga_errno = GA_ERR_INVALID_SHAPE;
            return NULL;
        }
        int64_t len = (e - s + st - 1) / st;
        new_shape[i] = len;
        new_strides[i] = tensor->strides[i] * st;
    }
    GATensor* view = (GATensor*)malloc(sizeof(GATensor));
    memcpy(view, tensor, sizeof(GATensor));
    view->ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        view->shape[i] = new_shape[i];
        view->strides[i] = new_strides[i];
    }
    size_t elem_size = ga_dtype_size(tensor->dtype);
    int64_t offset = 0;
    for (int i = 0; i < ndim; i++) offset += start[i] * tensor->strides[i];
    view->data = (uint8_t*)tensor->data + offset * elem_size;
    view->is_view = true;
    view->owns_data = false;
    view->refcount = 1;
    ga_tensor_retain(tensor);
    view->base = tensor;
    return view;
}

GATensor* ga_tensor_select(GATensor* tensor, int dim, int64_t index) {
    if (!tensor) return NULL;
    if (dim < 0) dim += tensor->ndim;
    if (dim < 0 || dim >= tensor->ndim || index < 0 || index >= tensor->shape[dim]) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    int64_t start[GA_MAX_DIMS];
    int64_t end[GA_MAX_DIMS];
    int64_t step[GA_MAX_DIMS];
    for (int i = 0; i < tensor->ndim; i++) {
        start[i] = 0;
        end[i] = tensor->shape[i];
        step[i] = 1;
    }
    start[dim] = index;
    end[dim] = index + 1;
    GATensor* slice = ga_tensor_slice(tensor, start, end, step);
    if (!slice) return NULL;
    // squeeze selected dimension
    return ga_tensor_squeeze(slice, dim);
}

GATensor* ga_tensor_index(GATensor* tensor, GATensor* indices) {
    // Minimal advanced indexing: indices is int64 1-D, gather along first dim
    if (!tensor || !indices || indices->dtype != GA_INT64 || indices->ndim != 1) {
        ga_errno = GA_ERR_NOT_IMPLEMENTED;
        return NULL;
    }
    int64_t out_ndim = tensor->ndim;
    int64_t out_shape[GA_MAX_DIMS];
    out_shape[0] = indices->shape[0];
    for (int i = 1; i < tensor->ndim; i++) out_shape[i] = tensor->shape[i];
    GATensor* out = ga_tensor_empty(out_ndim, out_shape, tensor->dtype);
    size_t elem_size = ga_dtype_size(tensor->dtype);
    for (int64_t i = 0; i < indices->shape[0]; i++) {
        int64_t idx = ((int64_t*)indices->data)[i];
        if (idx < 0 || idx >= tensor->shape[0]) {
            ga_errno = GA_ERR_OUT_OF_BOUNDS;
            ga_tensor_release(out);
            return NULL;
        }
        memcpy((uint8_t*)out->data + i * tensor->strides[0] * elem_size,
               (uint8_t*)tensor->data + idx * tensor->strides[0] * elem_size,
               tensor->strides[0] * elem_size);
    }
    return out;
}
