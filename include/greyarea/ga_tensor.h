/*
 * GreyML C API header: ga tensor.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_common.h"

struct GANode;
struct GATensor;

typedef struct GATensor {
    void* data;
    GADtype dtype;
    int ndim;
    int64_t shape[GA_MAX_DIMS];
    int64_t strides[GA_MAX_DIMS];
    int64_t size;
    bool owns_data;
    
    bool requires_grad;
    struct GATensor* grad;
    struct GANode* grad_fn;
    
    int32_t refcount;
    bool is_view;
    struct GATensor* base;
} GATensor;


GA_API GATensor* ga_tensor_empty(int ndim, const int64_t* shape, GADtype dtype);
GA_API GATensor* ga_tensor_zeros(int ndim, const int64_t* shape, GADtype dtype);
GA_API GATensor* ga_tensor_ones(int ndim, const int64_t* shape, GADtype dtype);
GA_API GATensor* ga_tensor_full(int ndim, const int64_t* shape, GADtype dtype, const void* fill_value);
GA_API GATensor* ga_tensor_from_data(int ndim, const int64_t* shape, GADtype dtype, void* data);
GA_API GATensor* ga_tensor_arange(int64_t start, int64_t stop, int64_t step, GADtype dtype);
GA_API GATensor* ga_tensor_linspace(float start, float stop, int64_t steps);
GA_API GATensor* ga_tensor_eye(int64_t n, int64_t m, GADtype dtype);
GA_API GATensor* ga_tensor_rand(int ndim, const int64_t* shape);
GA_API GATensor* ga_tensor_randn(int ndim, const int64_t* shape);

GA_API void ga_tensor_retain(GATensor* tensor);
GA_API void ga_tensor_release(GATensor* tensor);
GA_API GATensor* ga_tensor_clone(const GATensor* tensor);
GA_API GATensor* ga_tensor_contiguous(const GATensor* tensor);
GA_API GATensor* ga_tensor_detach(GATensor* tensor);

GA_API GATensor* ga_tensor_reshape(GATensor* tensor, int ndim, const int64_t* new_shape);
GA_API GATensor* ga_tensor_flatten(GATensor* tensor, int start_dim, int end_dim);
GA_API GATensor* ga_tensor_unsqueeze(GATensor* tensor, int dim);
GA_API GATensor* ga_tensor_squeeze(GATensor* tensor, int dim);
GA_API GATensor* ga_tensor_transpose(GATensor* tensor, int dim0, int dim1);
GA_API GATensor* ga_tensor_permute(GATensor* tensor, const int* dims);
GA_API GATensor* ga_tensor_expand(GATensor* tensor, int ndim, const int64_t* shape);

GA_API GATensor* ga_tensor_get(GATensor* tensor, const int64_t* indices);
GA_API void ga_tensor_set(GATensor* tensor, const int64_t* indices, const void* value);
GA_API GATensor* ga_tensor_slice(GATensor* tensor, const int64_t* start, const int64_t* end, const int64_t* step);
GA_API GATensor* ga_tensor_select(GATensor* tensor, int dim, int64_t index);
GA_API GATensor* ga_tensor_index(GATensor* tensor, GATensor* indices);

GA_API bool ga_tensor_is_contiguous(const GATensor* tensor);
GA_API bool ga_tensor_is_view(const GATensor* tensor);
GA_API bool ga_tensor_same_shape(const GATensor* a, const GATensor* b);
GA_API bool ga_tensor_broadcastable(const GATensor* a, const GATensor* b);

GA_API void* ga_tensor_data(GATensor* tensor);
GA_API float* ga_tensor_data_f32(GATensor* tensor);
GA_API double* ga_tensor_data_f64(GATensor* tensor);
GA_API void ga_tensor_copy_to(const GATensor* tensor, void* buffer);
GA_API void ga_tensor_copy_from(GATensor* tensor, const void* buffer);
GA_API void ga_tensor_fill(GATensor* tensor, const void* value);

GA_API int ga_tensor_ndim(GATensor* tensor);
GA_API void ga_tensor_get_shape(GATensor* tensor, int64_t* shape);
GA_API void ga_tensor_set_requires_grad(GATensor* tensor, bool value);
GA_API void ga_tensor_fill(GATensor* tensor, const void* value);
GA_API void ga_tensor_copy_from(GATensor* tensor, const void* data);
GA_API void ga_tensor_copy_to(const GATensor* tensor, void* buffer);

GA_API GATensor* ga_tensor_get_grad(GATensor* tensor);
GA_API void ga_tensor_zero_grad(GATensor* tensor);
