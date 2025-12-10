/*
 * GreyML C API header: ga ops.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"

GA_API GATensor* ga_add(GATensor* a, GATensor* b);
GA_API GATensor* ga_sub(GATensor* a, GATensor* b);
GA_API GATensor* ga_mul(GATensor* a, GATensor* b);
GA_API GATensor* ga_div(GATensor* a, GATensor* b);
GA_API GATensor* ga_pow(GATensor* a, GATensor* b);
GA_API GATensor* ga_add_scalar(GATensor* a, float scalar);
GA_API GATensor* ga_mul_scalar(GATensor* a, float scalar);

GA_API GATensor* ga_neg(GATensor* a);
GA_API GATensor* ga_exp(GATensor* a);
GA_API GATensor* ga_log(GATensor* a);
GA_API GATensor* ga_sqrt(GATensor* a);
GA_API GATensor* ga_abs(GATensor* a);
GA_API GATensor* ga_square(GATensor* a);
GA_API GATensor* ga_sin(GATensor* a);
GA_API GATensor* ga_cos(GATensor* a);
GA_API GATensor* ga_tanh(GATensor* a);
GA_API GATensor* ga_sigmoid(GATensor* a);
GA_API GATensor* ga_relu(GATensor* a);
GA_API GATensor* ga_leaky_relu(GATensor* a, float slope);
GA_API GATensor* ga_gelu(GATensor* a);
GA_API GATensor* ga_silu(GATensor* a);
GA_API GATensor* ga_softmax(GATensor* a, int dim);
GA_API GATensor* ga_log_softmax(GATensor* a, int dim);

GA_API GATensor* ga_sum(GATensor* a, int dim, bool keepdim);
GA_API GATensor* ga_mean(GATensor* a, int dim, bool keepdim);
GA_API GATensor* ga_var(GATensor* a, int dim, bool keepdim, bool unbiased);
GA_API GATensor* ga_max(GATensor* a, int dim, bool keepdim);
GA_API GATensor* ga_min(GATensor* a, int dim, bool keepdim);
GA_API GATensor* ga_argmax(GATensor* a, int dim);

GA_API GATensor* ga_matmul(GATensor* a, GATensor* b);
GA_API GATensor* ga_bmm(GATensor* a, GATensor* b);
GA_API GATensor* ga_dot(GATensor* a, GATensor* b);
GA_API GATensor* ga_outer(GATensor* a, GATensor* b);

GA_API GATensor* ga_conv1d(GATensor* input, GATensor* weight, GATensor* bias, int stride, int padding, int dilation, int groups);
GA_API GATensor* ga_conv2d(GATensor* input, GATensor* weight, GATensor* bias, int stride, int padding, int dilation, int groups);
GA_API GATensor* ga_conv_transpose2d(GATensor* input, GATensor* weight, GATensor* bias, int stride, int padding, int output_padding, int dilation, int groups);

GA_API GATensor* ga_max_pool2d(GATensor* input, int kernel_size, int stride, int padding, int dilation);
GA_API GATensor* ga_avg_pool2d(GATensor* input, int kernel_size, int stride, int padding);
GA_API GATensor* ga_adaptive_avg_pool2d(GATensor* input, int output_height, int output_width);

GA_API GATensor* ga_cat(GATensor** tensors, int num_tensors, int dim);
GA_API GATensor* ga_stack(GATensor** tensors, int num_tensors, int dim);
GA_API void ga_split(GATensor* tensor, int chunks, int dim, GATensor** outputs);
GA_API GATensor* ga_gather(GATensor* tensor, int dim, GATensor* index);
GA_API GATensor* ga_scatter(GATensor* tensor, int dim, GATensor* index, GATensor* src);
GA_API GATensor* ga_where(GATensor* cond, GATensor* x, GATensor* y);

GA_API GATensor* ga_transpose(GATensor* a);
GA_API GATensor* ga_reshape(GATensor* a, int ndim, const int64_t* shape);
GA_API GATensor* ga_flatten(GATensor* a);