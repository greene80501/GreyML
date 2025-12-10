/*
 * GreyML backend: ga ops conv.
 *
 * Numerical kernels for activation, unary/binary math, convolution, matmul, reduction, and transform operations.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <string.h>
#include "greyarea/ga_ops.h"
#include "greyarea/ga_common.h"
#include "greyarea/ga_autograd.h"

// Naive 2D convolution (NCHW). Supports stride/padding, dilation=1, groups=1 only.
GATensor* ga_conv2d(GATensor* input, GATensor* weight, GATensor* bias, int stride, int padding, int dilation, int groups) {
    assert(input && weight);
    assert(input->dtype == GA_FLOAT32 && weight->dtype == GA_FLOAT32);
    if (dilation != 1 || groups != 1) {
        ga_errno = GA_ERR_NOT_IMPLEMENTED;
        return NULL;
    }
    assert(input->ndim == 4);   // [N,C,H,W]
    assert(weight->ndim == 4);  // [O,I,kH,kW]
    int64_t N = input->shape[0];
    int64_t C = input->shape[1];
    int64_t H = input->shape[2];
    int64_t W = input->shape[3];
    int64_t O = weight->shape[0];
    int64_t kH = weight->shape[2];
    int64_t kW = weight->shape[3];

    int64_t outH = (H + 2 * padding - kH) / stride + 1;
    int64_t outW = (W + 2 * padding - kW) / stride + 1;
    int64_t out_shape[4] = {N, O, outH, outW};
    GATensor* out = ga_tensor_empty(4, out_shape, GA_FLOAT32);
    if (!out) return NULL;

    float* in = (float*)input->data;
    float* w = (float*)weight->data;
    float* b = bias ? (float*)bias->data : NULL;
    float* o = (float*)out->data;

    for (int64_t n = 0; n < N; n++) {
        for (int64_t oc = 0; oc < O; oc++) {
            for (int64_t oh = 0; oh < outH; oh++) {
                for (int64_t ow = 0; ow < outW; ow++) {
                    float sum = b ? b[oc] : 0.0f;
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;
                    for (int64_t ic = 0; ic < C; ic++) {
                        for (int64_t kh = 0; kh < kH; kh++) {
                            for (int64_t kw = 0; kw < kW; kw++) {
                                int64_t h_in = h_start + kh;
                                int64_t w_in = w_start + kw;
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    int64_t in_idx = ((n * C + ic) * H + h_in) * W + w_in;
                                    int64_t w_idx = ((oc * C + ic) * kH + kh) * kW + kw;
                                    sum += in[in_idx] * w[w_idx];
                                }
                            }
                        }
                    }
                    int64_t out_idx = ((n * O + oc) * outH + oh) * outW + ow;
                    o[out_idx] = sum;
                }
            }
        }
    }
    if (ga_is_grad_enabled() && ((input && input->requires_grad) || (weight && weight->requires_grad) || (bias && bias->requires_grad))) {
        GANode* node = ga_node_create("conv2d", 3, 0, 0, 4);
        if (node) {
            node->inputs[0] = input;
            node->inputs[1] = weight;
            node->inputs[2] = bias;
            ga_tensor_retain(input);
            ga_tensor_retain(weight);
            if (bias) ga_tensor_retain(bias);
            node->saved_ints[0] = stride;
            node->saved_ints[1] = padding;
            node->saved_ints[2] = dilation;
            node->saved_ints[3] = groups;
            node->backward_fn = ga_autograd_conv2d;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    return out;
}

static GATensor* pool2d(GATensor* input, int kernel, int stride, int padding, bool is_max) {
    assert(input && input->dtype == GA_FLOAT32);
    assert(input->ndim == 4); // [N,C,H,W]
    int64_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int64_t outH = (H + 2 * padding - kernel) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel) / stride + 1;
    int64_t out_shape[4] = {N, C, outH, outW};
    GATensor* out = ga_tensor_empty(4, out_shape, GA_FLOAT32);
    if (!out) return NULL;
    float* in = (float*)input->data;
    float* o = (float*)out->data;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            for (int64_t oh = 0; oh < outH; oh++) {
                for (int64_t ow = 0; ow < outW; ow++) {
                    float acc = is_max ? -INFINITY : 0.0f;
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;
                    for (int64_t kh = 0; kh < kernel; kh++) {
                        for (int64_t kw = 0; kw < kernel; kw++) {
                            int64_t h_in = h_start + kh;
                            int64_t w_in = w_start + kw;
                            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                int64_t idx = ((n * C + c) * H + h_in) * W + w_in;
                                float v = in[idx];
                                if (is_max) {
                                    if (v > acc) acc = v;
                                } else {
                                    acc += v;
                                }
                            }
                        }
                    }
                    if (!is_max) acc /= (float)(kernel * kernel);
                    int64_t out_idx = ((n * C + c) * outH + oh) * outW + ow;
                    o[out_idx] = acc;
                }
            }
        }
    }
    return out;
}

GATensor* ga_max_pool2d(GATensor* input, int kernel_size, int stride, int padding, int dilation) {
    (void)dilation; // unused in naive implementation
    GATensor* out = pool2d(input, kernel_size, stride, padding, true);
    if (ga_is_grad_enabled() && input && input->requires_grad) {
        GANode* node = ga_node_create("max_pool2d", 1, 0, 0, 4);
        if (node) {
            node->inputs[0] = input;
            ga_tensor_retain(input);
            node->saved_ints[0] = kernel_size;
            node->saved_ints[1] = stride;
            node->saved_ints[2] = padding;
            node->saved_ints[3] = dilation;
            node->backward_fn = ga_autograd_max_pool2d;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    return out;
}

GATensor* ga_avg_pool2d(GATensor* input, int kernel_size, int stride, int padding) {
    GATensor* out = pool2d(input, kernel_size, stride, padding, false);
    if (ga_is_grad_enabled() && input && input->requires_grad) {
        GANode* node = ga_node_create("avg_pool2d", 1, 0, 0, 3);
        if (node) {
            node->inputs[0] = input;
            ga_tensor_retain(input);
            node->saved_ints[0] = kernel_size;
            node->saved_ints[1] = stride;
            node->saved_ints[2] = padding;
            node->backward_fn = ga_autograd_avg_pool2d;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    return out;
}

GATensor* ga_adaptive_avg_pool2d(GATensor* input, int output_height, int output_width) {
    assert(input && input->dtype == GA_FLOAT32);
    assert(input->ndim == 4);
    int64_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int64_t out_shape[4] = {N, C, output_height, output_width};
    GATensor* out = ga_tensor_empty(4, out_shape, GA_FLOAT32);
    if (!out) return NULL;
    float* in = (float*)input->data;
    float* o = (float*)out->data;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    int h_start = (oh * H) / output_height;
                    int h_end = ((oh + 1) * H) / output_height;
                    int w_start = (ow * W) / output_width;
                    int w_end = ((ow + 1) * W) / output_width;
                    float acc = 0.0f;
                    int count = 0;
                    for (int h = h_start; h < h_end; h++) {
                        for (int w = w_start; w < w_end; w++) {
                            int64_t idx = ((n * C + c) * H + h) * W + w;
                            acc += in[idx];
                            count++;
                        }
                    }
                    if (count == 0) count = 1;
                    int64_t out_idx = ((n * C + c) * output_height + oh) * output_width + ow;
                    o[out_idx] = acc / (float)count;
                }
            }
        }
    }
    if (ga_is_grad_enabled() && input && input->requires_grad) {
        GANode* node = ga_node_create("adaptive_avg_pool2d", 1, 0, 0, 2);
        if (node) {
            node->inputs[0] = input;
            ga_tensor_retain(input);
            node->saved_ints[0] = output_height;
            node->saved_ints[1] = output_width;
            node->backward_fn = ga_autograd_adaptive_avg_pool2d;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    return out;
}
