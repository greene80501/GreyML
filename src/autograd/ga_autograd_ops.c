/*
 * GreyML backend: ga autograd ops.
 *
 * Automatic differentiation primitives: graph construction, tracing, and backward pass kernels.
 */

#include "greyarea/ga_autograd.h"
#include "greyarea/ga_ops.h"
#include <math.h>
#include <string.h>

// Helper to unbroadcast grad to target shape
static GATensor* ga_autograd_unbroadcast(GATensor* grad, const int64_t* target_shape, int target_ndim) {
    if (grad->ndim == target_ndim) {
        int same = 1;
        for (int i = 0; i < target_ndim; i++) {
            if (grad->shape[i] != target_shape[i]) {
                same = 0;
                break;
            }
        }
        if (same) {
            ga_tensor_retain(grad);
            return grad;
        }
    }

    GATensor* out = ga_tensor_zeros(target_ndim, target_shape, GA_FLOAT32);
    int grad_ndim = grad->ndim;
    int dim_offset = target_ndim - grad_ndim;

    for (int64_t idx = 0; idx < out->size; idx++) {
        int64_t grad_idx = 0;
        int64_t rem = idx;
        for (int d = target_ndim - 1; d >= 0; d--) {
            int64_t stride = 1;
            for (int k = d + 1; k < target_ndim; k++) stride *= target_shape[k];
            int64_t coord = rem / stride;
            rem = rem % stride;
            int g_dim = d - dim_offset;
            int64_t g_coord = (g_dim >= 0) ? (grad->shape[g_dim] == 1 ? 0 : coord) : 0;
            if (g_dim >= 0) {
                int64_t g_stride = 1;
                for (int k = g_dim + 1; k < grad_ndim; k++) g_stride *= grad->shape[k];
                grad_idx += g_coord * g_stride;
            }
        }
        ((float*)out->data)[idx] = ((float*)grad->data)[grad_idx];
    }
    return out;
}

static void maybe_release(GATensor* t) {
    if (t) ga_tensor_release(t);
}

void ga_autograd_add(GANode* node, GATensor* grad_output) {
    if (node->inputs[0] && node->inputs[0]->requires_grad) {
        GATensor* g0 = ga_tensor_clone(grad_output);
        ga_accumulate_grad(node->inputs[0], g0);
        maybe_release(g0);
    }
    if (node->inputs[1] && node->inputs[1]->requires_grad) {
        GATensor* g1 = ga_tensor_clone(grad_output);
        ga_accumulate_grad(node->inputs[1], g1);
        maybe_release(g1);
    }
}

void ga_autograd_sub(GANode* node, GATensor* grad_output) {
    if (node->inputs[0] && node->inputs[0]->requires_grad) {
        GATensor* g0 = ga_tensor_clone(grad_output);
        ga_accumulate_grad(node->inputs[0], g0);
        maybe_release(g0);
    }
    if (node->inputs[1] && node->inputs[1]->requires_grad) {
        GATensor* neg = ga_neg(ga_tensor_clone(grad_output));
        ga_accumulate_grad(node->inputs[1], neg);
        maybe_release(neg);
    }
}

void ga_autograd_mul(GANode* node, GATensor* grad_output) {
    if (node->inputs[0] && node->inputs[0]->requires_grad) {
        GATensor* g0 = ga_mul(ga_tensor_clone(grad_output), ga_tensor_clone(node->inputs[1]));
        ga_accumulate_grad(node->inputs[0], g0);
        maybe_release(g0);
    }
    if (node->inputs[1] && node->inputs[1]->requires_grad) {
        GATensor* g1 = ga_mul(ga_tensor_clone(grad_output), ga_tensor_clone(node->inputs[0]));
        ga_accumulate_grad(node->inputs[1], g1);
        maybe_release(g1);
    }
}

void ga_autograd_div(GANode* node, GATensor* grad_output) {
    if (node->inputs[0] && node->inputs[0]->requires_grad) {
        GATensor* g0 = ga_div(ga_tensor_clone(grad_output), ga_tensor_clone(node->inputs[1]));
        ga_accumulate_grad(node->inputs[0], g0);
        maybe_release(g0);
    }
    if (node->inputs[1] && node->inputs[1]->requires_grad) {
        GATensor* b = ga_tensor_clone(node->inputs[1]);
        GATensor* b2 = ga_mul(ga_tensor_clone(b), b);
        GATensor* num = ga_mul(ga_tensor_clone(grad_output), ga_tensor_clone(node->inputs[0]));
        GATensor* frac = ga_div(num, b2);
        GATensor* neg = ga_neg(frac);
        ga_accumulate_grad(node->inputs[1], neg);
        maybe_release(neg);
    }
}

void ga_autograd_matmul(GANode* node, GATensor* grad_output) {
    GATensor* a = node->inputs[0];
    GATensor* b = node->inputs[1];
    if (a && a->requires_grad) {
        GATensor* bT = ga_transpose(ga_tensor_clone(b));
        GATensor* g0 = ga_matmul(ga_tensor_clone(grad_output), bT);
        ga_accumulate_grad(a, g0);
        maybe_release(g0);
    }
    if (b && b->requires_grad) {
        GATensor* aT = ga_transpose(ga_tensor_clone(a));
        GATensor* g1 = ga_matmul(aT, ga_tensor_clone(grad_output));
        ga_accumulate_grad(b, g1);
        maybe_release(g1);
    }
}

void ga_autograd_relu(GANode* node, GATensor* grad_output) {
    GATensor* inp = node->inputs[0];
    if (inp && inp->requires_grad) {
        GATensor* mask = ga_tensor_clone(inp);
        float* data = (float*)mask->data;
        for (int64_t i = 0; i < mask->size; i++) data[i] = data[i] > 0 ? 1.0f : 0.0f;
        GATensor* g = ga_mul(mask, ga_tensor_clone(grad_output));
        ga_accumulate_grad(inp, g);
        maybe_release(g);
    }
}

void ga_autograd_sigmoid(GANode* node, GATensor* grad_output) {
    GATensor* out = node->saved_tensors[0]; // saved output
    if (node->inputs[0] && node->inputs[0]->requires_grad) {
        GATensor* one_minus = ga_sub(ga_tensor_clone(out), ga_tensor_full(out->ndim, out->shape, out->dtype, &(float){1.0f}));
        GATensor* grad = ga_mul(ga_tensor_clone(out), one_minus);
        grad = ga_mul(grad, ga_tensor_clone(grad_output));
        ga_accumulate_grad(node->inputs[0], grad);
        maybe_release(grad);
    }
}

void ga_autograd_log(GANode* node, GATensor* grad_output) {
    GATensor* inp = node->inputs[0];
    if (inp && inp->requires_grad) {
        GATensor* g = ga_div(ga_tensor_clone(grad_output), ga_tensor_clone(inp));
        ga_accumulate_grad(inp, g);
        maybe_release(g);
    }
}

void ga_autograd_sum(GANode* node, GATensor* grad_output) {
    GATensor* inp = node->inputs[0];
    if (!inp || !inp->requires_grad) return;
    GATensor* grad_expanded = ga_autograd_unbroadcast(grad_output, inp->shape, inp->ndim);
    ga_accumulate_grad(inp, grad_expanded);
    maybe_release(grad_expanded);
}

void ga_autograd_mean(GANode* node, GATensor* grad_output) {
    GATensor* inp = node->inputs[0];
    if (!inp || !inp->requires_grad) return;
    float scale = node->saved_scalars[0];
    GATensor* grad_scaled = ga_mul_scalar(ga_tensor_clone(grad_output), scale);
    GATensor* grad_expanded = ga_autograd_unbroadcast(grad_scaled, inp->shape, inp->ndim);
    ga_accumulate_grad(inp, grad_expanded);
    maybe_release(grad_scaled);
    maybe_release(grad_expanded);
}

void ga_autograd_softmax(GANode* node, GATensor* grad_output) {
    GATensor* inp = node->inputs[0];
    if (!inp || !inp->requires_grad) return;
    GATensor* out = node->saved_tensors[0];
    int dim = node->saved_ints[0];
    // grad_input = softmax * (grad - sum(grad*softmax))
    GATensor* prod = ga_mul(ga_tensor_clone(grad_output), ga_tensor_clone(out));
    GATensor* summed = ga_sum(prod, dim, true);
    GATensor* sub = ga_sub(ga_tensor_clone(grad_output), summed);
    GATensor* grad_input = ga_mul(ga_tensor_clone(out), sub);
    ga_accumulate_grad(inp, grad_input);
    maybe_release(prod);
    maybe_release(summed);
    maybe_release(sub);
    maybe_release(grad_input);
}

void ga_autograd_pow(GANode* node, GATensor* grad_output) {
    GATensor* a = node->inputs[0];
    GATensor* b = node->inputs[1];
    GATensor* out = node->saved_tensors[0];
    float eps = 1e-12f;

    GATensor* grad_a_full = NULL;
    GATensor* grad_b_full = NULL;
    if (a && a->requires_grad) grad_a_full = ga_tensor_zeros(out->ndim, out->shape, GA_FLOAT32);
    if (b && b->requires_grad) grad_b_full = ga_tensor_zeros(out->ndim, out->shape, GA_FLOAT32);

    float* go = (float*)grad_output->data;
    float* av = (float*)a->data;
    float* bv = (float*)b->data;
    float* ov = (float*)out->data;

    for (int64_t i = 0; i < out->size; i++) {
        int64_t a_idx = (a->size == 1) ? 0 : i % a->size;
        int64_t b_idx = (b->size == 1) ? 0 : i % b->size;
        float g = go[i];
        if (grad_a_full) {
            ((float*)grad_a_full->data)[i] = g * bv[b_idx] * powf(av[a_idx], bv[b_idx] - 1.0f);
        }
        if (grad_b_full) {
            float safe_a = fabsf(av[a_idx]) + eps;
            ((float*)grad_b_full->data)[i] = g * ov[i] * logf(safe_a);
        }
    }

    if (grad_a_full) {
        GATensor* g_unb = ga_autograd_unbroadcast(grad_a_full, a->shape, a->ndim);
        ga_accumulate_grad(a, g_unb);
        maybe_release(g_unb);
        maybe_release(grad_a_full);
    }
    if (grad_b_full) {
        GATensor* g_unb = ga_autograd_unbroadcast(grad_b_full, b->shape, b->ndim);
        ga_accumulate_grad(b, g_unb);
        maybe_release(g_unb);
        maybe_release(grad_b_full);
    }
}

void ga_autograd_conv2d(GANode* node, GATensor* grad_output) {
    GATensor* input = node->inputs[0];
    GATensor* weight = node->inputs[1];
    GATensor* bias = node->inputs[2];
    if (!input || !weight) return;

    int stride = node->saved_ints[0];
    int padding = node->saved_ints[1];
    int dilation = node->saved_ints[2];
    int groups = node->saved_ints[3];
    (void)dilation; // not used in naive backward
    (void)groups;   // groups=1 assumed

    int64_t N = input->shape[0];
    int64_t C = input->shape[1];
    int64_t H = input->shape[2];
    int64_t W = input->shape[3];
    int64_t O = weight->shape[0];
    int64_t kH = weight->shape[2];
    int64_t kW = weight->shape[3];
    int64_t outH = (H + 2 * padding - kH) / stride + 1;
    int64_t outW = (W + 2 * padding - kW) / stride + 1;

    float* go = (float*)grad_output->data;

    GATensor* grad_input = NULL;
    GATensor* grad_weight = NULL;
    GATensor* grad_bias = NULL;
    if (input->requires_grad) grad_input = ga_tensor_zeros(4, input->shape, GA_FLOAT32);
    if (weight->requires_grad) grad_weight = ga_tensor_zeros(4, weight->shape, GA_FLOAT32);
    if (bias && bias->requires_grad) grad_bias = ga_tensor_zeros(1, bias->shape, GA_FLOAT32);

    for (int64_t n = 0; n < N; n++) {
        for (int64_t oc = 0; oc < O; oc++) {
            for (int64_t oh = 0; oh < outH; oh++) {
                for (int64_t ow = 0; ow < outW; ow++) {
                    float g = go[((n * O + oc) * outH + oh) * outW + ow];
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;
                    if (grad_bias) {
                        ((float*)grad_bias->data)[oc] += g;
                    }
                    for (int64_t ic = 0; ic < C; ic++) {
                        for (int64_t kh = 0; kh < kH; kh++) {
                            for (int64_t kw = 0; kw < kW; kw++) {
                                int64_t h_in = h_start + kh;
                                int64_t w_in = w_start + kw;
                                if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;
                                float in_val = ((float*)input->data)[((n * C + ic) * H + h_in) * W + w_in];
                                float w_val = ((float*)weight->data)[((oc * C + ic) * kH + kh) * kW + kw];
                                if (grad_input) {
                                    ((float*)grad_input->data)[((n * C + ic) * H + h_in) * W + w_in] += w_val * g;
                                }
                                if (grad_weight) {
                                    ((float*)grad_weight->data)[((oc * C + ic) * kH + kh) * kW + kw] += in_val * g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (grad_input) {
        ga_accumulate_grad(input, grad_input);
        maybe_release(grad_input);
    }
    if (grad_weight) {
        ga_accumulate_grad(weight, grad_weight);
        maybe_release(grad_weight);
    }
    if (grad_bias) {
        ga_accumulate_grad(bias, grad_bias);
        maybe_release(grad_bias);
    }
}

void ga_autograd_max_pool2d(GANode* node, GATensor* grad_output) {
    GATensor* input = node->inputs[0];
    if (!input || !input->requires_grad) return;
    int kernel = node->saved_ints[0];
    int stride = node->saved_ints[1];
    int padding = node->saved_ints[2];
    int dilation = node->saved_ints[3];
    (void)dilation;

    int64_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int64_t outH = (H + 2 * padding - kernel) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel) / stride + 1;
    GATensor* grad_input = ga_tensor_zeros(4, input->shape, GA_FLOAT32);

    float* go = (float*)grad_output->data;
    float* gi = (float*)grad_input->data;
    float* in = (float*)input->data;

    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            for (int64_t oh = 0; oh < outH; oh++) {
                for (int64_t ow = 0; ow < outW; ow++) {
                    float max_val = -INFINITY;
                    int64_t max_h = -1, max_w = -1;
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;
                    for (int64_t kh = 0; kh < kernel; kh++) {
                        for (int64_t kw = 0; kw < kernel; kw++) {
                            int64_t h_in = h_start + kh;
                            int64_t w_in = w_start + kw;
                            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;
                            float v = in[((n * C + c) * H + h_in) * W + w_in];
                            if (v > max_val) {
                                max_val = v;
                                max_h = h_in;
                                max_w = w_in;
                            }
                        }
                    }
                    if (max_h >= 0 && max_w >= 0) {
                        gi[((n * C + c) * H + max_h) * W + max_w] += go[((n * C + c) * outH + oh) * outW + ow];
                    }
                }
            }
        }
    }
    ga_accumulate_grad(input, grad_input);
    maybe_release(grad_input);
}

void ga_autograd_avg_pool2d(GANode* node, GATensor* grad_output) {
    GATensor* input = node->inputs[0];
    if (!input || !input->requires_grad) return;
    int kernel = node->saved_ints[0];
    int stride = node->saved_ints[1];
    int padding = node->saved_ints[2];

    int64_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    int64_t outH = (H + 2 * padding - kernel) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel) / stride + 1;
    GATensor* grad_input = ga_tensor_zeros(4, input->shape, GA_FLOAT32);

    float* go = (float*)grad_output->data;
    float* gi = (float*)grad_input->data;

    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            for (int64_t oh = 0; oh < outH; oh++) {
                for (int64_t ow = 0; ow < outW; ow++) {
                    int64_t h_start = oh * stride - padding;
                    int64_t w_start = ow * stride - padding;
                    int count = 0;
                    for (int64_t kh = 0; kh < kernel; kh++) {
                        for (int64_t kw = 0; kw < kernel; kw++) {
                            int64_t h_in = h_start + kh;
                            int64_t w_in = w_start + kw;
                            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;
                            count++;
                        }
                    }
                    if (count == 0) count = 1;
                    float gshare = go[((n * C + c) * outH + oh) * outW + ow] / (float)count;
                    for (int64_t kh = 0; kh < kernel; kh++) {
                        for (int64_t kw = 0; kw < kernel; kw++) {
                            int64_t h_in = h_start + kh;
                            int64_t w_in = w_start + kw;
                            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) continue;
                            gi[((n * C + c) * H + h_in) * W + w_in] += gshare;
                        }
                    }
                }
            }
        }
    }
    ga_accumulate_grad(input, grad_input);
    maybe_release(grad_input);
}

void ga_autograd_adaptive_avg_pool2d(GANode* node, GATensor* grad_output) {
    GATensor* input = node->inputs[0];
    if (!input || !input->requires_grad) return;
    int outH = node->saved_ints[0];
    int outW = node->saved_ints[1];

    int64_t N = input->shape[0], C = input->shape[1], H = input->shape[2], W = input->shape[3];
    GATensor* grad_input = ga_tensor_zeros(4, input->shape, GA_FLOAT32);
    float* gi = (float*)grad_input->data;
    float* go = (float*)grad_output->data;

    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int h_start = (oh * H) / outH;
                    int h_end = ((oh + 1) * H) / outH;
                    int w_start = (ow * W) / outW;
                    int w_end = ((ow + 1) * W) / outW;
                    int count = (h_end - h_start) * (w_end - w_start);
                    if (count == 0) count = 1;
                    float gshare = go[((n * C + c) * outH + oh) * outW + ow] / (float)count;
                    for (int h = h_start; h < h_end; h++) {
                        for (int w = w_start; w < w_end; w++) {
                            gi[((n * C + c) * H + h) * W + w] += gshare;
                        }
                    }
                }
            }
        }
    }
    ga_accumulate_grad(input, grad_input);
    maybe_release(grad_input);
}

void ga_autograd_reshape(GANode* node, GATensor* grad_output) {
    GATensor* inp = node->inputs[0];
    if (!inp || !inp->requires_grad) return;
    int orig_ndim = node->saved_ints[0];
    int64_t shape[GA_MAX_DIMS];
    for (int i = 0; i < orig_ndim; i++) shape[i] = (int64_t)node->saved_ints[i + 1];
    GATensor* g = ga_tensor_reshape(grad_output, orig_ndim, shape);
    ga_accumulate_grad(inp, g);
    maybe_release(g);
}

void ga_autograd_transpose(GANode* node, GATensor* grad_output) {
    GATensor* inp = node->inputs[0];
    if (!inp || !inp->requires_grad) return;
    GATensor* g = ga_transpose(ga_tensor_clone(grad_output));
    ga_accumulate_grad(inp, g);
    maybe_release(g);
}

void ga_autograd_flatten(GANode* node, GATensor* grad_output) {
    GATensor* inp = node->inputs[0];
    if (!inp || !inp->requires_grad) return;
    int orig_ndim = node->saved_ints[0];
    int64_t shape[GA_MAX_DIMS];
    for (int i = 0; i < orig_ndim; i++) shape[i] = (int64_t)node->saved_ints[i + 1];
    GATensor* g = ga_tensor_reshape(grad_output, orig_ndim, shape);
    ga_accumulate_grad(inp, g);
    maybe_release(g);
}

void ga_autograd_nll_loss(GANode* node, GATensor* grad_output) {
    GATensor* log_probs = node->inputs[0];
    GATensor* target = node->saved_tensors[0];
    if (!log_probs || !log_probs->requires_grad || !target) return;
    int64_t batch = log_probs->shape[0];
    int64_t classes = log_probs->shape[1];
    GATensor* grad_log_probs = ga_tensor_zeros(log_probs->ndim, log_probs->shape, GA_FLOAT32);
    float* g_lp = (float*)grad_log_probs->data;
    float* go = (float*)grad_output->data;
    int64_t* tgt = (int64_t*)target->data;
    for (int64_t i = 0; i < batch; i++) {
        int64_t idx = tgt[i];
        if (idx < 0 || idx >= classes) continue;
        g_lp[i * classes + idx] = -go[i];
    }
    ga_accumulate_grad(log_probs, grad_log_probs);
    maybe_release(grad_log_probs);
}

// Register backward functions for each op
void ga_autograd_register_ops(void) {
    // Functions are referenced directly by ops when creating GANode
}
