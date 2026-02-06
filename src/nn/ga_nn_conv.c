/*
 * GreyML backend: ga nn conv.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include "greyarea/ga_nn.h"
#include "greyarea/ga_ops.h"

static void conv_forward_wrapper(void* self, GATensor* input, GATensor** output) {
    *output = ga_conv2d_forward((GAConv2D*)self, input);
}

GAConv2D* ga_conv2d_create(int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int groups, bool bias) {
    GAConv2D* conv = (GAConv2D*)calloc(1, sizeof(GAConv2D));
    conv->in_channels = in_channels;
    conv->out_channels = out_channels;
    conv->kernel_size[0] = kernel_size;
    conv->kernel_size[1] = kernel_size;
    conv->stride = stride;
    conv->padding = padding;
    conv->dilation = dilation;
    conv->groups = groups;
    
    int64_t weight_shape[4] = {out_channels, in_channels / groups, kernel_size, kernel_size};
    conv->weight = ga_tensor_empty(4, weight_shape, GA_FLOAT32);
    ga_tensor_set_requires_grad(conv->weight, true);
    ga_init_kaiming_uniform(conv->weight);

    if (bias) {
        int64_t bias_shape[1] = {out_channels};
        conv->bias = ga_tensor_empty(1, bias_shape, GA_FLOAT32);
        ga_tensor_set_requires_grad(conv->bias, true);
        ga_init_uniform(conv->bias, 0, 0.1f);
    }

    conv->base.n_params = bias ? 2 : 1;
    conv->base.parameters = (GATensor**)calloc(conv->base.n_params, sizeof(GATensor*));
    conv->base.parameters[0] = conv->weight;
    if (bias) conv->base.parameters[1] = conv->bias;
    conv->base.forward_fn = conv_forward_wrapper;
    
    return conv;
}

void ga_conv2d_free(GAConv2D* conv) {
    if (!conv) return;
    if (conv->weight) ga_tensor_release(conv->weight);
    if (conv->bias) ga_tensor_release(conv->bias);
    if (conv->base.parameters) free(conv->base.parameters);
    free(conv);
}

GATensor* ga_conv2d_forward(GAConv2D* conv, GATensor* input) {
    assert(conv && input);
    return ga_conv2d(input, conv->weight, conv->bias, conv->stride, conv->padding, conv->dilation, conv->groups);
}
