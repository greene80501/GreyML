/*
 * GreyML backend: ga nn linear.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#include "greyarea/ga_nn.h"
#include <math.h>
#include <stdlib.h>

static void linear_forward_wrapper(void* self, GATensor* input, GATensor** output) {
    *output = ga_linear_forward((GALinear*)self, input);
}

GALinear* ga_linear_create(int in_features, int out_features, bool use_bias) {
    GALinear* linear = (GALinear*)calloc(1, sizeof(GALinear));
    linear->in_features = in_features;
    linear->out_features = out_features;
    
    int64_t weight_shape[2] = {out_features, in_features};
    linear->weight = ga_tensor_empty(2, weight_shape, GA_FLOAT32);
    ga_tensor_set_requires_grad(linear->weight, true);
    ga_init_kaiming_uniform(linear->weight);
    
    if (use_bias) {
        int64_t bias_shape[1] = {out_features};
        linear->bias = ga_tensor_empty(1, bias_shape, GA_FLOAT32);
        ga_tensor_set_requires_grad(linear->bias, true);
        ga_init_uniform(linear->bias, 0, 0.1f);
    }

    // Wire parameters into module base for serialization/optimizers
    size_t n_params = use_bias ? 2 : 1;
    linear->base.n_params = n_params;
    linear->base.parameters = (GATensor**)calloc(n_params, sizeof(GATensor*));
    linear->base.parameters[0] = linear->weight;
    if (use_bias) linear->base.parameters[1] = linear->bias;
    linear->base.forward_fn = linear_forward_wrapper;
    
    return linear;
}

void ga_linear_free(GALinear* linear) {
    if (!linear) return;
    if (linear->weight) ga_tensor_release(linear->weight);
    if (linear->bias) ga_tensor_release(linear->bias);
    if (linear->base.parameters) free(linear->base.parameters);
    free(linear);
}

GATensor* ga_linear_forward(GALinear* linear, GATensor* input) {
    GATensor* weight_t = ga_transpose(linear->weight);  // FIXED: Was ga_tensor_transpose
    GATensor* output = ga_matmul(input, weight_t);
    ga_tensor_release(weight_t);
    
    if (linear->bias) {
        GATensor* bias_expanded = ga_tensor_unsqueeze(linear->bias, 0);
        GATensor* temp = ga_add(output, bias_expanded);
        ga_tensor_release(output);
        ga_tensor_release(bias_expanded);
        return temp;
    }
    return output;
}
