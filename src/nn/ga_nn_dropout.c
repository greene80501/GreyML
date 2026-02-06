/*
 * GreyML backend: ga nn dropout.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#include <stdlib.h>
#include "greyarea/ga_nn.h"
#include "greyarea/ga_random.h"

static void dropout_forward_wrapper(void* self, GATensor* input, GATensor** output) {
    GADropout* d = (GADropout*)self;
    if (!d->base.training || d->p <= 0.0f) {
        ga_tensor_retain(input);
        *output = input;
        return;
    }
    GATensor* mask = ga_tensor_empty(input->ndim, input->shape, GA_FLOAT32);
    float scale = 1.0f / (1.0f - d->p);
    float* m = (float*)mask->data;
    for (int64_t i = 0; i < input->size; i++) {
        float r = ga_random_float();
        m[i] = (r >= d->p) ? scale : 0.0f;
    }
    GATensor* scaled = ga_mul(input, mask);
    ga_tensor_release(mask);
    *output = scaled;
}

GADropout* ga_dropout_create(float p) {
    GADropout* d = (GADropout*)calloc(1, sizeof(GADropout));
    d->p = p;
    d->base.training = true;
    d->base.forward_fn = dropout_forward_wrapper;
    return d;
}

void ga_dropout_free(GADropout* drop) {
    if (!drop) return;
    free(drop);
}

GATensor* ga_dropout_forward(GADropout* drop, GATensor* input) {
    if (!drop) return NULL;
    GATensor* out = NULL;
    drop->base.forward_fn(drop, input, &out);
    return out;
}
