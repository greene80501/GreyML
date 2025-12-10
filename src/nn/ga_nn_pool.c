/*
 * GreyML backend: ga nn pool.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#include <stdlib.h>
#include "greyarea/ga_nn.h"
#include "greyarea/ga_ops.h"

static void pool_forward_wrapper(void* self, GATensor* input, GATensor** output) {
    GAPool2D* pool = (GAPool2D*)self;
    if (pool->is_max) {
        *output = ga_max_pool2d(input, pool->kernel_size, pool->stride, pool->padding, pool->dilation);
    } else {
        *output = ga_avg_pool2d(input, pool->kernel_size, pool->stride, pool->padding);
    }
}

GAPool2D* ga_maxpool2d_create(int kernel_size, int stride, int padding, int dilation) {
    GAPool2D* p = (GAPool2D*)calloc(1, sizeof(GAPool2D));
    p->kernel_size = kernel_size;
    p->stride = stride;
    p->padding = padding;
    p->dilation = dilation;
    p->is_max = true;
    p->base.forward_fn = pool_forward_wrapper;
    return p;
}

GAPool2D* ga_avgpool2d_create(int kernel_size, int stride, int padding) {
    GAPool2D* p = (GAPool2D*)calloc(1, sizeof(GAPool2D));
    p->kernel_size = kernel_size;
    p->stride = stride;
    p->padding = padding;
    p->dilation = 1;
    p->is_max = false;
    p->base.forward_fn = pool_forward_wrapper;
    return p;
}

void ga_pool2d_free(GAPool2D* pool) {
    if (!pool) return;
    free(pool);
}

GATensor* ga_pool2d_forward(GAPool2D* pool, GATensor* input) {
    if (!pool) return NULL;
    GATensor* out = NULL;
    pool->base.forward_fn(pool, input, &out);
    return out;
}
