/*
 * GreyML backend: ga nn init.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#include "greyarea/ga_nn.h"
#include "greyarea/ga_random.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void ga_init_uniform(GATensor* tensor, float low, float high) {
    ga_tensor_rand_(tensor);
    float* data = (float*)tensor->data;
    float range = high - low;
    for (int64_t i = 0; i < tensor->size; i++) {
        data[i] = data[i] * range + low;
    }
}

void ga_init_normal(GATensor* tensor, float mean, float std) {
    float* data = (float*)tensor->data;
    for (int64_t i = 0; i < tensor->size; i += 2) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * (float)M_PI * u2);
        data[i] = z0 * std + mean;
        if (i + 1 < tensor->size) {
            data[i + 1] = z1 * std + mean;
        }
    }
}

void ga_init_xavier_uniform(GATensor* tensor) {
    if (tensor->ndim < 2) return;
    int64_t fan_in = tensor->shape[tensor->ndim - 1];
    int64_t fan_out = tensor->shape[0];
    float bound = sqrtf(6.0f / (float)(fan_in + fan_out));
    ga_init_uniform(tensor, -bound, bound);
}

void ga_init_xavier_normal(GATensor* tensor) {
    if (tensor->ndim < 2) return;
    int64_t fan_in = tensor->shape[tensor->ndim - 1];
    int64_t fan_out = tensor->shape[0];
    float std = sqrtf(2.0f / (float)(fan_in + fan_out));
    ga_init_normal(tensor, 0.0f, std);
}

void ga_init_kaiming_uniform(GATensor* tensor) {
    if (tensor->ndim < 1) return;
    int64_t fan_in = tensor->shape[tensor->ndim - 1];
    float bound = sqrtf(1.0f / (float)fan_in);
    ga_init_uniform(tensor, -bound, bound);
}

void ga_init_kaiming_normal(GATensor* tensor) {
    if (tensor->ndim < 1) return;
    int64_t fan_in = tensor->shape[tensor->ndim - 1];
    float std = sqrtf(2.0f / (float)fan_in);
    ga_init_normal(tensor, 0.0f, std);
}