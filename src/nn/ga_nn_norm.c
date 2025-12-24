/*
 * GreyML backend: ga nn norm.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "greyarea/ga_nn.h"

static void batchnorm_forward_wrapper(void* self, GATensor* input, GATensor** output) {
    GABatchNorm2D* bn = (GABatchNorm2D*)self;
    int64_t N = input->shape[0];
    int64_t C = input->shape[1];
    int64_t H = input->shape[2];
    int64_t W = input->shape[3];

    GATensor* out = ga_tensor_empty(input->ndim, input->shape, input->dtype);
    float* x = (float*)input->data;
    float* y = (float*)out->data;
    float* running_mean = (float*)bn->running_mean->data;
    float* running_var = (float*)bn->running_var->data;
    float* gamma = (float*)bn->gamma->data;
    float* beta = (float*)bn->beta->data;

    const float momentum = bn->momentum;
    const float eps = bn->eps;
    int64_t spatial = H * W;
    int64_t batch_elems = N * spatial;

    if (bn->base.training) {
        for (int64_t c = 0; c < C; c++) {
            float mean = 0.0f;
            for (int64_t n = 0; n < N; n++) {
                for (int64_t hw = 0; hw < spatial; hw++) {
                    int64_t idx = ((n * C + c) * H * W) + hw;
                    mean += x[idx];
                }
            }
            mean /= (float)batch_elems;

            float var = 0.0f;
            for (int64_t n = 0; n < N; n++) {
                for (int64_t hw = 0; hw < spatial; hw++) {
                    int64_t idx = ((n * C + c) * H * W) + hw;
                    float diff = x[idx] - mean;
                    var += diff * diff;
                }
            }
            var /= (float)batch_elems;

            running_mean[c] = momentum * running_mean[c] + (1.0f - momentum) * mean;
            running_var[c] = momentum * running_var[c] + (1.0f - momentum) * var;

            float inv_std = 1.0f / sqrtf(var + eps);
            for (int64_t n = 0; n < N; n++) {
                for (int64_t hw = 0; hw < spatial; hw++) {
                    int64_t idx = ((n * C + c) * H * W) + hw;
                    float norm = (x[idx] - mean) * inv_std;
                    y[idx] = norm * gamma[c] + beta[c];
                }
            }
        }
    } else {
        for (int64_t c = 0; c < C; c++) {
            float mean = running_mean[c];
            float var = running_var[c];
            float inv_std = 1.0f / sqrtf(var + eps);

            for (int64_t n = 0; n < N; n++) {
                for (int64_t hw = 0; hw < spatial; hw++) {
                    int64_t idx = ((n * C + c) * H * W) + hw;
                    float norm = (x[idx] - mean) * inv_std;
                    y[idx] = norm * gamma[c] + beta[c];
                }
            }
        }
    }
    *output = out;
}

GABatchNorm2D* ga_batchnorm2d_create(int num_features, float momentum, float eps) {
    GABatchNorm2D* bn = (GABatchNorm2D*)calloc(1, sizeof(GABatchNorm2D));
    int64_t shape[1] = {num_features};
    bn->running_mean = ga_tensor_zeros(1, shape, GA_FLOAT32);
    bn->running_var = ga_tensor_full(1, shape, GA_FLOAT32, &(float){1.0f});
    bn->gamma = ga_tensor_full(1, shape, GA_FLOAT32, &(float){1.0f});
    bn->beta = ga_tensor_zeros(1, shape, GA_FLOAT32);
    bn->momentum = momentum;
    bn->eps = eps;
    bn->base.training = true;
    bn->base.forward_fn = batchnorm_forward_wrapper;
    bn->base.parameters = (GATensor**)calloc(2, sizeof(GATensor*));
    bn->base.parameters[0] = bn->gamma;
    bn->base.parameters[1] = bn->beta;
    bn->base.n_params = 2;
    return bn;
}

void ga_batchnorm2d_free(GABatchNorm2D* bn) {
    if (!bn) return;
    if (bn->running_mean) ga_tensor_release(bn->running_mean);
    if (bn->running_var) ga_tensor_release(bn->running_var);
    if (bn->gamma) ga_tensor_release(bn->gamma);
    if (bn->beta) ga_tensor_release(bn->beta);
    if (bn->base.parameters) free(bn->base.parameters);
    free(bn);
}

GATensor* ga_batchnorm2d_forward(GABatchNorm2D* bn, GATensor* input) {
    if (!bn) return NULL;
    GATensor* out = NULL;
    bn->base.forward_fn(bn, input, &out);
    return out;
}
