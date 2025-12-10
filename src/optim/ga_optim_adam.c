/*
 * GreyML backend: ga optim adam.
 *
 * Optimizer and scheduler kernels for updating model parameters.
 */

// Adam optimizer (per-parameter flattened buffers)
#include "greyarea/ga_optim.h"
#include <math.h>
#include <string.h>

GAAdam* ga_adam_create(GATensor** params, size_t n_params, float lr, float beta1, float beta2, float eps, float weight_decay) {
    GAAdam* adam = (GAAdam*)calloc(1, sizeof(GAAdam));
    adam->base.params = params;
    adam->base.n_params = n_params;
    adam->base.lr = lr;
    adam->beta1 = beta1;
    adam->beta2 = beta2;
    adam->eps = eps;
    adam->weight_decay = weight_decay;
    adam->t = 0;
    int64_t total = 0;
    for (size_t i = 0; i < n_params; i++) total += params[i]->size;
    adam->m = (float*)calloc((size_t)total, sizeof(float));
    adam->v = (float*)calloc((size_t)total, sizeof(float));
    return adam;
}

void ga_adam_step(GAAdam* adam) {
    adam->t += 1;
    int64_t offset = 0;
    for (size_t i = 0; i < adam->base.n_params; i++) {
        GATensor* p = adam->base.params[i];
        if (!p->grad) { offset += p->size; continue; }
        float* w = (float*)p->data;
        float* g = (float*)p->grad->data;
        for (int64_t j = 0; j < p->size; j++) {
            float grad = g[j];
            if (adam->weight_decay > 0) grad += adam->weight_decay * w[j];
            float* m = &adam->m[offset + j];
            float* v = &adam->v[offset + j];
            *m = adam->beta1 * (*m) + (1.0f - adam->beta1) * grad;
            *v = adam->beta2 * (*v) + (1.0f - adam->beta2) * grad * grad;
            float m_hat = (*m) / (1.0f - powf(adam->beta1, (float)adam->t));
            float v_hat = (*v) / (1.0f - powf(adam->beta2, (float)adam->t));
            w[j] -= adam->base.lr * m_hat / (sqrtf(v_hat) + adam->eps);
        }
        offset += p->size;
    }
}

void ga_adam_zero_grad(GAAdam* adam) {
    for (size_t i = 0; i < adam->base.n_params; i++) {
        if (adam->base.params[i]->grad) {
            ga_tensor_release(adam->base.params[i]->grad);
            adam->base.params[i]->grad = NULL;
        }
    }
}

void ga_adam_free(GAAdam* adam) {
    if (!adam) return;
    if (adam->m) free(adam->m);
    if (adam->v) free(adam->v);
    free(adam);
}
