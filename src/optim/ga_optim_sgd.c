/*
 * GreyML backend: ga optim sgd.
 *
 * Optimizer and scheduler kernels for updating model parameters.
*/

#include "greyarea/ga_optim.h"
#include <stdlib.h>

GASGD* ga_sgd_create(GATensor** params, size_t n_params, float lr, float momentum, float weight_decay) {
    GASGD* sgd = (GASGD*)calloc(1, sizeof(GASGD));
    sgd->base.params = params;
    sgd->base.n_params = n_params;
    sgd->base.lr = lr;
    sgd->momentum = momentum;
    sgd->weight_decay = weight_decay;
    // Allocate momentum buffer for all parameters (flattened)
    int64_t total = 0;
    for (size_t i = 0; i < n_params; i++) total += params[i]->size;
    sgd->momentum_buffers = (float*)calloc((size_t)total, sizeof(float));
    return sgd;
}

void ga_sgd_step(GASGD* sgd) {
    int64_t offset = 0;
    for (size_t i = 0; i < sgd->base.n_params; i++) {
        GATensor* param = sgd->base.params[i];
        if (!param->grad) { offset += param->size; continue; }
        
        float* p = (float*)param->data;
        float* g = (float*)param->grad->data;
        
        for (int64_t j = 0; j < param->size; j++) {
            if (sgd->weight_decay > 0) g[j] += sgd->weight_decay * p[j];
            
            float* v = &sgd->momentum_buffers[offset + j];
            float velocity = sgd->momentum * (*v) - sgd->base.lr * g[j];
            *v = velocity;
            p[j] += velocity;
        }
        offset += param->size;
    }
}

void ga_sgd_zero_grad(GASGD* sgd) {
    for (size_t i = 0; i < sgd->base.n_params; i++) {
        if (sgd->base.params[i]->grad) {
            ga_tensor_release(sgd->base.params[i]->grad);
            sgd->base.params[i]->grad = NULL;
        }
    }
}

void ga_sgd_free(GASGD* sgd) {
    if (!sgd) return;
    if (sgd->momentum_buffers) free(sgd->momentum_buffers);
    free(sgd);
}
