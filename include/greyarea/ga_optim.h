/*
 * GreyML C API header: ga optim.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"
#include "ga_nn.h"

typedef struct {
    GATensor** params;
    size_t n_params;
    float lr;
    void (*step_fn)(void* self);
    void (*zero_grad_fn)(void* self);
    void* state;
} GAOptimizer;

typedef struct {
    GAOptimizer base;
    float momentum;
    float weight_decay;
    float* momentum_buffers;
} GASGD;

typedef struct {
    GAOptimizer base;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float* m;
    float* v;
    int t;
} GAAdam;

GA_API GASGD* ga_sgd_create(GATensor** params, size_t n_params, float lr, float momentum, float weight_decay);
GA_API void ga_sgd_free(GASGD* sgd);
GA_API void ga_sgd_step(GASGD* sgd);
GA_API void ga_sgd_zero_grad(GASGD* sgd);

GA_API GAAdam* ga_adam_create(GATensor** params, size_t n_params, float lr, float beta1, float beta2, float eps, float weight_decay);
GA_API void ga_adam_free(GAAdam* adam);
GA_API void ga_adam_step(GAAdam* adam);
GA_API void ga_adam_zero_grad(GAAdam* adam);

typedef struct {
    GAOptimizer* base_opt;
    float base_lr;
    float gamma;
    int step_size;
    int last_epoch;
} GAStepLR;

GA_API GAStepLR* ga_scheduler_steplr_create(GAOptimizer* opt, float base_lr, int step_size, float gamma);
GA_API void ga_scheduler_step(GAStepLR* scheduler);