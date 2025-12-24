/*
 * GreyML backend: ga nn container.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#include <stdlib.h>
#include "greyarea/ga_nn.h"

static void forward_wrapper(void* self, GATensor* input, GATensor** output) {
    GASequential* seq = (GASequential*)self;
    GATensor* cur = input;
    for (size_t i = 0; i < seq->num_modules; i++) {
        GAModule* m = seq->modules[i];
        if (m && m->forward_fn) {
            GATensor* out = NULL;
            m->forward_fn(m, cur, &out);
            if (i != 0) {
                ga_tensor_release(cur);
            }
            cur = out;
        }
    }
    *output = cur;
}

GASequential* ga_sequential_create(GAModule** modules, size_t num_modules) {
    GASequential* seq = (GASequential*)calloc(1, sizeof(GASequential));
    seq->modules = (GAModule**)calloc(num_modules, sizeof(GAModule*));
    seq->num_modules = num_modules;
    for (size_t i = 0; i < num_modules; i++) seq->modules[i] = modules[i];
    seq->base.forward_fn = forward_wrapper;
    return seq;
}

void ga_sequential_free(GASequential* seq) {
    if (!seq) return;
    if (seq->modules) free(seq->modules);
    free(seq);
}

GATensor* ga_sequential_forward(GASequential* seq, GATensor* input) {
    if (!seq) return NULL;
    GATensor* out = NULL;
    seq->base.forward_fn(seq, input, &out);
    return out;
}

void ga_module_train(GAModule* module, bool training) {
    if (!module) return;
    module->training = training;
    if (module->children) {
        for (size_t i = 0; i < module->n_children; i++) {
            ga_module_train(module->children[i], training);
        }
    }
}

void ga_module_zero_grad(GAModule* module) {
    if (!module) return;
    for (size_t i = 0; i < module->n_params; i++) {
        if (module->parameters[i]) {
            ga_tensor_zero_grad(module->parameters[i]);
        }
    }
    if (module->children) {
        for (size_t i = 0; i < module->n_children; i++) {
            ga_module_zero_grad(module->children[i]);
        }
    }
}
