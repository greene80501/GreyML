/*
 * GreyML backend: ga autograd.
 *
 * Automatic differentiation primitives: graph construction, tracing, and backward pass kernels.
 */

#include "greyarea/ga_autograd.h"
#include <string.h>

GATape* ga_global_tape = NULL;
static bool grad_enabled = true;

void ga_set_grad_enabled(bool enabled) {
    grad_enabled = enabled;
}

bool ga_is_grad_enabled(void) {
    return grad_enabled;
}

// Utility: accumulate gradient into tensor->grad (float32 only for v0.1)
void ga_accumulate_grad(GATensor* tensor, GATensor* grad) {
    if (!tensor->grad) {
        tensor->grad = ga_tensor_clone(grad);
        return;
    }
    float* g_dst = (float*)tensor->grad->data;
    float* g_src = (float*)grad->data;
    for (int64_t i = 0; i < tensor->grad->size; i++) {
        g_dst[i] += g_src[i];
    }
}

GANode* ga_node_create(const char* op_name, int num_inputs, int num_saved, int num_scalars, int num_ints) {
    if (!grad_enabled) return NULL;
    
    GANode* node = (GANode*)calloc(1, sizeof(GANode));
    node->op_name = strdup(op_name);
    node->inputs = (GATensor**)calloc(num_inputs, sizeof(GATensor*));
    node->num_inputs = num_inputs;
    node->saved_tensors = (GATensor**)calloc(num_saved, sizeof(GATensor*));
    node->saved_scalars = (float*)calloc(num_scalars, sizeof(float));
    node->saved_ints = (int*)calloc(num_ints, sizeof(int));
    node->num_saved = num_saved;
    node->num_scalars = num_scalars;
    node->num_ints = num_ints;
    node->refcount = 1;
    return node;
}

void ga_node_save_tensor(GANode* node, int idx, GATensor* tensor) {
    if (!node || idx >= node->num_saved) return;
    node->saved_tensors[idx] = tensor;
    ga_tensor_retain(tensor);
}

void ga_node_save_scalar(GANode* node, int idx, float value) {
    if (!node || idx >= (int)node->num_scalars) return;
    node->saved_scalars[idx] = value;
}

void ga_node_save_int(GANode* node, int idx, int value) {
    if (!node || idx >= (int)node->num_ints) return;
    node->saved_ints[idx] = value;
}

// Topological backward
void ga_backward(GATensor* tensor, GATensor* grad) {
    if (!tensor || !tensor->requires_grad) return;
    
    // If no incoming grad, start with ones
    if (!grad) {
        int64_t shape[GA_MAX_DIMS];
        for (int i = 0; i < tensor->ndim; i++) shape[i] = tensor->shape[i];
        grad = ga_tensor_ones(tensor->ndim, shape, GA_FLOAT32);
    }
    
    ga_accumulate_grad(tensor, grad);

    // Build topological order
    GATensor** topo = NULL;
    size_t topo_size = 0;
    ga_build_topo(tensor, &topo, &topo_size);

    // Execute backward in reverse topo
    for (size_t i = topo_size; i-- > 0;) {
        GATensor* t = topo[i];
        if (!t || !t->grad_fn || !t->grad_fn->backward_fn || !t->grad) continue;
        bool prev = ga_is_grad_enabled();
        ga_set_grad_enabled(false);
        t->grad_fn->backward_fn(t->grad_fn, t->grad);
        ga_set_grad_enabled(prev);
    }

    free(topo);
}

void ga_detach(GATensor* tensor) {
    if (!tensor) return;
    tensor->requires_grad = false;
    tensor->grad = NULL;
    tensor->grad_fn = NULL;
}

void ga_no_grad(void) { ga_set_grad_enabled(false); }
void ga_enable_grad(void) { ga_set_grad_enabled(true); }
