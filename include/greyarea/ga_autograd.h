/*
 * GreyML C API header: ga autograd.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"
#include "ga_ops.h"

typedef struct GANode {
    char* op_name;
    void (*backward_fn)(struct GANode* node, GATensor* grad_output);
    GATensor** inputs;
    size_t num_inputs;
    GATensor* output;
    GATensor** saved_tensors;
    size_t num_saved;
    float* saved_scalars;
    int* saved_ints;
    size_t num_scalars;
    size_t num_ints;
    int refcount;
} GANode;

typedef struct {
    GANode** nodes;
    size_t capacity;
    size_t size;
    bool enabled;
} GATape;

extern GATape* ga_global_tape;

GA_API void ga_set_grad_enabled(bool enabled);
GA_API bool ga_is_grad_enabled(void);
GA_API void ga_backward(GATensor* tensor, GATensor* grad);
GA_API GANode* ga_node_create(const char* op_name, int num_inputs, int num_saved, int num_scalars, int num_ints);
GA_API void ga_node_save_tensor(GANode* node, int idx, GATensor* tensor);
GA_API void ga_node_save_scalar(GANode* node, int idx, float value);
GA_API void ga_node_save_int(GANode* node, int idx, int value);
GA_API void ga_accumulate_grad(GATensor* tensor, GATensor* grad);
GA_API void ga_detach(GATensor* tensor);
GA_API void ga_no_grad(void);
GA_API void ga_enable_grad(void);

// Topological traversal helpers
void ga_build_topo(GATensor* tensor, GATensor*** list, size_t* size);

// Backward function declarations (used by ops)
void ga_autograd_add(GANode* node, GATensor* grad_output);
void ga_autograd_sub(GANode* node, GATensor* grad_output);
void ga_autograd_mul(GANode* node, GATensor* grad_output);
void ga_autograd_div(GANode* node, GATensor* grad_output);
void ga_autograd_matmul(GANode* node, GATensor* grad_output);
void ga_autograd_relu(GANode* node, GATensor* grad_output);
void ga_autograd_sigmoid(GANode* node, GATensor* grad_output);
void ga_autograd_sum(GANode* node, GATensor* grad_output);
void ga_autograd_mean(GANode* node, GATensor* grad_output);
void ga_autograd_softmax(GANode* node, GATensor* grad_output);
void ga_autograd_log(GANode* node, GATensor* grad_output);
void ga_autograd_pow(GANode* node, GATensor* grad_output);
void ga_autograd_conv2d(GANode* node, GATensor* grad_output);
void ga_autograd_max_pool2d(GANode* node, GATensor* grad_output);
void ga_autograd_avg_pool2d(GANode* node, GATensor* grad_output);
void ga_autograd_adaptive_avg_pool2d(GANode* node, GATensor* grad_output);
void ga_autograd_reshape(GANode* node, GATensor* grad_output);
void ga_autograd_transpose(GANode* node, GATensor* grad_output);
void ga_autograd_flatten(GANode* node, GATensor* grad_output);
void ga_autograd_nll_loss(GANode* node, GATensor* grad_output);
