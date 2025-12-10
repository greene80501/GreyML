/*
 * GreyML backend: ga loss other.
 *
 * Built-in loss implementations intended for training loops and optimizer tests.
 */

// Other losses: L1, Huber, BCE, NLL, CrossEntropy
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include "greyarea/ga_loss.h"
#include "greyarea/ga_ops.h"
#include "greyarea/ga_common.h"
#include "greyarea/ga_autograd.h"

static GATensor* reduce_tensor(GATensor* t, int reduction) {
    if (reduction == GA_REDUCE_NONE) return t;
    GATensor* out = NULL;
    if (reduction == GA_REDUCE_MEAN) {
        out = ga_mean(t, -1, false);
    } else if (reduction == GA_REDUCE_SUM) {
        out = ga_sum(t, -1, false);
    }
    ga_tensor_release(t);
    return out;
}

GATensor* ga_l1_loss(GATensor* pred, GATensor* target, int reduction) {
    if (!pred || !target || pred->size != target->size) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    pred = ga_tensor_contiguous(pred);
    target = ga_tensor_contiguous(target);
    GATensor* diff = ga_sub(pred, target);
    GATensor* abs_diff = ga_abs(diff);
    return reduce_tensor(abs_diff, reduction);
}

GATensor* ga_binary_cross_entropy(GATensor* pred, GATensor* target, int reduction) {
    if (!pred || !target || pred->size != target->size) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    pred = ga_tensor_contiguous(pred);
    target = ga_tensor_contiguous(target);
    GATensor* out = ga_tensor_empty(pred->ndim, pred->shape, GA_FLOAT32);
    float* p = (float*)pred->data;
    float* y = (float*)target->data;
    float* o = (float*)out->data;
    const float eps = 1e-7f;
    for (int64_t i = 0; i < pred->size; i++) {
        float pi = p[i];
        pi = pi < eps ? eps : (pi > 1.0f - eps ? 1.0f - eps : pi);
        o[i] = -(y[i] * logf(pi) + (1.0f - y[i]) * logf(1.0f - pi));
    }
    ga_tensor_release(pred);
    ga_tensor_release(target);
    return reduce_tensor(out, reduction);
}

GATensor* ga_nll_loss(GATensor* log_probs, GATensor* target, int reduction) {
    if (!log_probs || !target || log_probs->ndim != 2 || target->ndim != 1 || log_probs->shape[0] != target->shape[0]) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    log_probs = ga_tensor_contiguous(log_probs);
    target = ga_tensor_contiguous(target);
    int64_t batch = log_probs->shape[0];
    int64_t classes = log_probs->shape[1];
    GATensor* out = ga_tensor_empty(1, &batch, GA_FLOAT32);
    float* lp = (float*)log_probs->data;
    float* o = (float*)out->data;
    int64_t* t = (int64_t*)target->data;
    for (int64_t i = 0; i < batch; i++) {
        int64_t idx = t[i];
        if (idx < 0 || idx >= classes) { ga_errno = GA_ERR_OUT_OF_BOUNDS; ga_tensor_release(out); return NULL; }
        o[i] = -lp[i * classes + idx];
    }
    if (ga_is_grad_enabled() && log_probs->requires_grad) {
        GANode* node = ga_node_create("nll_loss", 1, 1, 0, 0);
        if (node) {
            node->inputs[0] = log_probs;
            ga_tensor_retain(log_probs);
            node->saved_tensors[0] = target;
            ga_tensor_retain(target);
            node->backward_fn = ga_autograd_nll_loss;
            out->requires_grad = true;
            out->grad_fn = node;
        }
    }
    ga_tensor_release(log_probs);
    ga_tensor_release(target);
    return reduce_tensor(out, reduction);
}

GATensor* ga_cross_entropy_loss(GATensor* logits, GATensor* target, int reduction) {
    // CrossEntropy = log_softmax + nll
    GATensor* log_sm = ga_log_softmax(logits, -1);
    if (!log_sm) return NULL;
    return ga_nll_loss(log_sm, target, reduction);
}

GATensor* ga_huber_loss(GATensor* pred, GATensor* target, float delta, int reduction) {
    if (!pred || !target || pred->size != target->size) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }
    pred = ga_tensor_contiguous(pred);
    target = ga_tensor_contiguous(target);
    GATensor* diff = ga_sub(pred, target);
    GATensor* out = ga_tensor_empty(diff->ndim, diff->shape, diff->dtype);
    float* d = (float*)diff->data;
    float* o = (float*)out->data;
    for (int64_t i = 0; i < diff->size; i++) {
        float ad = fabsf(d[i]);
        if (ad <= delta) {
            o[i] = 0.5f * ad * ad;
        } else {
            o[i] = delta * (ad - 0.5f * delta);
        }
    }
    ga_tensor_release(diff);
    ga_tensor_release(pred);
    ga_tensor_release(target);
    return reduce_tensor(out, reduction);
}
