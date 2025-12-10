/*
 * GreyML backend: ga loss mse.
 *
 * Built-in loss implementations intended for training loops and optimizer tests.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include "greyarea/ga_loss.h"
#include "greyarea/ga_ops.h"

GATensor* ga_mse_loss(GATensor* pred, GATensor* target, int reduction) {
    assert(pred && target);
    assert(pred->dtype == target->dtype);
    
    pred = ga_tensor_contiguous(pred);
    target = ga_tensor_contiguous(target);
    
    if (pred->size != target->size) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        ga_tensor_release(pred);
        ga_tensor_release(target);
        return NULL;
    }
    
    // Calculate (pred - target)^2
    GATensor* diff = ga_sub(pred, target);
    GATensor* squared = ga_mul(diff, diff);
    
    if (reduction == GA_REDUCE_MEAN) {
        GATensor* result = ga_mean(squared, -1, false);
        ga_tensor_release(diff);
        ga_tensor_release(squared);
        return result;
    } else if (reduction == GA_REDUCE_SUM) {
        GATensor* result = ga_sum(squared, -1, false);
        ga_tensor_release(diff);
        ga_tensor_release(squared);
        return result;
    }
    
    ga_tensor_release(diff);
    return squared;
}