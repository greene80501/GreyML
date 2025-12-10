/*
 * GreyML backend: ga optim scheduler.
 *
 * Optimizer and scheduler kernels for updating model parameters.
 */

#include "greyarea/ga_optim.h"
#include <math.h>

GAStepLR* ga_scheduler_steplr_create(GAOptimizer* opt, float base_lr, int step_size, float gamma) {
    GAStepLR* sch = (GAStepLR*)calloc(1, sizeof(GAStepLR));
    sch->base_opt = opt;
    sch->base_lr = base_lr;
    sch->gamma = gamma;
    sch->step_size = step_size;
    sch->last_epoch = 0;
    return sch;
}

void ga_scheduler_step(GAStepLR* scheduler) {
    scheduler->last_epoch += 1;
    int k = scheduler->last_epoch / scheduler->step_size;
    float lr = scheduler->base_lr * powf(scheduler->gamma, (float)k);
    scheduler->base_opt->lr = lr;
}
