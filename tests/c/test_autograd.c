/*
 * GreyML C test: autograd.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include <assert.h>
#include "greyarea/greyarea.h"

int test_autograd_smoke(void) {
    int64_t shape[1] = {1};
    GATensor* x = ga_tensor_full(1, shape, GA_FLOAT32, &(float){2.0f});
    ga_tensor_set_requires_grad(x, true);
    GATensor* y = ga_mul(x, ga_tensor_full(1, shape, GA_FLOAT32, &(float){3.0f}));
    ga_backward(y, NULL);
    assert(x->grad != NULL);
    ga_tensor_release(y);
    ga_tensor_release(x);
    return 0;
}
