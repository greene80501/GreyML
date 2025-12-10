/*
 * GreyML C test: nn.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include <assert.h>
#include "greyarea/greyarea.h"

int test_nn_linear(void) {
    GALinear* lin = ga_linear_create(2, 1, true);
    int64_t shape[2] = {1, 2};
    GATensor* x = ga_tensor_ones(2, shape, GA_FLOAT32);
    GATensor* y = ga_linear_forward(lin, x);
    assert(y && y->ndim == 2 && y->shape[1] == 1);
    ga_tensor_release(y);
    ga_linear_free(lin);
    return 0;
}
