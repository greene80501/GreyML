/*
 * GreyML C test: tensor.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include <assert.h>
#include "greyarea/greyarea.h"

int test_tensor_basic(void) {
    int64_t shape[2] = {2, 2};
    GATensor* t = ga_tensor_ones(2, shape, GA_FLOAT32);
    assert(t);
    assert(t->size == 4);
    ga_tensor_release(t);
    return 0;
}
