/*
 * GreyML C test: ops.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include <assert.h>
#include "greyarea/greyarea.h"

int test_ops_basic(void) {
    int64_t shape[1] = {3};
    GATensor* a = ga_tensor_full(1, shape, GA_FLOAT32, &(float){2.0f});
    GATensor* b = ga_tensor_full(1, shape, GA_FLOAT32, &(float){3.0f});
    GATensor* c = ga_add(a, b);
    assert(c);
    float* v = (float*)c->data;
    for (int i = 0; i < 3; i++) assert(v[i] == 5.0f);
    ga_tensor_release(c);
    return 0;
}
