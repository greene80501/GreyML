/*
 * GreyML C test: io.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include <assert.h>
#include <stdio.h>
#include "greyarea/greyarea.h"

int test_io_tensor(void) {
    int64_t shape[1] = {2};
    GATensor* t = ga_tensor_full(1, shape, GA_FLOAT32, &(float){1.5f});
    GAStatus st = ga_tensor_save(t, "test_tmp.gat");
    assert(st == GA_SUCCESS);
    GATensor* loaded = ga_tensor_load("test_tmp.gat");
    assert(loaded);
    assert(loaded->size == t->size);
    ga_tensor_release(t);
    ga_tensor_release(loaded);
    // cleanup best-effort
    remove("test_tmp.gat");
    return 0;
}
