/*
 * GreyML benchmark: bench matmul.
 *
 * Measures the performance of this kernel in isolation for tuning work.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "greyarea/ga_ops.h"
#include "greyarea/ga_tensor.h"

static double now_seconds(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

int main(void) {
    int64_t shape[2] = {256, 256};
    GATensor* a = ga_tensor_rand(2, shape);
    GATensor* b = ga_tensor_rand(2, shape);

    double t0 = now_seconds();
    GATensor* c = ga_matmul(a, b);
    double t1 = now_seconds();
    printf("Matmul 256x256 took %.6f seconds\n", t1 - t0);

    ga_tensor_release(a);
    ga_tensor_release(b);
    ga_tensor_release(c);
    return 0;
}
