/*
 * GreyML backend: ga tensor print.
 *
 * Foundational runtime utilities including error handling, memory management, random utilities, and tensor lifecycle helpers.
 */

#include <stdio.h>
#include "greyarea/ga_tensor.h"

void ga_tensor_print(const GATensor* t) {
    printf("Tensor(");
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld", t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf(")[%s]\n", t->requires_grad ? "grad" : "no_grad");
    
    if (t->dtype == GA_FLOAT32 && t->ndim <= 2) {
        float* data = (float*)t->data;
        for (int i = 0; i < t->size; i++) {
            printf("%.4f ", data[i]);
            if ((i + 1) % t->shape[t->ndim - 1] == 0) printf("\n");
        }
    }
}