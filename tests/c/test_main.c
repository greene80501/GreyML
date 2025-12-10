/*
 * GreyML C test: main.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include <stdio.h>
#include <assert.h>
#include "greyarea/greyarea.h"

int test_tensor_basic(void);
int test_ops_basic(void);
int test_nn_linear(void);
int test_io_tensor(void);
int test_mem_alloc(void);
int test_autograd_smoke(void);
int test_cluster_smoke(void);
int test_svm_smoke(void);
int test_tree_smoke(void);
int test_knn_smoke(void);

int main(void) {
    assert(test_tensor_basic() == 0);
    assert(test_ops_basic() == 0);
    assert(test_nn_linear() == 0);
    assert(test_io_tensor() == 0);
    assert(test_mem_alloc() == 0);
    assert(test_autograd_smoke() == 0);
    assert(test_cluster_smoke() == 0);
    assert(test_svm_smoke() == 0);
    assert(test_tree_smoke() == 0);
    assert(test_knn_smoke() == 0);
    printf("C smoke tests passed.\n");
    return 0;
}
