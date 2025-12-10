/*
 * GreyML C test: neighbors.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include <assert.h>
#include "greyarea/greyarea.h"

int test_knn_smoke(void) {
    int64_t x_shape[2] = {4, 1};
    GATensor* X = ga_tensor_empty(2, x_shape, GA_FLOAT32);
    float* xd = (float*)X->data;
    xd[0] = 0.0f; xd[1] = 0.1f; xd[2] = 0.9f; xd[3] = 1.0f;
    int64_t y_shape[1] = {4};
    GATensor* y = ga_tensor_empty(1, y_shape, GA_INT64);
    int64_t* yd = (int64_t*)y->data;
    yd[0] = 0; yd[1] = 0; yd[2] = 1; yd[3] = 1;

    GAKNN* knn = ga_knn_classifier_create(1, KNN_WEIGHT_UNIFORM);
    ga_knn_fit(knn, X, y);
    GATensor* pred = ga_knn_predict(knn, X);
    int64_t* pd = (int64_t*)pred->data;
    assert(pd[0] == yd[0] && pd[3] == yd[3]);

    GATensor* dists = NULL;
    GATensor* idx = NULL;
    ga_knn_kneighbors(knn, X, 2, &dists, &idx);
    assert(dists && idx);

    ga_tensor_release(pred);
    ga_tensor_release(dists);
    ga_tensor_release(idx);
    ga_knn_free(knn);
    ga_tensor_release(X);
    ga_tensor_release(y);
    return 0;
}
