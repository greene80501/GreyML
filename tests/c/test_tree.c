/*
 * GreyML C test: tree.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include "greyarea/greyarea.h"
#include <assert.h>

int test_tree_smoke(void) {
    int64_t x_shape[2] = {4, 1};
    GATensor* X = ga_tensor_empty(2, x_shape, GA_FLOAT32);
    float* xd = (float*)X->data;
    xd[0] = 0.0f; xd[1] = 0.1f; xd[2] = 0.9f; xd[3] = 1.0f;
    int64_t y_shape[1] = {4};
    GATensor* y = ga_tensor_empty(1, y_shape, GA_INT64);
    int64_t* yd = (int64_t*)y->data;
    yd[0] = 0; yd[1] = 0; yd[2] = 1; yd[3] = 1;

    GATree* clf = ga_tree_classifier_create(3, 2, 1, 0, TREE_CRITERION_GINI);
    ga_tree_fit(clf, X, y);
    GATensor* pred = ga_tree_predict(clf, X);
    int64_t* pd = (int64_t*)pred->data;
    assert(pd[0] == yd[0] && pd[3] == yd[3]);

    // Random forest classification
    GAForest* rf = ga_forest_create(3, 3, 2);
    ga_forest_fit(rf, X, y);
    GATensor* pred_rf = ga_forest_predict(rf, X);
    int64_t* prf = (int64_t*)pred_rf->data;
    assert(prf[0] == yd[0]);

    ga_tensor_release(pred);
    ga_tensor_release(pred_rf);
    ga_tree_free(clf);
    ga_forest_free(rf);
    ga_tensor_release(X);
    ga_tensor_release(y);
    return 0;
}
