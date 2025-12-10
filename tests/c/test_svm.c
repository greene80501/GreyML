/*
 * GreyML C test: svm.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include "greyarea/greyarea.h"
#include <assert.h>

int test_svm_smoke(void) {
    int64_t x_shape[2] = {4, 1};
    GATensor* X = ga_tensor_empty(2, x_shape, GA_FLOAT32);
    float* xd = (float*)X->data;
    xd[0] = 0.0f; xd[1] = 0.2f; xd[2] = 0.8f; xd[3] = 1.0f;
    int64_t y_shape[1] = {4};
    GATensor* y = ga_tensor_empty(1, y_shape, GA_INT64);
    int64_t* yd = (int64_t*)y->data;
    yd[0] = 0; yd[1] = 0; yd[2] = 1; yd[3] = 1;

    GASVM* svc = ga_svc_create(1.0, SVM_KERNEL_LINEAR, 0.0, 3, 0.0, 1e-3, 100);
    ga_svm_fit(svc, X, y);
    GATensor* pred = ga_svm_predict(svc, X);
    int64_t* pd = (int64_t*)pred->data;
    assert(pd[0] == yd[0] && pd[3] == yd[3]);

    // SVR on y = x
    GATensor* yr = ga_tensor_empty(1, y_shape, GA_FLOAT32);
    float* yrd = (float*)yr->data;
    yrd[0] = 0.0f; yrd[1] = 0.2f; yrd[2] = 0.8f; yrd[3] = 1.0f;
    GASVM* svr = ga_svr_create(1.0, 0.05, SVM_KERNEL_RBF, 0.0, 3, 0.0, 1e-3, 100);
    ga_svm_fit(svr, X, yr);
    GATensor* pred_r = ga_svm_predict(svr, X);
    float* pr = (float*)pred_r->data;
    assert(pr[0] < pr[3]);

    ga_tensor_release(pred);
    ga_tensor_release(pred_r);
    ga_svm_free(svc);
    ga_svm_free(svr);
    ga_tensor_release(X);
    ga_tensor_release(y);
    ga_tensor_release(yr);
    return 0;
}
