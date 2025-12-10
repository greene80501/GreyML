/*
 * GreyML backend: ga svm.
 *
 * Classical machine learning algorithms built on the GreyML tensor core.
 */

#include "greyarea/ga_svm.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double ga_svm_kernel_eval(const double* x, const double* y, int len, const GASVM* svm) {
    double gamma = svm->gamma > 0 ? svm->gamma : 1.0 / (double)len;
    switch (svm->kernel) {
        case SVM_KERNEL_RBF: return ga_svm_kernel_rbf(x, y, len, gamma);
        case SVM_KERNEL_POLY: return ga_svm_kernel_poly(x, y, len, gamma, svm->degree > 0 ? svm->degree : 3, svm->coef0);
        case SVM_KERNEL_SIGMOID: {
            double dot = 0.0;
            for (int i = 0; i < len; i++) dot += x[i] * y[i];
            return tanh(gamma * dot + svm->coef0);
        }
        case SVM_KERNEL_LINEAR:
        default: return ga_svm_kernel_linear(x, y, len);
    }
}

static void ga_svm_build_kernel(const double* X, int64_t n, int64_t d, const GASVM* svm, double* K) {
    for (int64_t i = 0; i < n; i++) {
        const double* xi = X + i * d;
        for (int64_t j = 0; j <= i; j++) {
            const double* xj = X + j * d;
            double v = ga_svm_kernel_eval(xi, xj, (int)d, svm);
            K[i * n + j] = v;
            K[j * n + i] = v;
        }
    }
}

static double ga_svm_decision_idx_cls(const double* alpha, const double* y, const double* K, int64_t n, int64_t idx, double b) {
    double s = b;
    for (int64_t j = 0; j < n; j++) s += alpha[j] * y[j] * K[idx * n + j];
    return s;
}

static void ga_svm_store_model(GASVM* svm, const double* X, const double* coeff, int64_t n, int64_t d, double b) {
    // Count support vectors
    int64_t count = 0;
    for (int64_t i = 0; i < n; i++) {
        if (fabs(coeff[i]) > 1e-6) count++;
    }
    svm->n_support = (int)count;
    if (svm->support_vectors) ga_tensor_release(svm->support_vectors);
    if (svm->dual_coef) ga_tensor_release(svm->dual_coef);
    if (svm->intercept) free(svm->intercept);
    svm->intercept = (double*)calloc(1, sizeof(double));
    svm->intercept[0] = b;
    if (count == 0) return;
    int64_t sv_shape[2] = {count, d};
    svm->support_vectors = ga_tensor_empty(2, sv_shape, GA_FLOAT32);
    int64_t dc_shape[2] = {1, count};
    svm->dual_coef = ga_tensor_empty(2, dc_shape, GA_FLOAT32);
    float* svd = (float*)svm->support_vectors->data;
    float* dcd = (float*)svm->dual_coef->data;
    int64_t si = 0;
    for (int64_t i = 0; i < n; i++) {
        if (fabs(coeff[i]) <= 1e-6) continue;
        for (int64_t f = 0; f < d; f++) {
            svd[si * d + f] = (float)X[i * d + f];
        }
        dcd[si] = (float)coeff[i];
        si++;
    }
}

static void ga_svm_train_svc(GASVM* svm, const double* X, const double* y, int64_t n, int64_t d) {
    double* K = (double*)calloc((size_t)n * n, sizeof(double));
    double* alpha = (double*)calloc((size_t)n, sizeof(double));
    double b = 0.0;
    ga_svm_build_kernel(X, n, d, svm, K);
    int stagnation = 0;
    for (int iter = 0; iter < svm->max_iter && stagnation < 5; iter++) {
        int changed = 0;
        for (int64_t i = 0; i < n; i++) {
            double f_i = ga_svm_decision_idx_cls(alpha, y, K, n, i, b);
            double E_i = f_i - y[i];
            double yiEi = y[i] * E_i;
            if (!((yiEi < -svm->tol && alpha[i] < svm->C) || (yiEi > svm->tol && alpha[i] > 0.0))) continue;
            // pick j maximizing |Ei - Ej|
            int64_t j = -1;
            double best_diff = -1.0;
            for (int64_t cand = 0; cand < n; cand++) {
                if (cand == i) continue;
                double f_j = ga_svm_decision_idx_cls(alpha, y, K, n, cand, b);
                double diff = fabs(E_i - (f_j - y[cand]));
                if (diff > best_diff) {
                    best_diff = diff;
                    j = cand;
                }
            }
            if (j < 0) j = (i + 1) % n;

            double f_j = ga_svm_decision_idx_cls(alpha, y, K, n, j, b);
            double E_j = f_j - y[j];

            double alpha_i_old = alpha[i];
            double alpha_j_old = alpha[j];
            double L, H;
            if (y[i] != y[j]) {
                L = fmax(0.0, alpha_j_old - alpha_i_old);
                H = fmin(svm->C, svm->C + alpha_j_old - alpha_i_old);
            } else {
                L = fmax(0.0, alpha_i_old + alpha_j_old - svm->C);
                H = fmin(svm->C, alpha_i_old + alpha_j_old);
            }
            if (L == H) continue;
            double eta = 2.0 * K[i * n + j] - K[i * n + i] - K[j * n + j];
            if (eta >= 0) continue;
            double new_alpha_j = alpha_j_old - y[j] * (E_i - E_j) / eta;
            if (new_alpha_j > H) new_alpha_j = H;
            else if (new_alpha_j < L) new_alpha_j = L;
            if (fabs(new_alpha_j - alpha_j_old) < 1e-6) continue;
            double new_alpha_i = alpha_i_old + y[i] * y[j] * (alpha_j_old - new_alpha_j);
            alpha[i] = new_alpha_i;
            alpha[j] = new_alpha_j;
            double b1 = b - E_i - y[i] * (new_alpha_i - alpha_i_old) * K[i * n + i]
                        - y[j] * (new_alpha_j - alpha_j_old) * K[i * n + j];
            double b2 = b - E_j - y[i] * (new_alpha_i - alpha_i_old) * K[i * n + j]
                        - y[j] * (new_alpha_j - alpha_j_old) * K[j * n + j];
            if (new_alpha_i > 1e-6 && new_alpha_i < svm->C - 1e-6) b = b1;
            else if (new_alpha_j > 1e-6 && new_alpha_j < svm->C - 1e-6) b = b2;
            else b = 0.5 * (b1 + b2);
            changed++;
        }
        stagnation = (changed == 0) ? stagnation + 1 : 0;
    }
    double* coeff = (double*)calloc((size_t)n, sizeof(double));
    for (int64_t i = 0; i < n; i++) coeff[i] = alpha[i] * y[i];
    ga_svm_store_model(svm, X, coeff, n, d, b);
    free(coeff);
    free(alpha);
    free(K);
}

static double ga_svr_decision_idx(const double* coef, const double* K, int64_t n, int64_t idx, double b) {
    double s = b;
    for (int64_t j = 0; j < n; j++) s += coef[j] * K[idx * n + j];
    return s;
}

static double ga_svr_grad(double err, double eps) {
    if (err > eps) return err - eps;
    if (err < -eps) return err + eps;
    return 0.0;
}

static void ga_svm_train_svr(GASVM* svm, const double* X, const double* y, int64_t n, int64_t d) {
    double* K = (double*)calloc((size_t)n * n, sizeof(double));
    ga_svm_build_kernel(X, n, d, svm, K);
    double* coef = (double*)calloc((size_t)n, sizeof(double));  // represents alpha - alpha*
    double b = 0.0;
    for (int iter = 0; iter < svm->max_iter; iter++) {
        int changed = 0;
        for (int64_t i = 0; i < n; i++) {
            double f_i = ga_svr_decision_idx(coef, K, n, i, b);
            double err_i = f_i - y[i];
            double grad_i = ga_svr_grad(err_i, svm->epsilon);
            if (fabs(grad_i) <= svm->tol) continue;

            int64_t j = (i + 1) % n;
            double f_j = ga_svr_decision_idx(coef, K, n, j, b);
            double err_j = f_j - y[j];
            double grad_j = ga_svr_grad(err_j, svm->epsilon);

            double eta = K[i * n + i] + K[j * n + j] - 2.0 * K[i * n + j];
            if (eta <= 0) eta = 1e-12;
            double delta = (grad_i - grad_j) / eta;

            double new_ci = coef[i] - delta;
            double new_cj = coef[j] + delta;
            if (new_ci > svm->C) new_ci = svm->C;
            if (new_ci < -svm->C) new_ci = -svm->C;
            if (new_cj > svm->C) new_cj = svm->C;
            if (new_cj < -svm->C) new_cj = -svm->C;
            if (fabs(new_ci - coef[i]) < 1e-6 && fabs(new_cj - coef[j]) < 1e-6) continue;
            coef[i] = new_ci;
            coef[j] = new_cj;
            changed++;
        }
        if (changed == 0) break;
    }
    // Intercept: average residual for near-margin points
    double b_sum = 0.0;
    int b_count = 0;
    for (int64_t i = 0; i < n; i++) {
        double f_i = ga_svr_decision_idx(coef, K, n, i, 0.0);
        double err = y[i] - f_i;
        if (fabs(err) <= svm->epsilon + 1e-3) {
            b_sum += err;
            b_count++;
        }
    }
    if (b_count > 0) b = b_sum / b_count;
    ga_svm_store_model(svm, X, coef, n, d, b);
    free(coef);
    free(K);
}

GASVM* ga_svc_create(double C, SVMKernelType kernel, double gamma, int degree, double coef0, double tol, int max_iter) {
    GASVM* svm = (GASVM*)calloc(1, sizeof(GASVM));
    svm->C = C;
    svm->kernel = kernel;
    svm->gamma = gamma;
    svm->degree = degree;
    svm->coef0 = coef0;
    svm->tol = tol > 0 ? tol : 1e-3;
    svm->max_iter = max_iter > 0 ? max_iter : 1000;
    svm->is_regression = false;
    svm->epsilon = 0.0;
    svm->class_labels[0] = 0.0;
    svm->class_labels[1] = 1.0;
    return svm;
}

GASVM* ga_svr_create(double C, double epsilon, SVMKernelType kernel, double gamma, int degree, double coef0, double tol, int max_iter) {
    GASVM* svm = ga_svc_create(C, kernel, gamma, degree, coef0, tol, max_iter);
    svm->is_regression = true;
    svm->epsilon = epsilon > 0 ? epsilon : 0.1;
    return svm;
}

void ga_svm_free(GASVM* svm) {
    if (!svm) return;
    if (svm->support_vectors) ga_tensor_release(svm->support_vectors);
    if (svm->dual_coef) ga_tensor_release(svm->dual_coef);
    if (svm->intercept) free(svm->intercept);
    free(svm);
}

void ga_svm_fit(GASVM* svm, GATensor* X, GATensor* y) {
    if (!svm || !X || !y || X->ndim < 2) return;
    int64_t n = X->shape[0];
    int64_t d = X->shape[1];
    double* Xd = (double*)calloc((size_t)n * d, sizeof(double));
    for (int64_t i = 0; i < n * d; i++) Xd[i] = ((float*)X->data)[i];

    if (!svm->is_regression) {
        double cls[2] = {0.0, 0.0};
        int cls_count = 0;
        for (int64_t i = 0; i < n && cls_count < 2; i++) {
            double v = (y->dtype == GA_FLOAT32) ? (double)((float*)y->data)[i] : (double)((int64_t*)y->data)[i];
            bool found = false;
            for (int c = 0; c < cls_count; c++) if (fabs(cls[c] - v) < 1e-6) found = true;
            if (!found) cls[cls_count++] = v;
        }
        if (cls_count == 1) cls[1] = cls[0] + 1.0;
        svm->n_classes = cls_count;
        svm->class_labels[0] = cls[0];
        svm->class_labels[1] = (cls_count > 1) ? cls[1] : cls[0];

        double* yb = (double*)calloc((size_t)n, sizeof(double));
        for (int64_t i = 0; i < n; i++) {
            double v = (y->dtype == GA_FLOAT32) ? (double)((float*)y->data)[i] : (double)((int64_t*)y->data)[i];
            yb[i] = (v == svm->class_labels[1]) ? 1.0 : -1.0;
        }
        ga_svm_train_svc(svm, Xd, yb, n, d);
        free(yb);
    } else {
        double* yr = (double*)calloc((size_t)n, sizeof(double));
        if (y->dtype == GA_FLOAT32) {
            float* yd = (float*)y->data;
            for (int64_t i = 0; i < n; i++) yr[i] = yd[i];
        } else {
            int64_t* yd = (int64_t*)y->data;
            for (int64_t i = 0; i < n; i++) yr[i] = (double)yd[i];
        }
        ga_svm_train_svr(svm, Xd, yr, n, d);
        free(yr);
    }
    free(Xd);
}

GATensor* ga_svm_decision_function(GASVM* svm, GATensor* X) {
    if (!svm || !svm->support_vectors || !svm->dual_coef || !X) return NULL;
    int64_t n = X->shape[0];
    int64_t d = X->shape[1];
    int64_t sv = svm->support_vectors->shape[0];
    float* svd = (float*)svm->support_vectors->data;
    float* cd = (float*)svm->dual_coef->data;
    float* Xd = (float*)X->data;
    int64_t out_shape[1] = {n};
    GATensor* out = ga_tensor_empty(1, out_shape, GA_FLOAT32);
    float* od = (float*)out->data;
    for (int64_t i = 0; i < n; i++) {
        double sum = svm->intercept ? svm->intercept[0] : 0.0;
        for (int64_t s = 0; s < sv; s++) {
            sum += cd[s] * ga_svm_kernel_eval(svd + s * d, Xd + i * d, (int)d, svm);
        }
        od[i] = (float)sum;
    }
    return out;
}

GATensor* ga_svm_predict(GASVM* svm, GATensor* X) {
    GATensor* decision = ga_svm_decision_function(svm, X);
    if (!decision) return NULL;
    int64_t n = decision->shape[0];
    float* dd = (float*)decision->data;
    if (!svm->is_regression) {
        int64_t shape[1] = {n};
        GATensor* out = ga_tensor_empty(1, shape, GA_INT64);
        int64_t* od = (int64_t*)out->data;
        for (int64_t i = 0; i < n; i++) {
            od[i] = dd[i] >= 0 ? (int64_t)svm->class_labels[1] : (int64_t)svm->class_labels[0];
        }
        ga_tensor_release(decision);
        return out;
    }
    // Regression: return decision values
    return decision;
}
