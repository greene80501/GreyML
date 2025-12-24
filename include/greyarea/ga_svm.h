/*
 * GreyML C API header: ga svm.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"

typedef enum {
    SVM_KERNEL_LINEAR,
    SVM_KERNEL_RBF,
    SVM_KERNEL_POLY,
    SVM_KERNEL_SIGMOID,
} SVMKernelType;

typedef struct {
    SVMKernelType kernel;
    double C;
    double gamma;
    int degree;
    double coef0;
    double tol;
    int max_iter;
    bool shrinking;
    bool probability;
    bool is_regression;
    double epsilon;
    
    // Training data
    GATensor* support_vectors;
    GATensor* dual_coef;
    double* intercept;
    double class_labels[2];
    
    // Kernel params
    int n_support;
    int n_classes;
} GASVM;

GA_API GASVM* ga_svc_create(double C, SVMKernelType kernel, double gamma, int degree, double coef0, double tol, int max_iter);
GA_API GASVM* ga_svr_create(double C, double epsilon, SVMKernelType kernel, double gamma, int degree, double coef0, double tol, int max_iter);
GA_API void ga_svm_free(GASVM* svm);
GA_API void ga_svm_fit(GASVM* svm, GATensor* X, GATensor* y);
GA_API GATensor* ga_svm_predict(GASVM* svm, GATensor* X);
GA_API GATensor* ga_svm_decision_function(GASVM* svm, GATensor* X);

// Kernel helpers (exposed for testing and reuse)
GA_API double ga_svm_kernel_linear(const double* x, const double* y, int len);
GA_API double ga_svm_kernel_rbf(const double* x, const double* y, int len, double gamma);
GA_API double ga_svm_kernel_poly(const double* x, const double* y, int len, double gamma, int degree, double coef0);
