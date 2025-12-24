/*
 * GreyML backend: ga svm kernel.
 *
 * Classical machine learning algorithms built on the GreyML tensor core.
 */

#include "greyarea/ga_svm.h"
#include <math.h>

GA_API double ga_svm_kernel_linear(const double* x, const double* y, int len) {
    double sum = 0.0;
    for (int i = 0; i < len; i++) sum += x[i] * y[i];
    return sum;
}

GA_API double ga_svm_kernel_rbf(const double* x, const double* y, int len, double gamma) {
    double sum = 0.0;
    for (int i = 0; i < len; i++) {
        double d = x[i] - y[i];
        sum += d * d;
    }
    return exp(-gamma * sum);
}

GA_API double ga_svm_kernel_poly(const double* x, const double* y, int len, double gamma, int degree, double coef0) {
    double dot = 0.0;
    for (int i = 0; i < len; i++) dot += x[i] * y[i];
    return pow(gamma * dot + coef0, degree);
}
