/*
 * GreyML backend: ga tree split.
 *
 * Classical machine learning algorithms built on the GreyML tensor core.
 */

#include "greyarea/ga_tree.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float value;
    float target;
} Pair;

static int pair_cmp(const void* a, const void* b) {
    float va = ((const Pair*)a)->value;
    float vb = ((const Pair*)b)->value;
    return (va > vb) - (va < vb);
}

static float impurity_class(const int* counts, int n_classes, int total, GATreeCriterion criterion) {
    if (total == 0) return 0.0f;
    float imp = 0.0f;
    if (criterion == TREE_CRITERION_ENTROPY) {
        for (int c = 0; c < n_classes; c++) {
            if (counts[c] == 0) continue;
            float p = (float)counts[c] / (float)total;
            imp -= p * logf(p + 1e-12f);
        }
    } else { // Gini by default
        imp = 1.0f;
        float sumsq = 0.0f;
        for (int c = 0; c < n_classes; c++) {
            float p = (float)counts[c] / (float)total;
            sumsq += p * p;
        }
        imp -= sumsq;
    }
    return imp;
}

static float impurity_reg(float sum, float sumsq, int total, GATreeCriterion criterion) {
    if (total == 0) return 0.0f;
    if (criterion == TREE_CRITERION_MAE) {
        // Approximate MAE using variance proxy
        float mean = sum / (float)total;
        return fabsf(mean);
    }
    float mean = sum / (float)total;
    return (sumsq / (float)total) - mean * mean;
}

static void select_feature_subset(int* idx, int total, int choose) {
    // Fisher-Yates partial shuffle
    for (int i = 0; i < choose; i++) {
        int j = i + rand() % (total - i);
        int tmp = idx[i];
        idx[i] = idx[j];
        idx[j] = tmp;
    }
}

GATreeSplit ga_tree_find_best_split(GATensor* X, GATensor* y, int n_classes, GATreeCriterion criterion, int max_features, int min_samples_leaf, bool is_regression) {
    GATreeSplit best = {-1, 0.0f, FLT_MAX};
    int64_t n_samples = X->shape[0];
    int64_t n_features = X->shape[1];
    float* Xd = (float*)X->data;

    int feat_total = (int)n_features;
    int choose = (max_features > 0 && max_features < feat_total) ? max_features : feat_total;
    int* feat_idx = (int*)malloc(sizeof(int) * (size_t)feat_total);
    for (int i = 0; i < feat_total; i++) feat_idx[i] = i;
    if (choose < feat_total) select_feature_subset(feat_idx, feat_total, choose);

    Pair* pairs = (Pair*)malloc(sizeof(Pair) * (size_t)n_samples);

    if (!is_regression) {
        int64_t* yd = (int64_t*)y->data;
        for (int fi = 0; fi < choose; fi++) {
            int f = feat_idx[fi];
            for (int64_t i = 0; i < n_samples; i++) {
                pairs[i].value = Xd[i * n_features + f];
                pairs[i].target = (float)yd[i];
            }
            qsort(pairs, (size_t)n_samples, sizeof(Pair), pair_cmp);

            int* left_counts = (int*)calloc((size_t)n_classes, sizeof(int));
            int* right_counts = (int*)calloc((size_t)n_classes, sizeof(int));
            for (int64_t i = 0; i < n_samples; i++) right_counts[(int)pairs[i].target]++;

            for (int64_t i = 0; i < n_samples - 1; i++) {
                int cls = (int)pairs[i].target;
                left_counts[cls]++;
                right_counts[cls]--;
                if (pairs[i].value == pairs[i + 1].value) continue;
                int left_n = (int)(i + 1);
                int right_n = (int)(n_samples - left_n);
                if (left_n < min_samples_leaf || right_n < min_samples_leaf) continue;

                float imp_left = impurity_class(left_counts, n_classes, left_n, criterion);
                float imp_right = impurity_class(right_counts, n_classes, right_n, criterion);
                float impurity = (left_n * imp_left + right_n * imp_right) / (float)n_samples;
                if (impurity < best.impurity) {
                    best.feature_idx = f;
                    best.threshold = (pairs[i].value + pairs[i + 1].value) * 0.5f;
                    best.impurity = impurity;
                }
            }
            free(left_counts);
            free(right_counts);
        }
    } else {
        float* yd_f = (float*)y->data;
        for (int fi = 0; fi < choose; fi++) {
            int f = feat_idx[fi];
            float total_sum = 0.0f;
            float total_sumsq = 0.0f;
            for (int64_t i = 0; i < n_samples; i++) {
                total_sum += yd_f[i];
                total_sumsq += yd_f[i] * yd_f[i];
                pairs[i].value = Xd[i * n_features + f];
                pairs[i].target = yd_f[i];
            }
            qsort(pairs, (size_t)n_samples, sizeof(Pair), pair_cmp);

            float left_sum = 0.0f, left_sumsq = 0.0f;
            for (int64_t i = 0; i < n_samples - 1; i++) {
                float t = pairs[i].target;
                left_sum += t;
                left_sumsq += t * t;
                total_sum -= t;
                total_sumsq -= t * t;
                if (pairs[i].value == pairs[i + 1].value) continue;
                int left_n = (int)(i + 1);
                int right_n = (int)(n_samples - left_n);
                if (left_n < min_samples_leaf || right_n < min_samples_leaf) continue;
                float imp_left = impurity_reg(left_sum, left_sumsq, left_n, criterion);
                float imp_right = impurity_reg(total_sum, total_sumsq, right_n, criterion);
                float impurity = (left_n * imp_left + right_n * imp_right) / (float)n_samples;
                if (impurity < best.impurity) {
                    best.feature_idx = f;
                    best.threshold = (pairs[i].value + pairs[i + 1].value) * 0.5f;
                    best.impurity = impurity;
                }
            }
        }
    }

    free(pairs);
    free(feat_idx);
    return best;
}
