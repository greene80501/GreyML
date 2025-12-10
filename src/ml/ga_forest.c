/*
 * GreyML backend: ga forest.
 *
 * Classical machine learning algorithms built on the GreyML tensor core.
 */

#include "greyarea/ga_tree.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int ga_forest_default_max_features(bool is_regression, int n_features) {
    if (n_features <= 0) return 1;
    if (is_regression) {
        int v = (int)fmax(1.0, (double)n_features / 3.0);
        return v;
    }
    int v = (int)fmax(1.0, sqrt((double)n_features));
    return v;
}

GAForest* ga_forest_create(int n_trees, int max_depth, int n_classes) {
    GAForest* f = (GAForest*)calloc(1, sizeof(GAForest));
    f->n_trees = n_trees;
    f->max_depth = max_depth > 0 ? max_depth : 10;
    f->n_classes = n_classes;
    f->is_regression = false;
    f->trees = (GATree**)calloc((size_t)n_trees, sizeof(GATree*));
    return f;
}

GAForest* ga_forest_regressor_create(int n_trees, int max_depth, int max_features) {
    GAForest* f = (GAForest*)calloc(1, sizeof(GAForest));
    f->n_trees = n_trees;
    f->max_depth = max_depth > 0 ? max_depth : 10;
    f->n_classes = 0;
    f->is_regression = true;
    f->max_features = max_features;
    f->trees = (GATree**)calloc((size_t)n_trees, sizeof(GATree*));
    return f;
}

void ga_forest_free(GAForest* forest) {
    if (!forest) return;
    for (int i = 0; i < forest->n_trees; i++) {
        if (forest->trees[i]) ga_tree_free(forest->trees[i]);
    }
    free(forest->trees);
    free(forest);
}

void ga_forest_fit(GAForest* forest, GATensor* X, GATensor* y) {
    if (!forest || !X || !y) return;
    int64_t n = X->shape[0];
    int64_t d = X->shape[1];
    int max_features = forest->max_features > 0 ? forest->max_features : ga_forest_default_max_features(forest->is_regression, (int)d);

    for (int i = 0; i < forest->n_trees; i++) {
        int64_t x_shape[2] = {n, d};
        int64_t y_shape[1] = {n};
        GATensor* Xb = ga_tensor_empty(2, x_shape, GA_FLOAT32);
        GATensor* yb = ga_tensor_empty(1, y_shape, y->dtype);
        float* Xbd = (float*)Xb->data;
        float* Xd = (float*)X->data;
        char* ybd = (char*)yb->data;
        char* yd = (char*)y->data;
        size_t y_stride = y->dtype == GA_FLOAT32 ? sizeof(float) : sizeof(int64_t);
        for (int64_t r = 0; r < n; r++) {
            int idx = rand() % (int)n;
            memcpy(Xbd + r * d, Xd + idx * d, sizeof(float) * (size_t)d);
            memcpy(ybd + r * y_stride, yd + idx * y_stride, y_stride);
        }
        if (!forest->trees[i]) {
            forest->trees[i] = forest->is_regression ? ga_tree_regressor_create(forest->max_depth, 2, 1, max_features, TREE_CRITERION_MSE)
                                                     : ga_tree_classifier_create(forest->max_depth, 2, 1, max_features, TREE_CRITERION_GINI);
        } else {
            forest->trees[i]->max_features = max_features;
        }
        forest->trees[i]->n_classes = forest->n_classes;
        ga_tree_fit(forest->trees[i], Xb, yb);
        ga_tensor_release(Xb);
        ga_tensor_release(yb);
    }
}

GATensor* ga_forest_predict(GAForest* forest, GATensor* X) {
    if (!forest || forest->n_trees == 0) return NULL;
    int64_t n = X->shape[0];
    int64_t shape[1] = {n};

    if (forest->is_regression) {
        GATensor* out = ga_tensor_zeros(1, shape, GA_FLOAT32);
        float* od = (float*)out->data;
        for (int i = 0; i < forest->n_trees; i++) {
            GATensor* pred = ga_tree_predict(forest->trees[i], X);
            float* pd = (float*)pred->data;
            for (int64_t j = 0; j < n; j++) od[j] += pd[j];
            ga_tensor_release(pred);
        }
        float inv = 1.0f / (float)forest->n_trees;
        for (int64_t j = 0; j < n; j++) od[j] *= inv;
        return out;
    }

    GATensor* votes = ga_tensor_zeros(2, (int64_t[]){n, forest->n_classes}, GA_FLOAT32);
    float* vd = (float*)votes->data;
    for (int i = 0; i < forest->n_trees; i++) {
        GATensor* pred = ga_tree_predict(forest->trees[i], X);
        int64_t* pd = (int64_t*)pred->data;
        for (int64_t j = 0; j < n; j++) {
            int cls = (int)pd[j];
            if (cls >= 0 && cls < forest->n_classes) vd[j * forest->n_classes + cls] += 1.0f;
        }
        ga_tensor_release(pred);
    }
    GATensor* out = ga_tensor_empty(1, shape, GA_INT64);
    int64_t* od = (int64_t*)out->data;
    for (int64_t j = 0; j < n; j++) {
        int best = 0;
        float bestv = vd[j * forest->n_classes];
        for (int c = 1; c < forest->n_classes; c++) {
            float v = vd[j * forest->n_classes + c];
            if (v > bestv) { bestv = v; best = c; }
        }
        od[j] = best;
    }
    ga_tensor_release(votes);
    return out;
}
