/*
 * GreyML backend: ga tree.
 *
 * Classical machine learning algorithms built on the GreyML tensor core.
 */

#include "greyarea/ga_tree.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static float ga_tree_node_impurity(GATensor* y, GATree* tree) {
    int64_t n = y->shape[0];
    if (tree->is_regression) {
        float sum = 0.0f, sumsq = 0.0f;
        if (y->dtype == GA_FLOAT32) {
            float* yd = (float*)y->data;
            for (int64_t i = 0; i < n; i++) { sum += yd[i]; sumsq += yd[i] * yd[i]; }
        } else {
            int64_t* yd = (int64_t*)y->data;
            for (int64_t i = 0; i < n; i++) { float v = (float)yd[i]; sum += v; sumsq += v * v; }
        }
        float mean = sum / (float)n;
        return (sumsq / (float)n) - mean * mean;
    }
    int64_t* yd = (int64_t*)y->data;
    int* counts = (int*)calloc((size_t)tree->n_classes, sizeof(int));
    for (int64_t i = 0; i < n; i++) counts[yd[i]]++;
    float impurity = 0.0f;
    if (tree->criterion == TREE_CRITERION_ENTROPY) {
        for (int c = 0; c < tree->n_classes; c++) {
            if (counts[c] == 0) continue;
            float p = (float)counts[c] / (float)n;
            impurity -= p * logf(p + 1e-12f);
        }
    } else {
        float sumsq = 0.0f;
        for (int c = 0; c < tree->n_classes; c++) {
            float p = (float)counts[c] / (float)n;
            sumsq += p * p;
        }
        impurity = 1.0f - sumsq;
    }
    free(counts);
    return impurity;
}

static GATreeNode* ga_tree_create_leaf(GATensor* y, GATree* tree) {
    GATreeNode* node = (GATreeNode*)calloc(1, sizeof(GATreeNode));
    node->is_leaf = true;
    node->n_samples = (int)y->shape[0];
    node->impurity = ga_tree_node_impurity(y, tree);
    if (tree->is_regression) {
        node->value = (float*)calloc(1, sizeof(float));
        float sum = 0.0f;
        if (y->dtype == GA_FLOAT32) {
            float* yd = (float*)y->data;
            for (int64_t i = 0; i < y->shape[0]; i++) sum += yd[i];
        } else {
            int64_t* yd = (int64_t*)y->data;
            for (int64_t i = 0; i < y->shape[0]; i++) sum += (float)yd[i];
        }
        node->value[0] = sum / (float)y->shape[0];
    } else {
        node->value = (float*)calloc((size_t)tree->n_classes, sizeof(float));
        int64_t* yd = (int64_t*)y->data;
        for (int64_t i = 0; i < y->shape[0]; i++) {
            int cls = (int)yd[i];
            if (cls >= 0 && cls < tree->n_classes) node->value[cls] += 1.0f;
        }
    }
    return node;
}

static GATensor* ga_tree_filter_rows(GATensor* X, GATensor* y, int feature, float threshold, bool leq) {
    int64_t n = X->shape[0];
    int64_t d = X->shape[1];
    float* Xd = (float*)X->data;
    int count = 0;
    for (int64_t i = 0; i < n; i++) {
        float v = Xd[i * d + feature];
        if ((leq && v <= threshold) || (!leq && v > threshold)) count++;
    }
    if (count == 0) return NULL;
    int64_t x_shape[2] = {count, d};
    GATensor* Xout = ga_tensor_empty(2, x_shape, GA_FLOAT32);
    int64_t y_shape[1] = {count};
    GATensor* yout = ga_tensor_empty(1, y_shape, y->dtype);
    float* xo = (float*)Xout->data;
    char* yo = (char*)yout->data;
    size_t y_stride = y->dtype == GA_FLOAT32 ? sizeof(float) : sizeof(int64_t);

    int idx = 0;
    for (int64_t i = 0; i < n; i++) {
        float v = Xd[i * d + feature];
        if ((leq && v <= threshold) || (!leq && v > threshold)) {
            memcpy(xo + idx * d, Xd + i * d, sizeof(float) * (size_t)d);
            memcpy(yo + idx * y_stride, ((char*)y->data) + i * y_stride, y_stride);
            idx++;
        }
    }
    Xout->base = (GATensor*)yout;
    return Xout;
}

static void ga_tree_free_filtered(GATensor* Xf) {
    if (!Xf) return;
    if (Xf->base) ga_tensor_release(Xf->base);
    ga_tensor_release(Xf);
}

static GATreeNode* ga_tree_build(GATensor* X, GATensor* y, int depth, GATree* tree) {
    int64_t n = X->shape[0];
    float node_impurity = ga_tree_node_impurity(y, tree);
    bool stop_depth = tree->max_depth > 0 && depth >= tree->max_depth;
    if (stop_depth || n < tree->min_samples_split || node_impurity == 0.0f) {
        return ga_tree_create_leaf(y, tree);
    }

    if (!tree->is_regression) {
        int64_t* yd = (int64_t*)y->data;
        bool pure = true;
        for (int64_t i = 1; i < n; i++) {
            if (yd[i] != yd[0]) { pure = false; break; }
        }
        if (pure) return ga_tree_create_leaf(y, tree);
    }

    GATreeSplit split = ga_tree_find_best_split(X, y, tree->n_classes, tree->criterion, tree->max_features, tree->min_samples_leaf, tree->is_regression);
    if (split.feature_idx < 0) return ga_tree_create_leaf(y, tree);

    GATensor* X_left = ga_tree_filter_rows(X, y, split.feature_idx, split.threshold, true);
    GATensor* X_right = ga_tree_filter_rows(X, y, split.feature_idx, split.threshold, false);
    if (!X_left || !X_right) {
        ga_tree_free_filtered(X_left);
        ga_tree_free_filtered(X_right);
        return ga_tree_create_leaf(y, tree);
    }
    GATensor* y_left = X_left->base;
    GATensor* y_right = X_right->base;

    int64_t n_left = y_left->shape[0];
    int64_t n_right = y_right->shape[0];
    if (n_left < tree->min_samples_leaf || n_right < tree->min_samples_leaf) {
        ga_tree_free_filtered(X_left);
        ga_tree_free_filtered(X_right);
        return ga_tree_create_leaf(y, tree);
    }

    float imp_left = ga_tree_node_impurity(y_left, tree);
    float imp_right = ga_tree_node_impurity(y_right, tree);
    float decrease = node_impurity - ((float)n_left / (float)n) * imp_left - ((float)n_right / (float)n) * imp_right;
    if (tree->feature_importances && split.feature_idx < tree->n_features && decrease > 0) {
        tree->feature_importances[split.feature_idx] += decrease;
    }

    GATreeNode* node = (GATreeNode*)calloc(1, sizeof(GATreeNode));
    node->feature_idx = split.feature_idx;
    node->threshold = split.threshold;
    node->is_leaf = false;
    node->n_samples = (int)n;
    node->impurity = node_impurity;
    node->left = ga_tree_build(X_left, y_left, depth + 1, tree);
    node->right = ga_tree_build(X_right, y_right, depth + 1, tree);

    ga_tree_free_filtered(X_left);
    ga_tree_free_filtered(X_right);
    return node;
}

void ga_tree_fit(GATree* tree, GATensor* X, GATensor* y) {
    if (!tree || !X || !y) return;
    tree->n_features = (int)X->shape[1];
    if (!tree->is_regression) {
        int64_t* yd = (int64_t*)y->data;
        int max_cls = 0;
        for (int64_t i = 0; i < y->shape[0]; i++) if (yd[i] > max_cls) max_cls = (int)yd[i];
        tree->n_classes = max_cls + 1;
    } else {
        tree->n_classes = 0;
    }
    if (tree->feature_importances) free(tree->feature_importances);
    tree->feature_importances = (float*)calloc((size_t)tree->n_features, sizeof(float));
    tree->root = ga_tree_build(X, y, 0, tree);
}

static float ga_tree_predict_value(GATreeNode* node, float* xrow, bool regression, int n_classes) {
    if (node->is_leaf) {
        if (regression) return node->value[0];
        int best = 0;
        float bestv = node->value[0];
        for (int c = 1; c < n_classes; c++) {
            if (node->value[c] > bestv) { best = c; bestv = node->value[c]; }
        }
        return (float)best;
    }
    if (xrow[node->feature_idx] <= node->threshold) return ga_tree_predict_value(node->left, xrow, regression, n_classes);
    return ga_tree_predict_value(node->right, xrow, regression, n_classes);
}

GATensor* ga_tree_predict(GATree* tree, GATensor* X) {
    if (!tree || !tree->root) return NULL;
    int64_t n = X->shape[0];
    int64_t shape[1] = {n};
    GATensor* out = ga_tensor_empty(1, shape, tree->is_regression ? GA_FLOAT32 : GA_INT64);
    float* Xd = (float*)X->data;
    if (tree->is_regression) {
        float* od = (float*)out->data;
        for (int64_t i = 0; i < n; i++) {
            od[i] = ga_tree_predict_value(tree->root, Xd + i * X->shape[1], true, tree->n_classes);
        }
    } else {
        int64_t* od = (int64_t*)out->data;
        for (int64_t i = 0; i < n; i++) {
            od[i] = (int64_t)ga_tree_predict_value(tree->root, Xd + i * X->shape[1], false, tree->n_classes);
        }
    }
    return out;
}

GATensor* ga_tree_predict_proba(GATree* tree, GATensor* X) {
    if (!tree || tree->is_regression) return NULL;
    int64_t shape[2] = {X->shape[0], tree->n_classes};
    GATensor* out = ga_tensor_empty(2, shape, GA_FLOAT32);
    float* Xd = (float*)X->data;
    float* od = (float*)out->data;
    for (int64_t i = 0; i < shape[0]; i++) {
        GATreeNode* node = tree->root;
        while (!node->is_leaf) {
            if (Xd[i * X->shape[1] + node->feature_idx] <= node->threshold) node = node->left;
            else node = node->right;
        }
        float total = 0.0f;
        for (int c = 0; c < tree->n_classes; c++) total += node->value[c];
        for (int c = 0; c < tree->n_classes; c++) {
            float prob = (total > 0.0f) ? node->value[c] / total : 1.0f / tree->n_classes;
            od[i * tree->n_classes + c] = prob;
        }
    }
    return out;
}

GATensor* ga_tree_feature_importances(GATree* tree) {
    if (!tree || !tree->feature_importances) return NULL;
    int64_t shape[1] = {tree->n_features};
    GATensor* out = ga_tensor_empty(1, shape, GA_FLOAT32);
    float* od = (float*)out->data;
    float total = 0.0f;
    for (int i = 0; i < tree->n_features; i++) total += tree->feature_importances[i];
    for (int i = 0; i < tree->n_features; i++) {
        if (total > 0) od[i] = tree->feature_importances[i] / total;
        else od[i] = 0.0f;
    }
    return out;
}

GATree* ga_tree_classifier_create(int max_depth, int min_samples_split, int min_samples_leaf, int max_features, GATreeCriterion criterion) {
    GATree* t = (GATree*)calloc(1, sizeof(GATree));
    t->max_depth = max_depth > 0 ? max_depth : INT_MAX;
    t->min_samples_split = min_samples_split > 1 ? min_samples_split : 2;
    t->min_samples_leaf = min_samples_leaf > 0 ? min_samples_leaf : 1;
    t->max_features = max_features;
    t->criterion = criterion;
    t->is_regression = false;
    return t;
}

GATree* ga_tree_regressor_create(int max_depth, int min_samples_split, int min_samples_leaf, int max_features, GATreeCriterion criterion) {
    GATree* t = ga_tree_classifier_create(max_depth, min_samples_split, min_samples_leaf, max_features, criterion);
    t->is_regression = true;
    return t;
}

void ga_tree_free(GATree* tree) {
    if (!tree) return;
    GATreeNode* stack[1024];
    int top = 0;
    if (tree->root) stack[top++] = tree->root;
    while (top > 0) {
        GATreeNode* n = stack[--top];
        if (n->left) stack[top++] = n->left;
        if (n->right) stack[top++] = n->right;
        free(n->value);
        free(n);
    }
    if (tree->feature_importances) free(tree->feature_importances);
    free(tree);
}
