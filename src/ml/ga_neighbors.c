/*
 * GreyML backend: ga neighbors.
 *
 * Classical machine learning algorithms built on the GreyML tensor core.
 */

#include "greyarea/ga_neighbors.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float dist;
    int idx;
} Neighbor;

static void ga_knn_insert_neighbor(Neighbor* buf, int k, float dist, int idx) {
    for (int i = 0; i < k; i++) {
        if (dist < buf[i].dist) {
            for (int s = k - 1; s > i; s--) buf[s] = buf[s - 1];
            buf[i].dist = dist;
            buf[i].idx = idx;
            return;
        }
    }
}

GAKNN* ga_knn_classifier_create(int n_neighbors, KNNWeightType weights) {
    GAKNN* knn = (GAKNN*)calloc(1, sizeof(GAKNN));
    knn->n_neighbors = n_neighbors;
    knn->weights = weights;
    knn->is_regressor = false;
    return knn;
}

GAKNN* ga_knn_regressor_create(int n_neighbors, KNNWeightType weights) {
    GAKNN* knn = ga_knn_classifier_create(n_neighbors, weights);
    knn->is_regressor = true;
    return knn;
}

void ga_knn_free(GAKNN* knn) {
    if (!knn) return;
    if (knn->X_train) ga_tensor_release(knn->X_train);
    if (knn->y_train) ga_tensor_release(knn->y_train);
    free(knn);
}

void ga_knn_fit(GAKNN* knn, GATensor* X, GATensor* y) {
    if (!knn) return;
    knn->X_train = X;
    knn->y_train = y;
    if (X) ga_tensor_retain(X);
    if (y) ga_tensor_retain(y);
    // infer classes for classification
    if (!knn->is_regressor && y) {
        int64_t n = y->shape[0];
        int64_t* yd = (int64_t*)y->data;
        int max_cls = 0;
        for (int64_t i = 0; i < n; i++) if (yd[i] > max_cls) max_cls = (int)yd[i];
        knn->n_classes = max_cls + 1;
    }
}

static void ga_knn_find_neighbors(GAKNN* knn, float* Xt, int64_t n_feat, int64_t n_train, float* Xtr, int k, Neighbor* buf) {
    for (int i = 0; i < k; i++) { buf[i].dist = 1e30f; buf[i].idx = -1; }
    for (int64_t j = 0; j < n_train; j++) {
        float dist = 0.0f;
        for (int64_t f = 0; f < n_feat; f++) {
            float diff = Xt[f] - Xtr[j * n_feat + f];
            dist += diff * diff;
        }
        ga_knn_insert_neighbor(buf, k, dist, (int)j);
    }
}

GATensor* ga_knn_predict(GAKNN* knn, GATensor* X) {
    if (!knn || !knn->X_train || !knn->y_train) return NULL;
    int64_t n_train = knn->X_train->shape[0];
    int64_t n_feat = knn->X_train->shape[1];
    int64_t n = X->shape[0];
    float* Xtr = (float*)knn->X_train->data;
    float* Xt = (float*)X->data;
    int k = knn->n_neighbors;
    if (k > n_train) k = (int)n_train;
    Neighbor* buf = (Neighbor*)malloc(sizeof(Neighbor) * (size_t)k);

    if (knn->is_regressor) {
        float* ytr_f;
        float* tmp = NULL;
        if (knn->y_train->dtype == GA_FLOAT32) {
            ytr_f = (float*)knn->y_train->data;
        } else {
            tmp = (float*)calloc((size_t)n_train, sizeof(float));
            int64_t* yi = (int64_t*)knn->y_train->data;
            for (int64_t i = 0; i < n_train; i++) tmp[i] = (float)yi[i];
            ytr_f = tmp;
        }
        int64_t shape[1] = {n};
        GATensor* out = ga_tensor_empty(1, shape, GA_FLOAT32);
        float* od = (float*)out->data;
        for (int64_t i = 0; i < n; i++) {
            ga_knn_find_neighbors(knn, Xt + i * n_feat, n_feat, n_train, Xtr, k, buf);
            double num = 0.0, denom = 0.0;
            for (int t = 0; t < k; t++) {
                float dist = buf[t].dist;
                float w = (knn->weights == KNN_WEIGHT_DISTANCE) ? 1.0f / (sqrtf(dist) + 1e-8f) : 1.0f;
                num += w * ytr_f[buf[t].idx];
                denom += w;
            }
            od[i] = (float)(num / (denom > 0 ? denom : 1.0));
        }
        free(buf);
        if (tmp) free(tmp);
        return out;
    }

    int64_t shape[1] = {n};
    GATensor* out = ga_tensor_empty(1, shape, GA_INT64);
    int64_t* od = (int64_t*)out->data;
    int classes = knn->n_classes > 0 ? knn->n_classes : 32;
    float* votes = (float*)calloc((size_t)classes, sizeof(float));
    int64_t* ytr = (int64_t*)knn->y_train->data;

    for (int64_t i = 0; i < n; i++) {
        memset(votes, 0, sizeof(float) * (size_t)classes);
        ga_knn_find_neighbors(knn, Xt + i * n_feat, n_feat, n_train, Xtr, k, buf);
        for (int t = 0; t < k; t++) {
            float dist = buf[t].dist;
            float w = (knn->weights == KNN_WEIGHT_DISTANCE) ? 1.0f / (sqrtf(dist) + 1e-8f) : 1.0f;
            int cls = (int)ytr[buf[t].idx];
            if (cls >= 0 && cls < classes) votes[cls] += w;
        }
        int best_cls = 0;
        float best_vote = votes[0];
        for (int c = 1; c < classes; c++) {
            if (votes[c] > best_vote) { best_vote = votes[c]; best_cls = c; }
        }
        od[i] = best_cls;
    }
    free(votes);
    free(buf);
    return out;
}

void ga_knn_kneighbors(GAKNN* knn, GATensor* X, int k, GATensor** distances, GATensor** indices) {
    if (!knn || !knn->X_train || !X || !distances || !indices) return;
    int64_t n_train = knn->X_train->shape[0];
    int64_t n_feat = knn->X_train->shape[1];
    int64_t n = X->shape[0];
    if (k > n_train) k = (int)n_train;
    float* Xtr = (float*)knn->X_train->data;
    float* Xt = (float*)X->data;

    int64_t shape[2] = {n, k};
    *distances = ga_tensor_empty(2, shape, GA_FLOAT32);
    *indices = ga_tensor_empty(2, shape, GA_INT64);
    float* dd = (float*)(*distances)->data;
    int64_t* id = (int64_t*)(*indices)->data;
    Neighbor* buf = (Neighbor*)malloc(sizeof(Neighbor) * (size_t)k);

    for (int64_t i = 0; i < n; i++) {
        ga_knn_find_neighbors(knn, Xt + i * n_feat, n_feat, n_train, Xtr, k, buf);
        for (int t = 0; t < k; t++) {
            dd[i * k + t] = sqrtf(buf[t].dist);
            id[i * k + t] = buf[t].idx;
        }
    }
    free(buf);
}
