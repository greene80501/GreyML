/*
 * GreyML backend: ga cluster kmeans.
 *
 * Classical machine learning algorithms built on the GreyML tensor core.
 */

#include "greyarea/ga_cluster.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void ga_kmeans_copy_centroids(float* dst, const float* src, int clusters, int n_features) {
    memcpy(dst, src, sizeof(float) * (size_t)clusters * n_features);
}

static float ga_kmeans_assign(const float* Xd, int n_samples, int n_features, float* centroids, int k, int* labels, float* dist_buf) {
    float inertia = 0.0f;
    for (int i = 0; i < n_samples; i++) {
        float best = 1e30f;
        int best_idx = 0;
        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            for (int f = 0; f < n_features; f++) {
                float diff = Xd[i * n_features + f] - centroids[c * n_features + f];
                dist += diff * diff;
            }
            if (dist < best) { best = dist; best_idx = c; }
        }
        labels[i] = best_idx;
        dist_buf[i] = best;
        inertia += best;
    }
    return inertia;
}

static void ga_kmeans_reseed_empty(const float* Xd, int n_samples, int n_features, float* centroids, int k, int* labels, const float* dist_buf) {
    // Find farthest point overall and assign to empty clusters
    for (int c = 0; c < k; c++) {
        bool has_points = false;
        for (int i = 0; i < n_samples; i++) {
            if (labels[i] == c) { has_points = true; break; }
        }
        if (has_points) continue;
        // find farthest point
        int far_idx = 0;
        float far_dist = -1.0f;
        for (int i = 0; i < n_samples; i++) {
            if (dist_buf[i] > far_dist) {
                far_dist = dist_buf[i];
                far_idx = i;
            }
        }
        labels[far_idx] = c;
        for (int f = 0; f < n_features; f++) {
            centroids[c * n_features + f] = Xd[far_idx * n_features + f];
        }
    }
}

static float ga_kmeans_update_centroids(const float* Xd, int n_samples, int n_features, float* centroids, int k, const int* labels) {
    float* newC = (float*)calloc((size_t)k * n_features, sizeof(float));
    int* counts = (int*)calloc((size_t)k, sizeof(int));
    for (int i = 0; i < n_samples; i++) {
        int lbl = labels[i];
        counts[lbl]++;
        for (int f = 0; f < n_features; f++) {
            newC[lbl * n_features + f] += Xd[i * n_features + f];
        }
    }
    float shift = 0.0f;
    for (int c = 0; c < k; c++) {
        if (counts[c] == 0) continue;
        for (int f = 0; f < n_features; f++) {
            float updated = newC[c * n_features + f] / (float)counts[c];
            float diff = updated - centroids[c * n_features + f];
            shift += diff * diff;
            centroids[c * n_features + f] = updated;
        }
    }
    free(newC);
    free(counts);
    return shift;
}

GAKMeans* ga_kmeans_create(int n_clusters, int max_iter, float tol, int n_init) {
    GAKMeans* km = (GAKMeans*)calloc(1, sizeof(GAKMeans));
    km->n_clusters = n_clusters;
    km->max_iter = max_iter > 0 ? max_iter : 100;
    km->tol = tol > 0 ? tol : 1e-4f;
    km->n_init = n_init > 0 ? n_init : 1;
    km->inertia_ = 0.0f;
    km->n_iter_ = 0;
    return km;
}

void ga_kmeans_free(GAKMeans* kmeans) {
    if (!kmeans) return;
    if (kmeans->centroids) ga_tensor_release(kmeans->centroids);
    free(kmeans);
}

void ga_kmeans_fit(GAKMeans* kmeans, GATensor* X) {
    if (!kmeans || !X || X->ndim < 2) return;
    int n_samples = (int)X->shape[0];
    int n_features = (int)X->shape[1];
    float* Xd = (float*)X->data;

    float best_inertia = 1e30f;
    float* best_centroids = (float*)calloc((size_t)kmeans->n_clusters * n_features, sizeof(float));
    int best_iter = 0;

    int* labels = (int*)calloc((size_t)n_samples, sizeof(int));
    float* dist_buf = (float*)calloc((size_t)n_samples, sizeof(float));
    float* centroids = (float*)calloc((size_t)kmeans->n_clusters * n_features, sizeof(float));

    for (int init = 0; init < kmeans->n_init; init++) {
        // init centroids using random samples
        for (int c = 0; c < kmeans->n_clusters; c++) {
            int idx = rand() % n_samples;
            for (int f = 0; f < n_features; f++) {
                centroids[c * n_features + f] = Xd[idx * n_features + f];
            }
        }
        float inertia = 0.0f;
        float shift = 0.0f;
        int iter = 0;
        for (; iter < kmeans->max_iter; iter++) {
            inertia = ga_kmeans_assign(Xd, n_samples, n_features, centroids, kmeans->n_clusters, labels, dist_buf);
            ga_kmeans_reseed_empty(Xd, n_samples, n_features, centroids, kmeans->n_clusters, labels, dist_buf);
            shift = ga_kmeans_update_centroids(Xd, n_samples, n_features, centroids, kmeans->n_clusters, labels);
            if (shift < kmeans->tol) break;
        }
        if (inertia < best_inertia) {
            best_inertia = inertia;
            best_iter = iter + 1;
            ga_kmeans_copy_centroids(best_centroids, centroids, kmeans->n_clusters, n_features);
        }
    }

    if (kmeans->centroids) ga_tensor_release(kmeans->centroids);
    int64_t shape[2] = {kmeans->n_clusters, n_features};
    kmeans->centroids = ga_tensor_empty(2, shape, GA_FLOAT32);
    memcpy(kmeans->centroids->data, best_centroids, sizeof(float) * (size_t)kmeans->n_clusters * n_features);
    kmeans->inertia_ = best_inertia;
    kmeans->n_iter_ = best_iter;

    free(best_centroids);
    free(labels);
    free(dist_buf);
    free(centroids);
}

GATensor* ga_kmeans_predict(GAKMeans* kmeans, GATensor* X) {
    if (!kmeans || !kmeans->centroids) return NULL;
    int n_samples = (int)X->shape[0];
    int n_features = (int)X->shape[1];
    float* Xd = (float*)X->data;
    float* Cd = (float*)kmeans->centroids->data;
    int64_t shape[1] = {n_samples};
    GATensor* labels = ga_tensor_empty(1, shape, GA_INT64);
    int64_t* Ld = (int64_t*)labels->data;
    for (int i = 0; i < n_samples; i++) {
        float best = 1e30f;
        int best_idx = 0;
        for (int c = 0; c < kmeans->n_clusters; c++) {
            float dist = 0.0f;
            for (int f = 0; f < n_features; f++) {
                float d = Xd[i * n_features + f] - Cd[c * n_features + f];
                dist += d * d;
            }
            if (dist < best) { best = dist; best_idx = c; }
        }
        Ld[i] = best_idx;
    }
    return labels;
}
