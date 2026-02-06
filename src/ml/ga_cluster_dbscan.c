/*
 * GreyML backend: ga cluster dbscan.
 *
 * Classical machine learning algorithms built on the GreyML tensor core.
 */

#include "greyarea/ga_cluster.h"
#include <stdlib.h>
#include <stdbool.h>

GADBSCAN* ga_dbscan_create(float eps, int min_samples) {
    GADBSCAN* db = (GADBSCAN*)calloc(1, sizeof(GADBSCAN));
    db->eps = eps;
    db->min_samples = min_samples;
    return db;
}

void ga_dbscan_free(GADBSCAN* dbscan) {
    if (!dbscan) return;
    if (dbscan->labels_) free(dbscan->labels_);
    free(dbscan);
}

static int ga_dbscan_region_query(const float* X, int64_t n, int64_t d, int idx, float eps2, int* neighbors) {
    int count = 0;
    const float* xi = X + idx * d;
    for (int64_t j = 0; j < n; j++) {
        const float* xj = X + j * d;
        float dist = 0.0f;
        for (int64_t k = 0; k < d; k++) {
            float diff = xi[k] - xj[k];
            dist += diff * diff;
        }
        if (dist <= eps2) neighbors[count++] = (int)j;
    }
    return count;
}

GATensor* ga_dbscan_fit_predict(GADBSCAN* dbscan, GATensor* X) {
    if (!dbscan || !X || X->ndim < 2) return NULL;
    int64_t n = X->shape[0];
    int64_t d = X->shape[1];
    float* Xd = (float*)X->data;
    int64_t shape[1] = {n};
    GATensor* labels = ga_tensor_empty(1, shape, GA_INT64);
    int64_t* lbl = (int64_t*)labels->data;
    for (int64_t i = 0; i < n; i++) lbl[i] = -1;

    bool* visited = (bool*)calloc((size_t)n, sizeof(bool));
    int* neighbors = (int*)calloc((size_t)n, sizeof(int));
    int* queue = (int*)calloc((size_t)n, sizeof(int));
    int cluster_id = 0;
    float eps2 = dbscan->eps * dbscan->eps;

    for (int64_t i = 0; i < n; i++) {
        if (visited[i]) continue;
        visited[i] = true;
        int neighbor_count = ga_dbscan_region_query(Xd, n, d, (int)i, eps2, neighbors);
        if (neighbor_count < dbscan->min_samples) {
            lbl[i] = -1;
            continue;
        }
        lbl[i] = cluster_id;
        int q_head = 0, q_tail = 0;
        for (int ni = 0; ni < neighbor_count; ni++) queue[q_tail++] = neighbors[ni];
        while (q_head < q_tail) {
            int j = queue[q_head++];
            if (!visited[j]) {
                visited[j] = true;
                int nb2 = ga_dbscan_region_query(Xd, n, d, j, eps2, neighbors);
                if (nb2 >= dbscan->min_samples) {
                    for (int t = 0; t < nb2; t++) queue[q_tail++] = neighbors[t];
                }
            }
            if (lbl[j] == -1) lbl[j] = cluster_id;
        }
        cluster_id++;
    }
    dbscan->n_clusters = cluster_id;
    if (dbscan->labels_) free(dbscan->labels_);
    dbscan->labels_ = (int*)calloc((size_t)n, sizeof(int));
    for (int64_t i = 0; i < n; i++) dbscan->labels_[i] = (int)lbl[i];

    free(visited);
    free(neighbors);
    free(queue);
    return labels;
}
