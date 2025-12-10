/*
 * GreyML C test: cluster.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include <assert.h>
#include "greyarea/greyarea.h"

int test_cluster_smoke(void) {
    // KMeans two obvious blobs
    int64_t x_shape[2] = {4, 2};
    GATensor* X = ga_tensor_empty(2, x_shape, GA_FLOAT32);
    float* xd = (float*)X->data;
    float points[] = {0.0f, 0.0f, 0.1f, 0.0f, 5.0f, 5.0f, 5.1f, 5.0f};
    for (int i = 0; i < 8; i++) xd[i] = points[i];

    GAKMeans* km = ga_kmeans_create(2, 30, 1e-4f, 3);
    ga_kmeans_fit(km, X);
    GATensor* labels = ga_kmeans_predict(km, X);
    int64_t* ld = (int64_t*)labels->data;
    assert(ld[0] == ld[1]);
    assert(ld[2] == ld[3]);
    assert(ld[0] != ld[2]);

    // DBSCAN simple cluster + noise
    GADBSCAN* db = ga_dbscan_create(0.5f, 2);
    GATensor* db_labels = ga_dbscan_fit_predict(db, X);
    int64_t* dbld = (int64_t*)db_labels->data;
    assert(dbld[0] == dbld[1]);
    ga_tensor_release(labels);
    ga_tensor_release(db_labels);
    ga_kmeans_free(km);
    ga_dbscan_free(db);
    ga_tensor_release(X);
    return 0;
}
