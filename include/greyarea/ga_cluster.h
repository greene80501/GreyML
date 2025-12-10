/*
 * GreyML C API header: ga cluster.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"

typedef struct {
    int n_clusters;
    int max_iter;
    float tol;
    int n_init;
    GATensor* centroids;
    float inertia_;
    int n_iter_;
} GAKMeans;

typedef struct {
    float eps;
    int min_samples;
    int* labels_;
    int n_clusters;
} GADBSCAN;

GA_API GAKMeans* ga_kmeans_create(int n_clusters, int max_iter, float tol, int n_init);
GA_API void ga_kmeans_free(GAKMeans* kmeans);
GA_API void ga_kmeans_fit(GAKMeans* kmeans, GATensor* X);
GA_API GATensor* ga_kmeans_predict(GAKMeans* kmeans, GATensor* X);

GA_API GADBSCAN* ga_dbscan_create(float eps, int min_samples);
GA_API void ga_dbscan_free(GADBSCAN* dbscan);
GA_API GATensor* ga_dbscan_fit_predict(GADBSCAN* dbscan, GATensor* X);
