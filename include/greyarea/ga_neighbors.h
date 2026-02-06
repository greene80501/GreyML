/*
 * GreyML C API header: ga neighbors.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"

typedef enum {
    KNN_WEIGHT_UNIFORM,
    KNN_WEIGHT_DISTANCE,
} KNNWeightType;

typedef struct {
    int n_neighbors;
    KNNWeightType weights;
    GATensor* X_train;
    GATensor* y_train;
    int n_classes;
    bool is_regressor;
} GAKNN;

GA_API GAKNN* ga_knn_classifier_create(int n_neighbors, KNNWeightType weights);
GA_API GAKNN* ga_knn_regressor_create(int n_neighbors, KNNWeightType weights);
GA_API void ga_knn_free(GAKNN* knn);
GA_API void ga_knn_fit(GAKNN* knn, GATensor* X, GATensor* y);
GA_API GATensor* ga_knn_predict(GAKNN* knn, GATensor* X);
GA_API void ga_knn_kneighbors(GAKNN* knn, GATensor* X, int k, GATensor** distances, GATensor** indices);
