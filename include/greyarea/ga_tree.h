/*
 * GreyML C API header: ga tree.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"

typedef enum {
    TREE_CRITERION_GINI,
    TREE_CRITERION_ENTROPY,
    TREE_CRITERION_MSE,
    TREE_CRITERION_MAE,
} GATreeCriterion;

typedef struct GATreeNode {
    int feature_idx;
    float threshold;
    struct GATreeNode* left;
    struct GATreeNode* right;
    float* value;
    int n_samples;
    float impurity;
    bool is_leaf;
} GATreeNode;

typedef struct {
    GATreeNode* root;
    int max_depth;
    int min_samples_split;
    int min_samples_leaf;
    int max_features;
    GATreeCriterion criterion;
    int n_classes;
    int n_features;
    bool is_regression;
    float* feature_importances;
} GATree;

typedef struct {
    int feature_idx;
    float threshold;
    float impurity;
} GATreeSplit;

typedef struct {
    GATree** trees;
    int n_trees;
    int max_depth;
    int n_classes;
    int max_features;
    bool is_regression;
} GAForest;

GA_API GATree* ga_tree_classifier_create(int max_depth, int min_samples_split, int min_samples_leaf, int max_features, GATreeCriterion criterion);
GA_API GATree* ga_tree_regressor_create(int max_depth, int min_samples_split, int min_samples_leaf, int max_features, GATreeCriterion criterion);
GA_API void ga_tree_free(GATree* tree);
GA_API void ga_tree_fit(GATree* tree, GATensor* X, GATensor* y);
GA_API GATensor* ga_tree_predict(GATree* tree, GATensor* X);
GA_API GATensor* ga_tree_predict_proba(GATree* tree, GATensor* X);
GA_API GATensor* ga_tree_feature_importances(GATree* tree);

GA_API GAForest* ga_forest_create(int n_trees, int max_depth, int n_classes);
GA_API GAForest* ga_forest_regressor_create(int n_trees, int max_depth, int max_features);
GA_API void ga_forest_free(GAForest* forest);
GA_API void ga_forest_fit(GAForest* forest, GATensor* X, GATensor* y);
GA_API GATensor* ga_forest_predict(GAForest* forest, GATensor* X);

// Split helper
GA_API GATreeSplit ga_tree_find_best_split(GATensor* X, GATensor* y, int n_classes, GATreeCriterion criterion, int max_features, int min_samples_leaf, bool is_regression);
