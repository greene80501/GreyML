/*
 * GreyML C API header: ga io.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"
#include "ga_nn.h"

GA_API GAStatus ga_tensor_save(GATensor* tensor, const char* path);
GA_API GATensor* ga_tensor_load(const char* path);
GA_API GAStatus ga_model_save(GAModule* module, const char* path);
GA_API GAModule* ga_model_load(const char* path);
GA_API GATensor* ga_csv_load(const char* path, bool has_header, int* n_rows, int* n_cols);
