/*
 * GreyML C API header: ga random.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"

GA_API void ga_random_seed(uint64_t seed);
GA_API uint32_t ga_random_uint32(void);
GA_API float ga_random_float(void);
GA_API float ga_random_normal(void);
GA_API void ga_random_shuffle(int* array, size_t n);
GA_API void ga_tensor_rand_(GATensor* tensor);
GA_API void ga_tensor_randn_(GATensor* tensor);