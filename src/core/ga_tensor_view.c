/*
 * GreyML backend: ga tensor view.
 *
 * Foundational runtime utilities including error handling, memory management, random utilities, and tensor lifecycle helpers.
 */

#include "greyarea/ga_tensor.h"

// Stub file - implementations moved to ga_tensor.c to avoid duplicate symbols
// This file only contains declarations; actual code is in ga_tensor.c

// Forward declaration to suppress compiler warnings
extern GATensor* ga_tensor_slice(GATensor* tensor, const int64_t* start, const int64_t* end, const int64_t* step);