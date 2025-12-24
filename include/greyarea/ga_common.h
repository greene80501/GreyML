/*
 * GreyML C API header: ga common.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef _WIN32
    #ifdef GREYAREA_EXPORTS
        #define GA_API __declspec(dllexport)
    #else
        #define GA_API __declspec(dllimport)
    #endif
#else
    #define GA_API
#endif

typedef enum {
    GA_SUCCESS = 0,
    GA_ERR_ALLOC_FAILED = -1,
    GA_ERR_INVALID_SHAPE = -2,
    GA_ERR_INVALID_DTYPE = -3,
    GA_ERR_OUT_OF_BOUNDS = -4,
    GA_ERR_NOT_IMPLEMENTED = -5,
    GA_ERR_DIV_BY_ZERO = -6,
} GAStatus;

extern GAStatus ga_errno;

typedef enum {
    GA_FLOAT32 = 0,
    GA_FLOAT64 = 1,
    GA_INT32 = 2,
    GA_INT64 = 3,
    GA_UINT8 = 4,
    GA_BOOL = 5,
} GADtype;

static inline size_t ga_dtype_size(GADtype dtype) {
    static const size_t sizes[] = {4, 8, 4, 8, 1, 1};
    return sizes[dtype];
}

#define GA_MAX_DIMS 8
#define GA_ALIGN_SIZE 64

#define GA_REDUCE_NONE 0
#define GA_REDUCE_MEAN 1
#define GA_REDUCE_SUM 2

GA_API const char* ga_status_string(GAStatus status);
GA_API void ga_log_error(const char* msg);
GA_API void ga_log_errorf(const char* fmt, ...);
GA_API void ga_log_warning(const char* msg);
GA_API void ga_log_warningf(const char* fmt, ...);
GA_API void ga_log_info(const char* msg);
GA_API void ga_log_infof(const char* fmt, ...);
GA_API void ga_set_error(GAStatus status, const char* msg);