/*
 * GreyML backend: ga common.
 *
 * Foundational runtime utilities including error handling, memory management, random utilities, and tensor lifecycle helpers.
 */

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "greyarea/ga_common.h"

GAStatus ga_errno = GA_SUCCESS;

static const char* ga_status_messages[] = {
    [GA_SUCCESS] = "Success",
    [GA_ERR_ALLOC_FAILED] = "Memory allocation failed",
    [GA_ERR_INVALID_SHAPE] = "Invalid tensor shape or dimensions",
    [GA_ERR_INVALID_DTYPE] = "Invalid data type",
    [GA_ERR_OUT_OF_BOUNDS] = "Index out of bounds",
    [GA_ERR_NOT_IMPLEMENTED] = "Feature not yet implemented",
    [GA_ERR_DIV_BY_ZERO] = "Division by zero",
};

const char* ga_status_string(GAStatus status) {
    if (status >= 0 || status < GA_ERR_DIV_BY_ZERO) {
        return "Unknown error";
    }
    return ga_status_messages[-status];
}

void ga_log_error(const char* msg) {
    fprintf(stderr, "[GREYAREA ERROR] %s\n", msg);
}

void ga_log_errorf(const char* fmt, ...) {
    fprintf(stderr, "[GREYAREA ERROR] ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void ga_log_warning(const char* msg) {
    fprintf(stderr, "[GREYAREA WARNING] %s\n", msg);
}

void ga_log_warningf(const char* fmt, ...) {
    fprintf(stderr, "[GREYAREA WARNING] ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

void ga_log_info(const char* msg) {
    fprintf(stdout, "[GREYAREA INFO] %s\n", msg);
}

void ga_log_infof(const char* fmt, ...) {
    fprintf(stdout, "[GREYAREA INFO] ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
    fprintf(stdout, "\n");
}

void ga_set_error(GAStatus status, const char* msg) {
    ga_errno = status;
    if (msg) {
        ga_log_errorf("%s: %s", ga_status_string(status), msg);
    } else {
        ga_log_error(ga_status_string(status));
    }
}