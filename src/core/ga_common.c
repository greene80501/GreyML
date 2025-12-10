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
    "Success",                            // 0
    "Memory allocation failed",           // -1
    "Invalid tensor shape or dimensions", // -2
    "Invalid data type",                  // -3
    "Index out of bounds",                // -4
    "Feature not yet implemented",        // -5
    "Division by zero",                   // -6
};

const char* ga_status_string(GAStatus status) {
    if (status > GA_SUCCESS || status < GA_ERR_DIV_BY_ZERO) {
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
