/*
 * GreyML backend: ga io csv.
 *
 * File IO helpers for tensors, models, and CSV data used by the C backend and bindings.
 */

#include "greyarea/ga_io.h"
#include "greyarea/ga_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple CSV loader into a float32 tensor of shape [rows, cols].
// Only supports numeric values separated by commas. Lines with zero tokens are skipped.
GATensor* ga_csv_load(const char* path, bool has_header, int* n_rows, int* n_cols) {
    FILE* f = fopen(path, "r");
    if (!f) {
        ga_errno = GA_ERR_ALLOC_FAILED;
        return NULL;
    }

    char buffer[4096];
    if (has_header) {
        fgets(buffer, sizeof(buffer), f);
    }

    size_t capacity = 1024;
    size_t count = 0;
    float* values = (float*)malloc(capacity * sizeof(float));
    if (!values) {
        fclose(f);
        ga_errno = GA_ERR_ALLOC_FAILED;
        return NULL;
    }

    int rows = 0;
    int cols = -1;

    while (fgets(buffer, sizeof(buffer), f)) {
        if (buffer[0] == '\n' || buffer[0] == '\r' || buffer[0] == '\0') continue;

        int current_cols = 0;
        char* token = strtok(buffer, ",\n\r");
        while (token) {
            if (count >= capacity) {
                capacity *= 2;
                float* new_vals = (float*)realloc(values, capacity * sizeof(float));
                if (!new_vals) {
                    free(values);
                    fclose(f);
                    ga_errno = GA_ERR_ALLOC_FAILED;
                    return NULL;
                }
                values = new_vals;
            }
            values[count++] = (float)atof(token);
            current_cols++;
            token = strtok(NULL, ",\n\r");
        }

        if (current_cols == 0) continue;
        if (cols == -1) {
            cols = current_cols;
        } else if (current_cols != cols) {
            free(values);
            fclose(f);
            ga_errno = GA_ERR_INVALID_SHAPE;
            return NULL;
        }
        rows++;
    }

    fclose(f);

    if (rows == 0 || cols <= 0) {
        free(values);
        ga_errno = GA_ERR_INVALID_SHAPE;
        return NULL;
    }

    int64_t shape[2] = {rows, cols};
    GATensor* t = ga_tensor_empty(2, shape, GA_FLOAT32);
    if (!t) {
        free(values);
        return NULL;
    }

    memcpy(t->data, values, count * sizeof(float));
    free(values);

    if (n_rows) *n_rows = rows;
    if (n_cols) *n_cols = cols;
    return t;
}
