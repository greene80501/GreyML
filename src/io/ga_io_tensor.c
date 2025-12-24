/*
 * GreyML backend: ga io tensor.
 *
 * File IO helpers for tensors, models, and CSV data used by the C backend and bindings.
 */

#include "greyarea/ga_io.h"
#include <stdio.h>
#include <string.h>

// Simple binary tensor format:
// magic[4], version[u32], dtype[u32], ndim[u32], shape[ndim*i64], data

GAStatus ga_tensor_save(GATensor* tensor, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return GA_ERR_ALLOC_FAILED;
    
    const char magic[4] = {'G','A','T','\0'};
    fwrite(magic, 1, 4, f);
    uint32_t version = 1;
    fwrite(&version, sizeof(uint32_t), 1, f);
    uint32_t dtype = tensor->dtype;
    fwrite(&dtype, sizeof(uint32_t), 1, f);
    uint32_t ndim = tensor->ndim;
    fwrite(&ndim, sizeof(uint32_t), 1, f);
    fwrite(tensor->shape, sizeof(int64_t), ndim, f);
    fwrite(tensor->data, ga_dtype_size(tensor->dtype), tensor->size, f);
    fclose(f);
    return GA_SUCCESS;
}

GATensor* ga_tensor_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    fread(magic, 1, 4, f);
    if (strncmp(magic, "GAT", 3) != 0) { fclose(f); return NULL; }
    uint32_t version = 0, dtype = 0, ndim = 0;
    fread(&version, sizeof(uint32_t), 1, f);
    fread(&dtype, sizeof(uint32_t), 1, f);
    fread(&ndim, sizeof(uint32_t), 1, f);
    if (ndim > GA_MAX_DIMS) { fclose(f); return NULL; }
    int64_t shape[GA_MAX_DIMS];
    fread(shape, sizeof(int64_t), ndim, f);
    GATensor* t = ga_tensor_empty((int)ndim, shape, (GADtype)dtype);
    if (!t) { fclose(f); return NULL; }
    fread(t->data, ga_dtype_size(t->dtype), t->size, f);
    fclose(f);
    return t;
}
