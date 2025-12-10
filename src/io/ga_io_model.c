/*
 * GreyML backend: ga io model.
 *
 * File IO helpers for tensors, models, and CSV data used by the C backend and bindings.
 */

#include "greyarea/ga_io.h"
#include "greyarea/ga_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Model serialization with lightweight metadata:
// magic "GAM\0", version u32, n_params u32
// for each param: name_len u32, name bytes (no null), dtype u32, ndim u32, shape[i64]*ndim, raw data
GAStatus ga_model_save(GAModule* module, const char* path) {
    if (!module || !path) return GA_ERR_INVALID_SHAPE;
    FILE* f = fopen(path, "wb");
    if (!f) return GA_ERR_ALLOC_FAILED;

    const char magic[4] = {'G', 'A', 'M', '\0'};
    uint32_t version = 1;
    fwrite(magic, 1, 4, f);
    fwrite(&version, sizeof(uint32_t), 1, f);

    uint32_t n_params = (uint32_t)module->n_params;
    fwrite(&n_params, sizeof(uint32_t), 1, f);

    for (size_t i = 0; i < module->n_params; i++) {
        GATensor* p = module->parameters[i];
        if (!p) { fclose(f); return GA_ERR_INVALID_SHAPE; }
        const char* pname = (module->param_names && module->param_names[i]) ? module->param_names[i] : "";
        uint32_t name_len = (uint32_t)strlen(pname);
        fwrite(&name_len, sizeof(uint32_t), 1, f);
        if (name_len > 0) fwrite(pname, 1, name_len, f);

        uint32_t dtype = (uint32_t)p->dtype;
        uint32_t ndim = (uint32_t)p->ndim;
        fwrite(&dtype, sizeof(uint32_t), 1, f);
        fwrite(&ndim, sizeof(uint32_t), 1, f);
        fwrite(p->shape, sizeof(int64_t), ndim, f);
        fwrite(p->data, ga_dtype_size(p->dtype), p->size, f);
    }

    fclose(f);
    return GA_SUCCESS;
}

GAModule* ga_model_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || strncmp(magic, "GAM", 3) != 0) {
        fclose(f);
        return NULL;
    }
    uint32_t version = 0;
    fread(&version, sizeof(uint32_t), 1, f);
    uint32_t n_params = 0;
    fread(&n_params, sizeof(uint32_t), 1, f);

    GAModule* module = (GAModule*)calloc(1, sizeof(GAModule));
    module->n_params = n_params;
    module->parameters = (GATensor**)calloc(n_params, sizeof(GATensor*));
    module->param_names = (char**)calloc(n_params, sizeof(char*));

    for (uint32_t i = 0; i < n_params; i++) {
        uint32_t name_len = 0;
        fread(&name_len, sizeof(uint32_t), 1, f);
        if (name_len > 0) {
            module->param_names[i] = (char*)calloc(name_len + 1, sizeof(char));
            fread(module->param_names[i], 1, name_len, f);
            module->param_names[i][name_len] = '\0';
        }

        uint32_t dtype = 0, ndim = 0;
        fread(&dtype, sizeof(uint32_t), 1, f);
        fread(&ndim, sizeof(uint32_t), 1, f);
        if (ndim > GA_MAX_DIMS) { fclose(f); return NULL; }
        int64_t shape[GA_MAX_DIMS];
        fread(shape, sizeof(int64_t), ndim, f);
        GATensor* t = ga_tensor_empty((int)ndim, shape, (GADtype)dtype);
        if (!t) { fclose(f); return NULL; }
        fread(t->data, ga_dtype_size(t->dtype), t->size, f);
        module->parameters[i] = t;
    }

    fclose(f);
    return module;
}
