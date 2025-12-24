/*
 * GreyML backend: ga nn embed.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#include <stdlib.h>
#include "greyarea/ga_nn.h"
#include "greyarea/ga_common.h"

static void embedding_forward_wrapper(void* self, GATensor* indices, GATensor** output) {
    GAEmbedding* emb = (GAEmbedding*)self;
    int64_t nd = indices->ndim;
    if (nd != 1) {
        ga_errno = GA_ERR_INVALID_SHAPE;
        *output = NULL;
        return;
    }
    int64_t n = indices->shape[0];
    int64_t out_shape[2] = {n, emb->embedding_dim};
    GATensor* out = ga_tensor_empty(2, out_shape, GA_FLOAT32);
    int64_t* idx = (int64_t*)indices->data;
    float* dst = (float*)out->data;
    float* w = (float*)emb->weight->data;
    for (int64_t i = 0; i < n; i++) {
        int64_t id = idx[i];
        if (id < 0 || id >= emb->num_embeddings) { ga_errno = GA_ERR_OUT_OF_BOUNDS; continue; }
        memcpy(dst + i * emb->embedding_dim, w + id * emb->embedding_dim, emb->embedding_dim * sizeof(float));
    }
    *output = out;
}

GAEmbedding* ga_embedding_create(int num_embeddings, int embedding_dim) {
    GAEmbedding* emb = (GAEmbedding*)calloc(1, sizeof(GAEmbedding));
    emb->num_embeddings = num_embeddings;
    emb->embedding_dim = embedding_dim;
    int64_t wshape[2] = {num_embeddings, embedding_dim};
    emb->weight = ga_tensor_empty(2, wshape, GA_FLOAT32);
    ga_init_kaiming_uniform(emb->weight);
    emb->base.parameters = (GATensor**)calloc(1, sizeof(GATensor*));
    emb->base.parameters[0] = emb->weight;
    emb->base.n_params = 1;
    emb->base.forward_fn = embedding_forward_wrapper;
    return emb;
}

void ga_embedding_free(GAEmbedding* emb) {
    if (!emb) return;
    if (emb->weight) ga_tensor_release(emb->weight);
    if (emb->base.parameters) free(emb->base.parameters);
    free(emb);
}

GATensor* ga_embedding_forward(GAEmbedding* emb, GATensor* indices) {
    if (!emb) return NULL;
    GATensor* out = NULL;
    emb->base.forward_fn(emb, indices, &out);
    return out;
}
