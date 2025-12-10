/*
 * GreyML backend: ga nn attention.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#include <stdlib.h>
#include <math.h>
#include "greyarea/ga_nn.h"
#include "greyarea/ga_ops.h"

static void mha_forward_wrapper(void* self, GATensor* query, GATensor** output) {
    GAMultiheadAttention* attn = (GAMultiheadAttention*)self;
    GATensor* key = query;
    GATensor* value = query;

    if (query->ndim != 3) { ga_errno = GA_ERR_INVALID_SHAPE; *output = NULL; return; }

    int64_t batch = query->shape[0];
    int64_t seq_len = query->shape[1];
    int64_t embed_dim = query->shape[2];

    if (embed_dim != attn->embed_dim) { ga_errno = GA_ERR_INVALID_SHAPE; *output = NULL; return; }
    if (attn->embed_dim % attn->num_heads != 0) { ga_errno = GA_ERR_INVALID_SHAPE; *output = NULL; return; }

    int64_t head_dim = attn->embed_dim / attn->num_heads;

    GATensor* q_proj = ga_matmul(query, attn->w_q);
    GATensor* k_proj = ga_matmul(key, attn->w_k);
    GATensor* v_proj = ga_matmul(value, attn->w_v);

    int64_t qkv_shape[4] = {batch, seq_len, attn->num_heads, head_dim};
    GATensor* q_heads = ga_tensor_reshape(q_proj, 4, qkv_shape);
    GATensor* k_heads = ga_tensor_reshape(k_proj, 4, qkv_shape);
    GATensor* v_heads = ga_tensor_reshape(v_proj, 4, qkv_shape);

    int perm[4] = {0, 2, 1, 3};
    GATensor* q_t = ga_tensor_permute(q_heads, perm);
    GATensor* k_t = ga_tensor_permute(k_heads, perm);
    GATensor* v_t = ga_tensor_permute(v_heads, perm);

    int64_t attn_out_shape[3] = {batch * attn->num_heads, seq_len, head_dim};
    GATensor* q_r = ga_tensor_reshape(q_t, 3, attn_out_shape);
    GATensor* k_r = ga_tensor_reshape(k_t, 3, attn_out_shape);
    GATensor* v_r = ga_tensor_reshape(v_t, 3, attn_out_shape);

    GATensor* k_transpose = ga_transpose(k_r);
    GATensor* scores = ga_matmul(q_r, k_transpose);
    float scale = 1.0f / sqrtf((float)head_dim);
    GATensor* scaled = ga_mul_scalar(scores, scale);
    GATensor* weights = ga_softmax(scaled, -1);
    GATensor* ctx = ga_matmul(weights, v_r);

    int64_t ctx_reshape[4] = {batch, attn->num_heads, seq_len, head_dim};
    GATensor* ctx_4d = ga_tensor_reshape(ctx, 4, ctx_reshape);
    int perm2[4] = {0, 2, 1, 3};
    GATensor* ctx_t = ga_tensor_permute(ctx_4d, perm2);
    int64_t ctx_final_shape[3] = {batch, seq_len, attn->embed_dim};
    GATensor* ctx_final = ga_tensor_reshape(ctx_t, 3, ctx_final_shape);

    GATensor* out = ga_matmul(ctx_final, attn->w_o);

    ga_tensor_release(q_proj);
    ga_tensor_release(k_proj);
    ga_tensor_release(v_proj);
    ga_tensor_release(q_heads);
    ga_tensor_release(k_heads);
    ga_tensor_release(v_heads);
    ga_tensor_release(q_t);
    ga_tensor_release(k_t);
    ga_tensor_release(v_t);
    ga_tensor_release(q_r);
    ga_tensor_release(k_r);
    ga_tensor_release(v_r);
    ga_tensor_release(k_transpose);
    ga_tensor_release(scores);
    ga_tensor_release(scaled);
    ga_tensor_release(weights);
    ga_tensor_release(ctx);
    ga_tensor_release(ctx_4d);
    ga_tensor_release(ctx_t);
    ga_tensor_release(ctx_final);

    *output = out;
}

GAMultiheadAttention* ga_mha_create(int embed_dim, int num_heads) {
    GAMultiheadAttention* attn = (GAMultiheadAttention*)calloc(1, sizeof(GAMultiheadAttention));
    attn->embed_dim = embed_dim;
    attn->num_heads = num_heads;
    int64_t wshape[2] = {embed_dim, embed_dim};
    attn->w_q = ga_tensor_empty(2, wshape, GA_FLOAT32);
    attn->w_k = ga_tensor_empty(2, wshape, GA_FLOAT32);
    attn->w_v = ga_tensor_empty(2, wshape, GA_FLOAT32);
    attn->w_o = ga_tensor_empty(2, wshape, GA_FLOAT32);
    ga_init_xavier_uniform(attn->w_q);
    ga_init_xavier_uniform(attn->w_k);
    ga_init_xavier_uniform(attn->w_v);
    ga_init_xavier_uniform(attn->w_o);

    attn->base.parameters = (GATensor**)calloc(4, sizeof(GATensor*));
    attn->base.parameters[0] = attn->w_q;
    attn->base.parameters[1] = attn->w_k;
    attn->base.parameters[2] = attn->w_v;
    attn->base.parameters[3] = attn->w_o;
    attn->base.n_params = 4;
    attn->base.forward_fn = (void (*)(void*, GATensor*, GATensor**))mha_forward_wrapper;
    return attn;
}

void ga_mha_free(GAMultiheadAttention* attn) {
    if (!attn) return;
    if (attn->w_q) ga_tensor_release(attn->w_q);
    if (attn->w_k) ga_tensor_release(attn->w_k);
    if (attn->w_v) ga_tensor_release(attn->w_v);
    if (attn->w_o) ga_tensor_release(attn->w_o);
    if (attn->base.parameters) free(attn->base.parameters);
    free(attn);
}

GATensor* ga_mha_forward(GAMultiheadAttention* attn, GATensor* query, GATensor* key, GATensor* value, GATensor** attn_weights) {
    (void)key;
    (void)value;
    if (!attn) return NULL;
    GATensor* out = NULL;
    attn->base.forward_fn(attn, query, &out);
    if (attn_weights) *attn_weights = NULL; // weights already released; could clone if needed
    return out;
}
