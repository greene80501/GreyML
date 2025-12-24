/*
 * GreyML C API header: ga nn.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_tensor.h"
#include "ga_autograd.h"

typedef struct {
    char* name;
    GATensor** parameters;
    char** param_names;
    size_t n_params;
    struct GAModule** children;
    size_t n_children;
    bool training;
    void (*forward_fn)(void* self, GATensor* input, GATensor** output);
    void (*zero_grad_fn)(void* self);
    void (*to_device_fn)(void* self, int device);
    void (*train_fn)(void* self, bool training);
    void (*free_fn)(void* self);
} GAModule;

typedef struct {
    GAModule base;
    GATensor* weight;
    GATensor* bias;
    int in_features;
    int out_features;
} GALinear;

typedef struct {
    GAModule base;
    GATensor* weight;
    GATensor* bias;
    int in_channels;
    int out_channels;
    int kernel_size[2];
    int stride;
    int padding;
    int dilation;
    int groups;
} GAConv2D;

typedef struct {
    GAModule base;
    GATensor* running_mean;
    GATensor* running_var;
    GATensor* gamma;
    GATensor* beta;
    float momentum;
    float eps;
} GABatchNorm2D;

typedef struct {
    GAModule base;
    float p;
} GADropout;

typedef struct {
    GAModule base;
    GAModule** modules;
    size_t num_modules;
} GASequential;

typedef struct {
    GAModule base;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    bool is_max;
} GAPool2D;

typedef struct {
    GAModule base;
    GATensor* weight;
    int num_embeddings;
    int embedding_dim;
} GAEmbedding;

typedef struct {
    GAModule base;
    GATensor* weight_ih;
    GATensor* weight_hh;
    GATensor* bias_ih;
    GATensor* bias_hh;
    int input_size;
    int hidden_size;
} GARNN;

typedef struct {
    GAModule base;
    GATensor* weight_ih;
    GATensor* weight_hh;
    GATensor* bias_ih;
    GATensor* bias_hh;
    int input_size;
    int hidden_size;
} GALSTM;

typedef struct {
    GAModule base;
    GATensor* weight_ih;
    GATensor* weight_hh;
    GATensor* bias_ih;
    GATensor* bias_hh;
    int input_size;
    int hidden_size;
} GAGRU;

typedef struct {
    GAModule base;
    GATensor* w_q;
    GATensor* w_k;
    GATensor* w_v;
    GATensor* w_o;
    int embed_dim;
    int num_heads;
} GAMultiheadAttention;

GA_API GALinear* ga_linear_create(int in_features, int out_features, bool bias);
GA_API void ga_linear_free(GALinear* linear);
GA_API GATensor* ga_linear_forward(GALinear* linear, GATensor* input);

GA_API GAConv2D* ga_conv2d_create(int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int groups, bool bias);
GA_API void ga_conv2d_free(GAConv2D* conv);
GA_API GATensor* ga_conv2d_forward(GAConv2D* conv, GATensor* input);

GA_API GASequential* ga_sequential_create(GAModule** modules, size_t num_modules);
GA_API void ga_sequential_free(GASequential* seq);
GA_API GATensor* ga_sequential_forward(GASequential* seq, GATensor* input);

GA_API GABatchNorm2D* ga_batchnorm2d_create(int num_features, float momentum, float eps);
GA_API void ga_batchnorm2d_free(GABatchNorm2D* bn);
GA_API GATensor* ga_batchnorm2d_forward(GABatchNorm2D* bn, GATensor* input);

GA_API GADropout* ga_dropout_create(float p);
GA_API void ga_dropout_free(GADropout* drop);
GA_API GATensor* ga_dropout_forward(GADropout* drop, GATensor* input);

GA_API GAPool2D* ga_maxpool2d_create(int kernel_size, int stride, int padding, int dilation);
GA_API GAPool2D* ga_avgpool2d_create(int kernel_size, int stride, int padding);
GA_API void ga_pool2d_free(GAPool2D* pool);
GA_API GATensor* ga_pool2d_forward(GAPool2D* pool, GATensor* input);

GA_API GAEmbedding* ga_embedding_create(int num_embeddings, int embedding_dim);
GA_API void ga_embedding_free(GAEmbedding* emb);
GA_API GATensor* ga_embedding_forward(GAEmbedding* emb, GATensor* indices);

GA_API GARNN* ga_rnn_create(int input_size, int hidden_size);
GA_API void ga_rnn_free(GARNN* rnn);
GA_API GATensor* ga_rnn_forward(GARNN* rnn, GATensor* input, GATensor** last_hidden);

GA_API GALSTM* ga_lstm_create(int input_size, int hidden_size);
GA_API void ga_lstm_free(GALSTM* lstm);
GA_API GATensor* ga_lstm_forward(GALSTM* lstm, GATensor* input, GATensor** last_hidden, GATensor** last_cell);

GA_API GAGRU* ga_gru_create(int input_size, int hidden_size);
GA_API void ga_gru_free(GAGRU* gru);
GA_API GATensor* ga_gru_forward(GAGRU* gru, GATensor* input, GATensor** last_hidden);

GA_API GAMultiheadAttention* ga_mha_create(int embed_dim, int num_heads);
GA_API void ga_mha_free(GAMultiheadAttention* attn);
GA_API GATensor* ga_mha_forward(GAMultiheadAttention* attn, GATensor* query, GATensor* key, GATensor* value, GATensor** attn_weights);

GA_API void ga_init_uniform(GATensor* tensor, float low, float high);
GA_API void ga_init_normal(GATensor* tensor, float mean, float std);
GA_API void ga_init_xavier_uniform(GATensor* tensor);
GA_API void ga_init_xavier_normal(GATensor* tensor);
GA_API void ga_init_kaiming_uniform(GATensor* tensor);
GA_API void ga_init_kaiming_normal(GATensor* tensor);

GA_API void ga_module_train(GAModule* module, bool training);
GA_API void ga_module_zero_grad(GAModule* module);
