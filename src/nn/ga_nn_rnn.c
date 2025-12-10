/*
 * GreyML backend: ga nn rnn.
 *
 * Neural network kernels (attention, convolution, pooling, normalization, RNNs, etc.) that back the Python API.
 */

#include <stdlib.h>
#include <math.h>
#include "greyarea/ga_nn.h"
#include "greyarea/ga_ops.h"
#include "greyarea/ga_common.h"


static void rnn_forward_wrapper(void* self, GATensor* input, GATensor** output) {
    GARNN* rnn = (GARNN*)self;
    // input: [seq, batch, input_size]
    if (input->ndim != 3) { ga_errno = GA_ERR_INVALID_SHAPE; *output = NULL; return; }
    int64_t seq = input->shape[0];
    int64_t batch = input->shape[1];
    int64_t in_size = input->shape[2];
    if (in_size != rnn->input_size) { ga_errno = GA_ERR_INVALID_SHAPE; *output = NULL; return; }

    int64_t out_shape[3] = {seq, batch, rnn->hidden_size};
    GATensor* h_all = ga_tensor_empty(3, out_shape, GA_FLOAT32);
    float* hdata = (float*)h_all->data;
    float* x = (float*)input->data;
    float* wih = (float*)rnn->weight_ih->data;
    float* whh = (float*)rnn->weight_hh->data;
    float* bih = (float*)rnn->bias_ih->data;
    float* bhh = (float*)rnn->bias_hh->data;

    // initialize hidden to zeros
    float* h_prev = (float*)calloc((size_t)(batch * rnn->hidden_size), sizeof(float));

    for (int64_t t = 0; t < seq; t++) {
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t h = 0; h < rnn->hidden_size; h++) {
                float sum = bih[h] + bhh[h];
                // input contribution
                for (int64_t i = 0; i < rnn->input_size; i++) {
                    float xv = x[(t * batch + b) * in_size + i];
                    sum += xv * wih[h * rnn->input_size + i];
                }
                // hidden contribution
                for (int64_t hh = 0; hh < rnn->hidden_size; hh++) {
                    sum += h_prev[b * rnn->hidden_size + hh] * whh[h * rnn->hidden_size + hh];
                }
                float hval = tanhf(sum);
                hdata[(t * batch + b) * rnn->hidden_size + h] = hval;
            }
        }
        memcpy(h_prev, hdata + t * batch * rnn->hidden_size, sizeof(float) * batch * rnn->hidden_size);
    }
    free(h_prev);
    *output = h_all;
}

GARNN* ga_rnn_create(int input_size, int hidden_size) {
    GARNN* rnn = (GARNN*)calloc(1, sizeof(GARNN));
    rnn->input_size = input_size;
    rnn->hidden_size = hidden_size;
    int64_t wih_shape[2] = {hidden_size, input_size};
    int64_t whh_shape[2] = {hidden_size, hidden_size};
    int64_t b_shape[1] = {hidden_size};
    rnn->weight_ih = ga_tensor_empty(2, wih_shape, GA_FLOAT32);
    rnn->weight_hh = ga_tensor_empty(2, whh_shape, GA_FLOAT32);
    rnn->bias_ih = ga_tensor_zeros(1, b_shape, GA_FLOAT32);
    rnn->bias_hh = ga_tensor_zeros(1, b_shape, GA_FLOAT32);
    ga_init_kaiming_uniform(rnn->weight_ih);
    ga_init_kaiming_uniform(rnn->weight_hh);

    rnn->base.parameters = (GATensor**)calloc(4, sizeof(GATensor*));
    rnn->base.parameters[0] = rnn->weight_ih;
    rnn->base.parameters[1] = rnn->weight_hh;
    rnn->base.parameters[2] = rnn->bias_ih;
    rnn->base.parameters[3] = rnn->bias_hh;
    rnn->base.n_params = 4;
    rnn->base.forward_fn = rnn_forward_wrapper;
    return rnn;
}

void ga_rnn_free(GARNN* rnn) {
    if (!rnn) return;
    if (rnn->weight_ih) ga_tensor_release(rnn->weight_ih);
    if (rnn->weight_hh) ga_tensor_release(rnn->weight_hh);
    if (rnn->bias_ih) ga_tensor_release(rnn->bias_ih);
    if (rnn->bias_hh) ga_tensor_release(rnn->bias_hh);
    if (rnn->base.parameters) free(rnn->base.parameters);
    free(rnn);
}

GATensor* ga_rnn_forward(GARNN* rnn, GATensor* input, GATensor** last_hidden) {
    if (!rnn) return NULL;
    GATensor* out = NULL;
    rnn->base.forward_fn(rnn, input, &out);
    if (last_hidden) {
        int64_t batch = input->shape[1];
        int64_t h_shape[2] = {batch, rnn->hidden_size};
        GATensor* h_last = ga_tensor_empty(2, h_shape, GA_FLOAT32);
        memcpy(h_last->data, (float*)out->data + (input->shape[0] - 1) * batch * rnn->hidden_size,
               sizeof(float) * batch * rnn->hidden_size);
        *last_hidden = h_last;
    }
    return out;
}

static void lstm_forward_wrapper(void* self, GATensor* input, GATensor** output) {
    GALSTM* lstm = (GALSTM*)self;
    if (input->ndim != 3) { ga_errno = GA_ERR_INVALID_SHAPE; *output = NULL; return; }

    int64_t seq = input->shape[0];
    int64_t batch = input->shape[1];
    int64_t in_size = input->shape[2];
    if (in_size != lstm->input_size) { ga_errno = GA_ERR_INVALID_SHAPE; *output = NULL; return; }

    int64_t hid = lstm->hidden_size;
    int64_t out_shape[3] = {seq, batch, hid};
    GATensor* h_all = ga_tensor_empty(3, out_shape, GA_FLOAT32);
    float* hdata = (float*)h_all->data;
    float* x = (float*)input->data;
    float* wih = (float*)lstm->weight_ih->data;
    float* whh = (float*)lstm->weight_hh->data;
    float* bih = (float*)lstm->bias_ih->data;
    float* bhh = (float*)lstm->bias_hh->data;

    float* h_prev = (float*)calloc((size_t)(batch * hid), sizeof(float));
    float* c_prev = (float*)calloc((size_t)(batch * hid), sizeof(float));

    for (int64_t t = 0; t < seq; t++) {
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t h = 0; h < hid; h++) {
                float gi = bih[h] + bhh[h];
                float gf = bih[h + hid] + bhh[h + hid];
                float gg = bih[h + 2*hid] + bhh[h + 2*hid];
                float go = bih[h + 3*hid] + bhh[h + 3*hid];

                for (int64_t i = 0; i < in_size; i++) {
                    float xv = x[(t * batch + b) * in_size + i];
                    gi += xv * wih[h * in_size + i];
                    gf += xv * wih[(h + hid) * in_size + i];
                    gg += xv * wih[(h + 2*hid) * in_size + i];
                    go += xv * wih[(h + 3*hid) * in_size + i];
                }

                for (int64_t hh = 0; hh < hid; hh++) {
                    float hp = h_prev[b * hid + hh];
                    gi += hp * whh[h * hid + hh];
                    gf += hp * whh[(h + hid) * hid + hh];
                    gg += hp * whh[(h + 2*hid) * hid + hh];
                    go += hp * whh[(h + 3*hid) * hid + hh];
                }

                float i_gate = 1.0f / (1.0f + expf(-gi));
                float f_gate = 1.0f / (1.0f + expf(-gf));
                float g_gate = tanhf(gg);
                float o_gate = 1.0f / (1.0f + expf(-go));

                float c_new = f_gate * c_prev[b * hid + h] + i_gate * g_gate;
                float h_new = o_gate * tanhf(c_new);

                c_prev[b * hid + h] = c_new;
                hdata[(t * batch + b) * hid + h] = h_new;
            }
        }
        memcpy(h_prev, hdata + t * batch * hid, sizeof(float) * batch * hid);
    }

    free(h_prev);
    free(c_prev);
    *output = h_all;
}

GALSTM* ga_lstm_create(int input_size, int hidden_size) {
    GALSTM* lstm = (GALSTM*)calloc(1, sizeof(GALSTM));
    lstm->input_size = input_size;
    lstm->hidden_size = hidden_size;

    int64_t wih_shape[2] = {4 * hidden_size, input_size};
    int64_t whh_shape[2] = {4 * hidden_size, hidden_size};
    int64_t b_shape[1] = {4 * hidden_size};

    lstm->weight_ih = ga_tensor_empty(2, wih_shape, GA_FLOAT32);
    lstm->weight_hh = ga_tensor_empty(2, whh_shape, GA_FLOAT32);
    lstm->bias_ih = ga_tensor_zeros(1, b_shape, GA_FLOAT32);
    lstm->bias_hh = ga_tensor_zeros(1, b_shape, GA_FLOAT32);

    ga_init_kaiming_uniform(lstm->weight_ih);
    ga_init_kaiming_uniform(lstm->weight_hh);

    lstm->base.parameters = (GATensor**)calloc(4, sizeof(GATensor*));
    lstm->base.parameters[0] = lstm->weight_ih;
    lstm->base.parameters[1] = lstm->weight_hh;
    lstm->base.parameters[2] = lstm->bias_ih;
    lstm->base.parameters[3] = lstm->bias_hh;
    lstm->base.n_params = 4;
    lstm->base.forward_fn = lstm_forward_wrapper;
    return lstm;
}

void ga_lstm_free(GALSTM* lstm) {
    if (!lstm) return;
    if (lstm->weight_ih) ga_tensor_release(lstm->weight_ih);
    if (lstm->weight_hh) ga_tensor_release(lstm->weight_hh);
    if (lstm->bias_ih) ga_tensor_release(lstm->bias_ih);
    if (lstm->bias_hh) ga_tensor_release(lstm->bias_hh);
    if (lstm->base.parameters) free(lstm->base.parameters);
    free(lstm);
}

GATensor* ga_lstm_forward(GALSTM* lstm, GATensor* input, GATensor** last_hidden, GATensor** last_cell) {
    if (!lstm) return NULL;
    GATensor* out = NULL;
    lstm->base.forward_fn(lstm, input, &out);

    if (last_hidden) {
        int64_t batch = input->shape[1];
        int64_t h_shape[2] = {batch, lstm->hidden_size};
        GATensor* h_last = ga_tensor_empty(2, h_shape, GA_FLOAT32);
        memcpy(h_last->data, (float*)out->data + (input->shape[0] - 1) * batch * lstm->hidden_size,
               sizeof(float) * batch * lstm->hidden_size);
        *last_hidden = h_last;
    }

    if (last_cell) {
        *last_cell = NULL;
    }

    return out;
}

static void gru_forward_wrapper(void* self, GATensor* input, GATensor** output) {
    GAGRU* gru = (GAGRU*)self;
    if (input->ndim != 3) { ga_errno = GA_ERR_INVALID_SHAPE; *output = NULL; return; }

    int64_t seq = input->shape[0];
    int64_t batch = input->shape[1];
    int64_t in_size = input->shape[2];
    if (in_size != gru->input_size) { ga_errno = GA_ERR_INVALID_SHAPE; *output = NULL; return; }

    int64_t hid = gru->hidden_size;
    int64_t out_shape[3] = {seq, batch, hid};
    GATensor* h_all = ga_tensor_empty(3, out_shape, GA_FLOAT32);
    float* hdata = (float*)h_all->data;
    float* x = (float*)input->data;
    float* wih = (float*)gru->weight_ih->data;
    float* whh = (float*)gru->weight_hh->data;
    float* bih = (float*)gru->bias_ih->data;
    float* bhh = (float*)gru->bias_hh->data;

    float* h_prev = (float*)calloc((size_t)(batch * hid), sizeof(float));

    for (int64_t t = 0; t < seq; t++) {
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t h = 0; h < hid; h++) {
                float gr = bih[h] + bhh[h];
                float gz = bih[h + hid] + bhh[h + hid];
                float gn = bih[h + 2*hid];

                for (int64_t i = 0; i < in_size; i++) {
                    float xv = x[(t * batch + b) * in_size + i];
                    gr += xv * wih[h * in_size + i];
                    gz += xv * wih[(h + hid) * in_size + i];
                    gn += xv * wih[(h + 2*hid) * in_size + i];
                }

                for (int64_t hh = 0; hh < hid; hh++) {
                    float hp = h_prev[b * hid + hh];
                    gr += hp * whh[h * hid + hh];
                    gz += hp * whh[(h + hid) * hid + hh];
                }

                float r_gate = 1.0f / (1.0f + expf(-gr));
                float z_gate = 1.0f / (1.0f + expf(-gz));

                for (int64_t hh = 0; hh < hid; hh++) {
                    gn += r_gate * h_prev[b * hid + hh] * whh[(h + 2*hid) * hid + hh];
                }
                gn += bhh[h + 2*hid];

                float n_gate = tanhf(gn);
                float h_new = (1.0f - z_gate) * n_gate + z_gate * h_prev[b * hid + h];

                hdata[(t * batch + b) * hid + h] = h_new;
            }
        }
        memcpy(h_prev, hdata + t * batch * hid, sizeof(float) * batch * hid);
    }

    free(h_prev);
    *output = h_all;
}

GAGRU* ga_gru_create(int input_size, int hidden_size) {
    GAGRU* gru = (GAGRU*)calloc(1, sizeof(GAGRU));
    gru->input_size = input_size;
    gru->hidden_size = hidden_size;

    int64_t wih_shape[2] = {3 * hidden_size, input_size};
    int64_t whh_shape[2] = {3 * hidden_size, hidden_size};
    int64_t b_shape[1] = {3 * hidden_size};

    gru->weight_ih = ga_tensor_empty(2, wih_shape, GA_FLOAT32);
    gru->weight_hh = ga_tensor_empty(2, whh_shape, GA_FLOAT32);
    gru->bias_ih = ga_tensor_zeros(1, b_shape, GA_FLOAT32);
    gru->bias_hh = ga_tensor_zeros(1, b_shape, GA_FLOAT32);

    ga_init_kaiming_uniform(gru->weight_ih);
    ga_init_kaiming_uniform(gru->weight_hh);

    gru->base.parameters = (GATensor**)calloc(4, sizeof(GATensor*));
    gru->base.parameters[0] = gru->weight_ih;
    gru->base.parameters[1] = gru->weight_hh;
    gru->base.parameters[2] = gru->bias_ih;
    gru->base.parameters[3] = gru->bias_hh;
    gru->base.n_params = 4;
    gru->base.forward_fn = gru_forward_wrapper;
    return gru;
}

void ga_gru_free(GAGRU* gru) {
    if (!gru) return;
    if (gru->weight_ih) ga_tensor_release(gru->weight_ih);
    if (gru->weight_hh) ga_tensor_release(gru->weight_hh);
    if (gru->bias_ih) ga_tensor_release(gru->bias_ih);
    if (gru->bias_hh) ga_tensor_release(gru->bias_hh);
    if (gru->base.parameters) free(gru->base.parameters);
    free(gru);
}

GATensor* ga_gru_forward(GAGRU* gru, GATensor* input, GATensor** last_hidden) {
    if (!gru) return NULL;
    GATensor* out = NULL;
    gru->base.forward_fn(gru, input, &out);

    if (last_hidden) {
        int64_t batch = input->shape[1];
        int64_t h_shape[2] = {batch, gru->hidden_size};
        GATensor* h_last = ga_tensor_empty(2, h_shape, GA_FLOAT32);
        memcpy(h_last->data, (float*)out->data + (input->shape[0] - 1) * batch * gru->hidden_size,
               sizeof(float) * batch * gru->hidden_size);
        *last_hidden = h_last;
    }

    return out;
}
