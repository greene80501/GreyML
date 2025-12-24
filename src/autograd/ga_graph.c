/*
 * GreyML backend: ga graph.
 *
 * Automatic differentiation primitives: graph construction, tracing, and backward pass kernels.
 */

#include "greyarea/ga_autograd.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    GATensor** data;
    size_t size;
    size_t cap;
} TensorVec;

static void vec_init(TensorVec* v) {
    v->data = NULL;
    v->size = 0;
    v->cap = 0;
}

static void vec_push(TensorVec* v, GATensor* t) {
    if (v->size + 1 > v->cap) {
        size_t new_cap = v->cap ? v->cap * 2 : 16;
        v->data = (GATensor**)realloc(v->data, new_cap * sizeof(GATensor*));
        v->cap = new_cap;
    }
    v->data[v->size++] = t;
}

static int vec_contains(TensorVec* v, GATensor* t) {
    for (size_t i = 0; i < v->size; i++) {
        if (v->data[i] == t) return 1;
    }
    return 0;
}

static void dfs(GATensor* t, TensorVec* visited, TensorVec* order) {
    if (!t || vec_contains(visited, t)) return;
    vec_push(visited, t);
    if (t->grad_fn) {
        for (size_t i = 0; i < t->grad_fn->num_inputs; i++) {
            dfs(t->grad_fn->inputs[i], visited, order);
        }
    }
    vec_push(order, t);
}

void ga_build_topo(GATensor* tensor, GATensor*** list, size_t* size) {
    TensorVec visited;
    TensorVec order;
    vec_init(&visited);
    vec_init(&order);
    dfs(tensor, &visited, &order);
    *list = order.data;
    *size = order.size;
    free(visited.data);
}
