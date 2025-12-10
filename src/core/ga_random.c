/*
 * GreyML backend: ga random.
 *
 * Foundational runtime utilities including error handling, memory management, random utilities, and tensor lifecycle helpers.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include "greyarea/ga_random.h"

static uint64_t state = 0x853c49e6748fea9bULL;
static const uint64_t multiplier = 0x5851f42d4c957f2dULL;
static const uint64_t increment = 0x14057b7ef767814fULL;

void ga_random_seed(uint64_t seed) {
    state = seed;
}

static uint32_t pcg32_random() {
    uint64_t x = state;
    unsigned count = (unsigned)(x >> 61);
    state = x * multiplier + increment;
    x ^= x >> 22;
    return (uint32_t)(x >> (22 + count)) | (uint32_t)(x << (32 - (22 + count)));
}

float ga_random_float(void) {
    return (float)pcg32_random() / (float)(1ULL << 32);
}

float ga_random_normal(void) {
    // Box-Muller
    float u1 = ga_random_float();
    float u2 = ga_random_float();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

void ga_tensor_rand_(GATensor* tensor) {
    if (tensor->dtype == GA_FLOAT32) {
        float* data = (float*)tensor->data;
        for (int64_t i = 0; i < tensor->size; i++) {
            data[i] = ga_random_float();
        }
    }
}

void ga_tensor_randn_(GATensor* tensor) {
    if (tensor->dtype == GA_FLOAT32) {
        float* data = (float*)tensor->data;
        for (int64_t i = 0; i < tensor->size; i++) {
            data[i] = ga_random_normal();
        }
    }
}