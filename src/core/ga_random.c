/*
 * GreyML backend: ga random.
 *
 * Foundational runtime utilities including error handling, memory management, random utilities, and tensor lifecycle helpers.
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <stdint.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#include "greyarea/ga_random.h"

static const uint64_t multiplier = 0x5851f42d4c957f2dULL;
static const uint64_t default_increment = 0x14057b7ef767814fULL;

#if defined(_MSC_VER)
#define GA_THREAD_LOCAL __declspec(thread)
#else
#define GA_THREAD_LOCAL _Thread_local
#endif

static GA_THREAD_LOCAL uint64_t ga_rng_state = 0;
static GA_THREAD_LOCAL uint64_t ga_rng_inc = 0;
static GA_THREAD_LOCAL int ga_rng_seeded = 0;

static uint32_t pcg32_step(void) {
    uint64_t oldstate = ga_rng_state;
    ga_rng_state = oldstate * multiplier + ga_rng_inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((uint32_t)(-rot) & 31));
}

void ga_random_seed(uint64_t seed) {
    ga_rng_state = 0U;
    ga_rng_inc = (seed << 1u) | 1u;
    (void)pcg32_step();
    ga_rng_state += seed;
    (void)pcg32_step();
    ga_rng_seeded = 1;
}

static void ga_random_ensure_seeded(void) {
    if (ga_rng_seeded) return;
    uint64_t seed = (uint64_t)time(NULL);
    seed ^= (uint64_t)(uintptr_t)&ga_rng_state;
#if defined(_WIN32)
    seed ^= ((uint64_t)GetCurrentThreadId() << 32);
#endif
    if (seed == 0) seed = default_increment;
    ga_random_seed(seed);
}

uint32_t ga_random_uint32(void) {
    ga_random_ensure_seeded();
    return pcg32_step();
}

float ga_random_float(void) {
    return (float)ga_random_uint32() / (float)(1ULL << 32);
}

float ga_random_normal(void) {
    // Box-Muller
    float u1 = ga_random_float();
    float u2 = ga_random_float();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

void ga_random_shuffle(int* array, size_t n) {
    if (!array || n < 2) return;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(ga_random_uint32() % (uint32_t)(i + 1));
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
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
