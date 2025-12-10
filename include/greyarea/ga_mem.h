/*
 * GreyML C API header: ga mem.
 *
 * Declares the public interface for this subsystem so C and Python callers share one contract.
 */

#pragma once
#include "ga_common.h"

GA_API void* ga_aligned_alloc(size_t size, size_t alignment);
GA_API void ga_aligned_free(void* ptr);

typedef struct {
    uint8_t* base;
    size_t capacity;
    size_t used;
} GAArena;

GA_API GAArena* ga_arena_create(size_t capacity);
GA_API void ga_arena_destroy(GAArena* arena);
GA_API void* ga_arena_alloc(GAArena* arena, size_t size);
GA_API void ga_arena_reset(GAArena* arena);

typedef struct GAPool GAPool;
GA_API GAPool* ga_pool_create(size_t block_size, size_t count);
GA_API void ga_pool_destroy(GAPool* pool);
GA_API void* ga_pool_alloc(GAPool* pool);
GA_API void ga_pool_free(GAPool* pool, void* block);