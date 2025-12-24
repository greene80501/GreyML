/*
 * GreyML backend: ga mem.
 *
 * Foundational runtime utilities including error handling, memory management, random utilities, and tensor lifecycle helpers.
 */

#ifdef _WIN32
#include <windows.h>
#else
#include <stdlib.h>
#endif
#include <string.h>
#include "greyarea/ga_mem.h"

void* ga_aligned_alloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

void ga_aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

GAArena* ga_arena_create(size_t capacity) {
    GAArena* arena = (GAArena*)malloc(sizeof(GAArena));
    if (!arena) return NULL;
    arena->base = (uint8_t*)ga_aligned_alloc(capacity, GA_ALIGN_SIZE);
    if (!arena->base) {
        free(arena);
        return NULL;
    }
    arena->capacity = capacity;
    arena->used = 0;
    return arena;
}

void ga_arena_destroy(GAArena* arena) {
    if (!arena) return;
    ga_aligned_free(arena->base);
    free(arena);
}

void* ga_arena_alloc(GAArena* arena, size_t size) {
    size = (size + GA_ALIGN_SIZE - 1) & ~(GA_ALIGN_SIZE - 1);
    if (arena->used + size > arena->capacity) return NULL;
    void* ptr = arena->base + arena->used;
    arena->used += size;
    return ptr;
}

void ga_arena_reset(GAArena* arena) {
    arena->used = 0;
}

struct GAPool {
    uint8_t* blocks;
    void* free_list;
    size_t block_size;
    size_t count;
};

GAPool* ga_pool_create(size_t block_size, size_t count) {
    GAPool* pool = (GAPool*)malloc(sizeof(GAPool));
    block_size = (block_size + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
    pool->block_size = block_size;
    pool->count = count;
    pool->blocks = (uint8_t*)ga_aligned_alloc(block_size * count, GA_ALIGN_SIZE);
    
    uint8_t* p = pool->blocks;
    pool->free_list = p;
    for (size_t i = 0; i < count - 1; i++) {
        *(void**)p = p + block_size;
        p += block_size;
    }
    *(void**)p = NULL;
    return pool;
}

void ga_pool_destroy(GAPool* pool) {
    ga_aligned_free(pool->blocks);
    free(pool);
}

void* ga_pool_alloc(GAPool* pool) {
    if (!pool->free_list) return NULL;
    void* block = pool->free_list;
    pool->free_list = *(void**)block;
    return block;
}

void ga_pool_free(GAPool* pool, void* block) {
    *(void**)block = pool->free_list;
    pool->free_list = block;
}
