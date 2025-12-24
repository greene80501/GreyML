/*
 * GreyML backend: ga mem.
 *
 * Foundational runtime utilities including error handling, memory management, random utilities, and tensor lifecycle helpers.
 */

#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include "greyarea/ga_mem.h"

void* ga_aligned_alloc(size_t size, size_t alignment) {
    return _aligned_malloc(size, alignment);
}

void ga_aligned_free(void* ptr) {
    _aligned_free(ptr);
}

#define GA_CACHE_MIN_BYTES 64
#define GA_CACHE_BUCKETS 20
#define GA_CACHE_MAX_BLOCKS 64
#define GA_CACHE_LOCAL_MAX 16

static void* ga_cache_free_lists[GA_CACHE_BUCKETS];
static size_t ga_cache_counts[GA_CACHE_BUCKETS];
static SRWLOCK ga_cache_lock = SRWLOCK_INIT;

static _Thread_local void* ga_cache_local_lists[GA_CACHE_BUCKETS];
static _Thread_local size_t ga_cache_local_counts[GA_CACHE_BUCKETS];

static int ga_cache_bucket_index(size_t size) {
    if (size == 0) return -1;
    size_t bucket = GA_CACHE_MIN_BYTES;
    int idx = 0;
    while (bucket < size && idx < GA_CACHE_BUCKETS - 1) {
        bucket <<= 1;
        idx++;
    }
    if (bucket < size) return -1;
    return idx;
}

static size_t ga_cache_bucket_size(int idx) {
    return (size_t)GA_CACHE_MIN_BYTES << idx;
}

void* ga_cached_alloc(size_t size) {
    int idx = ga_cache_bucket_index(size);
    if (idx < 0) {
        return ga_aligned_alloc(size, GA_ALIGN_SIZE);
    }

    void* block = ga_cache_local_lists[idx];
    if (block) {
        ga_cache_local_lists[idx] = *(void**)block;
        ga_cache_local_counts[idx]--;
        return block;
    }

    AcquireSRWLockExclusive(&ga_cache_lock);
    block = ga_cache_free_lists[idx];
    if (block) {
        ga_cache_free_lists[idx] = *(void**)block;
        ga_cache_counts[idx]--;
    }
    ReleaseSRWLockExclusive(&ga_cache_lock);
    if (block) return block;
    return ga_aligned_alloc(ga_cache_bucket_size(idx), GA_ALIGN_SIZE);
}

void ga_cached_free(void* ptr, size_t size) {
    if (!ptr) return;
    int idx = ga_cache_bucket_index(size);
    if (idx < 0) {
        ga_aligned_free(ptr);
        return;
    }
    if (ga_cache_local_counts[idx] < GA_CACHE_LOCAL_MAX) {
        *(void**)ptr = ga_cache_local_lists[idx];
        ga_cache_local_lists[idx] = ptr;
        ga_cache_local_counts[idx]++;
        return;
    }

    AcquireSRWLockExclusive(&ga_cache_lock);
    if (ga_cache_counts[idx] < GA_CACHE_MAX_BLOCKS) {
        *(void**)ptr = ga_cache_free_lists[idx];
        ga_cache_free_lists[idx] = ptr;
        ga_cache_counts[idx]++;
        ReleaseSRWLockExclusive(&ga_cache_lock);
        return;
    }
    ReleaseSRWLockExclusive(&ga_cache_lock);
    ga_aligned_free(ptr);
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
