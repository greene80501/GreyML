/*
 * GreyML C test: mem.
 *
 * Exercises this subsystem end-to-end to catch regressions.
 */

#include <assert.h>
#include "greyarea/greyarea.h"

int test_mem_alloc(void) {
    GAArena* arena = ga_arena_create(1024);
    assert(arena);
    void* ptr = ga_arena_alloc(arena, 128);
    assert(ptr);
    ga_arena_destroy(arena);
    GAPool* pool = ga_pool_create(64, 4);
    assert(pool);
    void* p1 = ga_pool_alloc(pool);
    assert(p1);
    ga_pool_free(pool, p1);
    ga_pool_destroy(pool);
    return 0;
}
