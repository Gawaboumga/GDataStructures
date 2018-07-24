#ifndef XFASTTRIE_COMMON_CUH
#define XFASTTRIE_COMMON_CUH

#include "concurrent/allocators/default_allocator.cuh"

#include <cooperative_groups.h>

using allocator_type = gpu::concurrent::default_allocator;

template <class XFastTrie>
__global__ void initialize_allocator(allocator_type* allocator, char* memory, unsigned int memory_size, XFastTrie* xtrie, unsigned int maximal_number_of_insertions);

template <class XFastTrie>
__global__ void test_clear_allocator(allocator_type* allocator, XFastTrie* xtrie);

template <class XFastTrie>
__global__ void test_insert_increasing_order(XFastTrie* triePtr, unsigned int to_insert);

template <class XFastTrie>
__global__ void test_insert_with_duplicates(XFastTrie* triePtr);

template <class XFastTrie>
__global__ void test_insert_random(XFastTrie* triePtr, unsigned int to_insert);

template <class XFastTrie>
__global__ void test_retrieve_size(XFastTrie* xfasttrie, unsigned int to_insert);

#include "concurrent-xfasttrie-common-helper.cuh"

#endif // XFASTTRIE_COMMON_CUH
