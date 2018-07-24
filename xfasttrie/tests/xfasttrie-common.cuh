#ifndef XFASTTRIE_COMMON_CUH
#define XFASTTRIE_COMMON_CUH

#include "allocators/default_allocator.cuh"

#include <cooperative_groups.h>

template <class XFastTrie>
__global__ void initialize_allocator(gpu::default_allocator* allocator, char* memory, int memory_size, XFastTrie* xtrie);

template <class XFastTrie>
__global__ void test_insert_find(XFastTrie* triePtr);

template <class XFastTrie>
__global__ void test_insert_increasing_order(XFastTrie* triePtr);

template <class XFastTrie>
__global__ void test_insert_decreasing_order(XFastTrie* triePtr);

template <class XFastTrie>
__global__ void test_predecessor_successor(XFastTrie* triePtr);

template <class XFastTrie>
__global__ void test_random(XFastTrie* triePtr, int number_of_insertions);

#include "xfasttrie-common-helper.cuh"

#endif // XFASTTRIE_COMMON_CUH
