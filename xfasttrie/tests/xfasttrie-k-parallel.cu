//#include "xfasttrie-k-parallel.cuh"
//#include "Catch2/catch.hpp"
//#include "cuda/api_wrappers.h"
//
//#include "allocators/default_allocator.cuh"
//#include <cassert>
//#include <cooperative_groups.h>
//
//using XTrie = XFastTrieKParallel<unsigned char, int>;
//using XTrieKey = typename XTrie::key_type;
//using XTrie3 = XFastTrieKParallel<unsigned char, int, 3>;
//using XTrie3Key = typename XTrie3::key_type;
//using BigXTrie = XFastTrieKParallel<int, int>;
//using BigXTrieKey = typename BigXTrie::key_type;
//
//__global__ void XFastTrieKParallel_initialize_allocator_small(gpu::default_allocator* allocator, char* memory, int memory_size, XTrie3* xtrie)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//	if (block.thread_rank() == 0)
//		new (allocator) gpu::default_allocator(memory, memory_size);
//	block.sync();
//	new (xtrie) XTrie3(block, *allocator);
//}
//
//__global__ void XFastTrieKParallel_initialize_allocator(gpu::default_allocator* allocator, char* memory, int memory_size, XTrie* xtrie)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//	if (block.thread_rank() == 0)
//		new (allocator) gpu::default_allocator(memory, memory_size);
//	block.sync();
//	new (xtrie) XTrie(block, *allocator);
//}
//
//__global__ void XFastTrieKParallel_initialize_allocator_big(gpu::default_allocator* allocator, char* memory, int memory_size, BigXTrie* xtrie)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//	if (block.thread_rank() == 0)
//		new (allocator) gpu::default_allocator(memory, memory_size);
//	block.sync();
//	new (xtrie) BigXTrie(block, *allocator);
//}
//
//template <typename Key, typename Value, std::size_t Universe>
//__device__ void XFastTrieKParallel_ensure_value(const XFastTrieKParallel<Key, Value, Universe>& trie, typename XFastTrieKParallel<Key, Value, Universe>::iterator it, int expected_value)
//{
//	assert(it != trie.end());
//	assert(it->second == expected_value);
//}
//
//__global__ void XFastTrieKParallel_test_insert_find_2(XTrie* triePtr)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//
//	XTrie& trie = *triePtr;
//	auto convert = [](int value) -> XTrieKey { return value;  };
//	trie.insert(block, convert(3), 3);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(3)), 3);
//	trie.insert(block, convert(1), 1);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(1)), 1);
//	trie.insert(block, convert(6), 6);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(6)), 6);
//	trie.insert(block, convert(5), 5);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(5)), 5);
//}
//
//__global__ void XFastTrieKParallel_test_insert_find(XTrie* triePtr)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//
//	XTrie& trie = *triePtr;
//	auto convert = [](int value) -> XTrieKey { return value;  };
//	trie.insert(block, convert(255), 255);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(255)), 255);
//	trie.insert(block, convert(13), 13);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(13)), 13);
//	trie.insert(block, convert(251), 251);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(251)), 251);
//	trie.insert(block, convert(15), 15);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(15)), 15);
//}
//
//__global__ void XFastTrieKParallel_test_insert_find_small(XTrie3* triePtr)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//
//	XTrie3& trie = *triePtr;
//	auto convert = [](int value) -> XTrie3Key { return value;  };
//	trie.insert(block, convert(3), 3);
//	if (block.thread_rank() == 0)
//		trie.debug();
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(3)), 3);
//	trie.insert(block, convert(1), 1);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(1)), 1);
//	trie.insert(block, convert(6), 6);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(6)), 6);
//	printf("==");
//	trie.insert(block, convert(5), 5);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(5)), 5);
//	trie.insert(block, convert(7), 7);
//	auto it = trie.find(block, convert(7));
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(7)), 7);
//	trie.insert(block, convert(4), 4);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(4)), 4);
//	trie.insert(block, convert(0), 0);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(0)), 0);
//}
//
//__global__ void XFastTrieKParallel_test_insert_find_small_2(XTrie3* triePtr)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//
//	XTrie3& trie = *triePtr;
//	auto convert = [](int value) -> XTrie3Key { return value;  };
//	trie.insert(block, convert(7), 7);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(7)), 7);
//	trie.insert(block, convert(0), 0);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(0)), 0);
//	trie.insert(block, convert(3), 3);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(3)), 3);
//	trie.insert(block, convert(5), 5);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(5)), 5);
//	trie.insert(block, convert(7), 7);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(7)), 7);
//	trie.insert(block, convert(4), 4);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(4)), 4);
//	trie.insert(block, convert(0), 0);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(0)), 0);
//}
//
//__global__ void XFastTrieKParallel_test_insert_find_small_increasing_order(XTrie3* triePtr)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//
//	XTrie3& trie = *triePtr;
//	auto convert = [](int value) -> XTrie3Key { return value;  };
//	for (int i = 0; i != trie.size(); ++i)
//	{
//		trie.insert(block, convert(i), i);
//		XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(i)), i);
//	}
//}
//
//__global__ void XFastTrieKParallel_test_insert_find_small_decreasing_order(XTrie3* triePtr)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//
//	XTrie3& trie = *triePtr;
//	auto convert = [](int value) -> XTrie3Key { return value;  };
//	for (int i = trie.size() - 1; i != 0; --i)
//	{
//		trie.insert(block, convert(i), i);
//		XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(i)), i);
//	}
//}
//
//__global__ void XFastTrieKParallel_test_predecessor_successor_small(XTrie3* triePtr)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//
//	XTrie3& trie = *triePtr;
//	auto convert = [](int value) -> XTrie3Key { return value;  };
//	assert(trie.predecessor(block, convert(4)) == trie.end());
//	assert(trie.successor(block, convert(4)) == trie.end());
//
//	trie.insert(block, convert(2), 2);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(3)), 2);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(4)), 2);
//	XFastTrieKParallel_ensure_value(trie, trie.successor(block, convert(1)), 2);
//	assert(trie.predecessor(block, convert(1)) == trie.end());
//	assert(trie.successor(block, convert(3)) == trie.end());
//
//	trie.insert(block, convert(3), 3);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(3)), 3);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(4)), 3);
//	XFastTrieKParallel_ensure_value(trie, trie.successor(block, convert(2)), 2);
//}
//
//__global__ void XFastTrieKParallel_test_insert_find_big(BigXTrie* triePtr)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//
//	BigXTrie& trie = *triePtr;
//	auto convert = [](int value) -> BigXTrieKey { return value;  };
//	trie.insert(block, convert(3), 3);
//	/*if (block.thread_rank() == 0)
//		trie.debug();*/
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(3)), 3);
//	trie.insert(block, convert(1), 1);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(1)), 1);
//	trie.insert(block, convert(6), 6);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(6)), 6);
//	trie.insert(block, convert(5), 5);
//	XFastTrieKParallel_ensure_value(trie, trie.find(block, convert(5)), 5);
//}
//
//__global__ void XFastTrieKParallel_test_predecessor_successor(XTrie* triePtr)
//{
//	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
//
//	XTrie& trie = *triePtr;
//	auto convert = [](int value) -> XTrieKey { return value;  };
//	assert(trie.predecessor(block, convert(128)) == trie.end());
//	assert(trie.successor(block, convert(128)) == trie.end());
//
//	trie.insert(block, convert(2), 2);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(128)), 2);
//	XFastTrieKParallel_ensure_value(trie, trie.successor(block, convert(1)), 2);
//	assert(trie.predecessor(block, convert(1)) == trie.end());
//	assert(trie.successor(block, convert(3)) == trie.end());
//
//	trie.insert(block, convert(13), 13);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(128)), 13);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(13)), 13);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(12)), 2);
//	XFastTrieKParallel_ensure_value(trie, trie.successor(block, convert(3)), 13);
//	assert(trie.successor(block, convert(128)) == trie.end());
//
//	trie.insert(block, convert(251), 251);
//	if (block.thread_rank() == 0)
//		trie.debug();
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(128)), 13);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(253)), 251);
//	XFastTrieKParallel_ensure_value(trie, trie.successor(block, convert(1)), 2);
//	assert(trie.predecessor(block, convert(1)) == trie.end());
//	XFastTrieKParallel_ensure_value(trie, trie.successor(block, convert(3)), 13);
//	XFastTrieKParallel_ensure_value(trie, trie.successor(block, convert(128)), 251);
//	XFastTrieKParallel_ensure_value(trie, trie.successor(block, convert(248)), 251);
//	assert(trie.successor(block, convert(252)) == trie.end());
//
//	trie.insert(block, convert(190), 190);
//	if (block.thread_rank() == 0)
//		trie.debug();
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(189)), 13);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(190)), 190);
//	XFastTrieKParallel_ensure_value(trie, trie.predecessor(block, convert(250)), 190);
//	XFastTrieKParallel_ensure_value(trie, trie.successor(block, convert(191)), 251);
//}
//
//SCENARIO("X-FAST-TRIE-K-PARALLEL", "[XFASTTRIE][KPARALLEL]")
//{
//	int memory_size_allocated = 4 * 1024 * 1024;
//	auto current_device = cuda::device::current::get();
//	auto d_memory = cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated);
//	auto d_allocator = cuda::memory::device::make_unique<gpu::default_allocator>(current_device);
//	unsigned int number_of_warps = 2u;
//
//	GIVEN("A X-fast trie for 2^3")
//	{
//		auto d_xtrie3 = cuda::memory::device::make_unique<XTrie3>(current_device);
//		cuda::launch(
//			XFastTrieKParallel_initialize_allocator_small,
//			{ 1u, 1u },
//			d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie3.get()
//		);
//
//		WHEN("We add different values")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					XFastTrieKParallel_test_insert_find_small,
//					{ 1u, number_of_warps * 32u },
//					d_xtrie3.get()
//				);
//			}
//		}
//
//
//		WHEN("We try again")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					XFastTrieKParallel_test_insert_find_small_2,
//					{ 1u, number_of_warps * 32u },
//					d_xtrie3.get()
//				);
//			}
//		}
//
//		WHEN("We try again in increasing order")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					XFastTrieKParallel_test_insert_find_small_increasing_order,
//					{ 1u, number_of_warps * 32u },
//					d_xtrie3.get()
//				);
//			}
//		}
//
//		WHEN("We try again in decreasing order")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					XFastTrieKParallel_test_insert_find_small_decreasing_order,
//					{ 1u, number_of_warps * 32u },
//					d_xtrie3.get()
//				);
//			}
//		}
//
//		WHEN("We add different values")
//		{
//			THEN("Predecessor and successor should be conformed")
//			{
//				cuda::launch(
//					XFastTrieKParallel_test_predecessor_successor_small,
//					{ 1u, number_of_warps * 32u },
//					d_xtrie3.get()
//				);
//			}
//		}
//	}
//
//	GIVEN("A X-fast trie for 2^8")
//	{
//		auto d_xtrie = cuda::memory::device::make_unique<XTrie>(current_device);
//		cuda::launch(
//			XFastTrieKParallel_initialize_allocator,
//			{ 1u, number_of_warps * 32u },
//			d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get()
//		);
//
//		WHEN("We add different values")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					XFastTrieKParallel_test_insert_find,
//					{ 1u, number_of_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We add different values")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					XFastTrieKParallel_test_insert_find_2,
//					{ 1u, number_of_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We add different values")
//		{
//			THEN("Predecessor and successor should be conformed")
//			{
//				cuda::launch(
//					XFastTrieKParallel_test_predecessor_successor,
//					{ 1u, number_of_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//	}
//
//	GIVEN("A X-fast trie for 2^32")
//	{
//		auto d_xtrie = cuda::memory::device::make_unique<BigXTrie>(current_device);
//		cuda::launch(
//			XFastTrieKParallel_initialize_allocator_big,
//			{ 1u, number_of_warps * 32u },
//			d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get()
//		);
//
//		WHEN("We add different values")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					XFastTrieKParallel_test_insert_find_big,
//					{ 1u, number_of_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//	}
//}
