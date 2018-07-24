//#include "xfasttrie-warp-parallel.cuh"
//#include "Catch2/catch.hpp"
//#include "cuda/api_wrappers.h"
//
//#include "xfasttrie-common.cuh"
//
//using XTrie = XFastTrieWarpParallel<unsigned char, int>;
//using XTrieKey = typename XTrie::key_type;
//using XTrie3 = XFastTrieWarpParallel<unsigned char, int, 3>;
//using XTrie3Key = typename XTrie3::key_type;
//using MediumXTrie = XFastTrieWarpParallel<int, int, 16>;
//using MediumXTrieKey = typename MediumXTrie::key_type;
//using BigXTrie = XFastTrieWarpParallel<int, int>;
//using BigXTrieKey = typename BigXTrie::key_type;
//using HugeXTrie = XFastTrieWarpParallel<gpu::UInt64, int>;
//using HugeXTrieKey = typename HugeXTrie::key_type;
//
//SCENARIO("X-FAST-TRIE-WARP-PARALLEL", "[XFASTTRIE][WARPPARALLEL]")
//{
//	int memory_size_allocated = 1024u * 1024 * 1024;
//	auto current_device = cuda::device::current::get();
//	auto d_memory = cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated);
//	auto d_allocator = cuda::memory::device::make_unique<gpu::default_allocator>(current_device);
//	unsigned int number_warps = 1u;
//
//	GIVEN("A X-fast trie for 2^3")
//	{
//		auto d_xtrie3 = cuda::memory::device::make_unique<XTrie3>(current_device);
//		cuda::launch(
//			initialize_allocator<XTrie3>,
//			{ 1u, number_warps * 32u },
//			d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie3.get()
//		);
//
//		WHEN("We add different values")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_find<XTrie3>,
//					{ 1u, number_warps * 32u },
//					d_xtrie3.get()
//				);
//			}
//		}
//
//		WHEN("We try in increasing order")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_increasing_order<XTrie3>,
//					{ 1u, number_warps * 32u },
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
//					test_insert_decreasing_order<XTrie3>,
//					{ 1u, number_warps * 32u },
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
//					test_predecessor_successor<XTrie3>,
//					{ 1u, number_warps * 32u },
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
//			initialize_allocator<XTrie>,
//			{ 1u, number_warps * 32u },
//			d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get()
//		);
//
//		WHEN("We add different values")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_find<XTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We try in increasing order")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_increasing_order<XTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We try again in decreasing order")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_decreasing_order<XTrie>,
//					{ 1u, number_warps * 32u },
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
//					test_predecessor_successor<XTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We add random values")
//		{
//			THEN("It should be ok")
//			{
//				cuda::launch(
//					test_random<XTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get(), 10
//				);
//			}
//		}
//	}
//
//	GIVEN("A X-fast trie for 2^32")
//	{
//		auto d_xtrie = cuda::memory::device::make_unique<BigXTrie>(current_device);
//		cuda::launch(
//			initialize_allocator<BigXTrie>,
//			{ 1u, number_warps * 32u },
//			d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get()
//		);
//
//		WHEN("We add different values")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_find<BigXTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We try in increasing order")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_increasing_order<BigXTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We try again in decreasing order")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_decreasing_order<BigXTrie>,
//					{ 1u, number_warps * 32u },
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
//					test_predecessor_successor<BigXTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We add random values")
//		{
//			THEN("It should be ok")
//			{
//				cuda::launch(
//					test_random<BigXTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get(), 10
//				);
//			}
//		}
//	}
//
//	GIVEN("A X-fast trie for 2^64")
//	{
//		auto d_xtrie = cuda::memory::device::make_unique<HugeXTrie>(current_device);
//		cuda::launch(
//			initialize_allocator<HugeXTrie>,
//			{ 1u, number_warps * 32u },
//			d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get()
//		);
//
//		WHEN("We add different values")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_find<HugeXTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We try in increasing order")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_increasing_order<HugeXTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We try again in decreasing order")
//		{
//			THEN("We should be able to retrieve them")
//			{
//				cuda::launch(
//					test_insert_decreasing_order<HugeXTrie>,
//					{ 1u, number_warps * 32u },
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
//					test_predecessor_successor<HugeXTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get()
//				);
//			}
//		}
//
//		WHEN("We add random values")
//		{
//			THEN("It should be ok")
//			{
//				cuda::launch(
//					test_random<HugeXTrie>,
//					{ 1u, number_warps * 32u },
//					d_xtrie.get(), 10
//				);
//			}
//		}
//	}
//}
