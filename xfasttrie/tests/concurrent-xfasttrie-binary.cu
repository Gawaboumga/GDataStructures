//#include "concurrent-xfasttrie-binary.cuh"
//#include "Catch2/catch.hpp"
//#include "cuda/api_wrappers.h"
//
//#include "concurrent-xfasttrie-common.cuh"
//
//using key_type = unsigned int;
//using mapped_type = int;
//using XFastTrie = ConcurrentXFastTrieBinary<key_type, mapped_type, 3>;
//
//SCENARIO("CONCURRENT-X-FAST-TRIE-BINARY", "[XFASTTRIE][CONCURRENTBINARY]")
//{
//	unsigned int NUMBER_OF_WARPS = 2u;
//	int memory_size_allocated = 1u << 29u;
//	unsigned int to_insert = 1u << 5u;
//	auto current_device = cuda::device::current::get();
//	auto d_memory = cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated);
//	auto d_allocator = cuda::memory::device::make_unique<allocator_type>(current_device);
//
//	GIVEN("A XFastTrie")
//	{
//		auto d_xfasttrie = cuda::memory::device::make_unique<XFastTrie>(current_device);
//
//		cuda::launch(initialize_allocator<XFastTrie>,
//			{ 1u, NUMBER_OF_WARPS * 32u },
//			d_allocator.get(), d_memory.get(), memory_size_allocated, d_xfasttrie.get(), to_insert
//		);
//
//		WHEN("We add elements in increasing order")
//		{
//			THEN("It should be good")
//			{
//				cuda::launch(test_insert_increasing_order<XFastTrie>,
//					{ 1u, NUMBER_OF_WARPS * 32u },
//					d_xfasttrie.get()
//				);
//			}
//		}
//
//		WHEN("We add elements with dulpicates")
//		{
//			THEN("It should be good")
//			{
//				cuda::launch(test_insert_with_duplicates<XFastTrie>,
//				{ 1u, NUMBER_OF_WARPS * 32u },
//					d_xfasttrie.get()
//				);
//			}
//		}
//
//		WHEN("We add elements in random order")
//		{
//			THEN("It should be good")
//			{
//				cuda::launch(test_insert_random<XFastTrie>,
//				{ 1u, NUMBER_OF_WARPS * 32u },
//					d_xfasttrie.get()
//				);
//			}
//		}
//	}
//}
