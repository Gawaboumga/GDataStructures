#include "vEB.cuh"
#include "Catch2/catch.hpp"
#include "cuda/api_wrappers.h"

#include "concurrent-xfasttrie-common.cuh"

#include <ctime>
#include <iostream>

using key_type = gpu::UInt32;
using mapped_type = int;
using vanEB = vEB<key_type, mapped_type, 6, 11>;

SCENARIO("VAN-EMDE-BOAS-TRIE", "[VEB]")
{
	unsigned int NUMBER_OF_BLOCKS = 64u;
	unsigned int NUMBER_OF_WARPS = 16u;
	const unsigned int memory_size_allocated = 1u << 31u;
	unsigned int to_insert = 1u << 15u;
	auto current_device = cuda::device::current::get();
	auto d_memory = cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated);
	auto d_allocator = cuda::memory::device::make_unique<allocator_type>(current_device);

	GIVEN("A vEB")
	{
		auto d_vEB = cuda::memory::device::make_unique<vanEB>(current_device);

		cuda::launch(initialize_allocator<vanEB>,
			{ 1u, NUMBER_OF_WARPS * 32u },
			d_allocator.get(), d_memory.get(), memory_size_allocated, d_vEB.get(), to_insert
		);

		WHEN("We add elements in random order")
		{
			THEN("It should be good")
			{
				std::cout << to_insert << std::endl;
				const std::clock_t begin_time = std::clock();
				cuda::launch(test_insert_random<vanEB>,
					{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
					d_vEB.get(), to_insert
				);
				cuda::device::current::get().synchronize();
				std::cout << float(std::clock() - begin_time) / CLOCKS_PER_SEC;
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__);
				cuda::device::current::get().synchronize();
				/*cuda::launch(test_retrieve_size<vanEB>,
					{ 1u, 1u },
					d_vEB.get(), to_insert
				);
				cuda::device::current::get().synchronize();*/
				/*cuda::launch(test_post_condition<XFastTrie>,
					{ 1u, 32u },
					d_xfasttrie.get()
				);*/
			}
		}

		WHEN("We add elements in random order")
		{
			THEN("Ask for predecessors")
			{
				cuda::launch(test_predecessor_random<vanEB>,
					{ 1u, 32u },
					d_vEB.get(), to_insert
				);
				cuda::device::current::get().synchronize();
				fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__);
				cuda::device::current::get().synchronize();
				/*cuda::launch(test_retrieve_size<XFastTrie>,
				{ 1u, 1u },
				d_xfasttrie.get(), to_insert
				);*/
				cuda::device::current::get().synchronize();
				/*cuda::launch(test_post_condition<XFastTrie>,
				{ 1u, 32u },
				d_xfasttrie.get()
				);*/
			}
		}
	}
}
