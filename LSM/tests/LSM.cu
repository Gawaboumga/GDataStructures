#include "LSM.cuh"
#include "Catch2/catch.hpp"
#include "cuda/api_wrappers.h"

#include "concurrent/allocators/default_allocator.cuh"
#include "containers/hash_tables/default_hash_function.cuh"
#include <cooperative_groups.h>

namespace
{
	using key_type = int;
	using mapped_type = int;
	constexpr unsigned int NUMBER_OF_WARPS = 1u;
	constexpr unsigned int NUMBER_OF_THREADS_PER_WARP = 32u;
	using LSM = gpu::lsm<key_type, mapped_type, NUMBER_OF_WARPS * NUMBER_OF_THREADS_PER_WARP>;
	using allocator_type = gpu::concurrent::default_allocator;

	inline __device__ void lsm_ensure_value(LSM* lsm, typename LSM::iterator it, int expected_value)
	{
		ENSURE(!(it == lsm->end()));
		ENSURE(it->second == expected_value);
	}

	__global__ void lsm_initialize_allocator_small(allocator_type* allocator, char* memory, int memory_size, LSM* lsm, unsigned int maximal_size)
	{
		cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
		if (block.thread_rank() == 0)
			new (allocator) allocator_type(memory, memory_size);
		block.sync();
		new (lsm) LSM(block, *allocator, maximal_size);
	}

	__global__ void lsm_add_increasing_order(LSM* lsm)
	{
		auto block = cooperative_groups::this_thread_block();
		auto thid = block.thread_rank();

		unsigned int offset = 0u;
		for (int i = 0; i != 1 << 2; ++i)
		{
			lsm->insert(block, gpu::make_pair<key_type, mapped_type>(offset + thid, thid));

			/*block.sync();
			if (block.thread_rank() == 0)
				lsm->debug();
			block.sync();*/

			auto it = lsm->find(offset + thid);
			ENSURE(it->second == thid);

			offset += block.size();
		}

		/*block.sync();
		if (block.thread_rank() == 0)
			lsm->debug();*/
		/*
		if (thid == 0)
			lsm->debug();

		auto it = lsm->find(201);
		ENSURE(it == lsm->end());

		auto pred = lsm->predecessor(offset);
		ENSURE(pred != lsm->end() && pred->first == offset - 1);

		pred = lsm->predecessor(1);
		ENSURE(pred != lsm->end() && pred->first == 1);

		auto succ = lsm->successor(1);
		ENSURE(succ != lsm->end() && succ->first == 1);

		succ = lsm->successor(offset);
		ENSURE(succ == lsm->end());
		*/
	}

	__global__ void lsm_add_decreasing_order(LSM* lsm)
	{
		auto block = cooperative_groups::this_thread_block();
		auto thid = block.thread_rank();

		unsigned int to_insert = 6u;
		unsigned int offset = (to_insert - 1u) * NUMBER_OF_WARPS * 32u;
		for (int i = (to_insert - 1u); i != -1; --i)
		{
			lsm->insert(block, gpu::make_pair<key_type, mapped_type>(offset + thid, thid));

			auto it = lsm->find(offset + thid);
			ENSURE(it->second == thid);

			offset -= block.size();
		}

		auto it = lsm->find(201);
		ENSURE(it == lsm->end());
	}

	__global__ void lsm_add_with_duplicates(LSM* lsm)
	{
		auto block = cooperative_groups::this_thread_block();
		auto thid = block.thread_rank();

		unsigned int offset = 0u;
		for (int i = 0; i != 6; ++i)
		{
			key_type key = offset + thid;
			key = (key % 32 == 0) ? 0 : key;
			lsm->insert(block, gpu::make_pair<key_type, mapped_type>(key, offset + thid));

			if (thid == 0)
				lsm->debug();
			block.sync();

			auto it = lsm->find(key);
			ENSURE(it->second == offset + thid);

			offset += block.size();
		}
	}

	__global__ void lsm_add_random_order(LSM* lsm)
	{
		auto block = cooperative_groups::this_thread_block();
		auto thid = block.thread_rank();

		unsigned int offset = 0u;
		for (int i = 0; i != 6; ++i)
		{
			int hashed_i = int(gpu::hash<int>{}(offset + thid));
			lsm->insert(block, gpu::make_pair<key_type, mapped_type>(hashed_i, thid));

			auto it = lsm->find(offset + thid);
			ENSURE(it->second == thid);

			offset += block.size();
		}
	}

	/*
	__global__ void lsm_test_predecessor_successor(LSM* lsm)
	{
		threads warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

		ENSURE(lsm->predecessor(warp, 128) == lsm->end());
		ENSURE(lsm->successor(warp, 128) == lsm->end());

		lsm->insert(warp, 2, 2);

		lsm_ensure_value(lsm, lsm->predecessor(warp, 128), 2);
		ENSURE(lsm->successor(warp, 128) == lsm->end());
		ENSURE(lsm->predecessor(warp, 1) == lsm->end());
		lsm_ensure_value(lsm, lsm->successor(warp, 1), 2);

		lsm->insert(warp, 13, 13);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 128), 13);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 13), 13);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 12), 2);
		lsm_ensure_value(lsm, lsm->successor(warp, 3), 13);

		lsm->insert(warp, 251, 251);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 128), 13);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 253), 251);
		lsm_ensure_value(lsm, lsm->successor(warp, 128), 251);
		ENSURE(lsm->successor(warp, 252) == lsm->end());

		lsm->insert(warp, 190, 190);
		lsm_ensure_value(lsm, lsm->successor(warp, 191), 251);
		lsm_ensure_value(lsm, lsm->successor(warp, 190), 190);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 189), 13);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 250), 190);

		lsm->insert(warp, 17, 17);
		lsm->insert(warp, 35, 35);
		lsm->insert(warp, 51, 51);

		lsm_ensure_value(lsm, lsm->successor(warp, 51), 51);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 51), 51);
		lsm_ensure_value(lsm, lsm->successor(warp, 34), 35);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 36), 35);
		lsm_ensure_value(lsm, lsm->successor(warp, 36), 51);
		lsm_ensure_value(lsm, lsm->predecessor(warp, 34), 17);

		lsm_ensure_value(lsm, lsm->predecessor(warp, 190), 190); // It should be a split node
		lsm_ensure_value(lsm, lsm->successor(warp, 190), 190);
	}
	*/
}

SCENARIO("LSM", "[LSM]")
{
	int memory_size_allocated = 1u << 25u;
	unsigned int maximal_size = 1u << 20u;
	auto current_device = cuda::device::current::get();
	auto d_memory = cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated);
	auto d_allocator = cuda::memory::device::make_unique<allocator_type>(current_device);

	GIVEN("A LSM")
	{
		auto d_lsm = cuda::memory::device::make_unique<LSM>(current_device);

		cuda::launch(lsm_initialize_allocator_small,
			{ 1u, 1u },
			d_allocator.get(), d_memory.get(), memory_size_allocated, d_lsm.get(), maximal_size
		);

		WHEN("We add elements in increasing order")
		{
			THEN("It should be good")
			{
				cuda::launch(lsm_add_increasing_order,
					{ 1u, NUMBER_OF_WARPS * NUMBER_OF_THREADS_PER_WARP },
					d_lsm.get()
				);
			}
		}

		/*WHEN("We add elements in decreasing order")
		{
			THEN("It should be good")
			{
				cuda::launch(lsm_add_decreasing_order,
				{ 1u, NUMBER_OF_WARPS * 32u },
					d_lsm.get()
				);
			}
		}*/

		/*WHEN("We add elements with dulpicates")
		{
			THEN("It should be good")
			{
				cuda::launch(lsm_add_with_duplicates,
				{ 1u, NUMBER_OF_WARPS * 32u },
					d_lsm.get()
				);
			}
		}*/

		/*WHEN("We add elements in random order")
		{
			THEN("It should be good")
			{
				cuda::launch(lsm_add_random_order,
				{ 1u, NUMBER_OF_WARPS * 32u },
					d_lsm.get()
				);
			}
		}*/

		/*WHEN("We test for predecessor/successor")
		{
			THEN("It should be good")
			{
				cuda::launch(lsm_test_predecessor_successor,
				{ 1u, number_warps * 32u },
					d_lsm.get()
				);
			}
		}*/
	}
}
