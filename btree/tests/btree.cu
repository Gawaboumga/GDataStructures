#include "btree.cuh"
#include "Catch2/catch.hpp"
#include "cuda/api_wrappers.h"

#include "allocators/default_allocator.cuh"
#include "containers/hash_tables/default_hash_function.cuh"
#include <cooperative_groups.h>

using BTREE = gpu::BTree<gpu::Int64, gpu::Int64>;
using threads = cooperative_groups::thread_block_tile<32>;

inline __device__ void btree_ensure_value(BTREE* btree, typename BTREE::iterator it, int expected_value)
{
	ENSURE(!(it == btree->end()));
	ENSURE(it->second == expected_value);
}

__global__ void btree_initialize_allocator_small(gpu::default_allocator* allocator, char* memory, int memory_size, BTREE* btree)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	if (block.thread_rank() == 0)
		new (allocator) gpu::default_allocator(memory, memory_size);
	block.sync();
	new (btree) BTREE(block, *allocator);
}

__global__ void btree_add_increasing_order(BTREE* btree)
{
	threads warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

	for (int i = 0; i != 200; ++i)
	{
		btree->insert(warp, i, i);
		warp.sync();
	}

	/*auto it = btree->find(warp, 201);
	ENSURE(it == btree->end());

	it = btree->find(201);
	ENSURE(it == btree->end());*/
}

__global__ void btree_add_decreasing_order(BTREE* btree)
{
	threads warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

	for (int i = 200; i != 0; --i)
	{
		btree->insert(warp, i, i);
	}
}

__global__ void btree_add_random_order(BTREE* btree)
{
	threads warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

	for (int i = 500; i != 0; --i)
	{
		int hashed_i = int(gpu::hash<int>{}(i));
		btree->insert(warp, hashed_i, i);

		auto it = btree->find(warp, hashed_i);
		ENSURE(it->second == i);

		it = btree->find(hashed_i);
		ENSURE(it->second == i);
	}
}

__global__ void btree_test_predecessor_successor(BTREE* btree)
{
	threads warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

	ENSURE(btree->predecessor(warp, 128) == btree->end());
	ENSURE(btree->successor(warp, 128) == btree->end());

	btree->insert(warp, 2, 2);

	btree_ensure_value(btree, btree->predecessor(warp, 128), 2);
	ENSURE(btree->successor(warp, 128) == btree->end());
	ENSURE(btree->predecessor(warp, 1) == btree->end());
	btree_ensure_value(btree, btree->successor(warp, 1), 2);

	btree->insert(warp, 13, 13);
	btree_ensure_value(btree, btree->predecessor(warp, 128), 13);
	btree_ensure_value(btree, btree->predecessor(warp, 13), 13);
	btree_ensure_value(btree, btree->predecessor(warp, 12), 2);
	btree_ensure_value(btree, btree->successor(warp, 3), 13);

	btree->insert(warp, 251, 251);
	btree_ensure_value(btree, btree->predecessor(warp, 128), 13);
	btree_ensure_value(btree, btree->predecessor(warp, 253), 251);
	btree_ensure_value(btree, btree->successor(warp, 128), 251);
	ENSURE(btree->successor(warp, 252) == btree->end());

	btree->insert(warp, 190, 190);
	btree_ensure_value(btree, btree->successor(warp, 191), 251);
	btree_ensure_value(btree, btree->successor(warp, 190), 190);
	btree_ensure_value(btree, btree->predecessor(warp, 189), 13);
	btree_ensure_value(btree, btree->predecessor(warp, 250), 190);

	btree->insert(warp, 17, 17);
	btree->insert(warp, 35, 35);
	btree->insert(warp, 51, 51);

	btree_ensure_value(btree, btree->successor(warp, 51), 51);
	btree_ensure_value(btree, btree->predecessor(warp, 51), 51);
	btree_ensure_value(btree, btree->successor(warp, 34), 35);
	btree_ensure_value(btree, btree->predecessor(warp, 36), 35);
	btree_ensure_value(btree, btree->successor(warp, 36), 51);
	btree_ensure_value(btree, btree->predecessor(warp, 34), 17);

	btree_ensure_value(btree, btree->predecessor(warp, 190), 190); // It should be a split node
	btree_ensure_value(btree, btree->successor(warp, 190), 190);
}

SCENARIO("BTree", "[BTree]")
{
	int memory_size_allocated = 32 * 1024 * 1024;
	auto current_device = cuda::device::current::get();
	auto d_memory = cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated);
	auto d_allocator = cuda::memory::device::make_unique<gpu::default_allocator>(current_device);
	current_device.set_resource_limit(cudaLimitStackSize, 4000);
	unsigned int number_warps = 1u;

	GIVEN("A BTree")
	{
		auto d_btree = cuda::memory::device::make_unique<BTREE>(current_device);

		cuda::launch(btree_initialize_allocator_small,
			{ 1u, 1u },
			d_allocator.get(), d_memory.get(), memory_size_allocated, d_btree.get()
		);

		/*WHEN("We add some elements")
		{
			THEN("We should retrieve them")
			{
				cuda::launch(btree_add,
				{ 1u, number_warps * 32u },
				d_btree.get());
			}
		}*/

		WHEN("We add elements in increasing order")
		{
			THEN("It should be good")
			{
				cuda::launch(btree_add_increasing_order,
					{ 1u, number_warps * 32u },
					d_btree.get()
				);
			}
		}

		WHEN("We add elements in decreasing order")
		{
			THEN("It should be good")
			{
				cuda::launch(btree_add_decreasing_order,
				{ 1u, number_warps * 32u },
					d_btree.get()
				);
			}
		}

		WHEN("We add elements in random order")
		{
			THEN("It should be good")
			{
				cuda::launch(btree_add_random_order,
				{ 1u, number_warps * 32u },
					d_btree.get()
				);
			}
		}

		WHEN("We test for predecessor/successor")
		{
			THEN("It should be good")
			{
				cuda::launch(btree_test_predecessor_successor,
				{ 1u, number_warps * 32u },
					d_btree.get()
				);
			}
		}
	}
}
