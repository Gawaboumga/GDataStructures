#include "concurrent-xfasttrie-common.cuh"

#include "containers/hash_tables/default_hash_function.cuh"

template <class XFastTrie>
__global__ void initialize_allocator(allocator_type* allocator, char* memory, unsigned int memory_size, XFastTrie* xtrie, unsigned int maximal_number_of_insertions)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	if (block.thread_rank() == 0)
		new (allocator) allocator_type(memory, memory_size);
	block.sync();
	new (xtrie) XFastTrie(block, *allocator, maximal_number_of_insertions);
	block.sync();
}

template <class XFastTrie>
__global__ void test_clear_allocator(allocator_type* allocator, XFastTrie* xtrie)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	xtrie->clear(block);
	allocator->clear(block);
}

template <class XFastTrie>
__global__ void test_insert_increasing_order(XFastTrie* xfasttrie)
{
	auto block = cooperative_groups::this_thread_block();
	auto warp = cooperative_groups::tiled_partition<32>(block);
	auto thid = block.thread_rank();

	unsigned int offset = 2u;
	while (offset < 8)
	{
		typename XFastTrie::key_type key = offset + thid / 32;
		typename XFastTrie::mapped_type value = thid / 32;
		xfasttrie->insert(warp, key, value);

		auto it = xfasttrie->find(key);
		//ENSURE(it->second.value == value);

		/*block.sync();
		if (block.thread_rank() == 0)
			xfasttrie->debug();
		block.sync();
		*/
		offset += block.size() / 32;
	}

	/*block.sync();
	if (block.thread_rank() == 0)
		xfasttrie->debug();
	block.sync();*/

	/*auto it = xfasttrie->find(offset + 1);
	ENSURE(it == xfasttrie->end());

	auto pred = xfasttrie->predecessor(warp, offset);
	ENSURE(pred != xfasttrie->end() && pred->first == offset - 1);

	pred = xfasttrie->predecessor(warp, 1);
	ENSURE(pred != xfasttrie->end() && pred->first == 1);

	auto succ = xfasttrie->successor(warp, 1);
	ENSURE(succ != xfasttrie->end() && succ->first == 1);

	succ = xfasttrie->successor(warp, offset);
	ENSURE(succ == xfasttrie->end());

	ENSURE(xfasttrie->size() == 8);*/
}

template <class XFastTrie>
__global__ void test_insert_with_duplicates(XFastTrie* xfasttrie)
{
	auto block = cooperative_groups::this_thread_block();
	auto warp = cooperative_groups::tiled_partition<32>(block);
	auto thid = block.thread_rank();

	unsigned int offset = 0u;
	while (offset < 200)
	{
		typename XFastTrie::key_type key = offset + thid / 32;
		key = (key % 32 == 0) ? 0 : key;
		typename XFastTrie::mapped_type value = thid / 32;
		xfasttrie->insert(warp, key, value);

		auto it = xfasttrie->find(key);
		ENSURE(it->second.value == value);

		offset += block.size() / 32;
	}
}

template <class XFastTrie>
__global__ void test_insert_random(XFastTrie* xfasttrie, unsigned int to_insert)
{
	auto block = cooperative_groups::this_thread_block();
	auto warp = cooperative_groups::tiled_partition<32>(block);
	auto thid = threadIdx.x + blockIdx.x * blockDim.x;

	if (thid / 32 >= to_insert)
		return;

	unsigned int offset = 0u;
	while (offset < to_insert)
	{
		int hashed_i = int(gpu::hash<int>{}(offset + thid / 32));
		typename XFastTrie::key_type key = hashed_i; // offset + thid / 32;// 
		typename XFastTrie::mapped_type value = thid / 32;
		xfasttrie->insert(warp, key, value);
		offset += (blockDim.x * gridDim.x) / 32u;
	}
}

template <class XFastTrie>
__global__ void test_retrieve_size(XFastTrie* xfasttrie, unsigned int to_insert)
{
	printf("Size: %d expected: %d\n", xfasttrie->size(), to_insert);
	//xfasttrie->debug();
}

template <class XFastTrie>
__global__ void test_post_condition(XFastTrie* xfasttrie)
{
	auto block = cooperative_groups::this_thread_block();
	auto warp = cooperative_groups::tiled_partition<32>(block);
	xfasttrie->post_condition(warp);
}
