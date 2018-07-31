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
__global__ void test_insert_increasing_order(XFastTrie* xfasttrie, unsigned int to_insert)
{
	auto block = cooperative_groups::this_thread_block();
	auto warp = cooperative_groups::tiled_partition<32>(block);
	auto thid = threadIdx.x + blockIdx.x * blockDim.x;

	if (thid / 32 >= to_insert)
		return;

	unsigned int offset = 0u;
	while (offset < to_insert)
	{
		typename XFastTrie::key_type key = offset + thid / 32;
		typename XFastTrie::mapped_type value = thid / 32;
		xfasttrie->insert(warp, key, value);

		auto it = xfasttrie->find(key);
		//ENSURE(it->second == value);

		offset += (blockDim.x * gridDim.x) / 32u;
	}
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
__global__ void test_predecessor_random(XFastTrie* xfasttrie, unsigned int to_insert)
{
	auto block = cooperative_groups::this_thread_block();
	auto warp = cooperative_groups::tiled_partition<32>(block);
	auto thid = threadIdx.x + blockIdx.x * blockDim.x;

	if (thid / 32 >= to_insert)
		return;

	xfasttrie->insert(warp, 1024 + 5, 0);
	/*if (warp.thread_rank() == 0)
		xfasttrie->debug();*/
	ENSURE(xfasttrie->predecessor(warp, 0) == xfasttrie->end());
	auto pred_5000 = xfasttrie->predecessor(warp, 5000);
	ENSURE(pred_5000 != xfasttrie->end() && pred_5000->first == 1024 + 5);
	auto pred_1023 = xfasttrie->predecessor(warp, 1023);
	ENSURE(pred_1023 == xfasttrie->end());

	xfasttrie->insert(warp, 1024 + 7, 0);
	pred_5000 = xfasttrie->predecessor(warp, 5000);
	ENSURE(pred_5000 != xfasttrie->end() && pred_5000->first == 1024 + 7);
	auto pred_1030 = xfasttrie->predecessor(warp, 1030);
	ENSURE(pred_1030 != xfasttrie->end() && pred_1030->first == 1024 + 5);

	xfasttrie->insert(warp, 1686052074, 0);
	auto pred_1886052074 = xfasttrie->predecessor(warp, 1886052074);
	ENSURE(pred_1886052074 != xfasttrie->end() && pred_1886052074->first == 1686052074);
	auto pred_1686052073 = xfasttrie->predecessor(warp, 1686052073);
	ENSURE(pred_1686052073 != xfasttrie->end() && pred_1686052073->first == 1024 + 7);

	xfasttrie->insert(warp, 986052074, 0);
	auto pred_986052073 = xfasttrie->predecessor(warp, 986052073);
	ENSURE(pred_986052073 != xfasttrie->end() && pred_986052073->first == 1024 + 7);

	xfasttrie->insert(warp, 1686042074, 0);
	pred_1686052073 = xfasttrie->predecessor(warp, 1686052073);
	ENSURE(pred_1686052073 != xfasttrie->end() && pred_1686052073->first == 1686042074);
	

	/*unsigned int offset = 0u;
	while (offset < to_insert)
	{
		int hashed_i = int(gpu::hash<int>{}(offset + thid / 32)) * 451;
		typename XFastTrie::key_type key = hashed_i; // offset + thid / 32;// 
		typename XFastTrie::mapped_type value = thid / 32;
		xfasttrie->predecessor(warp, key);
		offset += (blockDim.x * gridDim.x) / 32u;
	}*/
}

template <class XFastTrie>
__global__ void test_retrieve_size(XFastTrie* xfasttrie, unsigned int to_insert)
{
	printf("Size: %d expected: %d\n", xfasttrie->size(), to_insert);
	xfasttrie->debug();
}

template <class XFastTrie>
__global__ void test_post_condition(XFastTrie* xfasttrie)
{
	auto block = cooperative_groups::this_thread_block();
	auto warp = cooperative_groups::tiled_partition<32>(block);
	xfasttrie->post_condition(warp);
}
