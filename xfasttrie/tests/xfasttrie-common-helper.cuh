#include "xfasttrie-common.cuh"

#include "containers/hash_tables/default_hash_function.cuh"

#include "utility/limits.cuh"

using threads = cooperative_groups::thread_block_tile<32>;

namespace detail
{
	template <class XFastTrie>
	inline __device__ void ensure_value(XFastTrie& trie, typename XFastTrie::iterator it, int expected_value)
	{
		ENSURE(it != trie.end());
		ENSURE(it->second.value == expected_value);
	}

	template <class XFastTrie>
	inline __device__ void insert_value(threads block, XFastTrie& trie, gpu::UInt64 value)
	{
		auto convert = [](gpu::UInt64 value) -> typename XFastTrie::key_type { return value;  };
		trie.insert(block, convert(value), value);
		auto it = trie.find(block, convert(value));
		ensure_value(trie, it, value);
	}

	template <class XFastTrie>
	inline __device__ void test_insert_find_small(threads block, XFastTrie& trie)
	{
		insert_value(block, trie, 3);
		insert_value(block, trie, 1);
		insert_value(block, trie, 6);
		insert_value(block, trie, 5);
	}

	template <class XFastTrie>
	inline __device__ void test_insert_find_small_2(threads block, XFastTrie& trie)
	{
		insert_value(block, trie, 6);
		insert_value(block, trie, 3);
		insert_value(block, trie, 1);
		insert_value(block, trie, 5);
		insert_value(block, trie, 7);
		insert_value(block, trie, 4);
		insert_value(block, trie, 0);
	}

	template <class XFastTrie>
	inline __device__ void test_insert_find_small_3(threads block, XFastTrie& trie)
	{
		insert_value(block, trie, 7);
		insert_value(block, trie, 0);
		insert_value(block, trie, 3);
		insert_value(block, trie, 5);
		insert_value(block, trie, 7);
		insert_value(block, trie, 4);
		insert_value(block, trie, 0);
	}

	template <class XFastTrie>
	inline __device__ void test_insert_find_small_4(threads block, XFastTrie& trie)
	{
		insert_value(block, trie, 3);
		insert_value(block, trie, 1);
		insert_value(block, trie, 6);
		insert_value(block, trie, 5);
	}

	template <class XFastTrie>
	inline __device__ void test_insert_find_medium(threads block, XFastTrie& trie)
	{
		insert_value(block, trie, 253);
		insert_value(block, trie, 13);
		insert_value(block, trie, 251);
		insert_value(block, trie, 15);
	}

	template <class XFastTrie>
	inline __device__ void test_warp_bug_big(threads block, XFastTrie& trie)
	{
		insert_value(block, trie, 950);
		insert_value(block, trie, 16311);
		insert_value(block, trie, 16169);
	}

	template <class XFastTrie>
	inline __device__ void test_insert_find_bad_sequence(threads block, XFastTrie& trie)
	{
		insert_value(block, trie, 182);
		insert_value(block, trie, 41);
		insert_value(block, trie, 46);
		insert_value(block, trie, 59);
	}

	template <class XFastTrie>
	inline __device__ void test_predecessor_successor_small(threads block, XFastTrie& trie)
	{
		auto convert = [](int value) -> typename XFastTrie::key_type { return value;  };
		ENSURE(trie.predecessor(block, convert(4)) == trie.end());
		ENSURE(trie.successor(block, convert(4)) == trie.end());

		insert_value(block, trie, 2);
		ensure_value(trie, trie.predecessor(block, convert(3)), 2);
		ensure_value(trie, trie.predecessor(block, convert(4)), 2);
		ensure_value(trie, trie.successor(block, convert(1)), 2);
		ENSURE(trie.predecessor(block, convert(1)) == trie.end());
		ENSURE(trie.successor(block, convert(3)) == trie.end());

		insert_value(block, trie, 3);
		ensure_value(trie, trie.predecessor(block, convert(3)), 3);
		ensure_value(trie, trie.predecessor(block, convert(4)), 3);
		ensure_value(trie, trie.successor(block, convert(2)), 2);
	}

	template <class XFastTrie>
	inline __device__ void test_predecessor_successor_medium(threads block, XFastTrie& trie)
	{
		auto convert = [](int value) -> typename XFastTrie::key_type { return value;  };
		ENSURE(trie.predecessor(block, convert(128)) == trie.end());
		ENSURE(trie.successor(block, convert(128)) == trie.end());

		insert_value(block, trie, 2);
		ensure_value(trie, trie.predecessor(block, convert(128)), 2);
		ensure_value(trie, trie.successor(block, convert(1)), 2);
		ENSURE(trie.predecessor(block, convert(1)) == trie.end());
		ENSURE(trie.successor(block, convert(3)) == trie.end());

		insert_value(block, trie, 13);
		ensure_value(trie, trie.predecessor(block, convert(128)), 13);
		ensure_value(trie, trie.predecessor(block, convert(13)), 13);
		ensure_value(trie, trie.predecessor(block, convert(12)), 2);
		ensure_value(trie, trie.successor(block, convert(3)), 13);
		ENSURE(trie.successor(block, convert(128)) == trie.end());

		insert_value(block, trie, 251);
		ensure_value(trie, trie.predecessor(block, convert(128)), 13);
		ensure_value(trie, trie.predecessor(block, convert(253)), 251);
		ensure_value(trie, trie.successor(block, convert(1)), 2);
		ENSURE(trie.predecessor(block, convert(1)) == trie.end());
		ensure_value(trie, trie.successor(block, convert(3)), 13);
		ensure_value(trie, trie.successor(block, convert(128)), 251);
		ensure_value(trie, trie.successor(block, convert(248)), 251);
		ENSURE(trie.successor(block, convert(252)) == trie.end());

		insert_value(block, trie, 190);
		ensure_value(trie, trie.predecessor(block, convert(189)), 13);
		ensure_value(trie, trie.predecessor(block, convert(190)), 190);
		ensure_value(trie, trie.predecessor(block, convert(250)), 190);
		ensure_value(trie, trie.successor(block, convert(191)), 251);

		insert_value(block, trie, 15);
		ensure_value(trie, trie.predecessor(block, convert(14)), 13);
		ensure_value(trie, trie.predecessor(block, convert(15)), 15);
		ensure_value(trie, trie.predecessor(block, convert(13)), 13);
		ensure_value(trie, trie.successor(block, convert(13)), 13);
		ensure_value(trie, trie.successor(block, convert(14)), 15);
	}
}

template <class XFastTrie>
__global__ void initialize_allocator(gpu::default_allocator* allocator, char* memory, int memory_size, XFastTrie* xtrie)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	if (block.thread_rank() == 0)
		new (allocator) gpu::default_allocator(memory, memory_size);
	block.sync();
	new (xtrie) XFastTrie(block, *allocator);
}

template <class XFastTrie>
__global__ void test_insert_find(XFastTrie* triePtr)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	threads tile32 = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

	XFastTrie& trie = *triePtr;
	if (XFastTrie::UNIVERSE_SIZE >= 3)
	{
		detail::test_insert_find_small(tile32, trie);
		trie.clear(block);
		detail::test_insert_find_small_2(tile32, trie);
		trie.clear(block);
		detail::test_insert_find_small_3(tile32, trie);
		trie.clear(block);
		detail::test_insert_find_small_4(tile32, trie);
		trie.clear(block);
	}

	if (XFastTrie::UNIVERSE_SIZE >= 8)
	{
		detail::test_insert_find_medium(tile32, trie);
		trie.clear(block);
		detail::test_insert_find_bad_sequence(tile32, trie);
		trie.clear(block);
	}
	
	if (XFastTrie::UNIVERSE_SIZE >= 32)
	{
		detail::test_warp_bug_big(tile32, trie);
		trie.clear(block);
	}
}

template <class XFastTrie>
__global__ void test_insert_increasing_order(XFastTrie* triePtr)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

	XFastTrie& trie = *triePtr;
	for (int i = 1; i <= XFastTrie::UNIVERSE_SIZE; ++i)
	{
		detail::insert_value(tile32, trie, i);
	}
}

template <class XFastTrie>
__global__ void test_insert_decreasing_order(XFastTrie* triePtr)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

	XFastTrie& trie = *triePtr;
	gpu::UInt64 upper_bound = trie.maximal_size() - 3;
	for (gpu::UInt64 i = upper_bound; i != upper_bound - XFastTrie::UNIVERSE_SIZE; --i)
	{
		detail::insert_value(tile32, trie, i);
	}
}

template <class XFastTrie>
__global__ void test_predecessor_successor(XFastTrie* triePtr)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	threads tile32 = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
	XFastTrie& trie = *triePtr;

	if (XFastTrie::UNIVERSE_SIZE >= 3)
	{
		detail::test_predecessor_successor_small(tile32, trie);
		trie.clear(block);
	}

	if (XFastTrie::UNIVERSE_SIZE >= 8)
	{
		detail::test_predecessor_successor_medium(tile32, trie);
		trie.clear(block);
	}
}

template <class XFastTrie>
__global__ void test_random(XFastTrie* triePtr, int number_of_insertions)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
	XFastTrie& trie = *triePtr;

	for (int i = 1; i != number_of_insertions; ++i)
	{
		gpu::UInt64 hashed_i = gpu::UInt64(gpu::hash<int>{}(i)) % gpu::UInt64(trie.maximal_size() - 3);
		detail::insert_value(tile32, trie, hashed_i);
	}
}
