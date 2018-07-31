#ifndef CONCURRENT_X_FAST_TRIE_FIXTURE_HPP
#define CONCURRENT_X_FAST_TRIE_FIXTURE_HPP

#include <cuda/api_wrappers.h>
#include <hayai/hayai.hpp>

#include "concurrent/allocators/default_allocator.cuh"
#include "concurrent/containers/hash_tables/default_hash_function.cuh"
#include "utility/pair.cuh"

#include <cooperative_groups.h>

enum class Structure
{
	BTREE,
	LSM,
	XFASTTRIE
};

using allocator_type = gpu::concurrent::default_allocator;

template <class XFastTrie>
inline __global__ void benchmark_xfasttrie_fixture_initialize_allocator(allocator_type* allocator, char* memory, unsigned int memory_size, XFastTrie* xtrie, unsigned int expected_number_of_elements)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	if (block.thread_rank() == 0)
		new (allocator) allocator_type(memory, memory_size);
	block.sync();
	new (xtrie) XFastTrie(block, *allocator, expected_number_of_elements);
}

template <class XFastTrie>
inline __global__ void benchmark_clear(allocator_type* allocator, XFastTrie* xtrie)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	allocator->clear(block);
	xtrie->clear(block);
}

template <class XFastTrie>
inline __global__ void benchmark_get_thread(XFastTrie* trie, unsigned int number_of_insertions, gpu::UInt64 upper_bound, unsigned int random_offset)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	for (unsigned int offset = 0; offset < number_of_insertions; offset += blockDim.x * gridDim.x)
	{
		gpu::UInt64 hashed_i = gpu::UInt64(gpu::hash<unsigned int>{}(offset + thid + random_offset)) % upper_bound;
		volatile auto it = trie->find(hashed_i);
	}
}

template <class XFastTrie>
inline __global__ void benchmark_get_warp(XFastTrie* trie, unsigned int number_of_insertions, gpu::UInt64 upper_bound, unsigned int random_offset)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	for (unsigned int i = thid / 32u; i < number_of_insertions; i += blockDim.x * gridDim.x / 32u)
	{
		gpu::UInt64 hashed_i = gpu::UInt64(gpu::hash<unsigned int>{}(i + random_offset)) % upper_bound;
		volatile auto it = trie->find(warp, hashed_i);
	}
}

template <class XFastTrie>
inline __global__ void benchmark_predecessor_thread(XFastTrie* trie, unsigned int number_of_insertions, gpu::UInt64 upper_bound, unsigned int random_offset)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	for (unsigned int offset = 0; offset < number_of_insertions; offset += blockDim.x * gridDim.x)
	{
		gpu::UInt64 hashed_i = gpu::UInt64(gpu::hash<unsigned int>{}(offset + thid)) % upper_bound;
		hashed_i ^= random_offset;
		volatile auto it = trie->predecessor(hashed_i);
	}
}

template <class XFastTrie>
inline __global__ void benchmark_predecessor_warp(XFastTrie* trie, unsigned int number_of_insertions, gpu::UInt64 upper_bound, unsigned int random_offset)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	for (unsigned int i = thid / 32u; i < number_of_insertions; i += blockDim.x * gridDim.x / 32u)
	{
		gpu::UInt64 hashed_i = gpu::UInt64(gpu::hash<unsigned int>{}(i + random_offset)) % upper_bound;
		hashed_i ^= random_offset;
		volatile auto it = trie->predecessor(warp, hashed_i);
	}
}

template <class XFastTrie>
inline __global__ void benchmark_successor_thread(XFastTrie* trie, unsigned int number_of_insertions, gpu::UInt64 upper_bound, unsigned int random_offset)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	for (unsigned int offset = 0; offset < number_of_insertions; offset += blockDim.x * gridDim.x)
	{
		gpu::UInt64 hashed_i = gpu::UInt64(gpu::hash<unsigned int>{}(offset + thid)) % upper_bound;
		hashed_i ^= random_offset;
		volatile auto it = trie->successor(hashed_i);
	}
}

template <class XFastTrie>
inline __global__ void benchmark_successor_warp(XFastTrie* trie, unsigned int number_of_insertions, gpu::UInt64 upper_bound, unsigned int random_offset)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	for (unsigned int i = thid / 32u; i < number_of_insertions; i += blockDim.x * gridDim.x / 32u)
	{
		gpu::UInt64 hashed_i = gpu::UInt64(gpu::hash<unsigned int>{}(i + random_offset)) % upper_bound;
		hashed_i ^= random_offset;
		volatile auto it = trie->successor(warp, hashed_i);
	}
}

template <class XFastTrie>
__global__ void benchmark_insert(XFastTrie* trie, unsigned int number_of_insertions, gpu::UInt64 upper_bound, unsigned int random_offset)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);

	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid / 32 > number_of_insertions)
		return;

	for (unsigned int i = thid / 32u; i < number_of_insertions; i += blockDim.x * gridDim.x / 32u)
	{
		gpu::UInt64 hashed_i = gpu::UInt64(gpu::hash<unsigned int>{}(i + random_offset)) % upper_bound;
		trie->insert(warp, hashed_i, i);
	}
}

template <class XFastTrie>
__global__ void benchmark_insert_block(XFastTrie* trie, unsigned int number_of_insertions, gpu::UInt64 upper_bound, unsigned int random_offset)
{
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	auto block = cooperative_groups::this_thread_block();
	for (unsigned int i = 0; i < number_of_insertions; i += blockDim.x * gridDim.x)
	{
		gpu::UInt64 hashed_i = gpu::UInt64(gpu::hash<unsigned int>{}(thid + i + random_offset)) % upper_bound;
		trie->insert(block, gpu::make_pair<typename XFastTrie::key_type, typename XFastTrie::mapped_type>(hashed_i, i));

		//printf("| %u %d | ", trie->number_of_batches(), trie->number_of_batches());
	}

	/*if (thid == 0)
		gpu::print(trie->size(), " ", blockDim.x);*/
}

template <class XFastTrie>
__global__ void benchmark_test(XFastTrie* trie)
{
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid == 0)
	{
		printf("%d ", trie->size());
		printf("%d ", trie->memory_consummed());
	}
}

template <class XFastTrie>
__global__ void benchmark_test_test(XFastTrie* trie)
{
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid == 0)
	{
		printf("%d ", trie->size());
		printf("%d \n", trie->number_of_batches());
	}
}

using key_type = gpu::UInt32;
using mapped_type = int;

constexpr unsigned int HEIGHT = 32u;
constexpr gpu::UInt64 UNIVERSE = (1u << HEIGHT) - 3u;
constexpr unsigned int MEMORY_ALLOCATED = 1u << 31u;
constexpr unsigned int NUMBER_OF_INSERTIONS = 1u << 19u;
constexpr unsigned int NUMBER_OF_ITERATIONS = 1u;// 5u;
constexpr unsigned int NUMBER_OF_RUNS = 30u;// 20u;
constexpr unsigned int NUMBER_OF_WARPS = 16u;
constexpr unsigned int NUMBER_OF_BLOCKS = 32u;

constexpr unsigned int STACK_SIZE = 4000u;
static unsigned int seed = 1;
static unsigned int random_offset = 1;

template <class DataStructure, Structure structure, bool WORST = false>
struct HelperInsert
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset);
};

template <class DataStructure>
struct HelperInsert<DataStructure, Structure::LSM, false>
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset)
	{
		gpu::UInt64 upper_bound = UNIVERSE;
		cuda::launch(
			benchmark_insert_block<DataStructure>,
			{ 1u, 16u * 32u },
			trie, NUMBER_OF_INSERTIONS, upper_bound, random_offset
		);
		cuda::device::current::get().synchronize();
	}
};

template <class DataStructure>
struct HelperInsert<DataStructure, Structure::LSM, true>
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset)
	{
		gpu::UInt64 upper_bound = UNIVERSE;
		cuda::launch(
			benchmark_insert_block<DataStructure>,
			{ 1u, 16u * 32u },
			trie, NUMBER_OF_INSERTIONS - 16 * 32u, upper_bound, random_offset
		);
		cuda::device::current::get().synchronize();
	}
};

template <class DataStructure>
struct HelperInsert<DataStructure, Structure::BTREE>
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset)
	{
		gpu::UInt64 upper_bound = UNIVERSE;
		cuda::launch(
			benchmark_insert<DataStructure>,
			{ 1u, 32u },
			trie, NUMBER_OF_INSERTIONS, upper_bound, random_offset
		);
		cuda::device::current::get().synchronize();
	}
};

template <class DataStructure>
struct HelperInsert<DataStructure, Structure::XFASTTRIE>
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset)
	{
		gpu::UInt64 upper_bound = UNIVERSE;
		cuda::launch(
			benchmark_insert<DataStructure>,
			{ NUMBER_OF_BLOCKS, NUMBER_OF_WARPS * 32u },
			trie, NUMBER_OF_INSERTIONS, upper_bound, random_offset
		);
		cuda::device::current::get().synchronize();
		/*cuda::launch(
			benchmark_test<DataStructure>,
			{ 1u, 1u },
			trie
		);
		cuda::device::current::get().synchronize();*/
	}
};

template <class DataStructure, bool WARP>
struct HelperPredecessor
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset);
};

template <class DataStructure>
struct HelperPredecessor<DataStructure, true>
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset)
	{
		gpu::UInt64 upper_bound = UNIVERSE;
		cuda::launch(
			benchmark_predecessor_warp<DataStructure>,
			{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
			trie, NUMBER_OF_INSERTIONS, upper_bound, random_offset
		);
		cuda::device::current::get().synchronize();
	}
};

template <class DataStructure>
struct HelperPredecessor<DataStructure, false>
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset)
	{
		gpu::UInt64 upper_bound = UNIVERSE;
		cuda::launch(
			benchmark_predecessor_thread<DataStructure>,
			{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
			trie, NUMBER_OF_INSERTIONS, upper_bound, random_offset
		);
		cuda::device::current::get().synchronize();
	}
};

template <class DataStructure, bool WARP>
struct HelperSuccessor
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset);
};

template <class DataStructure>
struct HelperSuccessor<DataStructure, true>
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset)
	{
		gpu::UInt64 upper_bound = UNIVERSE;
		cuda::launch(
			benchmark_successor_warp<DataStructure>,
			{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
			trie, NUMBER_OF_INSERTIONS, upper_bound, random_offset
		);
		cuda::device::current::get().synchronize();
	}
};

template <class DataStructure>
struct HelperSuccessor<DataStructure, false>
{
	void operator()(DataStructure* trie, unsigned int number_of_insertions, unsigned int random_offset)
	{
		gpu::UInt64 upper_bound = UNIVERSE;
		cuda::launch(
			benchmark_successor_thread<DataStructure>,
			{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
			trie, NUMBER_OF_INSERTIONS, upper_bound, random_offset
		);
		cuda::device::current::get().synchronize();
	}
};

template <class XFastTrie, Structure structure>
class XTrieInsertionFixture : public ::hayai::Fixture
{
	public:
		XTrieInsertionFixture() :
			::hayai::Fixture()
		{
			unsigned int memory_size_allocated = MEMORY_ALLOCATED;
			auto current_device = cuda::device::current::get();
			d_memory = std::move(cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated));
			d_allocator = std::move(cuda::memory::device::make_unique<allocator_type>(current_device));

			d_xtrie = std::move(cuda::memory::device::make_unique<XFastTrie>(current_device));
			cuda::launch(
				benchmark_xfasttrie_fixture_initialize_allocator<XFastTrie>,
				{ 1u, NUMBER_OF_WARPS * 32u },
				d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get(), NUMBER_OF_INSERTIONS
			);
			cuda::device::current::get().synchronize();

			random_offset = std::hash<unsigned int>{}(seed);
			seed = (seed % NUMBER_OF_RUNS) + 1u;
		}

		void insert()
		{
			HelperInsert<XFastTrie, structure>{}(d_xtrie.get(), NUMBER_OF_INSERTIONS, random_offset);
		}

		virtual void TearDown()
		{
			cuda::device::current::get().synchronize();
			cuda::launch(
				benchmark_clear<XFastTrie>,
				{ 1u, NUMBER_OF_WARPS * 32u },
				d_allocator.get(), d_xtrie.get()
			);
			cuda::device::current::get().synchronize();
		}

		cuda::memory::device::unique_ptr<char[]> d_memory;
		cuda::memory::device::unique_ptr<gpu::concurrent::default_allocator> d_allocator;
		cuda::memory::device::unique_ptr<XFastTrie> d_xtrie;
};

template <class XFastTrie, Structure structure, bool WORST = false>
class XTrieGetThreadFixture : public ::hayai::Fixture
{
	public:
		XTrieGetThreadFixture() :
			::hayai::Fixture()
		{
			unsigned int memory_size_allocated = MEMORY_ALLOCATED;
			auto current_device = cuda::device::current::get();
			d_memory = std::move(cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated));
			d_allocator = std::move(cuda::memory::device::make_unique<allocator_type>(current_device));
			current_device.set_resource_limit(cudaLimitStackSize, STACK_SIZE);

			d_xtrie = std::move(cuda::memory::device::make_unique<XFastTrie>(current_device));
			cuda::launch(
				benchmark_xfasttrie_fixture_initialize_allocator<XFastTrie>,
				{ 1u, NUMBER_OF_WARPS * 32u },
				d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get(), NUMBER_OF_INSERTIONS
			);
			cuda::device::current::get().synchronize();

			random_offset = std::hash<unsigned int>{}(seed);
			seed = (seed % NUMBER_OF_RUNS) + 1u;

			insert();
		}

		void get_thread()
		{
			gpu::UInt64 upper_bound = UNIVERSE;
			cuda::launch(
				benchmark_get_thread<XFastTrie>,
				{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
				d_xtrie.get(), NUMBER_OF_INSERTIONS, upper_bound, random_offset
			);
			cuda::device::current::get().synchronize();
		}

		void insert()
		{
			HelperInsert<XFastTrie, structure, WORST>{}(d_xtrie.get(), NUMBER_OF_INSERTIONS, random_offset);
		}

		cuda::memory::device::unique_ptr<char[]> d_memory;
		cuda::memory::device::unique_ptr<gpu::concurrent::default_allocator> d_allocator;
		cuda::memory::device::unique_ptr<XFastTrie> d_xtrie;
};

template <class XFastTrie, Structure structure, bool WORST = false>
class XTrieGetWarpFixture : public ::hayai::Fixture
{
	public:
		XTrieGetWarpFixture() :
			::hayai::Fixture()
		{
			unsigned int memory_size_allocated = MEMORY_ALLOCATED;
			auto current_device = cuda::device::current::get();
			d_memory = std::move(cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated));
			d_allocator = std::move(cuda::memory::device::make_unique<allocator_type>(current_device));
			current_device.set_resource_limit(cudaLimitStackSize, STACK_SIZE);

			d_xtrie = std::move(cuda::memory::device::make_unique<XFastTrie>(current_device));
			cuda::launch(
				benchmark_xfasttrie_fixture_initialize_allocator<XFastTrie>,
				{ 1u, NUMBER_OF_WARPS * 32u },
				d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get(), NUMBER_OF_INSERTIONS
			);
			cuda::device::current::get().synchronize();

			random_offset = std::hash<unsigned int>{}(seed);
			seed = (seed % NUMBER_OF_RUNS) + 1u;

			insert();
		}

		void get_warp()
		{
			gpu::UInt64 upper_bound = UNIVERSE;
			cuda::launch(
				benchmark_get_warp<XFastTrie>,
				{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
				d_xtrie.get(), NUMBER_OF_INSERTIONS, upper_bound, random_offset
			);
			cuda::device::current::get().synchronize();
		}

		void insert()
		{
			HelperInsert<XFastTrie, structure, WORST>{}(d_xtrie.get(), NUMBER_OF_INSERTIONS, random_offset);
		}

		cuda::memory::device::unique_ptr<char[]> d_memory;
		cuda::memory::device::unique_ptr<gpu::concurrent::default_allocator> d_allocator;
		cuda::memory::device::unique_ptr<XFastTrie> d_xtrie;
};

template <class XFastTrie, Structure structure, bool WARP = true, bool WORST = false>
class XTriePredecessorFixture : public ::hayai::Fixture
{
	public:
		XTriePredecessorFixture() :
			::hayai::Fixture()
		{
			unsigned int memory_size_allocated = MEMORY_ALLOCATED;
			auto current_device = cuda::device::current::get();
			d_memory = std::move(cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated));
			d_allocator = std::move(cuda::memory::device::make_unique<allocator_type>(current_device));
			current_device.set_resource_limit(cudaLimitStackSize, STACK_SIZE);

			d_xtrie = std::move(cuda::memory::device::make_unique<XFastTrie>(current_device));
			cuda::launch(
				benchmark_xfasttrie_fixture_initialize_allocator<XFastTrie>,
				{ 1u, NUMBER_OF_WARPS * 32u },
				d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get(), NUMBER_OF_INSERTIONS
			);
			cuda::device::current::get().synchronize();

			random_offset = std::hash<unsigned int>{}(seed);
			seed = (seed % NUMBER_OF_RUNS) + 1u;

			insert();
		}

		void predecessor()
		{
			HelperPredecessor<XFastTrie, WARP>{}(d_xtrie.get(), NUMBER_OF_INSERTIONS, random_offset);
		}

		void insert()
		{
			HelperInsert<XFastTrie, structure, WORST>{}(d_xtrie.get(), NUMBER_OF_INSERTIONS, random_offset);
		}

		cuda::memory::device::unique_ptr<char[]> d_memory;
		cuda::memory::device::unique_ptr<gpu::concurrent::default_allocator> d_allocator;
		cuda::memory::device::unique_ptr<XFastTrie> d_xtrie;
};

template <class XFastTrie, Structure structure, bool WARP = true, bool WORST = false>
class XTrieSuccessorFixture : public ::hayai::Fixture
{
	public:
		XTrieSuccessorFixture() :
			::hayai::Fixture()
		{
			unsigned int memory_size_allocated = MEMORY_ALLOCATED;
			auto current_device = cuda::device::current::get();
			d_memory = std::move(cuda::memory::device::make_unique<char[]>(current_device, memory_size_allocated));
			d_allocator = std::move(cuda::memory::device::make_unique<allocator_type>(current_device));
			current_device.set_resource_limit(cudaLimitStackSize, STACK_SIZE);

			d_xtrie = std::move(cuda::memory::device::make_unique<XFastTrie>(current_device));
			cuda::launch(
				benchmark_xfasttrie_fixture_initialize_allocator<XFastTrie>,
				{ 1u, NUMBER_OF_WARPS * 32u },
				d_allocator.get(), d_memory.get(), memory_size_allocated, d_xtrie.get(), NUMBER_OF_INSERTIONS
			);
			cuda::device::current::get().synchronize();

			random_offset = std::hash<unsigned int>{}(seed);
			seed = (seed % NUMBER_OF_RUNS) + 1u;

			insert();
		}

		void successor()
		{
			HelperSuccessor<XFastTrie, WARP>{}(d_xtrie.get(), NUMBER_OF_INSERTIONS, random_offset);
		}

		void insert()
		{
			HelperInsert<XFastTrie, structure, WORST>{}(d_xtrie.get(), NUMBER_OF_INSERTIONS, random_offset);
		}

		cuda::memory::device::unique_ptr<char[]> d_memory;
		cuda::memory::device::unique_ptr<gpu::concurrent::default_allocator> d_allocator;
		cuda::memory::device::unique_ptr<XFastTrie> d_xtrie;
};

#endif // CONCURRENT_X_FAST_TRIE_FIXTURE_HPP
