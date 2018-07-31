#ifndef FAST_INTEGER_FIXTURE_HPP
#define FAST_INTEGER_FIXTURE_HPP

#include <cuda/api_wrappers.h>
#include <hayai/hayai.hpp>

#include "concurrent/containers/hash_tables/default_hash_function.cuh"
#include "concurrent/allocators/default_allocator.cuh"

#include "utility/pair.cuh"

using allocator_type = gpu::concurrent::default_allocator;

template <class HashMap>
inline __global__ void benchmark_hash_map_fixture_initialize_allocator(allocator_type* allocator, char* memory, unsigned int memory_size, HashMap* map, unsigned int number_of_insertions)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(block);

	if (block.thread_rank() == 0)
		new (allocator) allocator_type(memory, memory_size);
	block.sync();
	new (map) HashMap(warp, *allocator, number_of_insertions);
}

template <class HashMap>
inline __global__ void benchmark_hash_map_fixture_clear(allocator_type* allocator, HashMap* map)
{
	cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
	allocator->clear(block);
	map->clear(warp);
}

template <class HashMap>
inline __global__ void benchmark_hash_map_insert(HashMap* map, unsigned int number_of_insertions, unsigned int random_offset)
{
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	for (unsigned int offset = 0; offset < number_of_insertions; offset += blockDim.x * gridDim.x)
	{
		unsigned int hashed_i = gpu::hash<unsigned int>{}(random_offset + offset + thid) % static_cast<unsigned int>(-3);
		map->insert(gpu::make_pair(hashed_i, thid));
	}
}

template <class HashMap>
inline __global__ void benchmark_hash_map_get(HashMap* map, unsigned int number_of_insertions, unsigned int random_offset)
{
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	for (unsigned int offset = 0; offset < number_of_insertions; offset += blockDim.x * gridDim.x)
	{
		int hashed_i = int(gpu::hash<unsigned int>{}(random_offset + offset + thid));
		volatile auto it = map->find(hashed_i);
	}
}

template <class HashMap>
inline __global__ void benchmark_hash_map_get_unsuccessful(HashMap* map, unsigned int number_of_insertions, unsigned int random_offset)
{
	unsigned int thid = blockDim.x * blockIdx.x + threadIdx.x;
	if (thid > number_of_insertions)
		return;

	for (unsigned int offset = 0; offset < number_of_insertions; offset += blockDim.x * gridDim.x)
	{
		volatile auto it = map->find(random_offset + offset + thid);
	}

}

using key_type = gpu::UInt32;
using mapped_type = gpu::UInt32;

constexpr unsigned int MEMORY_ALLOCATED = 1u << (26u + 5u);// 25u;
constexpr unsigned int NUMBER_OF_INSERTIONS = 1u << 20u;// 20u; // Min = 2^15
constexpr unsigned int NUMBER_OF_ITERATIONS = 10u;
constexpr unsigned int NUMBER_OF_RUNS = 30u;
constexpr unsigned int NUMBER_OF_WARPS = 32u;// 32u;
constexpr unsigned int NUMBER_OF_BLOCKS = 32u;// 32u;

static unsigned int seed = 1;
static unsigned int random_offset = 0;

template <class HashMap>
class HashMapInsertionFixture : public ::hayai::Fixture
{
	public:
		HashMapInsertionFixture() :
			::hayai::Fixture()
		{
			auto current_device = cuda::device::current::get();
			d_memory = std::move(cuda::memory::device::make_unique<char[]>(current_device, MEMORY_ALLOCATED));
			d_allocator = std::move(cuda::memory::device::make_unique<allocator_type>(current_device));

			d_hash_map = std::move(cuda::memory::device::make_unique<HashMap>(current_device));
			cuda::launch(
				benchmark_hash_map_fixture_initialize_allocator<HashMap>,
				{ 1u, 32u },
				d_allocator.get(), d_memory.get(), MEMORY_ALLOCATED, d_hash_map.get(), NUMBER_OF_INSERTIONS
			);
			cuda::device::current::get().synchronize();

			random_offset = std::hash<unsigned int>{}(seed);
			seed = (seed % NUMBER_OF_RUNS) + 1u;
		}

		virtual void TearDown()
		{
			cuda::device::current::get().synchronize();
			cuda::launch(
				benchmark_hash_map_fixture_clear<HashMap>,
				{ 1u, 32u },
				d_allocator.get(), d_hash_map.get()
			);
			cuda::device::current::get().synchronize();
		}

		void insert()
		{
			cuda::launch(
				benchmark_hash_map_insert<HashMap>,
				{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
				d_hash_map.get(), NUMBER_OF_INSERTIONS, random_offset
			);
			cuda::device::current::get().synchronize();
		}

		cuda::memory::device::unique_ptr<char[]> d_memory;
		cuda::memory::device::unique_ptr<allocator_type> d_allocator;
		cuda::memory::device::unique_ptr<HashMap> d_hash_map;
};

template <class HashMap>
class HashMapGetFixture : public ::hayai::Fixture
{
	public:
		HashMapGetFixture() :
			::hayai::Fixture()
		{
			auto current_device = cuda::device::current::get();
			d_memory = std::move(cuda::memory::device::make_unique<char[]>(current_device, MEMORY_ALLOCATED));
			d_allocator = std::move(cuda::memory::device::make_unique<allocator_type>(current_device));

			d_hash_map = std::move(cuda::memory::device::make_unique<HashMap>(current_device));
			cuda::launch(
				benchmark_hash_map_fixture_initialize_allocator<HashMap>,
				{ 1u, 32u },
				d_allocator.get(), d_memory.get(), MEMORY_ALLOCATED, d_hash_map.get(), NUMBER_OF_INSERTIONS
			);
			cuda::device::current::get().synchronize();

			insert();

			random_offset = std::hash<unsigned int>{}(seed);
			seed = (seed % NUMBER_OF_RUNS) + 1u;
		}

		virtual void TearDown()
		{
			cuda::device::current::get().synchronize();
			cuda::launch(
				benchmark_hash_map_fixture_clear<HashMap>,
				{ 1u, 32u },
				d_allocator.get(), d_hash_map.get()
			);
			cuda::device::current::get().synchronize();
		}

		void get()
		{
			cuda::launch(
				benchmark_hash_map_get<HashMap>,
				{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
				d_hash_map.get(), NUMBER_OF_INSERTIONS, random_offset
			);
			cuda::device::current::get().synchronize();
		}

	private:
		void insert()
		{
			cuda::launch(
				benchmark_hash_map_insert<HashMap>,
				{ 32u * 1u, 32u * 32u },
				d_hash_map.get(), NUMBER_OF_INSERTIONS, random_offset
			);
			cuda::device::current::get().synchronize();
		}

		cuda::memory::device::unique_ptr<char[]> d_memory;
		cuda::memory::device::unique_ptr<allocator_type> d_allocator;
		cuda::memory::device::unique_ptr<HashMap> d_hash_map;
};

template <class HashMap>
class HashMapGetUnsuccessfulFixture : public ::hayai::Fixture
{
	public:
		HashMapGetUnsuccessfulFixture() :
			::hayai::Fixture()
		{
			auto current_device = cuda::device::current::get();
			d_memory = std::move(cuda::memory::device::make_unique<char[]>(current_device, MEMORY_ALLOCATED));
			d_allocator = std::move(cuda::memory::device::make_unique<allocator_type>(current_device));

			d_hash_map = std::move(cuda::memory::device::make_unique<HashMap>(current_device));
			cuda::launch(
				benchmark_hash_map_fixture_initialize_allocator<HashMap>,
				{ 1u, 32u },
				d_allocator.get(), d_memory.get(), MEMORY_ALLOCATED, d_hash_map.get(), NUMBER_OF_INSERTIONS
			);
			cuda::device::current::get().synchronize();

			insert();

			random_offset = std::hash<unsigned int>{}(seed);
			seed = (seed % NUMBER_OF_RUNS) + 1u;
		}

		virtual void TearDown()
		{
			cuda::device::current::get().synchronize();
			cuda::launch(
				benchmark_hash_map_fixture_clear<HashMap>,
				{ 1u, 32u },
				d_allocator.get(), d_hash_map.get()
			);
			cuda::device::current::get().synchronize();
		}

		void unsuccessful_get()
		{
			cuda::launch(
				benchmark_hash_map_get_unsuccessful<HashMap>,
				{ NUMBER_OF_BLOCKS * 1u, NUMBER_OF_WARPS * 32u },
				d_hash_map.get(), NUMBER_OF_INSERTIONS, random_offset
			);
			cuda::device::current::get().synchronize();
		}

	private:
		void insert()
		{
			cuda::launch(
				benchmark_hash_map_insert<HashMap>,
				{ 32u * 1u, 32u * 32u },
				d_hash_map.get(), NUMBER_OF_INSERTIONS, random_offset
			);
			cuda::device::current::get().synchronize();
		}

		cuda::memory::device::unique_ptr<char[]> d_memory;
		cuda::memory::device::unique_ptr<allocator_type> d_allocator;
		cuda::memory::device::unique_ptr<HashMap> d_hash_map;
};

#endif // FAST_INTEGER_FIXTURE_HPP
