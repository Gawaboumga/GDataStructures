#ifndef GPU_LSM_CUH
#define GPU_LSM_CUH

#include "concurrent/allocators/default_allocator.cuh"
#include "containers/array.cuh"
#include "utility/pair.cuh"

namespace gpu
{
	template <typename Key, typename Value, unsigned int N = 32>
	class lsm
	{
		public:
			using key_type = Key;
			using mapped_type = Value;
			using value_type = pair<key_type, mapped_type>;
			using size_type = typename memory_view<value_type>::size_type;
			using pointer = value_type*;
			using const_pointer = const value_type*;
			using iterator = pointer;
			using const_iterator = const_pointer;
			using allocator_type = concurrent::default_allocator;

			using block_threads = cooperative_groups::thread_block;
			using threads = cooperative_groups::thread_block_tile<32>;

		private:
			using internal_storage = memory_view<value_type>;
			using buffer_storage = array<value_type, N>;

		public:
			__device__ iterator end();
			__device__ const_iterator end() const;
			__device__ const_iterator cend() const;

		public:
			__device__ lsm() = default;
			__device__ lsm(block_threads group, allocator_type& allocator, size_type expected_number_of_elements);
			__device__ lsm(threads group, allocator_type& allocator, size_type expected_number_of_elements);
			__device__ lsm(lsm&& other) = default;

			__device__ void clear(block_threads group);

			__device__ iterator find(const key_type& key);
			__device__ const_iterator find(const key_type& key) const;
			__device__ iterator find(threads group, const key_type& key);
			__device__ const_iterator find(threads group, const key_type& key) const;

			__device__ void insert(block_threads group, value_type value);

			__device__ lsm& operator=(lsm&& other) = default;

			__device__ size_type number_of_batches() const;

			__device__ iterator predecessor(const key_type& key);
			__device__ iterator predecessor(threads group, const key_type& key);

			__device__ size_type size() const;
			__device__ iterator successor(const key_type& key);
			__device__ iterator successor(threads group, const key_type& key);

			__device__ void debug() const;

		private:
			__device__ internal_storage& current_buffer();
			__device__ const internal_storage& current_buffer() const;

			__device__ static key_type FREE();
			__device__ void full_empty(block_threads group, unsigned int offset, unsigned int number_of_elements_at_level_i);

			__device__ bool is_level_full(unsigned int level) const;

			__device__ unsigned int number_of_levels() const;

			__device__ pointer level(unsigned int level);
			__device__ const_pointer level(unsigned int level) const;

			__device__ void merge(block_threads group, unsigned int offset, unsigned int number_of_elements_at_level_i);

			__device__ void sort(block_threads group, value_type value);

			size_type m_number_of_batches;
			internal_storage m_storage;
			internal_storage m_buffers[2];
			bool m_current_buffer;
	};
}

#include "LSM.cu"

#endif // GPU_LSM_CUH