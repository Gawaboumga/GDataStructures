#include "LSM.cuh"

#include "algorithms/binary_search.cuh"
#include "algorithms/for_each.cuh"
#include "algorithms/merge.cuh"
#include "algorithms/move.cuh"
#include "algorithms/set.cuh"
#include "utility/limits.cuh"
#include "utility/print.cuh"

#include "cub/block/block_radix_sort.cuh"

namespace gpu
{
	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::iterator lsm<Key, Value, N>::end()
	{
		return iterator(nullptr);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::const_iterator lsm<Key, Value, N>::end() const
	{
		return const_iterator(nullptr);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::const_iterator lsm<Key, Value, N>::cend() const
	{
		return const_iterator(nullptr);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ lsm<Key, Value, N>::lsm(block_threads group, allocator_type& allocator, size_type expected_number_of_elements) :
		m_number_of_batches{ 0u },
		m_storage{},
		m_current_buffer{ false }
	{
		m_storage = allocator.allocate<value_type>(group, expected_number_of_elements);
		m_buffers[0] = allocator.allocate<value_type>(group, expected_number_of_elements / 2u);
		m_buffers[1] = allocator.allocate<value_type>(group, expected_number_of_elements / 2u);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ lsm<Key, Value, N>::lsm(threads group, allocator_type& allocator, size_type expected_number_of_elements) :
		m_number_of_batches{ 0u },
		m_storage{},
		m_current_buffer{false}
	{
		m_storage = allocator.allocate<value_type>(group, expected_number_of_elements);
		m_buffers[0] = allocator.allocate<value_type>(group, expected_number_of_elements / 2u);
		m_buffers[1] = allocator.allocate<value_type>(group, expected_number_of_elements / 2u);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ void lsm<Key, Value, N>::clear(block_threads group)
	{
		if (group.thread_rank() == 0)
			m_number_of_batches = 0u;
		group.sync();
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::iterator lsm<Key, Value, N>::find(const key_type& key)
	{
		for (unsigned int i = 0u; i != number_of_levels(); ++i)
		{
			if (m_number_of_batches & (1 << i))
			{
				pointer end_level = level(i + 1);
				pointer result = lower_bound(level(i), end_level, key, [](const value_type& lhs, const key_type& rhs) {
					return lhs.first < rhs;
				});
				if (result != end_level && result->first == key)
					return result;
			}
		}
		return end();
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::const_iterator lsm<Key, Value, N>::find(const key_type& key) const
	{
		for (unsigned int i = 0u; i != number_of_levels(); ++i)
		{
			if (m_number_of_batches & (1 << i))
			{
				const_pointer end_level = level(i + 1);
				const_pointer result = lower_bound(level(i), end_level, key, [](const value_type& lhs, const key_type& rhs) {
					return lhs.first < rhs;
				});
				if (result != end_level && result->first == key)
					return result;
			}
		}
		return end();
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::iterator lsm<Key, Value, N>::find(threads g, const key_type& key)
	{
		iterator it;
		if (g.thread_rank() == 0)
			it = find(key);
		return reinterpret_cast<pointer>(g.shfl(reinterpret_cast<std::uintptr_t>(it), 0));
	}

		template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::const_iterator lsm<Key, Value, N>::find(threads g, const key_type& key) const
	{
		const_iterator it;
		if (g.thread_rank() == 0)
			it = find(key);
		return reinterpret_cast<pointer>(g.shfl(reinterpret_cast<std::uintptr_t>(it), 0));
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ void lsm<Key, Value, N>::insert(block_threads group, value_type value)
	{
		sort(group, value);
		unsigned int i = 0u;
		unsigned int offset = 0u;
		unsigned int number_of_elements_at_level_i = N;

		while (is_level_full(i))
		{
			merge(group, offset, number_of_elements_at_level_i);
			full_empty(group, offset, number_of_elements_at_level_i);

			++i;
			offset += number_of_elements_at_level_i;
			number_of_elements_at_level_i <<= 1u;
		}

		gpu::move(group, current_buffer().begin(), current_buffer().begin() + number_of_elements_at_level_i, m_storage.begin() + offset);

		++m_number_of_batches;
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::size_type lsm<Key, Value, N>::number_of_batches() const
	{
		return m_number_of_batches;
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::iterator lsm<Key, Value, N>::predecessor(const key_type& key)
	{
		iterator previous_max = end();
		for (unsigned int i = 0u; i != number_of_levels(); ++i)
		{
			if (m_number_of_batches & (1 << i))
			{
				pointer start_level = level(i);
				pointer end_level = level(i + 1);
				pointer result = upper_bound(start_level, end_level, key, [](const key_type& lhs, const value_type& rhs) -> bool {
					return lhs < rhs.first;
				});
				if (result != start_level)
					--result;
				if (result->first <= key)
				{
					if (previous_max == end())
						previous_max = result;
					else if (result->first > previous_max->first)
						previous_max = result;
				}
			}
		}
		return previous_max;
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::iterator lsm<Key, Value, N>::predecessor(threads g, const key_type& key)
	{
		iterator it = end();
		if (g.thread_rank() == 0)
			it = predecessor(key);
		it = reinterpret_cast<iterator>(g.shfl(reinterpret_cast<std::uintptr_t>(it), 0));
		return it;
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::size_type lsm<Key, Value, N>::size() const
	{
		return m_number_of_batches * N;
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::iterator lsm<Key, Value, N>::successor(const key_type& key)
	{
		iterator successor_min = end();
		for (unsigned int i = 0u; i != number_of_levels(); ++i)
		{
			if (m_number_of_batches & (1 << i))
			{
				pointer end_level = level(i + 1);
				pointer result = lower_bound(level(i), end_level, key, [](const value_type& lhs, const key_type& rhs) -> bool {
					return lhs.first < rhs;
				});
				if (result != end_level)
				{
					if (successor_min == end())
						successor_min = result;
					else if (result->first < successor_min->first)
						successor_min = result;
				}
			}
		}
		return successor_min;
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::iterator lsm<Key, Value, N>::successor(threads g, const key_type& key)
	{
		return successor(key);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ void lsm<Key, Value, N>::debug() const
	{
		print("Number of elements (", m_number_of_batches * N, ")\n");

		for (unsigned int i = 0u; i != number_of_levels(); ++i)
		{
			if (m_number_of_batches & (1 << i))
			{
				print("Dictonary number (", i, "): ");
				for (auto it = level(i); it != level(i + 1); ++it)
					print(" { ", it->first, ": ", it->second, " }, ");
				print("\n");
			}
		}
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::internal_storage& lsm<Key, Value, N>::current_buffer()
	{
		return m_buffers[m_current_buffer];
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ const typename lsm<Key, Value, N>::internal_storage& lsm<Key, Value, N>::current_buffer() const
	{
		return m_buffers[m_current_buffer];
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::key_type lsm<Key, Value, N>::FREE()
	{
		return numeric_limits<key_type>::max();
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ void lsm<Key, Value, N>::full_empty(block_threads group, unsigned int offset, unsigned int number_of_elements_at_level_i)
	{
		gpu::for_each(group, m_storage.begin() + offset, m_storage.begin() + offset + number_of_elements_at_level_i, [](value_type& value) {
			value.first = FREE();
		});
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ bool lsm<Key, Value, N>::is_level_full(unsigned int level) const
	{
		return m_number_of_batches & (1 << level);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ unsigned int lsm<Key, Value, N>::number_of_levels() const
	{
		return 32 - __clz(m_number_of_batches);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::pointer lsm<Key, Value, N>::level(unsigned int level)
	{
		return m_storage.begin() + N * ((1 << level) - 1u);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ typename lsm<Key, Value, N>::const_pointer lsm<Key, Value, N>::level(unsigned int level) const
	{
		return m_storage.begin() + N * ((1 << level) - 1u);
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ void lsm<Key, Value, N>::merge(block_threads group, unsigned int offset, unsigned int number_of_elements_at_level_i)
	{
		auto& other_buffer = m_buffers[!m_current_buffer];
		auto start_next_buffer = other_buffer.begin();

		auto start_first_it = m_storage.begin() + offset;
		auto start_second_it = current_buffer().begin();
		gpu::merge(group, start_first_it, start_first_it + number_of_elements_at_level_i,
		      start_second_it, start_second_it + number_of_elements_at_level_i,
		      other_buffer.begin(),
		      [](const value_type& x, const value_type& y) {
		          return (x.first >> 1u) < (y.first >> 1u);
		});
		group.sync();

		if (group.thread_rank() == 0)
			m_current_buffer = !m_current_buffer;
	}

	template <typename Key, typename Value, unsigned int N>
	__device__ void lsm<Key, Value, N>::sort(block_threads group, value_type value)
	{
		// Specialize BlockRadixSort for a 1D block of N threads owning 1 integer items each
		typedef cub::BlockRadixSort<key_type, N, 1, mapped_type> BlockRadixSort;
		// Allocate shared memory for BlockRadixSort
		__shared__ typename BlockRadixSort::TempStorage temp_storage;

		key_type thread_key[1] = { std::move(value.first) };
		mapped_type thread_value[1] = { std::move(value.second) };

		// Collectively sort the keys
		BlockRadixSort(temp_storage).Sort(thread_key, thread_value);

		auto thid = group.thread_rank();
		current_buffer()[thid].first = std::move(thread_key[0]);
		current_buffer()[thid].second = std::move(thread_value[0]);
	}
}
