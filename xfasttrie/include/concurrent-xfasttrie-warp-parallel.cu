#include "concurrent-xfasttrie-warp-parallel.cuh"

#include "algorithms/find.cuh"
#include "utility/print.cuh"
#include "utility/warp_value.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::begin()
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::const_iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::begin() const
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::end()
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::const_iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::end() const
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::ConcurrentXFastTrieWarpParallel(block_threads block, allocator_type& allocator, unsigned int expected_number_of_elements)
{
	unsigned int power_of_two;
	if (expected_number_of_elements == 0)
		power_of_two = 10u;
	else
		power_of_two = __ffs(expected_number_of_elements) - 1u;

	for (int rank = 0; rank != SUBRANK; ++rank)
	{
		unsigned int preallocate = rank < power_of_two ? 1u << (rank + 1u) : 1u << power_of_two;
		new (&m_maps[rank]) Keyset{ block, allocator, preallocate };
	}
	new (&m_bottom) Map{ block, allocator, 1u << power_of_two };

	if (block.thread_rank() == 0)
	{
		m_head.store_unatomically(INVALID_PREDECESSOR());
		m_tail.store_unatomically(INVALID_SUCCESSOR());
	}
	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::ConcurrentXFastTrieWarpParallel(threads group, allocator_type& allocator, unsigned int expected_number_of_elements)
{
	unsigned int power_of_two;
	if (expected_number_of_elements == 0)
		power_of_two = 10u;
	else
		power_of_two = __ffs(expected_number_of_elements) - 1u;

	for (int rank = 0; rank != SUBRANK; ++rank)
	{
		unsigned int preallocate = rank < power_of_two ? 1u << (rank + 1u) : 1u << power_of_two;
		new (&m_maps[rank]) Keyset{ block, allocator, preallocate };
	}
	new (&m_bottom) Map{ block, allocator, 1u << power_of_two };

	m_head.store_unatomically(INVALID_PREDECESSOR());
	m_tail.store_unatomically(INVALID_SUCCESSOR());
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::clear(block_threads block)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);

	if (block.thread_rank() < 32)
		clear(tile32);

	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::clear(threads group)
{
	for (int rank = 0; rank != SUBRANK; ++rank)
		m_maps[rank].clear(group);

	m_bottom.clear(group);

	m_head.store_unatomically(INVALID_PREDECESSOR());
	m_tail.store_unatomically(INVALID_SUCCESSOR());
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::find(key_type key)
{
	key &= (1u << UNIVERSE) - 1u;
	return m_bottom.find(key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::find(threads group, key_type key)
{
	key &= (1u << UNIVERSE) - 1u;
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::const_iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::find(threads group, key_type key) const
{
	key &= (1u << UNIVERSE) - 1u;
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::insert(threads group, key_type key, mapped_type value)
{
	key &= (1u << UNIVERSE) - 1u;

	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	auto old_values = find_or_update(group, key);
	key_type old_minimal = old_values.first;
	key_type old_maximal = old_values.second;

	gpu::warp_value::ensure(group, old_minimal);
	gpu::warp_value::ensure(group, old_maximal);

	if (!has_data())
	{
		auto current_it = insert_at_bottom(group, key, value, INVALID_PREDECESSOR(), INVALID_SUCCESSOR());
		key_type old_head = m_head.compare_and_swap(group, INVALID_PREDECESSOR(), key);
		if (old_head == INVALID_PREDECESSOR())
		{
			key_type old_tail = m_tail.compare_and_swap(group, INVALID_SUCCESSOR(), key);
			if (old_tail == INVALID_SUCCESSOR())
			{
				return current_it;
			}

			if (key > old_tail)
				return insert_after(group, old_tail, current_it);
			else
				return insert_before(group, old_tail, current_it);
		}

		if (key > old_head)
			return insert_after(group, old_head, current_it);
		else if (key < old_head)
			return insert_before(group, old_head, current_it);
		else
			return current_it;
	}
	else
	{
		if (key < m_head)
		{
			auto current_it = insert_at_bottom(group, key, value, INVALID_PREDECESSOR(), m_head);
			return insert_before(group, m_head, current_it);
		}
		else if (key > m_tail)
		{
			auto current_it = insert_at_bottom(group, key, value, m_tail, INVALID_SUCCESSOR());
			return insert_after(group, m_tail, current_it);
		}
		else
		{
			if (key < old_minimal)
			{
				//Map_iterator successor_it = spinlock_for_value(group, old_minimal);
				auto current_it = insert_at_bottom(group, key, value, old_minimal, old_minimal);
				return insert_before(group, old_minimal, current_it);
			}
			else if (key > old_maximal)
			{
				//Map_iterator predecessor_it = spinlock_for_value(group, old_maximal);
				auto current_it = insert_at_bottom(group, key, value, old_maximal, old_maximal);
				return insert_after(group, old_maximal, current_it);
			}
			else
			{
				auto current_it = insert_at_bottom(group, key, value, old_minimal, old_maximal);
				return insert_between(group, old_minimal, old_maximal, current_it);
			}
		}
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::size_type ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::maximal_size() const
{
	return 1 << RANK;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::predecessor(threads group, key_type key)
{
	if (!has_data())
		return end();

	if (key < m_head)
		return end();
	if (key >= m_tail)
		return m_bottom.find(group, m_tail);

	auto u = binary_search(group, key);
	if (u.second == m_bottom.end())
	{
		if (key < u.first->second.minimal_left)
		{
			auto predecessor_it = m_bottom.find(group, u.first->second.minimal_left);
			return m_bottom.find(group, predecessor_it->second.predecessor);
		}
		else
			return m_bottom.find(group, u.first->second.maximal_right);
	}
	else
		return u.second;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::size_type ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::size() const
{
	return m_bottom.size();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::successor(threads group, key_type key)
{
	if (!has_data())
		return end();

	if (key > m_tail)
		return end();
	if (key <= m_head)
		return m_bottom.find(group, m_head);

	auto u = binary_search(group, key);
	if (u.second == m_bottom.end())
	{
		if (key < u.first->second.minimal_left)
			return m_bottom.find(group, u.first->second.minimal_left);
		else
		{
			auto predecessor_it = m_bottom.find(group, u.first->second.minimal_left);
			return m_bottom.find(group, predecessor_it->second.successor);
		}
	}
	else
		return u.second;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::debug() const
{
	using gpu::print;
	for (int i = 0; i != SUBRANK; ++i)
	{
		print("HashMap (", i, "): ");
		const auto& map = m_maps[i];
		map.for_each([](const auto& it) {
			print("{", key_type(it->first), "|", key_type(it->second.minimal_left), ", ", key_type(it->second.maximal_right), "}");
		});
		print("\n");
	}

	print("Bottom: ");
	m_bottom.for_each([this](const auto& it) {
		auto& value = it->second;
		if (value.predecessor != INVALID_PREDECESSOR() && value.successor != INVALID_SUCCESSOR())
			print("{", key_type(it->first), "|", value.value, "=>[", key_type(value.predecessor), "|", key_type(value.successor), "]}");
		else if (value.predecessor != INVALID_PREDECESSOR())
			print("{", key_type(it->first), "|", value.value, "=>[", key_type(value.predecessor), "|#]}");
		else if (value.successor != INVALID_SUCCESSOR())
			print("{", key_type(it->first), "|", value.value, "=>[#|", key_type(value.successor), "]}");
		else
			print("{", key_type(it->first), "|", value.value, "=>[#|#]}");
	});

	print("\nHead/Tail: ");
	if (m_head != INVALID_PREDECESSOR())
		print("Head: ", key_type(m_head), " ");
	if (m_tail != INVALID_SUCCESSOR())
		print("Tail: ", key_type(m_tail), " ");
	print("\n");
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ auto ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::binary_search(threads group, key_type key) -> gpu::pair<keyset_iterator, Map_iterator>
{
	auto thid = group.thread_rank();
	keyset_iterator found_it;
	Map_iterator bottom_it;

	unsigned int warp_offset = 0u;
	do
	{
		bool has_value = false;
		if (warp_offset + thid < SUBRANK)
		{
			key_type bits = extract_i_upper_bits(key, warp_offset + thid);
			found_it = m_maps[warp_offset + thid].find(bits);

			// We try to find out the last place such that T T T F <- We want the third T.
			has_value = found_it != m_maps[warp_offset + thid].end();
		}
		else if (warp_offset + thid == SUBRANK)
		{
			bottom_it = m_bottom.find(key);
		}

		unsigned int matching_bits = group.ballot(has_value);
		if (matching_bits)
		{
			// The idea is that we want the min, max of the lowest node in the tree where there is data
			unsigned int insert_update_separation = 31u - __clz(matching_bits);
			found_it.shfl(group, insert_update_separation);
			break;
		}
		warp_offset += group.size();
	} while (warp_offset < SUBRANK);

	bottom_it.shfl(group, SUBRANK % 32);
	return { found_it, bottom_it };
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::key_type ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::extract_i_upper_bits(key_type key, int number_of_bits) const
{
	key_type result = key >> (SUBRANK - number_of_bits);
	return result;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ auto ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::find_or_update(threads group, key_type key) -> gpu::pair<key_type, key_type>
{
	auto thid = group.thread_rank();

	key_type old_minimum = INVALID_PREDECESSOR();
	key_type old_maximum = INVALID_SUCCESSOR();

	{
		unsigned int warp_offset = 0u;
		do
		{
			bool has_value = false;
			if (warp_offset + thid < SUBRANK)
			{
				key_type bits = extract_i_upper_bits(key, warp_offset + thid);
				keyset_iterator found_it = m_maps[warp_offset + thid].find(bits);

				// We try to find out the last place such that T T T F <- We want the third T.
				has_value = found_it != m_maps[warp_offset + thid].end();
				unsigned int matching_bits = group.ballot(has_value);
				unsigned int insert_update_separation = 32u - __clz(matching_bits);

				if (has_value)
				{
					old_minimum = found_it->second.minimal_left;
					old_maximum = found_it->second.maximal_right;
				}

				if (thid >= insert_update_separation && !has_value) // Not just, should be related to group
				{
					m_maps[warp_offset + thid].insert_or_update(gpu::make_pair<key_type, Children>(bits, { key, key }),
						[](Children& lhs, Children&& rhs) {
						if (rhs.minimal_left < lhs.minimal_left)
							lhs.minimal_left.min(rhs.minimal_left);
						else if (rhs.maximal_right > lhs.maximal_right)
							lhs.maximal_right.max(rhs.maximal_right);
					});
				}
				else
				{
					while (found_it == m_maps[warp_offset + thid].end()) // Should be unlikely
						found_it = m_maps[warp_offset + thid].find(bits);

					if (key < found_it->second.minimal_left)
						found_it->second.minimal_left.min(key);
					if (key > found_it->second.maximal_right)
						found_it->second.maximal_right.max(key);
				}
			}

			group.sync();
			unsigned int matching_bits = group.ballot(has_value);
			if (matching_bits)
			{
				// The idea is that we want the min, max of the lowest node in the tree where there is data
				unsigned int insert_update_separation = 31u - __clz(matching_bits);
				old_minimum = group.shfl(old_minimum, insert_update_separation);
				old_maximum = group.shfl(old_maximum, insert_update_separation);
			}

			warp_offset += group.size();
		} while (warp_offset < SUBRANK);
	}

	return { old_minimum, old_maximum };
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::Map_iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::insert_at_bottom(threads group, key_type key, mapped_type value, key_type predecessor, key_type successor)
{
	return m_bottom.insert_or_update(group, gpu::make_pair<key_type, Node>(key, { value, predecessor, successor }), [](auto& lhs, auto&& rhs) {
		lhs.value = std::move(rhs.value);
		/*if (rhs.predecessor < lhs.predecessor)
			lhs.predecessor.min(rhs.predecessor);
		else if (rhs.successor > lhs.successor)
			lhs.successor.max(rhs.successor);*/
	});
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ bool ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::has_data() const
{
	return m_head != INVALID_PREDECESSOR() || m_tail != INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::key_type ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::INVALID_PREDECESSOR() const
{
	return gpu::numeric_limits<key_type>::max();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::key_type ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::INVALID_SUCCESSOR() const
{
	return gpu::numeric_limits<key_type>::min();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::insert_after(threads group, key_type key, Map_iterator to_insert)
{
	auto predecessor_it = spinlock_for_value(group, key);
	if (predecessor_it == m_bottom.end())
		return to_insert;

	int i = 0;
	//while (true)
	{
		auto last_predecessor_it = end();
		while (predecessor_it->second.successor != INVALID_SUCCESSOR() && predecessor_it->second.successor < to_insert->first && predecessor_it != last_predecessor_it)
		{
			last_predecessor_it = predecessor_it;
			predecessor_it = spinlock_for_value(group, predecessor_it->second.successor);
			++i;

			if (predecessor_it == m_bottom.end() || i > 20)
				return to_insert;
		}

		if (predecessor_it->second.successor == INVALID_SUCCESSOR())
		{
			to_insert->second.predecessor.store(group, predecessor_it->first);
			to_insert->second.successor.store(group, INVALID_SUCCESSOR());

			if (predecessor_it->second.successor.compare_and_swap(group, INVALID_SUCCESSOR(), to_insert->first) == INVALID_SUCCESSOR())
			{
				m_tail.max(group, to_insert->first);
				return to_insert;
			}

			//continue;
			return to_insert;
		}

		if (predecessor_it == m_bottom.end() || predecessor_it->second.successor == to_insert->first)
			return to_insert;

#ifdef GPU_XFASTTRIE_DEBUG
		ENSURE(predecessor_it->second.successor != INVALID_SUCCESSOR());
#endif // GPU_XFASTTRIE_DEBUG
		auto successor_it = spinlock_for_value(group, predecessor_it->second.successor);

		if (successor_it == m_bottom.end() || successor_it->second.predecessor == to_insert->first)
			return to_insert;

		key_type old_successor = predecessor_it->second.successor;
		key_type old_predecessor = successor_it->second.predecessor;

#ifdef GPU_XFASTTRIE_DEBUG
		ENSURE(old_successor >= to_insert->first && old_predecessor <= to_insert->first);
#endif // GPU_XFASTTRIE_DEBUG

		to_insert->second.predecessor.store(group, predecessor_it->first);
		to_insert->second.successor.store(group, successor_it->first);

		if (predecessor_it->second.successor.compare_and_swap(group, old_successor, to_insert->first) == old_successor)
		{
			if (successor_it->second.predecessor.compare_and_swap(group, old_predecessor, to_insert->first) == old_predecessor)
			{
				return to_insert;
			}
		}
		return to_insert;
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::insert_before(threads group, key_type key, Map_iterator to_insert)
{
	auto successor_it = spinlock_for_value(group, key);
	if (successor_it == m_bottom.end())
		return to_insert;

	int i = 0;
	//while (true)
	{
		auto last_successor_it = end();
		while (successor_it->second.predecessor != INVALID_PREDECESSOR() && successor_it->second.predecessor > to_insert->first && successor_it != last_successor_it)
		{
			last_successor_it = successor_it;
			successor_it = spinlock_for_value(group, successor_it->second.predecessor);
			++i;

			if (successor_it == m_bottom.end() || i > 20)
				return to_insert;
		}

		if (successor_it->second.predecessor == INVALID_PREDECESSOR())
		{
			to_insert->second.predecessor.store(group, INVALID_PREDECESSOR());
			to_insert->second.successor.store(group, successor_it->first);

			if (successor_it->second.predecessor.compare_and_swap(group, INVALID_PREDECESSOR(), to_insert->first) == INVALID_PREDECESSOR())
			{
				m_head.min(group, to_insert->first);
				return to_insert;
			}

			//continue;
			return to_insert;
		}

		if (successor_it == m_bottom.end() || successor_it->second.predecessor == to_insert->first)
			return to_insert;

#ifdef GPU_XFASTTRIE_DEBUG
		ENSURE(successor_it->second.predecessor != INVALID_PREDECESSOR());
#endif // GPU_XFASTTRIE_DEBUG
		auto predecessor_it = spinlock_for_value(group, successor_it->second.predecessor);
		if (predecessor_it == m_bottom.end())
			return to_insert;

		key_type old_successor = predecessor_it->second.successor;
		key_type old_predecessor = successor_it->second.predecessor;

#ifdef GPU_XFASTTRIE_DEBUG
		ENSURE(old_successor >= to_insert->first && old_predecessor <= to_insert->first);
#endif // GPU_XFASTTRIE_DEBUG

		to_insert->second.predecessor.store(group, predecessor_it->first);
		to_insert->second.successor.store(group, successor_it->first);

		if (predecessor_it->second.successor.compare_and_swap(group, old_successor, to_insert->first) == old_successor)
		{
			if (successor_it->second.predecessor.compare_and_swap(group, old_predecessor, to_insert->first) == old_predecessor)
			{
				return to_insert;
			}
		}
		return to_insert;
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::insert_between(threads group, key_type predecessor_it_key, key_type successor_it_key, Map_iterator to_insert)
{
	auto predecessor_it = spinlock_for_value(group, predecessor_it_key);
	auto successor_it = spinlock_for_value(group, successor_it_key);

	if (predecessor_it == m_bottom.end() || successor_it == m_bottom.end())
		return to_insert;

	int i = 0;
	//while (true)
	{
		while (predecessor_it->second.successor < to_insert->first)
		{
			predecessor_it = spinlock_for_value(group, predecessor_it->second.successor);
			++i;

			if (predecessor_it == m_bottom.end() || i > 20)
				return to_insert;
		}

		while (successor_it->second.predecessor > to_insert->first)
		{
			successor_it = spinlock_for_value(group, successor_it->second.predecessor);
			++i;

			if (successor_it == m_bottom.end() || i > 20)
				return to_insert;
		}

		key_type old_successor = predecessor_it->second.successor;
		key_type old_predecessor = successor_it->second.predecessor;

#ifdef GPU_XFASTTRIE_DEBUG
		ENSURE(old_successor >= to_insert->first && old_predecessor <= to_insert->first);
#endif // GPU_XFASTTRIE_DEBUG

		to_insert->second.predecessor.store(group, predecessor_it->first);
		to_insert->second.successor.store(group, successor_it->first);

		if (predecessor_it->second.successor.compare_and_swap(group, old_successor, to_insert->first) == old_successor)
		{
			if (successor_it->second.predecessor.compare_and_swap(group, old_predecessor, to_insert->first) == old_predecessor)
			{
				return to_insert;
			}
		}
		return to_insert;
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::Map_iterator ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::spinlock_for_value(threads group, key_type key)
{
	int i = 0;
	Map_iterator result_it;
	do
	{
		result_it = m_bottom.find(group, key);
		++i;
		if (i > 32)
			return m_bottom.end();
	} while (result_it == m_bottom.end());
	return result_it;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void ConcurrentXFastTrieWarpParallel<Key, Value, UNIVERSE>::post_condition(threads group)
{
#ifdef GPU_XFASTTRIE_DEBUG
	for (int i = 0; i != SUBRANK; ++i)
	{
		auto& map = m_maps[i];
		map.for_each([this, group, i](const auto& it) {
			if (i + 1 < SUBRANK)
			{
				auto left_child = m_maps[i + 1].find(group, it->first << 1);
				auto right_child = m_maps[i + 1].find(group, (it->first << 1) | 1);
				ENSURE(left_child != m_maps[i + 1].end() || right_child != m_maps[i + 1].end());
			}
		});
	}

	if (m_head != INVALID_PREDECESSOR())
	{
		ENSURE(m_head == m_tail || m_tail != INVALID_SUCCESSOR());
	}
	else
		ENSURE(m_tail == INVALID_SUCCESSOR());

	if (has_data())
	{
		m_bottom.for_each([this, group](const auto& it) {
			if (it->first != m_head)
			{
				auto predecessor_it = m_bottom.find(group, it->second.predecessor);
				ENSURE(it->first > predecessor_it->first);
				ENSURE(it->first == predecessor_it->second.successor);
			}
			if (it->first != m_tail)
			{
				auto successor_it = m_bottom.find(group, it->second.successor);
				ENSURE(it->first < successor_it->first);
				ENSURE(it->first == successor_it->second.predecessor);
			}
		});
	}
#endif // GPU_XFASTTRIE_DEBUG
}
