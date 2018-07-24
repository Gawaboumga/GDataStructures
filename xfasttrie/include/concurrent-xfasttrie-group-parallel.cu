#include "concurrent-xfasttrie-group-parallel.cuh"

#include "algorithms/find.cuh"
#include "utility/limits.cuh"
#include "utility/print.cuh"
#include "utility/warp_value.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::begin()
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::const_iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::begin() const
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::end()
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::const_iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::end() const
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::ConcurrentXFastTrieGroupParallel(block_threads block, allocator_type& allocator, unsigned int expected_number_of_elements)
{
	unsigned int power_of_two;
	if (expected_number_of_elements == 0)
		power_of_two = 10u;
	else
		power_of_two = __ffs(expected_number_of_elements) - 1u;

	for (int rank = 0; rank != NUMBER_OF_KEYSETS; ++rank)
	{
		unsigned int preallocate = rank * (GROUP_SIZE + 1u) < power_of_two ? 1u << (rank * (GROUP_SIZE + 1u) + 2u) : 1u << power_of_two;
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::ConcurrentXFastTrieGroupParallel(threads group, allocator_type& allocator, unsigned int expected_number_of_elements)
{
	unsigned int power_of_two;
	if (expected_number_of_elements == 0)
		power_of_two = 10u;
	else
		power_of_two = __ffs(expected_number_of_elements) - 1u;

	for (int rank = 0; rank != NUMBER_OF_KEYSETS; ++rank)
	{
		unsigned int preallocate = rank * (GROUP_SIZE + 1u) < power_of_two ? 1u << (rank * (GROUP_SIZE + 1u) + 2u) : 1u << power_of_two;
		new (&m_maps[rank]) Keyset{ group, allocator, preallocate };
	}
	new (&m_bottom) Map{ group, allocator, 1u << power_of_two };

	if (group.thread_rank() == 0)
	{
		m_head.store_unatomically(INVALID_PREDECESSOR());
		m_tail.store_unatomically(INVALID_SUCCESSOR());
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::clear(block_threads block)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);

	if (block.thread_rank() < 32)
		clear(tile32);

	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::clear(threads group)
{
	for (int rank = 0; rank != NUMBER_OF_KEYSETS; ++rank)
		m_maps[rank].clear(group);

	m_bottom.clear(group);

	m_head.store_unatomically(INVALID_PREDECESSOR());
	m_tail.store_unatomically(INVALID_SUCCESSOR());
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::find(key_type key)
{
	key &= (1u << UNIVERSE) - 1u;
	return m_bottom.find(key);
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::find(threads group, key_type key)
{
	key &= (1u << UNIVERSE) - 1u;
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::const_iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::find(threads group, key_type key) const
{
	key &= (1u << UNIVERSE) - 1u;
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::insert(threads group, key_type key, mapped_type value)
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
		key_type old_head = m_head.compare_and_swap(INVALID_PREDECESSOR(), key);
		if (old_head == INVALID_PREDECESSOR())
		{
			key_type old_tail = m_tail.compare_and_swap(INVALID_SUCCESSOR(), key);
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::size_type ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::maximal_size() const
{
	return 1 << RANK;
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::predecessor(threads group, key_type key)
{
	if (!has_data())
		return end();

	if (key < m_head)
		return end();
	if (key >= m_tail)
		return m_bottom.find(group, m_tail);

	BinarySearchResult u = binary_search(group, key);
	if (u.bottom_it == m_bottom.end())
	{
		const keyset_iterator& it = u.it;
		unsigned int index = u.index;
		if (key < it->second.minimal_left[index])
		{
			auto predecessor_it = m_bottom.find(group, it->second.minimal_left[index]);
			return m_bottom.find(group, predecessor_it->second.predecessor);
		}
		else
			return m_bottom.find(group, it->second.maximal_right[index]);
	}
	else
		return u.bottom_it;
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::size_type ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::size() const
{
	return m_bottom.size();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::successor(threads group, key_type key)
{
	if (!has_data())
		return end();

	if (key > m_tail)
		return end();
	if (key <= m_head)
		return m_bottom.find(group, m_head);

	BinarySearchResult u = binary_search(group, key);
	if (u.bottom_it == m_bottom.end())
	{
		const keyset_iterator& it = u.it;
		unsigned int index = u.index;

		if (key < it->second.minimal_left[index])
			return m_bottom.find(group, it->second.minimal_left[index]);
		else
		{
			auto predecessor_it = m_bottom.find(group, it->second.minimal_left[index]);
			return m_bottom.find(group, predecessor_it->second.successor);
		}
	}
	else
		return u.bottom_it;
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::debug() const
{
	using gpu::print;
	for (int i = 0; i != NUMBER_OF_KEYSETS; ++i)
	{
		print("HashMap (", i, "): ");
		const auto& map = m_maps[i];
		map.for_each([this, i, NUMBER_OF_KEYSETS = NUMBER_OF_KEYSETS, TAIL_GROUP_SIZE = TAIL_GROUP_SIZE, GROUP_SIZE = GROUP_SIZE](const auto& it) {
			print("{", key_type(it->first), "|");
			print("0: (", key_type(it->second.minimal_left[0]), ", ", key_type(it->second.maximal_right[0]), ")");
			unsigned int group_size = (i == NUMBER_OF_KEYSETS - 1u) ? TAIL_GROUP_SIZE : GROUP_SIZE;
			if (group_size)
			{
				for (unsigned int g = 0u; g != group_size; ++g)
				{
					print(" ", g + 1, ": ");
					unsigned int offset = (1u << (g + 1u)) - 1u;
					for (unsigned int i = 0u; i != 1u << (g + 1u); ++i)
					{
						print("(", key_type(it->second.minimal_left[offset + i]), ", ", key_type(it->second.maximal_right[offset + i]), ")");
					}
				}
			}
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ auto ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::binary_search(threads group, key_type key) -> BinarySearchResult
{
	auto thid = group.thread_rank();
	bool has_value = false;
	unsigned int index;
	keyset_iterator found_it;
	Map_iterator bottom_it;
	if (thid < NUMBER_OF_KEYSETS)
	{
		key_type group_bits;
		key_type tail_bits;
		if (thid == NUMBER_OF_KEYSETS - 1u)
		{
			group_bits = key >> (TAIL_GROUP_SIZE + 1u);
			tail_bits = key & ((1u << (TAIL_GROUP_SIZE + 1u)) - 1u);
			tail_bits >>= 1u; // Avoid last bit
			found_it = m_maps[thid].find(group_bits);
		}
		else
		{
			unsigned int shift_value = (TAIL_GROUP_SIZE + 1u) + (NUMBER_OF_KEYSETS - thid - 2u) * (GROUP_SIZE + 1) + 1u; // Don't forget last bit
			group_bits = key >> shift_value;
			tail_bits = group_bits & ((1u << GROUP_SIZE) - 1u);
			group_bits = group_bits >> GROUP_SIZE; // We get the upper bits
			found_it = m_maps[thid].find(group_bits);
		}

		// We try to find out the last place such that T T T F <- We want the third T.
		has_value = found_it != m_maps[thid].end();

		if (has_value)
			index = get_highest_index(tail_bits, found_it->second, thid);
	}
	else if (thid == NUMBER_OF_KEYSETS)
	{
		bottom_it = m_bottom.find(key);
	}

	unsigned int matching_bits = group.ballot(has_value);
	if (matching_bits)
	{
		// The idea is that we want the min, max of the lowest node in the tree where there is data
		unsigned int insert_update_separation = 31u - __clz(matching_bits);
		index = group.shfl(index, insert_update_separation);
		found_it.shfl(group, insert_update_separation);
	}
	bottom_it.shfl(group, NUMBER_OF_KEYSETS);
	return { found_it, bottom_it, index };
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::key_type ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::extract_i_upper_bits(key_type key, int number_of_bits) const
{
	key_type result = key >> (SUBRANK - number_of_bits);
	return result;
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ auto ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::find_or_update(threads group, key_type key) -> gpu::pair<key_type, key_type>
{
	auto thid = group.thread_rank();

	key_type old_minimum = INVALID_PREDECESSOR();
	key_type old_maximum = INVALID_SUCCESSOR();
	bool has_value = false;
	if (thid < NUMBER_OF_KEYSETS)
	{
		key_type group_bits;
		key_type tail_bits;
		keyset_iterator found_it;
		if (thid == NUMBER_OF_KEYSETS - 1u)
		{
			group_bits = key >> (TAIL_GROUP_SIZE + 1u);
			tail_bits = key & ((1u << (TAIL_GROUP_SIZE + 1u)) - 1u);
			tail_bits >>= 1u; // Avoid last bit
		}
		else
		{
			unsigned int shift_value = (TAIL_GROUP_SIZE + 1u) + (NUMBER_OF_KEYSETS - thid - 2u) * (GROUP_SIZE + 1) + 1u; // Don't forget last bit
			group_bits = key >> shift_value;
			tail_bits = group_bits & ((1u << GROUP_SIZE) - 1u);
			group_bits = group_bits >> GROUP_SIZE; // We get the upper bits
		}
		found_it = m_maps[thid].find(group_bits);

		// We try to find out the last place such that T T T F <- We want the third T.
		has_value = found_it != m_maps[thid].end();

		unsigned int matching_bits = group.ballot(has_value);
		unsigned int insert_update_separation = 32u - __clz(matching_bits);

		if (has_value)
		{
			unsigned int i = get_highest_index(tail_bits, found_it->second, thid);
			old_minimum = found_it->second.minimal_left[i];
			old_maximum = found_it->second.maximal_right[i];
		}

		if (thid >= insert_update_separation && !has_value) // Not just, should be related to group
		{
			m_maps[thid].insert_or_update(gpu::make_pair<key_type, Child>(group_bits, make_children(tail_bits, key, thid)),
				[thid, NUMBER_OF_KEYSETS = NUMBER_OF_KEYSETS, TAIL_GROUP_SIZE = TAIL_GROUP_SIZE, GROUP_SIZE = GROUP_SIZE]
			(Child& lhs, Child&& rhs) {
				unsigned int upper_bound = thid == NUMBER_OF_KEYSETS - 1u ? TAIL_GROUP_SIZE : GROUP_SIZE;
				for (unsigned int i = 0u; i != (1u << (upper_bound + 1u)) - 1u; ++i)
				{
					if (rhs.minimal_left[i] < lhs.minimal_left[i])
						lhs.minimal_left[i].min(rhs.minimal_left[i]);
					if (rhs.maximal_right[i] > lhs.maximal_right[i])
						lhs.maximal_right[i].max(rhs.maximal_right[i]);
				}
			});
		}
		else
		{
			while (found_it == m_maps[thid].end()) // Should be unlikely
				found_it = m_maps[thid].find(group_bits);

			update_key(found_it->second, tail_bits, key, thid);
		}
	}

	unsigned int matching_bits = group.ballot(has_value);
	if (matching_bits)
	{
		// The idea is that we want the min, max of the lowest node in the tree where there is data
		unsigned int insert_update_separation = 31u - __clz(matching_bits);
		old_minimum = group.shfl(old_minimum, insert_update_separation);
		old_maximum = group.shfl(old_maximum, insert_update_separation);
	}

	return { old_minimum, old_maximum };
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ unsigned int ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::get_highest_index(key_type tail_bits, const Child& value, unsigned int thid)
{
	unsigned int upper_bound = thid == NUMBER_OF_KEYSETS - 1u ? TAIL_GROUP_SIZE : GROUP_SIZE;
	unsigned int offset = (1u << upper_bound) - 1u;
	while (offset)
	{
		unsigned int position = offset + tail_bits;
		if (value.minimal_left[position] != INVALID_PREDECESSOR())
			return position;
		offset = offset >> 1u;
		tail_bits = tail_bits >> 1u;
	}
	return 0u;
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ auto ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::make_children(key_type tail_bits, const key_type& key, unsigned int thid) -> Child
{
	Child child;
	for (unsigned int i = 0u; i != (1u << (GROUP + 1u)) - 1u; ++i)
	{
		child.minimal_left[i].store_unatomically(INVALID_PREDECESSOR());
		child.maximal_right[i].store_unatomically(INVALID_SUCCESSOR());
	}

	unsigned int upper_bound = thid == NUMBER_OF_KEYSETS - 1u ? TAIL_GROUP_SIZE : GROUP_SIZE;
	unsigned int offset = (1u << upper_bound) - 1u;
	while (offset)
	{
		unsigned int position = offset + tail_bits;
		child.minimal_left[position].store_unatomically(key);
		child.maximal_right[position].store_unatomically(key);
		offset = offset >> 1u;
		tail_bits = tail_bits >> 1u;
	}
	child.minimal_left[0u].store_unatomically(key);
	child.maximal_right[0u].store_unatomically(key);
	return child;
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::update_key(Child& child, key_type tail_bits, const key_type& key, unsigned int thid)
{
	unsigned int upper_bound = thid == NUMBER_OF_KEYSETS - 1u ? TAIL_GROUP_SIZE : GROUP_SIZE;
	unsigned int offset = (1u << upper_bound) - 1u;
	while (offset)
	{
		unsigned int position = offset + tail_bits;
		if (key < child.minimal_left[position])
			child.minimal_left[position].min(key);
		if (key > child.maximal_right[position])
			child.maximal_right[position].max(key);
		offset = offset >> 1u;
		tail_bits = tail_bits >> 1u;
	}
	if (key < child.minimal_left[0u])
		child.minimal_left[0u].min(key);
	if (key > child.maximal_right[0u])
		child.maximal_right[0u].max(key);
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::Map_iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::insert_at_bottom(threads group, key_type key, mapped_type value, key_type predecessor, key_type successor)
{
	return m_bottom.insert_or_update(group, gpu::make_pair<key_type, Node>(key, { value, predecessor, successor }), [](auto& lhs, auto&& rhs) {
		lhs.value = std::move(rhs.value);
	});
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ bool ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::has_data() const
{
	return m_head != INVALID_PREDECESSOR() || m_tail != INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::key_type ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::INVALID_PREDECESSOR() const
{
	return gpu::numeric_limits<key_type>::max();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::key_type ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::INVALID_SUCCESSOR() const
{
	return gpu::numeric_limits<key_type>::min();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::insert_after(threads group, key_type key, Map_iterator to_insert)
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::insert_before(threads group, key_type key, Map_iterator to_insert)
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::insert_between(threads group, key_type predecessor_it_key, key_type successor_it_key, Map_iterator to_insert)
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::Map_iterator ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::spinlock_for_value(threads group, const key_type& key)
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void ConcurrentXFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::post_condition(threads group)
{
#ifdef GPU_XFASTTRIE_DEBUG
	if (m_head != INVALID_PREDECESSOR())
	{
		ENSURE(m_head == m_tail || m_tail != INVALID_SUCCESSOR());
		auto buffer_start = &(*m_bottom.end()) - m_bottom.capacity();
		auto head = m_bottom.find(group, m_head);
		auto tail = m_bottom.find(group, m_tail);
		ENSURE(&(*head) >= buffer_start && &(*head) < &(*m_bottom.end()));
		ENSURE(&(*tail) >= buffer_start && &(*tail) < &(*m_bottom.end()));
	}
	else
		ENSURE(m_tail == INVALID_SUCCESSOR());

	/*if (has_data())
	{
		m_bottom.for_each([this, &group](const auto& it) {
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
	}*/
#endif // GPU_XFASTTRIE_DEBUG
}
