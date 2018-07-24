#include "concurrent-xfasttrie-binary.cuh"

#include "utility/limits.cuh"
#include "utility/print.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::begin()
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::const_iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::begin() const
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::end()
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::const_iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::end() const
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::ConcurrentXFastTrieBinary(block_threads block, allocator_type& allocator, unsigned int expected_number_of_elements)
{
	unsigned int power_of_two;
	if (expected_number_of_elements == 0)
		power_of_two = 10u;
	else
		power_of_two = __ffs(expected_number_of_elements) - 1u;

	for (int rank = 0; rank != SUBRANK; ++rank)
	{
		unsigned int preallocate = rank < power_of_two ? 1u << (rank + 2u) : 1u << power_of_two;
		new (&m_maps[rank]) Keyset{ block, allocator, preallocate };
	}
	new (&m_bottom) Map{ block, allocator, 1u << power_of_two };

	if (block.thread_rank() == 0)
	{
		m_head = INVALID_PREDECESSOR();
		m_tail = INVALID_SUCCESSOR();
	}
	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::ConcurrentXFastTrieBinary(threads group, allocator_type& allocator, unsigned int expected_number_of_elements)
{
	unsigned int max_allocate = 1 << 19u;
	for (int rank = 0; rank != SUBRANK; ++rank)
	{
		unsigned int preallocate = rank < 17u ? 1u << (rank + 2u) : max_allocate;
		m_maps[rank] = Keyset{ group, allocator, preallocate };
	}
	m_bottom = Map{ group, allocator, max_allocate };

	m_head = INVALID_PREDECESSOR();
	m_tail = INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::clear(block_threads block)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);

	if (block.thread_rank() < 32)
		clear(tile32);

	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::clear(threads group)
{
	for (int rank = 0; rank != SUBRANK; ++rank)
		m_maps[rank].clear(group);

	m_bottom.clear(group);

	m_head = INVALID_PREDECESSOR();
	m_tail = INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::find(key_type key)
{
	return m_bottom.find(key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::find(threads group, key_type key)
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::const_iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::find(threads group, key_type key) const
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::insert(threads group, key_type key, value_type value)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	auto u = binary_search(group, key);
	auto old_it = m_maps[u.second].end(); // since when we reallocate data, pointer will change

	size_type loop = u.second;
	while (loop != SUBRANK)
	{
		key_type bits = extract_i_upper_bits(key, loop);
		auto found_it = m_maps[loop].find(group, bits);
		auto end_it = m_maps[loop].end();
		if (found_it == end_it)
		{
			m_maps[loop].insert_or_update(gpu::make_pair<key_type, Children>(bits, { key, key }),
				[](Children& lhs, Children&& rhs) {
				lhs.minimal_left.min(rhs.minimal_left);
				lhs.maximal_right.min(rhs.maximal_right);
			});
		}
		++loop;
	}

	walk_up(group, key, u.second);

	auto old_minimal = u.first->second.minimal_left;
	auto old_maximal = u.first->second.maximal_right;

	if (!has_data())
	{
		auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, INVALID_PREDECESSOR(), INVALID_SUCCESSOR() }));
		key_type old_head = m_head.min(group, key);
		if (old_head != INVALID_PREDECESSOR())
			return insert_around(group, old_head, current_it);
		key_type old_tail = m_tail.max(group, key);
		if (old_tail != INVALID_SUCCESSOR())
			return insert_around(group, old_tail, current_it);
		post_condition(group);
		return current_it;
	}
	else
	{
		if (key < m_head)
		{
			auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, INVALID_PREDECESSOR(), m_head }));
			return insert_around(group, m_head, current_it);
		}
		else if (key > m_tail)
		{
			auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, m_tail, INVALID_SUCCESSOR() }));
			return insert_around(group, m_tail, current_it);
		}
		else
		{
			if (key < old_minimal)
			{
				Map_iterator successor_it = spinlock_for_value(group, old_minimal);
				auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, successor_it->second.predecessor, old_minimal }));
				return insert_around(group, old_minimal, current_it);
			}
			else if (key > old_maximal)
			{
				Map_iterator predecessor_it = spinlock_for_value(group, old_maximal);
				auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, old_maximal, predecessor_it->second.successor }));
				return insert_around(group, old_maximal, current_it);
			}
			else
			{
				/*if (group.thread_rank() == 3)
				{
				gpu::print(" > ", key, " o ", old_minimal, " p ", old_maximal, "\n");
				}
				group.sync();
				if (group.thread_rank() == 0)
				{
				gpu::print(" > ", key, " o ", old_minimal, " p ", old_maximal, "\n");
				}
				group.sync();

				ENSURE(false);*/
				return end();
			}
		}
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::size_type ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::maximal_size() const
{
	return 1 << RANK;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::predecessor(threads group, key_type key)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	if (!has_data())
		return end();

	if (key < m_head)
		return end();
	if (key >= m_tail)
		return m_bottom.find(group, m_tail);

	auto u = binary_search(group, key);
	if (key < u.first->second.minimal_left)
	{
		auto predecessor_it = m_bottom.find(group, u.first->second.minimal_left);
		return m_bottom.find(group, predecessor_it->second.predecessor);
	}
	else
		return m_bottom.find(group, u.first->second.maximal_right);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::size_type ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::size() const
{
	return m_bottom.size();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::successor(threads group, key_type key)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	if (!has_data())
		return end();

	if (key > m_tail)
		return end();
	if (key <= m_head)
		return m_bottom.find(group, m_head);

	auto u = binary_search(group, key);
	if (key < u.first->second.minimal_left)
		return m_bottom.find(group, u.first->second.minimal_left);
	else
	{
		auto predecessor_it = m_bottom.find(group, u.first->second.minimal_left);
		return m_bottom.find(group, predecessor_it->second.successor);
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::debug() const
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
__device__ auto ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::binary_search(threads group, key_type key) -> gpu::pair<keyset_iterator, size_type>
{
	int low = 0;
	int high = SUBRANK - 1;
	keyset_iterator u = m_maps[low].end();
	while (low <= high)
	{
		int mid = (low + high) / 2;
		key_type bits = extract_i_upper_bits(key, mid);
		auto v = m_maps[mid].find(group, bits);
		if (v == m_maps[mid].end())
		{
			high = mid - 1;
		}
		else
		{
			u = v;
			low = mid + 1;
		}
	}

	return { u, low };
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::key_type ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::extract_i_upper_bits(key_type key, int number_of_bits) const
{
	key_type result = key >> (SUBRANK - number_of_bits);
	return result;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::get_predecessor(threads group, key_type key, keyset_iterator u)
{
	auto& value = u->second;
	return m_bottom.find(group, value.minimal_left);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::get_successor(threads group, key_type key, keyset_iterator u)
{
	auto& value = u->second;
	return m_bottom.find(group, value.maximal_right);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ bool ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::has_data() const
{
	return m_head != INVALID_PREDECESSOR() && m_tail != INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::key_type ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::INVALID_PREDECESSOR() const
{
	return gpu::numeric_limits<key_type>::max();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::key_type ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::INVALID_SUCCESSOR() const
{
	return gpu::numeric_limits<key_type>::min();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::insert_around(threads group, const key_type& key, Map_iterator to_insert)
{
	while (true)
	{
		Map_iterator predecessor = m_bottom.end();
		Map_iterator successor = m_bottom.end();

		predecessor = spinlock_for_value(group, key);
		while (predecessor->second.predecessor != INVALID_PREDECESSOR() && predecessor->second.predecessor > to_insert->first)
		{
			successor = predecessor;
			predecessor = spinlock_for_value(group, predecessor->second.predecessor);
		}
		while (predecessor->second.successor != INVALID_SUCCESSOR() && predecessor->second.successor < to_insert->first)
		{
			successor = spinlock_for_value(group, predecessor->second.successor);
			predecessor = successor;
		}

		if (predecessor->second.predecessor == INVALID_PREDECESSOR())
		{
			if (predecessor->first < to_insert->first)
			{
				to_insert->second.predecessor = predecessor->first;
				to_insert->second.successor = predecessor->second.successor;

				if (predecessor->second.successor.compare_and_swap(group, INVALID_SUCCESSOR(), to_insert->first) == INVALID_SUCCESSOR())
				{
					m_tail.max(group, to_insert->first);
					return to_insert;
				}
			}
			else
			{
				to_insert->second.predecessor = predecessor->second.predecessor;
				to_insert->second.successor = predecessor->first;

				if (predecessor->second.predecessor.compare_and_swap(group, INVALID_PREDECESSOR(), to_insert->first) == INVALID_PREDECESSOR())
				{
					m_head.min(group, to_insert->first);
					return to_insert;
				}
			}
		}
		else
		{
			to_insert->second.predecessor = predecessor->first;
			to_insert->second.successor = predecessor->second.successor;

			if (predecessor->second.successor == INVALID_SUCCESSOR())
			{
				if (predecessor->second.successor.compare_and_swap(group, INVALID_SUCCESSOR(), to_insert->first) == INVALID_SUCCESSOR())
				{
					m_tail.max(group, to_insert->first);
					return to_insert;
				}
			}

			successor = spinlock_for_value(group, predecessor->second.successor);
			key_type old_successor = predecessor->second.successor;
			key_type old_predecessor = successor->second.predecessor;
			if (predecessor->second.successor.compare_and_swap(group, old_successor, to_insert->first) == old_successor)
			{
				if (successor->second.predecessor.compare_and_swap(group, old_predecessor, to_insert->first) == old_predecessor)
				{
					return to_insert;
				}
			}
		}
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::Map_iterator ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::spinlock_for_value(threads group, const key_type& key)
{
	Map_iterator result_it;
	do
	{
		result_it = m_bottom.find(group, key);
	} while (result_it == m_bottom.end());
	return result_it;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::walk_up(threads group, key_type key, size_type from)
{
	while (from != 0)
	{
		--from;
		key_type bits = extract_i_upper_bits(key, from);
		auto it = m_maps[from].find(group, bits);
		it->second.minimal_left.min(key);
		it->second.maximal_right.max(key);
		if (it->second.minimal_left != key && it->second.maximal_right != key)
			return;
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void ConcurrentXFastTrieBinary<Key, Value, UNIVERSE>::post_condition(threads group)
{
#ifdef GPU_XFASTTRIE_DEBUG
	for (int i = 0; i != SUBRANK; ++i)
	{
		auto& map = m_maps[i];
		for (auto it = map.begin(); it != map.end(); ++it)
		{
			if (i + 1 < SUBRANK)
			{
				auto left_child = m_maps[i + 1].find(group, it->first << 1);
				auto right_child = m_maps[i + 1].find(group, (it->first << 1) | 1);
				ENSURE(left_child != m_maps[i + 1].end() || right_child != m_maps[i + 1].end());
			}
		}
	}

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

	if (has_data())
	{
		for (auto it = m_bottom.begin(); it != m_bottom.end(); ++it)
		{
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
		}
	}
#endif // GPU_XFASTTRIE_DEBUG
}
