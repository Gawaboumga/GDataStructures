#include "xfasttrie-binary.cuh"

#include "utility/limits.cuh"
#include "utility/print.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::iterator XFastTrieBinary<Key, Value, UNIVERSE>::begin()
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::const_iterator XFastTrieBinary<Key, Value, UNIVERSE>::begin() const
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::iterator XFastTrieBinary<Key, Value, UNIVERSE>::end()
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::const_iterator XFastTrieBinary<Key, Value, UNIVERSE>::end() const
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ XFastTrieBinary<Key, Value, UNIVERSE>::XFastTrieBinary(block_threads block, gpu::default_allocator& allocator)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);
	auto thid = block.thread_rank();

	if (thid < tile32.size())
	{
		unsigned int max_allocate = 1 << 18u;
		for (int rank = 0; rank != SUBRANK; ++rank)
		{
			unsigned int preallocate = rank < 17u ? 1u << (rank + 2u) : max_allocate;
			m_maps[rank] = Keyset{ tile32, allocator, preallocate };
		}
		m_bottom = Map{ tile32, allocator, max_allocate };

		m_head = INVALID_PREDECESSOR();
		m_tail = INVALID_SUCCESSOR();
	}
	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ XFastTrieBinary<Key, Value, UNIVERSE>::XFastTrieBinary(threads group, gpu::default_allocator& allocator)
{
	unsigned int max_allocate = 1 << 18u;
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
__device__ void XFastTrieBinary<Key, Value, UNIVERSE>::clear(block_threads block)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);

	if (block.thread_rank() < 32)
		clear(tile32);

	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void XFastTrieBinary<Key, Value, UNIVERSE>::clear(threads group)
{
	for (int rank = 0; rank != SUBRANK; ++rank)
		m_maps[rank].clear(group);

	m_bottom.clear(group);

	m_head = INVALID_PREDECESSOR();
	m_tail = INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::iterator XFastTrieBinary<Key, Value, UNIVERSE>::find(key_type key)
{
	return m_bottom.find(key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::iterator XFastTrieBinary<Key, Value, UNIVERSE>::find(threads group, key_type key)
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::const_iterator XFastTrieBinary<Key, Value, UNIVERSE>::find(threads group, key_type key) const
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::iterator XFastTrieBinary<Key, Value, UNIVERSE>::insert(threads group, key_type key, value_type value)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	auto u = binary_search(group, key);
	auto old_it = m_maps[u.second].end(); // since when we reallocate data, pointer will change
	auto old_minimal = u.first->second.minimal_left;
	auto old_maximal = u.first->second.maximal_right;

	size_type loop = u.second;
	while (loop != SUBRANK)
	{
		key_type bits = extract_i_upper_bits(key, loop);
		auto found_it = m_maps[loop].find(group, bits);
		auto end_it = m_maps[loop].end();
		if (found_it == end_it)
		{
			m_maps[loop].insert(group, gpu::make_pair<key_type, Children>(bits, { key, key }));
		}
		++loop;
	}

	walk_up(group, key, u.second);

	if (!has_data())
	{
		auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, INVALID_PREDECESSOR(), INVALID_SUCCESSOR() }));
		m_head = key;
		m_tail = key;
		post_condition(group);
		return current_it;
	}
	else
	{
		if (key < m_head)
		{
			auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, INVALID_PREDECESSOR(), m_head }));
			auto it = m_bottom.find(group, m_head);
			it->second.predecessor = key;
			m_head = key;
			post_condition(group);
			return current_it;
		}
		else if (key > m_tail)
		{
			auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, m_tail, INVALID_SUCCESSOR() }));
			auto it = m_bottom.find(group, m_tail);
			it->second.successor = key;
			m_tail = key;
			post_condition(group);
			return current_it;
		}
		else
		{
			if (key < old_minimal)
			{
				auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value,{}, old_minimal }));
				auto successor_it = m_bottom.find(group, old_minimal);
				current_it->second.predecessor = successor_it->second.predecessor;
				auto predecessor_it = m_bottom.find(group, successor_it->second.predecessor);

				if (group.thread_rank() == 0)
				{
					successor_it->second.predecessor = key;
					predecessor_it->second.successor = key;
				}
				group.sync();
				post_condition(group);
				return current_it;
			}
			else if (key > old_maximal)
			{
				auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, old_maximal,{} }));
				auto predecessor_it = m_bottom.find(group, old_maximal);
				current_it->second.successor = predecessor_it->second.successor;
				auto successor_it = m_bottom.find(group, predecessor_it->second.successor);

				if (group.thread_rank() == 0)
				{
					successor_it->second.predecessor = key;
					predecessor_it->second.successor = key;
				}
				group.sync();
				post_condition(group);
				return current_it;
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
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::size_type XFastTrieBinary<Key, Value, UNIVERSE>::maximal_size() const
{
	return 1 << RANK;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::iterator XFastTrieBinary<Key, Value, UNIVERSE>::predecessor(threads group, key_type key)
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
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::size_type XFastTrieBinary<Key, Value, UNIVERSE>::size() const
{
	return m_bottom.size();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::iterator XFastTrieBinary<Key, Value, UNIVERSE>::successor(threads group, key_type key)
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
__device__ void XFastTrieBinary<Key, Value, UNIVERSE>::debug() const
{
	using gpu::print;
	for (int i = 0; i != SUBRANK; ++i)
	{
		print("HashMap (", i, "): ");
		const auto& map = m_maps[i];
		for (auto it = map.begin(); it != map.end(); ++it)
		{
			print("{", it->first, "|", it->second.minimal_left, ", ", it->second.maximal_right, "}");
		}
		print("\n");
	}

	print("Bottom: ");
	for (auto it = m_bottom.begin(); it != m_bottom.end(); ++it)
	{
		auto& value = it->second;
		if (value.predecessor != INVALID_PREDECESSOR() && value.successor != INVALID_SUCCESSOR())
			print("{", it->first, "|", value.value, "=>[", value.predecessor, "|", value.successor, "]}");
		else if (value.predecessor != INVALID_PREDECESSOR())
			print("{", it->first, "|", value.value, "=>[", value.predecessor, "|#]}");
		else if (value.successor != INVALID_SUCCESSOR())
			print("{", it->first, "|", value.value, "=>[#|", value.successor, "]}");
		else
			print("{", it->first, "|", value.value, "=>[#|#]}");
	}

	print("\nHead/Tail: ");
	if (m_head != INVALID_PREDECESSOR())
		print("Head: ", m_head, " ");
	if (m_tail != INVALID_SUCCESSOR())
		print("Tail: ", m_tail, " ");
	print("\n");
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ auto XFastTrieBinary<Key, Value, UNIVERSE>::binary_search(threads group, key_type key) -> gpu::pair<keyset_iterator, size_type>
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
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::key_type XFastTrieBinary<Key, Value, UNIVERSE>::extract_i_upper_bits(key_type key, int number_of_bits) const
{
	key_type result = key >> (SUBRANK - number_of_bits);
	return result;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::iterator XFastTrieBinary<Key, Value, UNIVERSE>::get_predecessor(threads group, key_type key, keyset_iterator u)
{
	auto& value = u->second;
	return m_bottom.find(group, value.minimal_left);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::iterator XFastTrieBinary<Key, Value, UNIVERSE>::get_successor(threads group, key_type key, keyset_iterator u)
{
	auto& value = u->second;
	return m_bottom.find(group, value.maximal_right);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ bool XFastTrieBinary<Key, Value, UNIVERSE>::has_data() const
{
	return m_head != INVALID_PREDECESSOR() && m_tail != INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::key_type XFastTrieBinary<Key, Value, UNIVERSE>::INVALID_PREDECESSOR() const
{
	return gpu::numeric_limits<key_type>::max();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieBinary<Key, Value, UNIVERSE>::key_type XFastTrieBinary<Key, Value, UNIVERSE>::INVALID_SUCCESSOR() const
{
	return gpu::numeric_limits<key_type>::min();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void XFastTrieBinary<Key, Value, UNIVERSE>::walk_up(threads group, key_type key, size_type from)
{
	while (from != 0)
	{
		--from;
		key_type bits = extract_i_upper_bits(key, from);
		auto it = m_maps[from].find(group, bits);
		it->second.minimal_left = min(it->second.minimal_left, key);
		it->second.maximal_right = max(it->second.maximal_right, key);
		if (it->second.minimal_left != key && it->second.maximal_right != key)
			return;
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void XFastTrieBinary<Key, Value, UNIVERSE>::post_condition(threads group)
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
