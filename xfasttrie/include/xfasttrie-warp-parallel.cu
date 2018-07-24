#include "xfasttrie-warp-parallel.cuh"

#include "algorithms/find.cuh"
#include "utility/print.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::begin()
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::const_iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::begin() const
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::end()
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::const_iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::end() const
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ XFastTrieWarpParallel<Key, Value, UNIVERSE>::XFastTrieWarpParallel(block_threads block, gpu::default_allocator& allocator)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);
	auto thid = block.thread_rank();

	if (thid < tile32.size())
	{
		//unsigned int max_allocate = 4096u;
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
__device__ XFastTrieWarpParallel<Key, Value, UNIVERSE>::XFastTrieWarpParallel(threads group, gpu::default_allocator& allocator)
{
	//unsigned int max_allocate = 4096u;
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
__device__ void XFastTrieWarpParallel<Key, Value, UNIVERSE>::clear(block_threads block)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);

	if (block.thread_rank() < 32)
		clear(tile32);

	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void XFastTrieWarpParallel<Key, Value, UNIVERSE>::clear(threads group)
{
	for (int rank = 0; rank != SUBRANK; ++rank)
		m_maps[rank].clear(group);

	m_bottom.clear(group);

	m_head = INVALID_PREDECESSOR();
	m_tail = INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::find(key_type key)
{
	return m_bottom.find(key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::find(threads group, key_type key)
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::const_iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::find(threads group, key_type key) const
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::insert(threads group, key_type key, mapped_type value)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	auto old_values = find_or_update(group, key);
	key_type old_minimal = old_values.first;
	key_type old_maximal = old_values.second;

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
				auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, {}, old_minimal }));
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
				auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, old_maximal, {} }));
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
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::size_type XFastTrieWarpParallel<Key, Value, UNIVERSE>::maximal_size() const
{
	return 1 << RANK;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::predecessor(threads group, key_type key)
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
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::size_type XFastTrieWarpParallel<Key, Value, UNIVERSE>::size() const
{
	return m_bottom.size();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::iterator XFastTrieWarpParallel<Key, Value, UNIVERSE>::successor(threads group, key_type key)
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
__device__ void XFastTrieWarpParallel<Key, Value, UNIVERSE>::debug() const
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
__device__ auto XFastTrieWarpParallel<Key, Value, UNIVERSE>::binary_search(threads group, key_type key) -> gpu::pair<keyset_iterator, Map_iterator>
{
	auto thid = group.thread_rank();
	keyset_iterator found_it;
	Map_iterator bottom_it;

	unsigned int warp_offset = 0u;
	do
	{
		bool has_no_value = true;
		if (warp_offset + thid < SUBRANK)
		{
			key_type bits = extract_i_upper_bits(key, warp_offset + thid);
			found_it = m_maps[warp_offset + thid].find(bits);

			// We try to find out the last place such that T T T F <- We want the third T.
			has_no_value = found_it == m_maps[warp_offset + thid].end();
		}
		else if (warp_offset + thid == SUBRANK)
		{
			bottom_it = m_bottom.find(key);
		}

		unsigned int matching_bits = group.ballot(has_no_value);
		if (matching_bits)
		{
			// The idea is that we want the min, max of the lowest node in the tree where there is data
			unsigned int insert_update_separation = __ffs(matching_bits) - 2u;
			found_it.shfl(group, insert_update_separation);
			break;
		}
		warp_offset += group.size();
	} while (warp_offset < SUBRANK);

	bottom_it.shfl(group, SUBRANK % 32);
	return { found_it, bottom_it };
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::key_type XFastTrieWarpParallel<Key, Value, UNIVERSE>::extract_i_upper_bits(key_type key, int number_of_bits) const
{
	key_type result = key >> (SUBRANK - number_of_bits);
	return result;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ auto XFastTrieWarpParallel<Key, Value, UNIVERSE>::find_or_update(threads group, key_type key) -> gpu::pair<key_type, key_type>
{
	auto thid = group.thread_rank();

	{
		unsigned int warp_offset = 0u;
		do
		{
			bool resize = false;
			if (warp_offset + thid < SUBRANK)
			{
				resize = m_maps[warp_offset + thid].pending_resize();
			}
			group.sync();
			unsigned int matching_bits = group.ballot(resize);
			while (matching_bits)
			{
				unsigned int rank = __ffs(matching_bits) - 1u;
				m_maps[warp_offset + rank].resize(group);
				matching_bits ^= (1 << rank); // Bits are put in little endian
			}
			warp_offset += group.size();
		} while (warp_offset < SUBRANK);
	}

	key_type old_minimum = INVALID_PREDECESSOR();
	key_type old_maximum = INVALID_SUCCESSOR();

	{
		unsigned int warp_offset = 0u;
		do
		{
			bool has_no_value = true;
			if (warp_offset + thid < SUBRANK)
			{
				key_type bits = extract_i_upper_bits(key, warp_offset + thid);
				keyset_iterator found_it = m_maps[warp_offset + thid].find(bits);

				// We try to find out the last place such that T T T F <- We want the third T.
				has_no_value = found_it == m_maps[warp_offset + thid].end();
				unsigned int matching_bits = group.ballot(has_no_value);
				unsigned int insert_update_separation = __ffs(matching_bits);

				if (!has_no_value)
				{
					old_minimum = found_it->second.minimal_left;
					old_maximum = found_it->second.maximal_right;
				}

				if (thid >= insert_update_separation - 1 && has_no_value) // Not just, should be related to group
				{
					m_maps[warp_offset + thid].insert(gpu::make_pair<key_type, Children>(bits, { key, key }));
				}
				else
				{
					found_it->second.minimal_left = min(found_it->second.minimal_left, key);
					found_it->second.maximal_right = max(found_it->second.maximal_right, key);
				}
			}

			group.sync();

			unsigned int matching_bits = group.ballot(has_no_value);
			if (matching_bits)
			{
				// The idea is that we want the min, max of the lowest node in the tree where there is data
				unsigned int insert_update_separation = __ffs(matching_bits) - 2u;
				old_minimum = group.shfl(old_minimum, insert_update_separation);
				old_maximum = group.shfl(old_maximum, insert_update_separation);
			}

			warp_offset += group.size();
		} while (warp_offset < SUBRANK);
	}

	return { old_minimum, old_maximum };
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ bool XFastTrieWarpParallel<Key, Value, UNIVERSE>::has_data() const
{
	return m_head != INVALID_PREDECESSOR() && m_tail != INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::key_type XFastTrieWarpParallel<Key, Value, UNIVERSE>::INVALID_PREDECESSOR() const
{
	return key_type{ -1 };
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieWarpParallel<Key, Value, UNIVERSE>::key_type XFastTrieWarpParallel<Key, Value, UNIVERSE>::INVALID_SUCCESSOR() const
{
	return key_type{ 0 };
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void XFastTrieWarpParallel<Key, Value, UNIVERSE>::walk_up(threads group, key_type key, size_type from)
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
__device__ void XFastTrieWarpParallel<Key, Value, UNIVERSE>::post_condition(threads group)
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
