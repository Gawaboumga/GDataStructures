#include "xfasttrie-group-parallel.cuh"

#include "algorithms/find.cuh"
#include "utility/limits.cuh"
#include "utility/print.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::begin()
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::const_iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::begin() const
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::end()
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::const_iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::end() const
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::XFastTrieGroupParallel(block_threads block, gpu::default_allocator& allocator)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);
	auto thid = block.thread_rank();

	if (thid < tile32.size())
	{
		unsigned int max_allocate = 1 << 18u;
		for (int rank = 0; rank != NUMBER_OF_KEYSETS; ++rank)
		{
			unsigned int preallocate = rank * GROUP_SIZE < 17u ? 1u << (rank * GROUP_SIZE + 2u) : max_allocate;
			m_maps[rank] = Keyset{ tile32, allocator, preallocate };
		}
		m_bottom = Map{ tile32, allocator, max_allocate };

		m_head = INVALID_PREDECESSOR();
		m_tail = INVALID_SUCCESSOR();
	}
	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::XFastTrieGroupParallel(threads group, gpu::default_allocator& allocator)
{
	unsigned int max_allocate = 1 << 18u;
	for (int rank = 0; rank != NUMBER_OF_KEYSETS; ++rank)
	{
		unsigned int preallocate = rank * GROUP_SIZE < 17u ? 1u << (rank * GROUP_SIZE + 2u) : max_allocate;
		m_maps[rank] = Keyset{ group, allocator, preallocate };
	}
	m_bottom = Map{ group, allocator, max_allocate };

	m_head = INVALID_PREDECESSOR();
	m_tail = INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::clear(block_threads block)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);

	if (block.thread_rank() < 32)
		clear(tile32);

	block.sync();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::clear(threads group)
{
	for (int rank = 0; rank != NUMBER_OF_KEYSETS; ++rank)
		m_maps[rank].clear(group);

	m_bottom.clear(group);

	m_head = INVALID_PREDECESSOR();
	m_tail = INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::find(key_type key)
{
	return m_bottom.find(key);
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::find(threads group, key_type key)
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::const_iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::find(threads group, key_type key) const
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::insert(threads group, key_type key, mapped_type value)
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::size_type XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::maximal_size() const
{
	return 1 << RANK;
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::predecessor(threads group, key_type key)
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
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::size_type XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::size() const
{
	return m_bottom.size();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::iterator XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::successor(threads group, key_type key)
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
__device__ void XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::debug() const
{
	using gpu::print;
	for (int i = 0; i != NUMBER_OF_KEYSETS; ++i)
	{
		print("HashMap (", i, "): ");
		const auto& map = m_maps[i];
		for (auto it = map.begin(); it != map.end(); ++it)
		{
			print("{", it->first, "|");
			print("0: (", it->second.minimal_left[0], ", ", it->second.maximal_right[0], ")");
			unsigned int group_size = (i == NUMBER_OF_KEYSETS - 1u) ? TAIL_GROUP_SIZE : GROUP_SIZE;
			if (group_size)
			{
				for (unsigned int g = 0u; g != group_size; ++g)
				{
					print(" ", g + 1, ": ");
					unsigned int offset = (1u << (g + 1u)) - 1u;
					for (unsigned int i = 0u; i != 1u << (g + 1u); ++i)
					{
						print("(", it->second.minimal_left[offset + i], ", ", it->second.maximal_right[offset + i], ")");
					}
				}
			}
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ auto XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::binary_search(threads group, key_type key) -> BinarySearchResult
{
	auto thid = group.thread_rank();
	bool has_no_value = true;
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
		has_no_value = found_it == m_maps[thid].end();

		if (!has_no_value)
			index = get_highest_index(tail_bits, found_it->second, thid);
	}
	else if (thid == NUMBER_OF_KEYSETS)
	{
		bottom_it = m_bottom.find(key);
	}

	unsigned int matching_bits = group.ballot(has_no_value);
	if (matching_bits)
	{
		// The idea is that we want the min, max of the lowest node in the tree where there is data
		unsigned int insert_update_separation = __ffs(matching_bits) - 2u;
		index = group.shfl(index, insert_update_separation);
		found_it.shfl(group, insert_update_separation);
	}
	bottom_it.shfl(group, NUMBER_OF_KEYSETS);
	return { found_it, bottom_it, index };
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::key_type XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::extract_i_upper_bits(key_type key, int number_of_bits) const
{
	key_type result = key >> (SUBRANK - number_of_bits);
	return result;
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ auto XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::find_or_update(threads group, key_type key) -> gpu::pair<key_type, key_type>
{
	auto thid = group.thread_rank();

	{
		bool resize = false;
		if (thid < NUMBER_OF_KEYSETS)
		{
			resize = m_maps[thid].pending_resize();
		}
		group.sync();
		unsigned int matching_bits = group.ballot(resize);
		while (matching_bits)
		{
			unsigned int rank = __ffs(matching_bits) - 1u;
			m_maps[rank].resize(group);
			matching_bits ^= (1 << rank); // Bits are put in little endian
		}
	}

	key_type old_minimum = INVALID_PREDECESSOR();
	key_type old_maximum = INVALID_SUCCESSOR();
	bool has_no_value = true;
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
		has_no_value = found_it == m_maps[thid].end();
		unsigned int matching_bits = group.ballot(has_no_value);
		unsigned int insert_update_separation = __ffs(matching_bits);

		if (!has_no_value)
		{
			unsigned int i = get_highest_index(tail_bits, found_it->second, thid);
			old_minimum = found_it->second.minimal_left[i];
			old_maximum = found_it->second.maximal_right[i];
		}

		if (thid >= insert_update_separation - 1 && has_no_value) // Not just, should be related to group
		{
			m_maps[thid].insert(gpu::make_pair<key_type, Child>(group_bits, make_children(tail_bits, key, thid)));
		}
		else
		{
			update_key(found_it->second, tail_bits, key, thid);
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

	return { old_minimum, old_maximum };
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ unsigned int XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::get_highest_index(key_type tail_bits, const Child& value, unsigned int thid)
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
__device__ auto XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::make_children(key_type tail_bits, const key_type& key, unsigned int thid) -> Child
{
	Child child;
	for (unsigned int i = 0u; i != (1u << (GROUP + 1u)) - 1u; ++i)
	{
		child.minimal_left[i] = INVALID_PREDECESSOR();
		child.maximal_right[i] = INVALID_SUCCESSOR();
	}

	unsigned int upper_bound = thid == NUMBER_OF_KEYSETS - 1u ? TAIL_GROUP_SIZE : GROUP_SIZE;
	unsigned int offset = (1u << upper_bound) - 1u;
	while (offset)
	{
		unsigned int position = offset + tail_bits;
		child.minimal_left[position] = key;
		child.maximal_right[position] = key;
		offset = offset >> 1u;
		tail_bits = tail_bits >> 1u;
	}
	child.minimal_left[0u] = key;
	child.maximal_right[0u] = key;
	return child;
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::update_key(Child& child, key_type tail_bits, const key_type& key, unsigned int thid)
{
	unsigned int upper_bound = thid == NUMBER_OF_KEYSETS - 1u ? TAIL_GROUP_SIZE : GROUP_SIZE;
	unsigned int offset = (1u << upper_bound) - 1u;
	while (offset)
	{
		unsigned int position = offset + tail_bits;
		child.minimal_left[position] = min(child.minimal_left[position], key);
		child.maximal_right[position] = max(child.maximal_right[position], key);
		offset = offset >> 1u;
		tail_bits = tail_bits >> 1u;
	}
	child.minimal_left[0u] = min(child.minimal_left[0u], key);
	child.maximal_right[0u] = max(child.maximal_right[0u], key);
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ bool XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::has_data() const
{
	return m_head != INVALID_PREDECESSOR() && m_tail != INVALID_SUCCESSOR();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::key_type XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::INVALID_PREDECESSOR() const
{
	return gpu::numeric_limits<key_type>::max();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ typename XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::key_type XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::INVALID_SUCCESSOR() const
{
	return gpu::numeric_limits<key_type>::min();
}

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::walk_up(threads group, key_type key, size_type from)
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

template <typename Key, typename Value, std::size_t UNIVERSE, std::size_t GROUP>
__device__ void XFastTrieGroupParallel<Key, Value, UNIVERSE, GROUP>::post_condition(threads group)
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
