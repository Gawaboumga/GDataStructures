#include "xfasttrie-k-parallel.cuh"

#include "algorithms/find.cuh"
#include "containers/array.cuh"
#include "utility/print.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::iterator XFastTrieKParallel<Key, Value, UNIVERSE>::begin()
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::const_iterator XFastTrieKParallel<Key, Value, UNIVERSE>::begin() const
{
	return m_bottom.begin();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::iterator XFastTrieKParallel<Key, Value, UNIVERSE>::end()
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::const_iterator XFastTrieKParallel<Key, Value, UNIVERSE>::end() const
{
	return m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ XFastTrieKParallel<Key, Value, UNIVERSE>::XFastTrieKParallel(threads group, gpu::default_allocator& allocator)
{
	unsigned int max_allocate = 1024;
	for (int rank = 0; rank != SUBRANK; ++rank)
	{
		unsigned int preallocate = rank < 10 ? 1 << (rank + 2) : max_allocate;
		m_maps[rank] = Keyset{ group, allocator, preallocate };
	}
	m_bottom = Map{ group, allocator, max_allocate };

	m_head = m_bottom.end();
	m_tail = m_bottom.end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void XFastTrieKParallel<Key, Value, UNIVERSE>::clear(threads group)
{
	for (int rank = 0; rank != SUBRANK; ++rank)
		m_maps[rank].clear(group);

	m_bottom.clear(group);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::iterator XFastTrieKParallel<Key, Value, UNIVERSE>::find(threads group, key_type key)
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::const_iterator XFastTrieKParallel<Key, Value, UNIVERSE>::find(threads group, key_type key) const
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::iterator XFastTrieKParallel<Key, Value, UNIVERSE>::insert(threads group, key_type key, value_type value)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	auto u = binary_search(group, key);

	size_type loop = u.second;
	while (loop != SUBRANK)
	{
		key_type bits = extract_i_upper_bits(key, loop);
		auto found_it = m_maps[loop].find(group, bits);
		auto end_it = m_maps[loop].end();
		if (found_it == end_it)
		{
			m_maps[loop].insert(group, gpu::make_pair<key_type, key_type>(bits, key));
		}
		++loop;
	}

	if (!m_head && !m_tail)
	{
		auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, m_bottom.end(), m_bottom.end() }));
		m_head = current_it;
		m_tail = current_it;
		post_condition(group);
		return current_it;
	}
	else if (u.first == m_maps[u.second].end())
	{
		if (key < m_head->first)
		{
			auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, m_bottom.end(), m_head }));
			m_head->second.predecessor = current_it;
			m_head = current_it;
			post_condition(group);
			return current_it;
		}
		else
		{
			auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, m_tail, m_bottom.end() }));
			m_tail->second.successor = current_it;
			m_tail = current_it;
			post_condition(group);
			return current_it;
		}
	}
	else
	{
		const auto& predecessor_key = u.first->second;
		auto predecessor_it = m_bottom.find(group, predecessor_key);
		if (key > predecessor_key)
		{
			if (key > m_tail->first)
			{
				auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, m_tail, m_bottom.end() }));
				m_tail->second.successor = current_it;
				m_tail = current_it;
				post_condition(group);
				return current_it;
			}
			else
			{
				auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, predecessor_it, predecessor_it->second.successor }));
				current_it->second.successor->second.predecessor = current_it;
				predecessor_it->second.successor = current_it;
				post_condition(group);
				return current_it;
			}
		}
		else
		{
			while (predecessor_it->second.predecessor && predecessor_it->second.predecessor->first > key)
				predecessor_it = predecessor_it->second.predecessor; // In case of 6 7 and we add 5

			auto current_it = m_bottom.insert(group, gpu::make_pair<key_type, Node>(key, { value, predecessor_it->second.predecessor, predecessor_it }));
			if (predecessor_it->second.predecessor)
				current_it->second.predecessor->second.successor = current_it;
			predecessor_it->second.predecessor = current_it;

			if (m_head == predecessor_it)
				m_head = current_it;

			post_condition(group);
			return current_it;
		}
	}
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::iterator XFastTrieKParallel<Key, Value, UNIVERSE>::predecessor(threads group, key_type key)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	if (!m_head)
		return end();

	if (key < m_head->first)
		return end();
	if (key >= m_tail->first)
		return m_tail;

	auto u = binary_search(group, key);

	return get_predecessor(group, key, u.first);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::size_type XFastTrieKParallel<Key, Value, UNIVERSE>::size() const
{
	return 1 << RANK;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::iterator XFastTrieKParallel<Key, Value, UNIVERSE>::successor(threads group, key_type key)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	if (!m_tail)
		return end();

	if (key > m_tail->first)
		return end();
	if (key <= m_head->first)
		return m_head;

	auto u = binary_search(group, key);

	return get_successor(group, key, u.first);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ auto XFastTrieKParallel<Key, Value, UNIVERSE>::binary_search(threads group, key_type key) -> gpu::pair<keyset_iterator, size_type>
{
	group.sync();

	__shared__ gpu::pair<keyset_iterator, size_type> result;
	__shared__ gpu::array<keyset_iterator, SUBRANK> results_iterator;
	auto thid = group.thread_rank();
	auto warp_id = thid / 32;
	results_iterator[warp_id] = m_maps[0].end();

	/*if (group.thread_rank() == 0)
		debug();
	group.sync();*/

	cooperative_groups::thread_block_tile<32> warp = cooperative_groups::tiled_partition<32>(group);

	if (thid < 32 * SUBRANK)
	{
		int offset = 0;
		do
		{
			key_type bits = extract_i_upper_bits(key, warp_id + offset);
			auto found_it = m_maps[warp_id + offset].find(warp, bits);
			warp.sync();
			if (warp.thread_rank() == 0)
				results_iterator[warp_id + offset] = found_it;
			/*group.sync();
			if (warp.thread_rank() == 0)
				printf("%d %d %d %d %p %p ", thid, warp_id, offset, group.size(), &(*results_iterator[0]), &(*results_iterator[1]));*/
			offset += group.size() / 32;
		} while (warp_id + offset < SUBRANK);
		group.sync();

		if (warp.thread_rank() == 0)
			printf("%d %d %d %p %p ", thid, warp_id, offset, &(*results_iterator[0]), &(*results_iterator[1]));

		if (warp_id == 0)
		{
			// We try to find out the last place such that T T T F <- We want the third T.
			// So we search the first F
			auto res = gpu::find_if(warp, results_iterator.begin(), results_iterator.end(), [this, thid](const keyset_iterator& it) {
				return it == this->m_maps[thid % 32].end();
			}); // We need to find the first invalid

			//printf("%d %p ", group.thread_rank(), res);

			// If there are none -> F F F F then we return head.
			if (res == results_iterator.begin() && thid == 0)
				result = { *results_iterator.begin(), 0 };
			// Otherwise we return the previous one, as in T F, we get 1, we need to return 0
			else if (thid == 0)
			{
				//printf("%p %p ; ", results_iterator.begin(), res);
				--res;
				assert(res >= results_iterator.begin());
				result = std::move(gpu::make_pair<keyset_iterator, size_type>(*res, res - results_iterator.begin()));
			}
		}
	}
	group.sync();

	gpu::pair<keyset_iterator, size_type> copy = result;
	return copy;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void XFastTrieKParallel<Key, Value, UNIVERSE>::debug() const
{
	using gpu::print;
	for (int i = 0; i != SUBRANK; ++i)
	{
		print("HashMap (", i, "): ");
		const auto& map = m_maps[i];
		for (auto it = map.begin(); it != map.end(); ++it)
		{
			print("{", it->first, "|", it->second, "}");
		}
		print("\n");
	}

	print("Bottom: ");
	for (auto it = m_bottom.begin(); it != m_bottom.end(); ++it)
	{
		auto& value = it->second;
		if (value.predecessor && value.successor)
			print("{", it->first, "|", value.value, "=>[", value.predecessor->first, "|", value.successor->first, "]}");
		else if (value.predecessor)
			print("{", it->first, "|", value.value, "=>[", value.predecessor->first, "|#]}");
		else if (value.successor)
			print("{", it->first, "|", value.value, "=>[#|", value.successor->first, "]}");
		else
			print("{", it->first, "|", value.value, "=>[#|#]}");
	}

	print("\nHead/Tail: ");
	if (m_head)
		print("Head: ", m_head->second.value, " ");
	if (m_tail)
		print("Tail: ", m_tail->second.value, " ");
	print("\n");
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::key_type XFastTrieKParallel<Key, Value, UNIVERSE>::extract_i_upper_bits(key_type key, int number_of_bits) const
{
	key_type result = key >> (SUBRANK - number_of_bits);
	return result;
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::iterator XFastTrieKParallel<Key, Value, UNIVERSE>::get_predecessor(threads group, key_type key, keyset_iterator u)
{
	auto& value = u->second;
	if (key < value)
	{
		auto it = m_bottom.find(group, value);
		if (it->second.predecessor)
			return it->second.predecessor;
		else
			return end();
	}
	return m_bottom.find(group, value);
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ typename XFastTrieKParallel<Key, Value, UNIVERSE>::iterator XFastTrieKParallel<Key, Value, UNIVERSE>::get_successor(threads group, key_type key, keyset_iterator u)
{
	auto& value = u->second;
	auto it = m_bottom.find(group, value);
	if (it->first > key)
		return it;
	if (it->second.successor)
		return it->second.successor;
	else
		return end();
}

template <typename Key, typename Value, std::size_t UNIVERSE>
__device__ void XFastTrieKParallel<Key, Value, UNIVERSE>::post_condition(threads group)
{
	for (int i = 0; i != SUBRANK; ++i)
	{
		auto& map = m_maps[i];
		for (auto it = map.begin(); it != map.end(); ++it)
		{
			if (i + 1 < SUBRANK)
			{
				auto left_child = m_maps[i + 1].find(group, it->first << 1);
				auto right_child = m_maps[i + 1].find(group, (it->first << 1) | 1);
				assert(left_child != m_maps[i + 1].end() || right_child != m_maps[i + 1].end());
			}
		}
	}

	if (m_head)
		assert(m_tail);
	else
		assert(!m_tail);

	if (m_head)
	{
		for (auto it = m_bottom.begin(); it != m_bottom.end(); ++it)
		{
			if (it->second.predecessor)
			{
				if (it != m_head)
					assert(it->first > it->second.predecessor->first);
				assert(it->second.predecessor->second.successor == it);
			}

			if (it->second.successor)
			{
				if (it != m_tail)
					assert(it->first < it->second.successor->first);
				assert(it->second.successor->second.predecessor == it);
			}
		}
	}
	group.sync();
}
