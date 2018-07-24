#include "vEB.cuh"

#include "utility/print.cuh"

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::end()
{
	return m_bottom.end();
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::const_iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::end() const
{
	return m_bottom.end();
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::vEB(block_threads block, allocator_type& allocator, unsigned int expected_number_of_elements) :
	m_allocator(&allocator)
{
	unsigned int power_of_two;
	if (expected_number_of_elements == 0)
		power_of_two = 10u;
	else
		power_of_two = __ffs(expected_number_of_elements) - 1u;

	auto warp = cooperative_groups::tiled_partition<32>(block);
	if (block.thread_rank() < warp.size())
		clear(warp, &m_vEB);

	new (&m_bottom) Map{ block, allocator, 1u << power_of_two };
	block.sync();
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::vEB(threads group, allocator_type& allocator, unsigned int expected_number_of_elements) :
	m_allocator(&allocator)
{
	unsigned int power_of_two;
	if (expected_number_of_elements == 0)
		power_of_two = 10u;
	else
		power_of_two = __ffs(expected_number_of_elements) - 1u;

	clear(group, &m_vEB);

	new (&m_bottom) Map{ block, allocator, 1u << power_of_two };
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ void vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::clear(block_threads block)
{
	threads tile32 = cooperative_groups::tiled_partition<32>(block);

	if (block.thread_rank() < tile32.size())
		clear(tile32);

	block.sync();
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ void vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::clear(threads group)
{
	clear(group, &m_vEB);

	m_bottom.clear(group);
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::find(key_type key)
{
	return m_bottom.find(key);
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::find(threads group, key_type key)
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::const_iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::find(threads group, key_type key) const
{
	return m_bottom.find(group, key);
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::insert(threads group, key_type key, mapped_type value)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	unsigned int depth = 0u;
	Node* current_node = &m_vEB;
#ifdef VAN_EMDE_BOAS_DEBUG
	if (group.thread_rank() == 0)
		printf("%d\n\n", key);
#endif // VAN_EMDE_BOAS_DEBUG
	while (true)
	{
	#ifdef VAN_EMDE_BOAS_DEBUG
		if (group.thread_rank() == 0)
			printf("%d %d %p ", depth, is_leaf(depth), current_node);
	#endif // VAN_EMDE_BOAS_DEBUG
		if (is_leaf(depth))
		{
			LeafNode* leaf_node = reinterpret_cast<LeafNode*>(current_node);
		#ifdef VAN_EMDE_BOAS_DEBUG
			if (group.thread_rank() == 0)
				printf("END: %p\n", leaf_node);
		#endif // VAN_EMDE_BOAS_DEBUG
			return set(group, leaf_node, key, value);
		}
		else
		{
			size_type bits = extract_bits(key, depth);
		#ifdef VAN_EMDE_BOAS_DEBUG
			if (group.thread_rank() == 0)
				printf("%d ", bits);
		#endif // VAN_EMDE_BOAS_DEBUG
			InternalNode* internal_node = reinterpret_cast<InternalNode*>(current_node);
			gpu::atomic<Node*>& ptr = internal_node->nodes[bits];
			do
			{
				if (!ptr)
				{
					if (ptr.compare_and_swap(group, nullptr, INSERTING()) == nullptr)
					{
						++depth;
						Node* next_node;
						if (is_leaf(depth))
						{
							LeafNode* leaf_node = m_allocator->allocate<LeafNode>(group);
						#ifdef VAN_EMDE_BOAS_DEBUG
							if (group.thread_rank() == 0)
								printf("Leaf %p\n", leaf_node);
						#endif // VAN_EMDE_BOAS_DEBUG
							clear(group, leaf_node);
							next_node = leaf_node;
						}
						else
						{
							InternalNode* internal_node = m_allocator->allocate<InternalNode>(group);
						#ifdef VAN_EMDE_BOAS_DEBUG
							if (group.thread_rank() == 0)
								printf("int %p\n", internal_node);
						#endif // VAN_EMDE_BOAS_DEBUG
							clear(group, internal_node);
							next_node = internal_node;
						}
						ptr.store(group, next_node);
						current_node = next_node;
					}
				}
				else
				{
					if (ptr != INSERTING())
					{
						++depth;
						current_node = ptr;
					#ifdef VAN_EMDE_BOAS_DEBUG
						if (group.thread_rank() == 0)
							printf("\n");
					#endif // VAN_EMDE_BOAS_DEBUG
						break;
					}
				}
			} while (ptr == INSERTING());
		}
	}
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::predecessor(threads group, key_type key)
{
	auto it = m_bottom.find(group, key);
	if (it != m_bottom.end())
		return it;

	key_type discovered_bits = 0u;
	unsigned int depth = 0u;
	unsigned int predecessor_depth = 0u;
	Node* predecessor_node = nullptr;
	Node* current_node = &m_vEB;
	while (true)
	{
		if (is_leaf(depth))
		{
			LeafNode* leaf_node = reinterpret_cast<LeafNode*>(current_node);
			auto leaf_predecessor_info = predecessor_leaf(group, leaf_node, key);
			if (leaf_predecessor_info.in_leaf)
				return m_bottom.find(leaf_predecessor_info.key);
			else
				break;
		}
			
		else
		{
			size_type bits = extract_bits(key, depth);
			InternalNode* internal_node = reinterpret_cast<InternalNode*>(current_node);
			auto predecessor_info = find_predecessor(group, internal_node, bits);
			bool has_predecessor = predecessor_info.has_predecessor;
			Node* current_predecessor = predecessor_info.current_predecessor;
			Node* ptr = predecessor_info.ptr;
		#ifdef VAN_EMDE_BOAS_DEBUG
			if (group.thread_rank() == 0)
				printf("%p %p %d \n", ptr, current_predecessor, has_predecessor);
		#endif // VAN_EMDE_BOAS_DEBUG
			if (has_predecessor)
			{
				predecessor_depth = depth;
				predecessor_node = current_predecessor;
				if (depth == 0u)
					discovered_bits = predecessor_info.predecessor_index;
				else
				{
					discovered_bits = key >> (UNIVERSE - depth * NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO);
					discovered_bits = (discovered_bits << relative_shift(depth)) + predecessor_info.predecessor_index;
				}
			}

			++depth;

			if (!ptr || ptr == INSERTING())
				break;

			current_node = ptr;
		}
	}

	if (predecessor_node)
		return find_max(group, predecessor_node, discovered_bits, predecessor_depth + 1);
	else
		return m_bottom.end();
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::size_type vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::size() const
{
	return m_bottom.size();
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ void vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::debug() const
{
	internal_debug(&m_vEB, 0, 0u);
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ void vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::internal_debug(const Node* node, unsigned int depth, key_type key) const
{
	if (is_leaf(depth))
	{
		const LeafNode* leaf_node = reinterpret_cast<const LeafNode*>(node);

		for (unsigned int i = 0u; i != NUMBER_OF_ELEMENTS_AT_BOTTOM; ++i)
		{
			unsigned int data = leaf_node->bits[i];
			for (unsigned j = 0u; j != 8 * sizeof(gpu::UInt32); ++j)
			{
				if (data & (1u << j))
					gpu::print(key * NUMBER_OF_BITS_PER_NODE + i * 32 + j, " ");
			}
		}
		gpu::print("\n");
	}
	else
	{
		for (unsigned int i = 0u; i != NUMBER_OF_CHILDREN_PER_NODE; ++i)
		{
			const InternalNode* internal_node = reinterpret_cast<const InternalNode*>(node);
			const Node* ptr = internal_node->nodes[i];
			if (ptr && ptr != INSERTING())
			{
				unsigned int search_bits = UNIVERSE - NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO;
				unsigned int delta = (search_bits / NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO);
				delta = delta * NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO;
				delta = (UNIVERSE - NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO) - delta;

				if (!is_leaf(depth + 1))
					internal_debug(internal_node->nodes[i], depth + 1, (key << NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO) + i);
				else
					internal_debug(internal_node->nodes[i], depth + 1, (key << delta) + i);
				
			}
				
		}
	}
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::Node* vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::INSERTING() const
{
	return reinterpret_cast<Node*>(10);
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::size_type vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::extract_bits(key_type key, unsigned int depth) const
{
	unsigned int shift = (UNIVERSE - NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO * (depth + 1u));
	unsigned int shifted_key = key >> shift;
	unsigned int masked_key = shifted_key & (NUMBER_OF_CHILDREN_PER_NODE - 1u);
	if (depth + 1u == NUMBER_OF_LEVELS)
	{
		unsigned int search_bits = UNIVERSE - NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO;
		unsigned int delta = (search_bits / NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO);
		delta = delta * NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO;
		delta = (UNIVERSE - NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO) - delta;
		masked_key = masked_key >> (NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO - delta);
	}
	return masked_key;
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ bool vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::is_leaf(unsigned int depth) const
{
	return NUMBER_OF_LEVELS == depth;
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ void vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::clear(threads g, InternalNode* internal_node)
{
	unsigned int offset = 0u;
	while (offset < NUMBER_OF_CHILDREN_PER_NODE)
	{
		internal_node->nodes[offset + g.thread_rank()].store_unatomically(nullptr);
		offset += g.size();
	}
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ void vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::clear(threads g, LeafNode* leaf_node)
{
	for (unsigned int offset = 0u; offset != NUMBER_OF_ELEMENTS_AT_BOTTOM; offset += g.size())
		leaf_node->bits[offset + g.thread_rank()].store_unatomically(0u);
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ unsigned int vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::relative_shift(unsigned int depth) const
{
	if (depth + 1u == NUMBER_OF_LEVELS)
	{
		unsigned int search_bits = UNIVERSE - NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO;
		unsigned int delta = (search_bits / NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO);
		delta = delta * NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO;
		delta = (UNIVERSE - NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO) - delta;
		return delta;
	}
	else
		return NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO;
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::set(threads g, LeafNode* current_node, key_type key, mapped_type value)
{
	auto local_bits = key & (NUMBER_OF_BITS_PER_NODE - 1u);
	auto local_thid = local_bits / g.size();
	auto local_bit = local_bits % g.size();
	if (g.thread_rank() == (local_thid % g.size()))
		current_node->bits[local_thid].fetch_or(1u << local_bit);
	
	return m_bottom.insert(g, gpu::make_pair<key_type, mapped_type>(key, value));
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::find_max(threads g, Node* current_node, key_type discovered_bits, unsigned int depth)
{
	while (true)
	{
		if (is_leaf(depth))
		{
			LeafNode* leaf_node = reinterpret_cast<LeafNode*>(current_node);
			return find_max(g, leaf_node, discovered_bits);
		}
		else
		{
			InternalNode* internal_node = reinterpret_cast<InternalNode*>(current_node);
			auto result = find_max(g, internal_node);
			if (!result.next_node)
				return end();

			current_node = result.next_node;
			discovered_bits = (discovered_bits << relative_shift(depth)) + result.index;
			++depth;
		}
	}
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::MaxInfo vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::find_max(threads g, InternalNode* internal_node)
{
	unsigned int work_to_do = NUMBER_OF_CHILDREN_PER_NODE;
	unsigned int already_done = 0u;
	unsigned int pos = 0u;
	do
	{
		pos = work_to_do - g.size() - already_done + g.thread_rank();
		Node* ptr = internal_node->nodes[pos];
		unsigned int warp_result = g.ballot(ptr != nullptr);
		if (warp_result)
		{
			unsigned int winner_thid = 31u - __clz(warp_result);
			ptr = reinterpret_cast<Node*>(g.shfl(reinterpret_cast<std::uintptr_t>(ptr), winner_thid));
			pos = g.shfl(pos, winner_thid);
			return { ptr, pos };
		}
		already_done += g.size();
	} while (already_done < work_to_do);
	return { nullptr, -1 };
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::iterator vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::find_max(threads g, LeafNode* leaf_node, key_type key)
{
	unsigned int local_offset = (NUMBER_OF_ELEMENTS_AT_BOTTOM / g.size()) - 1u;
	do
	{
		unsigned int local_index = local_offset * g.size() + g.thread_rank();
		bool has_predecessor = leaf_node->bits[local_index];
		unsigned int warp_result = g.ballot(has_predecessor);
		if (!warp_result)
		{
			if (local_offset == 0u)
				return end();
			else
				--local_offset;
		}
		else
		{
			unsigned int winner_thid = __ffs(warp_result) - 1;
			unsigned int data = leaf_node->bits[local_offset * g.size() + winner_thid];
			unsigned int shift = 31 - __clz(data);
			key_type result_key = key * NUMBER_OF_BITS_PER_NODE + winner_thid * 32 + local_offset * 1024 + shift;

			return m_bottom.find(g, result_key);
		}
	} while (local_offset != 0);
	return end();
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::PredecessorInfo vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::find_predecessor(threads g, InternalNode* internal_node, size_type bits)
{
	Node* previous = nullptr;
	Node* current = nullptr;

	unsigned int local_offset = bits / g.size();
	do
	{
		unsigned int local_index = local_offset * g.size() + g.thread_rank();
		if (local_index < bits)
			previous = internal_node->nodes[local_index];
		else if (local_index == bits)
			current = internal_node->nodes[bits];

		current = reinterpret_cast<Node*>(g.shfl(reinterpret_cast<std::uintptr_t>(current), bits % g.size()));
		unsigned int warp_result = g.ballot(previous != nullptr);

		if (!warp_result)
		{
			if (local_offset == 0u)
				return { current, nullptr, -1, false };
			else
				--local_offset;
		}
		else
		{
			unsigned int pos = 31 - __clz(warp_result);
			previous = reinterpret_cast<Node*>(g.shfl(reinterpret_cast<std::uintptr_t>(previous), pos));
			return { current, previous, local_offset * g.size() + pos, true };
		}
	} while (local_offset != 0u);
	return { current, nullptr, -1, false };
}

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>
__device__ typename vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::PredecessorLeafInfo vEB<Key, Value, NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO, NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO>::predecessor_leaf(threads g, LeafNode* leaf_node, key_type key)
{
	unsigned int local_bits = key & (NUMBER_OF_BITS_PER_NODE - 1u);
	unsigned int local_offset = local_bits / g.size();
	unsigned int local_bit = local_bits % g.size();
	bool has_predecessor = false;
	do
	{
		unsigned int local_index = local_offset + g.thread_rank();
		unsigned int msb = 0u;
		if (local_index < local_offset && leaf_node->bits[local_index])
		{
			auto data = leaf_node->bits[local_index];
			msb = 32u - __clz(data);
			has_predecessor = true;
		}
		else if (local_index == local_offset)
		{
			unsigned shift = 1u << local_bit;
			unsigned int mask = shift - 1u;
			unsigned int resulting_data = leaf_node->bits[local_index] & mask;
			msb = 31u - __clz(resulting_data);
			has_predecessor = resulting_data != 0u;
		}

		unsigned int warp_result = g.ballot(has_predecessor);
		if (!warp_result)
		{
			if (local_offset == 0u)
				return { {}, false };
			else
				--local_offset;
		}
		else
		{
			unsigned int winner_thid = __ffs(warp_result) - 1;
			msb = g.shfl(msb, winner_thid);
			key_type resulting_key = local_offset * g.size() + winner_thid * 32 + msb;
			key_type upper_bits = ~(NUMBER_OF_BITS_PER_NODE - 1u);
			resulting_key = resulting_key + (key & upper_bits);
			return { resulting_key, true };
		}

	} while (local_offset != 0u);
	return { {}, false };
}
