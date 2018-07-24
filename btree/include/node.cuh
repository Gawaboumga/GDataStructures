#ifndef GPU_NODE_HPP
#define GPU_NODE_HPP

class LeafNode;
class InternalNode;

struct Node
{
	using key_type = Key;
	using mapped_type = Value;
	using allocator_type = allocator_type;

	template <class Wrapped>
	struct Wrapper
	{
		__device__ operator Wrapped()
		{
			return *data;
		}

		const Wrapped* data;
	};

	struct NodeIterator
	{
		Wrapper<key_type> first;
		Wrapper<mapped_type> second;

		__device__ NodeIterator* operator->()
		{
			return this;
		}

		__device__ bool operator==(const NodeIterator& other) const
		{
			return first.data == other.first.data && second.data == other.second.data;
		}
	};

	using const_iterator = const NodeIterator;
	using iterator = NodeIterator;

	__device__ const_iterator end() const
	{
		return { nullptr, nullptr };
	}

	__device__ iterator end()
	{
		return { nullptr, nullptr };
	}

	__device__ iterator find(key_type key) const
	{
		if (is_leaf())
			return static_cast<const LeafNode*>(this)->find(key);
		else
			return static_cast<const InternalNode*>(this)->find(key);
	}

	__device__ iterator find(threads group, key_type key) const
	{
		if (is_leaf())
			return static_cast<const LeafNode*>(this)->find(group, key);
		else
			return static_cast<const InternalNode*>(this)->find(group, key);
	}

	__device__ bool is_leaf() const
	{
		return type == NodeType::Leaf;
	}

	__device__ iterator insert(BTree<Key, Value>& btree, threads group, key_type key, mapped_type value)
	{
		if (is_leaf())
			return static_cast<LeafNode*>(this)->insert(btree, group, key, value);
		else
			return static_cast<InternalNode*>(this)->insert(btree, group, key, value);
	}

	__device__ bool needs_split() const
	{
		if (is_leaf())
			return count >= NUMBER_ELEMENTS_PER_NODE;
		else
			return count >= NUMBER_ELEMENTS_PER_NODE;
	}

	__device__ iterator predecessor(threads group, key_type key) const
	{
		if (is_leaf())
			return static_cast<const LeafNode*>(this)->predecessor(group, key);
		else
			return static_cast<const InternalNode*>(this)->predecessor(group, key);
	}

	__device__ size_type size() const
	{
		if (is_leaf())
			return static_cast<const LeafNode*>(this)->size();
		else
			return static_cast<const InternalNode*>(this)->size();
	}

	__device__ iterator successor(threads group, key_type key) const
	{
		if (is_leaf())
			return static_cast<const LeafNode*>(this)->successor(group, key);
		else
			return static_cast<const InternalNode*>(this)->successor(group, key);
	}

	__device__ void split(BTree<Key, Value>& btree, threads group, key_type* key, Node** left, Node** right)
	{
		if (is_leaf())
			return static_cast<LeafNode*>(this)->split(btree, group, key, left, right);
		else
			return static_cast<InternalNode*>(this)->split(btree, group, key, left, right);
	}

	__device__ size_type memory_consumed() const
	{
		if (is_leaf())
			return static_cast<const LeafNode*>(this)->memory_consumed();
		else
			return static_cast<const InternalNode*>(this)->memory_consumed();
	}

	__device__ void debug(int level = 0) const
	{
		if (is_leaf())
			static_cast<const LeafNode*>(this)->debug(level);
		else
			static_cast<const InternalNode*>(this)->debug(level);
	}

	__device__ void post_condition(threads group) const
	{
		if (is_leaf())
			static_cast<const LeafNode*>(this)->post_condition(group);
		else
			static_cast<const InternalNode*>(this)->post_condition(group);
	}

	using keys_type = gpu::array<key_type, NUMBER_ELEMENTS_PER_NODE>;
	using values_type = gpu::array<mapped_type, NUMBER_ELEMENTS_PER_NODE>;
	using nodes_type = gpu::array<Node*, NUMBER_ELEMENTS_PER_NODE>;

	__device__ unsigned int search_key(key_type key) const
	{
		unsigned int low = 0u;
		unsigned int high = count - 1u;

		while (low <= high)
		{
			unsigned int mid = (low + high) / 2u;
			if (keys[mid] == key)
			{
				return mid;
			}
			else if (key < keys[mid])
			{
				if (mid == 0u)
					return low;

				high = mid - 1u;
			}
			else
			{
				low = mid + 1u;
			}
		}
		return low;
	}

	__device__ unsigned int search_key(threads group, key_type key) const
	{
		unsigned int thid = group.thread_rank();
		bool lower = true;
		if (thid < count)
			lower = key <= keys[thid];
		group.sync();
		unsigned int matching_bits = group.ballot(lower);
		return __ffs(matching_bits) - 1u;
	}

	enum class NodeType
	{
		Leaf,
		Internal = 1,
	};

	keys_type keys;
	NodeType type;
	int count;
};

__device__ static void append(threads group, InternalNode& in, key_type key)
{
	if (group.thread_rank() == 0)
	{
		in.keys[in.count] = key;
	}
	group.sync();
}

__device__ static void append(threads group, InternalNode& in, Node* node)
{
	if (group.thread_rank() == 0)
	{
		in.nodes[in.count] = node;
		++in.count;
	}
	group.sync();
}

__device__ static void append(threads group, LeafNode& leaf, key_type key, mapped_type value)
{
	if (group.thread_rank() == 0)
	{
		leaf.keys[leaf.count] = key;
		leaf.values[leaf.count] = value;
		++leaf.count;
	}
	group.sync();
}

__device__ static void insert_at(threads group, InternalNode& in, unsigned int index, key_type key)
{
	if (in.count == index)
	{
		if (group.thread_rank() == 0)
		{
			in.keys[index] = key;
		}
		group.sync();
		return;
	}

	gpu::insert_at(group, in.keys.begin(), in.keys.begin() + in.count, index, key);
}

__device__ static void insert_at(threads group, InternalNode& in, unsigned int index, Node* node)
{
	if (in.count == index)
	{
		if (group.thread_rank() == 0)
		{
			in.nodes[index] = node;
			++in.count;
		}
		group.sync();
		return;
	}

	gpu::insert_at(group, in.nodes.begin(), in.nodes.begin() + in.count, index, node);
	if (group.thread_rank() == 0)
	{
		++in.count;
	}
	group.sync();
}

__device__ static void insert_at(threads group, LeafNode& leaf, unsigned int index, key_type key, mapped_type value)
{
	if (leaf.count == index)
	{
		if (group.thread_rank() == 0)
		{
			leaf.keys[index] = key;
			leaf.values[index] = value;
			++leaf.count;
		}
		group.sync();
		return;
	}

	gpu::insert_at(group, leaf.keys.begin(), leaf.keys.begin() + leaf.count, index, key);
	gpu::insert_at(group, leaf.values.begin(), leaf.values.begin() + leaf.count, index, value);
	if (group.thread_rank() == 0)
	{
		++leaf.count;
	}
	group.sync();
}

__device__ static InternalNode* make_internal(BTree& btree, threads group)
{
	InternalNode* in = btree.m_allocator->allocate<InternalNode>(group);
	if (group.thread_rank() == 0)
	{
		in->type = Node::NodeType::Internal;
		in->keys[1u] = gpu::numeric_limits<key_type>::max();
		in->count = 0;
	}
	group.sync();
	gpu::fill(group, in->nodes.begin(), in->nodes.end(), nullptr);
	return in;
}

__device__ static LeafNode* make_leaf(BTree& btree, threads group)
{
	LeafNode* leaf = btree.m_allocator->allocate<LeafNode>(group);
	if (group.thread_rank() == 0)
	{
		leaf->type = Node::NodeType::Leaf;
		leaf->count = 0;
	}
	group.sync();
	return leaf;
}

__device__ static Node* esplit(BTree& btree, threads group, Node* parent, Node* child)
{
	if (!child->needs_split())
		return parent;

	key_type key;
	Node* left;
	Node* right;
	child->split(btree, group, &key, &left, &right);
	if (!parent)
	{
		InternalNode* in = make_internal(btree, group);
		append(group, *in, key);
		append(group, *in, left);
		append(group, *in, right);
		return in;
	}

	InternalNode& in = *static_cast<InternalNode*>(parent);
	unsigned int index = in.search_key(group, key);

	insert_at(group, in, index, key);
	if (group.thread_rank() == 0)
	{
		in.nodes[index] = left;
	}
	group.sync();
	insert_at(group, in, index + 1u, right);

	return &in;
}

struct InternalNode : public Node
{
	using const_iterator = typename Node::const_iterator;
	using iterator = typename Node::iterator;

	__device__ iterator find(key_type key) const
	{
		unsigned int index = search_key(key);
		if (index >= count)
			return nodes[count - 1u]->find(key);
		else if (key >= keys[index])
			return nodes[index + 1u]->find(key);
		else
			return nodes[index]->find(key);
	}

	__device__ iterator find(threads group, key_type key) const
	{
		unsigned int index = search_key(group, key);
		if (index >= count)
			return nodes[count - 1u]->find(group, key);
		else if (key >= keys[index])
			return nodes[index + 1u]->find(group, key);
		else
			return nodes[index]->find(group, key);
	}

	__device__ iterator insert(BTree& btree, threads group, key_type key, mapped_type value)
	{
		unsigned int index = search_key(group, key);
		Node* child;
		if (index == count)
			child = nodes[count - 1u];
		else if (key == keys[index])
			child = nodes[index + 1u];
		else
			child = nodes[index];

		auto it = child->insert(btree, group, key, value);
		if (child->needs_split())
			esplit(btree, group, this, child);
		return it;
	}

	__device__ iterator predecessor(threads group, key_type key) const
	{
		unsigned int index = search_key(group, key);
		Node* child;
		if (index >= count)
			child = nodes[count - 1u];
		else if (key == keys[index])
			child = nodes[index + 1u];
		else
			child = nodes[index];

		return child->predecessor(group, key);
	}

	__device__ void split(BTree<Key, Value>& btree, threads group, key_type* key, Node** left, Node** right)
	{
	#ifdef GPU_BTREE_DEBUG
		ENSURE(count >= 3);
	#endif // GPU_BTREE_DEBUG

		unsigned int i = NUMBER_ELEMENTS_PER_NODE / 2u;
		*key = keys[i - 1u];

		InternalNode* in = make_internal(btree, group);
		gpu::copy(group, keys.begin(), keys.begin() + i, in->keys.begin());
		gpu::copy(group, nodes.begin(), nodes.begin() + i, in->nodes.begin());
		if (group.thread_rank() == 0)
		{
			in->count = i;
		}
		group.sync();
		*left = in;

		count = count - i;
		gpu::copy(group, keys.begin() + i, keys.end(), keys.begin());
		gpu::copy(group, nodes.begin() + i, nodes.end(), nodes.begin());
		if (group.thread_rank() == 0)
		{
			keys[count] = gpu::numeric_limits<key_type>::max();
		}
		group.sync();

		*right = this;
	}

	__device__ size_type size() const
	{
		size_type result = 0u;
		for (unsigned int i = 0; i != count; ++i)
		{
			result += nodes[i]->size();
		}
		return result;
	}

	__device__ iterator successor(threads group, key_type key) const
	{
		unsigned int index = search_key(group, key);
		Node* child;
		if (index >= count)
			child = nodes[count - 1u];
		else if (key == keys[index])
			child = nodes[index + 1u];
		else
			child = nodes[index];

		return child->successor(group, key);
	}

	__device__ void debug(int level) const
	{
		for (unsigned int i = 0; i != count; ++i)
		{
			for (unsigned int l = 0u; l != level; ++l)
				gpu::print("  ");
			gpu::print(keys[i], "\n");
			nodes[i]->debug(level + 1);
		}
	}

	__device__ size_type memory_consumed() const
	{
		size_type total = NUMBER_ELEMENTS_PER_NODE;
		for (unsigned int i = 0; i != count; ++i)
		{
			total += nodes[i]->memory_consumed();
		}
		return total;
	}

	__device__ void post_condition(threads group) const
	{
		for (unsigned int i = 0u; i != count - 1u; ++i)
		{
			ENSURE(nodes[i]);
			Node* node = nodes[i];
			ENSURE(keys[i] >= node->keys[node->count - 1u]);

			node->post_condition(group);
		}
		nodes[count - 1u]->post_condition(group);
	}

	nodes_type nodes;
};

struct LeafNode : public Node
{
	using const_iterator = typename Node::const_iterator;
	using iterator = typename Node::iterator;

	__device__ iterator find(key_type key) const
	{
		unsigned int index = search_key(key);
		if (index == count)
		{
			if (keys[index] == key)
				return { keys.begin() + index, values.begin() + index };
			else
				return end();
		}
		return { keys.begin() + index, values.begin() + index };
	}

	__device__ iterator find(threads group, key_type key) const
	{
		unsigned int index = search_key(group, key);
		if (index == count)
		{
			if (keys[index] == key)
				return { keys.begin() + index, values.begin() + index };
			else
				return end();
		}
		return { keys.begin() + index, values.begin() + index };
	}

	__device__ iterator insert(BTree& btree, threads group, key_type key, mapped_type value)
	{
		unsigned int index = search_key(group, key);
		if (index == size())
		{
			append(group, *this, key, value);
		}
		else if (keys[index] == key)
		{
			values[index] = value;
		}
		else
		{
			insert_at(group, *this, index, key, value);
		}
		return { keys.begin() + index, values.begin() + index };
	}

	__device__ iterator predecessor(threads group, key_type key) const
	{
		unsigned int index = search_key(group, key);
		if (keys[min(index, count)] == key)
			return { keys.begin() + index, values.begin() + index };
		else if (index >= 1u)
			return { keys.begin() + index - 1u, values.begin() + index - 1u };
		else
			return end();
	}

	__device__ size_type size() const
	{
		return count;
	}

	__device__ void split(BTree& btree, threads group, key_type* key, Node** left, Node** right)
	{
	#ifdef GPU_BTREE_DEBUG
		ENSURE(count >= 2);
	#endif // GPU_BTREE_DEBUG

		unsigned int i = NUMBER_ELEMENTS_PER_NODE / 2u;
		*key = keys[i];

		LeafNode* leaf = make_leaf(btree, group);
		gpu::copy(group, keys.begin(), keys.begin() + i, leaf->keys.begin());
		gpu::copy(group, values.begin(), values.begin() + i, leaf->values.begin());
		if (group.thread_rank() == 0)
		{
			leaf->count = i;
		}
		group.sync();
		*left = leaf;

		if (group.thread_rank() == 0)
		{
			count = count - i;
		}
		group.sync();
		gpu::copy(group, keys.begin() + i, keys.end(), keys.begin());
		gpu::copy(group, values.begin() + i, values.end(), values.begin());
		if (group.thread_rank() == 0)
		{
			keys[count] = 0u;
		}
		group.sync();

		*right = this;
	}

	__device__ iterator successor(threads group, key_type key) const
	{
		unsigned int index = search_key(group, key);
		if (index < size())
			return { keys.begin() + index, values.begin() + index };
		else
			return end();
	}

	__device__ void debug(int level) const
	{
		for (unsigned int i = 0; i != count; ++i)
		{
			for (unsigned int l = 0u; l != level; ++l)
				gpu::print("  ");
			gpu::print("{", keys[i], ", ", values[i], "}\n");
		}
	}

	__device__ size_type memory_consumed() const
	{
		size_type total = NUMBER_ELEMENTS_PER_NODE;
		return total;
	}

	__device__ void post_condition(threads group) const
	{
		unsigned int thid = group.thread_rank();

		bool ok = true;
		if (thid + 1u < size())
			ok = keys[thid] < keys[thid + 1u];
		group.sync();
		ENSURE(!group.ballot(!ok));
	}

	values_type values;
};

#endif // GPU_NODE_HPP
