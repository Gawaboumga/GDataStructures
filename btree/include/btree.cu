#include "btree.cuh"

namespace gpu
{
	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::iterator BTree<Key, Value>::begin()
	{
		if (m_root)
			return m_root->begin();
		else
			return end();
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::const_iterator BTree<Key, Value>::begin() const
	{
		if (m_root)
			return m_root->begin();
		else
			return end();
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::iterator BTree<Key, Value>::end()
	{
		return { nullptr, nullptr };
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::const_iterator BTree<Key, Value>::end() const
	{
		return { nullptr, nullptr };
	}

	template <class Key, class Value>
	__device__ BTree<Key, Value>::BTree(block_threads group, allocator_type& allocator, unsigned int expected_number_of_elements) :
		m_allocator{ &allocator },
		m_root{ nullptr },
		m_number_of_elements{ 0u }
	{
	}

	template <class Key, class Value>
	__device__ BTree<Key, Value>::BTree(threads group, allocator_type& allocator, unsigned int expected_number_of_elements) :
		m_allocator{ &allocator },
		m_root{ nullptr },
		m_number_of_elements{ 0u }
	{
	}

	template <class Key, class Value>
	__device__ void BTree<Key, Value>::clear(block_threads group)
	{
		if (m_root)
			m_root = nullptr;
	}

	template <class Key, class Value>
	__device__ void BTree<Key, Value>::clear(threads group)
	{
		if (m_root)
			m_root = nullptr;
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::iterator BTree<Key, Value>::find(const key_type& key) const
	{
		if (!m_root)
			return end();

		return m_root->find(key);
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::iterator BTree<Key, Value>::find(threads group, const key_type& key) const
	{
		if (!m_root)
			return end();

		return m_root->find(group, key);
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::iterator BTree<Key, Value>::insert(threads group, key_type key, mapped_type value)
	{
		if (!m_root)
		{
			leaf_type* node = make_leaf(*this, group);
			m_root = node;
			auto it = node->insert(*this, group, key, value);
		#ifdef GPU_BTREE_DEBUG
			m_root->post_condition(group);
		#endif // GPU_BTREE_DEBUG
			return it;
		}

		auto it = m_root->insert(*this, group, key, value);
		if (m_root->needs_split())
			m_root = esplit(*this, group, nullptr, m_root);
	#ifdef GPU_BTREE_DEBUG
		m_root->post_condition(group);
	#endif // GPU_BTREE_DEBUG
		return it;
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::iterator BTree<Key, Value>::predecessor(threads group, const key_type& key) const
	{
		if (!m_root)
			return end();

		return m_root->predecessor(group, key);
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::size_type BTree<Key, Value>::size() const
	{
		if (!m_root)
			return 0;

		return m_root->size();
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::iterator BTree<Key, Value>::successor(threads group, const key_type& key) const
	{
		if (!m_root)
			return end();

		return m_root->successor(group, key);
	}

	template <class Key, class Value>
	__device__ void BTree<Key, Value>::debug() const
	{
		if (m_root)
			m_root->debug();
	}

	template <class Key, class Value>
	__device__ typename BTree<Key, Value>::size_type BTree<Key, Value>::memory_consumed() const
	{
		if (m_root)
			return m_root->memory_consumed();
		return 0;
	}

	template <class Key, class Value>
	__device__ void BTree<Key, Value>::post_condition(threads group) const
	{
		if (m_root)
			m_root->post_condition(group);
	}
}
