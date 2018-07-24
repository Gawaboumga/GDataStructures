#ifndef GPU_BTREE_HPP
#define GPU_BTREE_HPP

#include "algorithms/copy.cuh"
#include "algorithms/fill.cuh"
#include "algorithms/insert_at.cuh"
#include "concurrent/allocators/default_allocator.cuh"
//#include "allocators/default_allocator.cuh"
#include "containers/array.cuh"
#include "utility/limits.cuh"
#include "utility/pair.cuh"
#include "utility/print.cuh"

#include <cooperative_groups.h>

namespace gpu
{
	template <class Key, class Value>
	class BTree
	{
		public:
			using key_type = Key;
			using mapped_type = Value;
			using size_type = unsigned int;
			using allocator_type = concurrent::default_allocator;
			//using allocator_type = default_allocator;

			using block_threads = cooperative_groups::thread_block;
			using threads = cooperative_groups::thread_block_tile<32>;

		private:
			static constexpr unsigned int NUMBER_ELEMENTS_PER_NODE = 32u;

			friend class Node;
			friend class InternalNode;
			friend class LeafNode;

			#include "node.cuh"

			using internal_node_type = InternalNode;
			using leaf_type = LeafNode;
			using node_type = Node;

		public:
			using const_iterator = typename Node::const_iterator;
			using iterator = typename Node::iterator;

		public:
			__device__ iterator begin();
			__device__ const_iterator begin() const;
			__device__ iterator end();
			__device__ const_iterator end() const;

			__device__ BTree() = default;
			__device__ BTree(block_threads group, allocator_type& allocator, unsigned int expected_number_of_elements = 0);
			__device__ BTree(threads group, allocator_type& allocator, unsigned int expected_number_of_elements = 0);

			__device__ void clear(block_threads group);
			__device__ void clear(threads group);

			__device__ iterator find(const key_type& key) const;
			__device__ iterator find(threads group, const key_type& key) const;

			__device__ iterator insert(threads group, key_type key, mapped_type value);

			__device__ iterator predecessor(threads group, const key_type& key) const;

			__device__ size_type size() const;
			__device__ iterator successor(threads group, const key_type& key) const;

			__device__ void debug() const;

			__device__ size_type memory_consumed() const;

		private:
			__device__ void post_condition(threads group) const;

			allocator_type* m_allocator;
			Node* m_root;
			size_type m_number_of_elements;
	};
}

#include "btree.cu"

#endif // GPU_BTREE_HPP
