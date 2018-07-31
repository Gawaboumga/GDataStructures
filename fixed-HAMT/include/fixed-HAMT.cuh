#ifndef CONCURRENT_HAMT_HPP
#define CONCURRENT_HAMT_HPP

#include "concurrent/allocators/default_allocator.cuh"
#include "concurrent/containers/hash_tables/fixed_fast_integer.cuh"
#include "utility/pair.cuh"

template <typename Key, typename Value, unsigned int NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO = 5u, unsigned int NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO = 10u>
class HAMT
{
	static_assert(NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO >= 5, "Warp-based");
	static_assert(NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO >= 10, "Warp-based");
	static constexpr unsigned int NUMBER_OF_CHILDREN_PER_NODE = 1u << NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO;
	static constexpr unsigned int NUMBER_OF_BITS_PER_NODE = 1u << NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO;
	static constexpr unsigned int NUMBER_OF_ELEMENTS_AT_BOTTOM = NUMBER_OF_BITS_PER_NODE / 32u;
	static constexpr unsigned int UNIVERSE = sizeof(Key) * 8;
	static constexpr unsigned int NUMBER_OF_LEVELS = ((UNIVERSE - NUMBER_OF_BITS_PER_NODE_POWER_OF_TWO) / NUMBER_OF_CHILDREN_PER_NODE_POWER_OF_TWO) + 1u;

	public:
		using key_type = Key;
		using mapped_type = Value;
		using size_type = typename std::conditional<(UNIVERSE > 31), gpu::UInt64, gpu::UInt32>::type;

	private:
		struct Node
		{
		};

		struct InternalNode : Node
		{
			gpu::atomic<Node*> nodes[NUMBER_OF_CHILDREN_PER_NODE];
		};

		struct LeafNode : Node
		{
			gpu::atomic<gpu::UInt32> bits[NUMBER_OF_ELEMENTS_AT_BOTTOM];
		};

		InternalNode m_HAMT;
		using Map = gpu::concurrent::fixed_fast_integer<key_type, mapped_type>;
		using Map_iterator = typename Map::iterator;
		Map m_bottom;

	public:
		using block_threads = cooperative_groups::thread_block;
		using threads = cooperative_groups::thread_block_tile<32>;
		using iterator = typename Map::iterator;
		using const_iterator = typename Map::const_iterator;
		using allocator_type = gpu::concurrent::default_allocator;

		__device__ iterator end();
		__device__ const_iterator end() const;

		__device__ HAMT() = default;
		__device__ HAMT(block_threads group, allocator_type& allocator, unsigned int expected_number_of_elements);
		__device__ HAMT(threads group, allocator_type& allocator, unsigned int expected_number_of_elements);
		__device__ HAMT(HAMT&& other) = default;

		__device__ void clear(block_threads group);
		__device__ void clear(threads group);

		__device__ iterator find(key_type key);
		__device__ iterator find(threads group, key_type key);
		__device__ const_iterator find(threads group, key_type key) const;

		__device__ iterator insert(threads group, key_type key, mapped_type value);

		__device__ HAMT& operator=(HAMT&& other) = default;

		__device__ iterator predecessor(threads group, key_type key);

		__device__ size_type size() const;

		__device__ void debug() const;

		__device__ void post_condition(threads group);

	private:
		__device__ void internal_debug(const Node* node, unsigned int depth, key_type key) const;
		__device__ Node* INSERTING() const;

		__device__ size_type extract_bits(key_type key, unsigned int depth) const;

		__device__ bool is_leaf(unsigned int depth) const;

		__device__ void clear(threads g, LeafNode* leaf_node);
		__device__ void clear(threads g, InternalNode* internal_node);
		__device__ unsigned int relative_shift(unsigned int depth) const;
		__device__ iterator set(threads g, LeafNode* current_node, key_type key, mapped_type value);

		struct PredecessorLeafInfo
		{
			key_type key;
			bool in_leaf;
		};

		struct PredecessorInfo
		{
			Node* ptr;
			Node* current_predecessor;
			unsigned int predecessor_index;
			bool has_predecessor;
		};

		struct MaxInfo
		{
			Node* next_node;
			unsigned int index;
		};

		__device__ iterator find_max(threads g, Node* current_node, key_type key, unsigned int depth);
		__device__ MaxInfo find_max(threads g, InternalNode* internal_node);
		__device__ iterator find_max(threads g, LeafNode* leaf_node, key_type key);
		__device__ PredecessorInfo find_predecessor(threads g, InternalNode* internal_node, size_type bits);
		__device__ PredecessorLeafInfo predecessor_leaf(threads g, LeafNode* leaf_node, key_type key);
		

		allocator_type* m_allocator;
};

#include "fixed-HAMT.cu"

#endif // CONCURRENT_HAMT_HPP
