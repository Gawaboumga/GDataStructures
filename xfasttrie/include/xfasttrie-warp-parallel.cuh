#ifndef X_FAST_TRIE_WARP_PARALLEL_HPP
#define X_FAST_TRIE_WARP_PARALLEL_HPP

#include "allocators/default_allocator.cuh"
#include "containers/hash_tables/fast_integer.cuh"
#include "utility/pair.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE = (sizeof(Key) * 8)>
class XFastTrieWarpParallel
{
	static_assert(sizeof(UNIVERSE) <= sizeof(Key) * 8, "It would be meaningless, don't you find ?");

	public:
		using key_type = Key;
		using mapped_type = Value;

		static constexpr std::size_t UNIVERSE_SIZE = UNIVERSE;

	private:
		static constexpr std::size_t RANK = UNIVERSE;
		static constexpr std::size_t SUBRANK = UNIVERSE - 1;
		static constexpr std::size_t UPRANK = UNIVERSE + 1;

		struct Children
		{
			bool operator==(const Children& other) const
			{
				return minimal_left == other.minimal_left && maximal_right == other.maximal_right;
			}
			key_type minimal_left;
			key_type maximal_right;
		};
		using Keyset = gpu::fast_integer<key_type, Children>;
		using keyset_iterator = typename Keyset::iterator;
		using Keysets = Keyset[SUBRANK];
		Keysets m_maps;
		struct Node
		{
			__device__ Node() = default;
			__device__ Node(mapped_type val) :
				value{ val }
			{
			}

			__device__ Node(mapped_type val, key_type p, key_type s) :
				value{ val },
				predecessor{ p },
				successor{ s }
			{
			}

			__device__ Node(const Node&) = default;
			__device__ Node(Node&&) = default;

			__device__ Node& operator=(const Node&) = default;
			__device__ Node& operator=(Node&&) = default;

			__device__ bool operator==(const Node& other) const
			{
				return value == other.value;
			}

			mapped_type value;
			key_type predecessor;
			key_type successor;
		};
		using Map = gpu::fast_integer<key_type, Node>;
		using Map_iterator = typename Map::iterator;
		Map m_bottom;

		key_type m_head;
		key_type m_tail;

	public:
		using block_threads = cooperative_groups::thread_block;
		using threads = cooperative_groups::thread_block_tile<32>;
		using iterator = typename Map::iterator;
		using const_iterator = typename Map::const_iterator;
		using size_type = typename std::conditional<(RANK > 31), gpu::UInt64, gpu::UInt32>::type;

		__device__ iterator begin();
		__device__ const_iterator begin() const;
		__device__ iterator end();
		__device__ const_iterator end() const;

		__device__ XFastTrieWarpParallel() = default;
		__device__ XFastTrieWarpParallel(block_threads group, gpu::default_allocator& allocator);
		__device__ XFastTrieWarpParallel(threads group, gpu::default_allocator& allocator);
		__device__ XFastTrieWarpParallel(const XFastTrieWarpParallel& other) = default;
		__device__ XFastTrieWarpParallel(XFastTrieWarpParallel&& other) = default;

		__device__ void clear(block_threads group);
		__device__ void clear(threads group);

		__device__ iterator find(key_type key);
		__device__ iterator find(threads group, key_type key);
		__device__ const_iterator find(threads group, key_type key) const;

		__device__ iterator insert(threads group, key_type key, mapped_type value);

		__device__ size_type maximal_size() const;

		__device__ XFastTrieWarpParallel& operator=(const XFastTrieWarpParallel& other) = default;
		__device__ XFastTrieWarpParallel& operator=(XFastTrieWarpParallel&& other) = default;

		__device__ iterator predecessor(threads group, key_type key);

		__device__ size_type size() const;
		__device__ iterator successor(threads group, key_type key);

		__device__ void debug() const;

		__host__ __device__ static unsigned int universe() { return UNIVERSE; }

	private:
		__device__ gpu::pair<keyset_iterator, Map_iterator> binary_search(threads group, key_type key);

		__device__ key_type extract_i_upper_bits(key_type key, int number_of_bits) const;

		__device__  gpu::pair<key_type, key_type> find_or_update(threads group, key_type key);

		__device__ bool has_data() const;

		__device__ key_type INVALID_PREDECESSOR() const;
		__device__ key_type INVALID_SUCCESSOR() const;

		__device__ void walk_up(threads group, key_type key, size_type from);

		__device__ void post_condition(threads group);
};

#include "xfasttrie-warp-parallel.cu"

#endif // X_FAST_TRIE_WARP_PARALLEL_HPP
