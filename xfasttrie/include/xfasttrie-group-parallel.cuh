#ifndef X_FAST_TRIE_GROUP_PARALLEL_HPP
#define X_FAST_TRIE_GROUP_PARALLEL_HPP

#include "allocators/default_allocator.cuh"
#include "containers/array.cuh"
#include "containers/hash_tables/fast_integer.cuh"
#include "utility/pair.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE = (sizeof(Key) * 8), std::size_t GROUP = 2>
class XFastTrieGroupParallel
{
	static_assert(sizeof(UNIVERSE) <= sizeof(Key) * 8, "It would be meaningless, don't you find ?");

	public:
		using key_type = Key;
		using mapped_type = Value;

		static constexpr std::size_t UNIVERSE_SIZE = UNIVERSE;
		static constexpr std::size_t GROUP_SIZE = GROUP;
		static constexpr std::size_t TAIL_GROUP_SIZE = (UNIVERSE - 1u) % (GROUP_SIZE + 1u) == 0u ? GROUP_SIZE : (UNIVERSE - 1u) % (GROUP_SIZE + 1u) - 1u;
		static_assert(GROUP_SIZE < 8u, "Not tested beyond this amount");
		static_assert(TAIL_GROUP_SIZE <= GROUP_SIZE, "Looks surprising");

	private:
		static constexpr std::size_t RANK = UNIVERSE;
		static constexpr std::size_t SUBRANK = UNIVERSE - 1;
		static constexpr std::size_t UPRANK = UNIVERSE + 1;
		static constexpr std::size_t NUMBER_OF_KEYSETS = (SUBRANK + GROUP) / (GROUP + 1u);

		template <std::size_t N>
		struct Children
		{
			gpu::array<key_type, N> minimal_left;
			gpu::array<key_type, N> maximal_right;
		};
		using Child = Children<(1u << (GROUP + 1u)) - 1u>;
		using Keyset = gpu::fast_integer<key_type, Child>;
		using keyset_iterator = typename Keyset::iterator;
		using Keysets = Keyset[NUMBER_OF_KEYSETS];
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
			__device__ Node& operator=(Node&& other) = default;

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

		__device__ XFastTrieGroupParallel() = default;
		__device__ XFastTrieGroupParallel(block_threads group, gpu::default_allocator& allocator);
		__device__ XFastTrieGroupParallel(threads group, gpu::default_allocator& allocator);
		__device__ XFastTrieGroupParallel(const XFastTrieGroupParallel& other) = default;
		__device__ XFastTrieGroupParallel(XFastTrieGroupParallel&& other) = default;

		__device__ void clear(block_threads group);
		__device__ void clear(threads group);

		__device__ iterator find(key_type key);
		__device__ iterator find(threads group, key_type key);
		__device__ const_iterator find(threads group, key_type key) const;

		__device__ iterator insert(threads group, key_type key, mapped_type value);

		__device__ size_type maximal_size() const;

		__device__ XFastTrieGroupParallel& operator=(const XFastTrieGroupParallel& other) = default;
		__device__ XFastTrieGroupParallel& operator=(XFastTrieGroupParallel&& other) = default;

		__device__ iterator predecessor(threads group, key_type key);

		__device__ size_type size() const;
		__device__ iterator successor(threads group, key_type key);

		__device__ void debug() const;

		__host__ __device__ static unsigned int universe() { return UNIVERSE; }

	private:
		struct BinarySearchResult
		{
			keyset_iterator it;
			Map_iterator bottom_it;
			unsigned int index;
		};
		__device__ BinarySearchResult binary_search(threads group, key_type key);

		__device__ key_type extract_i_upper_bits(key_type key, int number_of_bits) const;

		__device__  gpu::pair<key_type, key_type> find_or_update(threads group, key_type key);

		__device__ unsigned int get_highest_index(key_type tail_bits, const Child& value, unsigned int thid);
		__device__ Child make_children(key_type tail_bits, const key_type& key, unsigned int thid);
		__device__ void update_key(Child& value, key_type tail_bits, const key_type& key, unsigned int thid);

		__device__ bool has_data() const;

		__device__ key_type INVALID_PREDECESSOR() const;
		__device__ key_type INVALID_SUCCESSOR() const;

		__device__ void walk_up(threads group, key_type key, size_type from);

		__device__ void post_condition(threads group);
};

#include "xfasttrie-group-parallel.cu"

#endif // X_FAST_TRIE_GROUP_PARALLEL_HPP
