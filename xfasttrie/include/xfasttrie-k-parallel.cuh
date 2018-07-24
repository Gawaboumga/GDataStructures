#ifndef X_FAST_TRIE_K_PARALLEL_HPP
#define X_FAST_TRIE_K_PARALLEL_HPP

#include "allocators/default_allocator.cuh"
#include "containers/hash_tables/fast_integer.cuh"
#include "utility/pair.cuh"

template <typename Key, typename Value, std::size_t UNIVERSE = (sizeof(Key) * 8)>
class XFastTrieKParallel
{
	public:
		using key_type = Key;
		using mapped_type = Value;
		using value_type = mapped_type;

	private:
		static constexpr std::size_t RANK = UNIVERSE;
		static constexpr std::size_t SUBRANK = UNIVERSE - 1;
		static constexpr std::size_t UPRANK = UNIVERSE + 1;

		using Keyset = gpu::fast_integer<key_type, key_type>;
		using keyset_iterator = typename Keyset::iterator;
		using Keysets = Keyset[SUBRANK];
		Keysets m_maps;
		struct Node
		{
			__device__ Node() = default;
			__device__ Node(mapped_type val) :
				value(val)
			{
			}

			__device__ Node(mapped_type val, typename gpu::fast_integer<key_type, Node>::iterator p, typename gpu::fast_integer<key_type, Node>::iterator s) :
				value(val),
				predecessor(p),
				successor(s)
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
			typename gpu::fast_integer<key_type, Node>::iterator predecessor;
			typename gpu::fast_integer<key_type, Node>::iterator successor;
		};
		using Map = gpu::fast_integer<key_type, Node>;
		Map m_bottom;

		using iterator_bottom_element = typename gpu::fast_integer<key_type, Node>::iterator;
		iterator_bottom_element m_head;
		iterator_bottom_element m_tail;

	public:
		using threads = cooperative_groups::thread_block;
		using iterator = typename Map::iterator;
		using const_iterator = typename Map::const_iterator;
		using size_type = unsigned int;

		__device__ iterator begin();
		__device__ const_iterator begin() const;
		__device__ iterator end();
		__device__ const_iterator end() const;

		__device__ XFastTrieKParallel() = default;
		__device__ XFastTrieKParallel(threads group, gpu::default_allocator& allocator);
		__device__ XFastTrieKParallel(const XFastTrieKParallel& other) = default;
		__device__ XFastTrieKParallel(XFastTrieKParallel&& other) = default;

		__device__ void clear(threads group);

		__device__ iterator find(threads group, key_type key);
		__device__ const_iterator find(threads group, key_type key) const;

		__device__ iterator insert(threads group, key_type key, value_type value);

		__device__ XFastTrieKParallel& operator=(const XFastTrieKParallel& other) = default;
		__device__ XFastTrieKParallel& operator=(XFastTrieKParallel&& other) = default;

		__device__ iterator predecessor(threads group, key_type key);

		__device__ size_type size() const;
		__device__ iterator successor(threads group, key_type key);

		__device__ void debug() const;

	private:
		__device__ gpu::pair<keyset_iterator, size_type> binary_search(threads group, key_type key);

		__device__ key_type extract_i_upper_bits(key_type key, int number_of_bits) const;
		__device__ iterator get_predecessor(threads group, key_type key, keyset_iterator u);
		__device__ iterator get_successor(threads group, key_type key, keyset_iterator u);

		__device__ void post_condition(threads group);
};

#include "xfasttrie-k-parallel.cu"

#endif // X_FAST_TRIE_K_PARALLEL_HPP
