//#include <hayai/hayai.hpp>
//
//#include "concurrent/containers/hash_tables/chaining.cuh"
//
//#include "hash_map-fixture.cu"
//
//using Chaining = gpu::concurrent::chaining<key_type, mapped_type, gpu::hash<key_type>>;
//using ChainingInsertionFixture = HashMapInsertionFixture<Chaining>;
//using ChainingGetFixture = HashMapGetFixture<Chaining>;
//using ChainingGetUnsuccessfulFixture = HashMapGetUnsuccessfulFixture<Chaining>;
//
//BENCHMARK_F(ChainingInsertionFixture, Chaining, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}
//
//BENCHMARK_F(ChainingGetFixture, Chaining, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get();
//}
//
//BENCHMARK_F(ChainingGetUnsuccessfulFixture, Chaining, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	unsuccessful_get();
//}
