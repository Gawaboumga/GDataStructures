//#include <hayai/hayai.hpp>
//
//#include "concurrent/containers/hash_tables/fixed_cuckoo.cuh"
//#include "concurrent/containers/hash_tables/fixed_bucket_cuckoo.cuh"
//
//#include "hash_map-fixture.cu"
///*
//using Cuckoo2 = gpu::concurrent::fixed_cuckoo<key_type, mapped_type, gpu::hash<key_type>, 2>;
//using Cuckoo2InsertionFixture = HashMapInsertionFixture<Cuckoo2>;
//using Cuckoo2GetFixture = HashMapGetFixture<Cuckoo2>;
//using Cuckoo2GetUnsuccessfulFixture = HashMapGetUnsuccessfulFixture<Cuckoo2>;
//
//BENCHMARK_F(Cuckoo2InsertionFixture, Cuckoo2, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}
//
//BENCHMARK_F(Cuckoo2GetFixture, Cuckoo2, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get();
//}
//
//BENCHMARK_F(Cuckoo2GetUnsuccessfulFixture, Cuckoo2, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	unsuccessful_get();
//}
//*/
//using Cuckoo3 = gpu::concurrent::fixed_cuckoo<key_type, mapped_type, gpu::hash<key_type>, 3>;
//using Cuckoo3InsertionFixture = HashMapInsertionFixture<Cuckoo3>;
//using Cuckoo3GetFixture = HashMapGetFixture<Cuckoo3>;
//using Cuckoo3GetUnsuccessfulFixture = HashMapGetUnsuccessfulFixture<Cuckoo3>;
//
//BENCHMARK_F(Cuckoo3InsertionFixture, Cuckoo3, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}
//
//BENCHMARK_F(Cuckoo3GetFixture, Cuckoo3, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get();
//}
//
//BENCHMARK_F(Cuckoo3GetUnsuccessfulFixture, Cuckoo3, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	unsuccessful_get();
//}
//
//using Cuckoo4 = gpu::concurrent::fixed_cuckoo<key_type, mapped_type, gpu::hash<key_type>, 4>;
//using Cuckoo4InsertionFixture = HashMapInsertionFixture<Cuckoo4>;
//using Cuckoo4GetFixture = HashMapGetFixture<Cuckoo4>;
//using Cuckoo4GetUnsuccessfulFixture = HashMapGetUnsuccessfulFixture<Cuckoo4>;
//
//BENCHMARK_F(Cuckoo4InsertionFixture, Cuckoo4, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}
//
//BENCHMARK_F(Cuckoo4GetFixture, Cuckoo4, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get();
//}
//
//BENCHMARK_F(Cuckoo4GetUnsuccessfulFixture, Cuckoo4, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	unsuccessful_get();
//}
//
//using BucketCuckoo4 = gpu::concurrent::fixed_bucket_cuckoo<key_type, mapped_type, gpu::hash<key_type>, 4>;
//using BucketCuckoo4InsertionFixture = HashMapInsertionFixture<BucketCuckoo4>;
//using BucketCuckoo4GetFixture = HashMapGetFixture<BucketCuckoo4>;
//using BucketCuckoo4GetUnsuccessfulFixture = HashMapGetUnsuccessfulFixture<BucketCuckoo4>;
//
//
//BENCHMARK_F(BucketCuckoo4InsertionFixture, BucketCuckoo4, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}
//
//BENCHMARK_F(BucketCuckoo4GetFixture, BucketCuckoo4, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get();
//}
//
//BENCHMARK_F(BucketCuckoo4GetUnsuccessfulFixture, BucketCuckoo4, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	unsuccessful_get();
//}
