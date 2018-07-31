//#include <hayai/hayai.hpp>
//
//#include "concurrent-xfasttrie-binary.cuh"
//
//#include "concurrent-xfasttrie-fixture.cu"
//
//using Binary = ConcurrentXFastTrieBinary<key_type, mapped_type, HEIGHT>;
//using BinaryInsertionFixture = XTrieInsertionFixture<Binary>;
//using BinaryGetThreadFixture = XTrieGetThreadFixture<Binary>;
//using BinaryGetWarpFixture = XTrieGetWarpFixture<Binary>;
//using BinaryPredecessorFixture = XTriePredecessorFixture<Binary>;
//using BinarySuccessorFixture = XTrieSuccessorFixture<Binary>;
//
//BENCHMARK_F(BinaryInsertionFixture, InsertionBinary, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}
//
///*
//BENCHMARK_F(BinaryGetThreadFixture, GetThreadBinary, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_thread();
//}
//
//BENCHMARK_F(BinaryGetWarpFixture, GetWarpBinary, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_warp();
//}
//*/
//
//BENCHMARK_F(BinaryPredecessorFixture, PredecessorBinary, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	predecessor();
//}
//
//BENCHMARK_F(BinarySuccessorFixture, SuccessorBinary, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	successor();
//}
