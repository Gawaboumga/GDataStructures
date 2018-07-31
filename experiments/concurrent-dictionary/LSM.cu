//#include <hayai/hayai.hpp>
//
//#include "LSM.cuh"
//
//#include "concurrent-xfasttrie-fixture.cu"
//
//using LSM = gpu::lsm<key_type, mapped_type, 16u * 32u>;
//using LSMInsertionFixture = XTrieInsertionFixture<LSM, Structure::LSM>;
//using LSMGetThreadFixture = XTrieGetThreadFixture<LSM, Structure::LSM>;
//using LSMGetWorstThreadFixture = XTrieGetThreadFixture<LSM, Structure::LSM, true>;
//using LSMGetWarpFixture = XTrieGetWarpFixture<LSM, Structure::LSM>;
//using LSMGetWorstWarpFixture = XTrieGetWarpFixture<LSM, Structure::LSM, true>;
//using LSMPredecessorThreadFixture = XTriePredecessorFixture<LSM, Structure::LSM, false>;
//using LSMPredecessorWarpFixture = XTriePredecessorFixture<LSM, Structure::LSM, true>;
//using LSMPredecessorWorstWarpFixture = XTriePredecessorFixture<LSM, Structure::LSM, true, true>;
//using LSMSuccessorThreadFixture = XTrieSuccessorFixture<LSM, Structure::LSM, false>;
//using LSMSuccessorWarpFixture = XTrieSuccessorFixture<LSM, Structure::LSM, true>;
//
///*BENCHMARK_F(LSMInsertionFixture, LSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}*/
//
//BENCHMARK_F(LSMGetThreadFixture, GetThreadLSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_thread();
//}
//
//BENCHMARK_F(LSMGetWorstThreadFixture, GetWorstThreadLSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_thread();
//}
//
//BENCHMARK_F(LSMGetWarpFixture, GetWarpLSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_warp();
//}
//
//BENCHMARK_F(LSMGetWorstWarpFixture, GetWorstWarpLSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_warp();
//}
///*
//BENCHMARK_F(LSMPredecessorThreadFixture, PredecessorThreadLSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	predecessor();
//}
//*/
//BENCHMARK_F(LSMPredecessorWarpFixture, PredecessorWarpLSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	predecessor();
//}
//
//BENCHMARK_F(LSMPredecessorWorstWarpFixture, PredecessorWorstWarpLSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	predecessor();
//}
///*
//BENCHMARK_F(LSMSuccessorThreadFixture, SuccessorThreadLSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	successor();
//}
//
//BENCHMARK_F(LSMSuccessorWarpFixture, SuccessorWarpLSM, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	successor();
//}
//*/
