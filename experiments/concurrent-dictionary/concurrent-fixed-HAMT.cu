//#include <hayai/hayai.hpp>
//
//#include "fixed-HAMT.cuh"
//
//#include "concurrent-xfasttrie-fixture.cu"
//
//using HAMT5 = HAMT<key_type, mapped_type, 5>;
//using HAMT5InsertionFixture = XTrieInsertionFixture<HAMT5, Structure::XFASTTRIE>;
//using HAMT5GetWarpFixture = XTrieGetWarpFixture<HAMT5, Structure::XFASTTRIE>;
//using HAMT5PredecessorFixture = XTriePredecessorFixture<HAMT5, Structure::XFASTTRIE, true>;
//
//BENCHMARK_F(HAMT5InsertionFixture, HAMT5, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}
///*
//BENCHMARK_F(HAMT5GetWarpFixture, GetWarpHAMT5, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_warp();
//}
//
//BENCHMARK_F(HAMT5PredecessorFixture, PredecessorHAMT5, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	predecessor();
//}*/
//
//using HAMT6 = HAMT<key_type, mapped_type, 6>;
//using HAMT6InsertionFixture = XTrieInsertionFixture<HAMT6, Structure::XFASTTRIE>;
//using HAMT6GetWarpFixture = XTrieGetWarpFixture<HAMT6, Structure::XFASTTRIE>;
//using HAMT6PredecessorFixture = XTriePredecessorFixture<HAMT6, Structure::XFASTTRIE, true>;
//
//BENCHMARK_F(HAMT6InsertionFixture, HAMT6, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}
///*
//BENCHMARK_F(HAMT6GetWarpFixture, GetWarpHAMT6, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_warp();
//}
//
//BENCHMARK_F(HAMT6PredecessorFixture, PredecessorHAMT6, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	predecessor();
//}*/
