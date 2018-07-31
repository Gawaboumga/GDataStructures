//#include <hayai/hayai.hpp>
//
//#include "btree.cuh"
//
//#include "concurrent-xfasttrie-fixture.cu"
//
//using BTREE = gpu::BTree<key_type, mapped_type>;
//using BTreeInsertionFixture = XTrieInsertionFixture<BTREE, Structure::BTREE>;
//using BTreeGetThreadFixture = XTrieGetThreadFixture<BTREE, Structure::BTREE>;
//using BTreeGetWarpFixture = XTrieGetWarpFixture<BTREE, Structure::BTREE>;
//using BTreePredecessorFixture = XTriePredecessorFixture<BTREE, Structure::BTREE, true>;
//using BTreeSuccessorFixture = XTrieSuccessorFixture<BTREE, Structure::BTREE, true>;
//
//BENCHMARK_F(BTreeInsertionFixture, InsertionBtree, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	insert();
//}
///*
//BENCHMARK_F(BTreeGetThreadFixture, GetThreadBtree, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_thread();
//}
//
//BENCHMARK_F(BTreeGetWarpFixture, GetWarpBtree, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	get_warp();
//}
//
//BENCHMARK_F(BTreePredecessorFixture, PredecessorBtree, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	predecessor();
//}*/
///*
//BENCHMARK_F(BTreeSuccessorFixture, SuccessorBtree, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
//{
//	successor();
//}*/
