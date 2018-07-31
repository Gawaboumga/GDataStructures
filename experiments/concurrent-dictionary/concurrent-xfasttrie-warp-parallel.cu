#include <hayai/hayai.hpp>

#include "concurrent-xfasttrie-warp-parallel.cuh"

#include "concurrent-xfasttrie-fixture.cu"

using Warp = ConcurrentXFastTrieWarpParallel<key_type, mapped_type, HEIGHT>;
using WarpInsertionFixture = XTrieInsertionFixture<Warp, Structure::XFASTTRIE>;
using WarpGetThreadFixture = XTrieGetThreadFixture<Warp, Structure::XFASTTRIE>;
using WarpGetWarpFixture = XTrieGetWarpFixture<Warp, Structure::XFASTTRIE>;
using WarpPredecessorFixture = XTriePredecessorFixture<Warp, Structure::XFASTTRIE, true>;
using WarpSuccessorFixture = XTrieSuccessorFixture<Warp, Structure::XFASTTRIE, true>;

BENCHMARK_F(WarpInsertionFixture, Warp, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	insert();
}

BENCHMARK_F(WarpGetThreadFixture, GetThreadWarp, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	get_thread();
}

BENCHMARK_F(WarpGetWarpFixture, GetWarpWarp, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	get_warp();
}

BENCHMARK_F(WarpPredecessorFixture, PredecessorWarp, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	predecessor();
}
/*
BENCHMARK_F(WarpSuccessorFixture, SuccessorWarp, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	successor();
}
*/
