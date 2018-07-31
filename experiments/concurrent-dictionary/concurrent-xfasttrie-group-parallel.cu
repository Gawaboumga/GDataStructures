#include <hayai/hayai.hpp>

#include "concurrent-xfasttrie-group-parallel.cuh"

#include "concurrent-xfasttrie-fixture.cu"

using Group1 = ConcurrentXFastTrieGroupParallel<key_type, mapped_type, HEIGHT, 1u>;
using Group1InsertionFixture = XTrieInsertionFixture<Group1, Structure::XFASTTRIE>;
using Group1GetWarpFixture = XTrieGetWarpFixture<Group1, Structure::XFASTTRIE>;
using Group1PredecessorFixture = XTriePredecessorFixture<Group1, Structure::XFASTTRIE, true>;
using Group1SuccessorFixture = XTrieSuccessorFixture<Group1, Structure::XFASTTRIE, true>;

BENCHMARK_F(Group1InsertionFixture, Group1, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	insert();
}
/*
BENCHMARK_F(Group1GetWarpFixture, GetWarpGroup1, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	get_warp();
}
*/
BENCHMARK_F(Group1PredecessorFixture, PredecessorGroup1, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	predecessor();
}
/*
BENCHMARK_F(Group1SuccessorFixture, SuccessorGroup1, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	successor();
}
*/
using Group2 = ConcurrentXFastTrieGroupParallel<key_type, mapped_type, HEIGHT, 2u>;
using Group2InsertionFixture = XTrieInsertionFixture<Group2, Structure::XFASTTRIE>;
using Group2GetWarpFixture = XTrieGetWarpFixture<Group2, Structure::XFASTTRIE>;
using Group2PredecessorFixture = XTriePredecessorFixture<Group2, Structure::XFASTTRIE, true>;
using Group2SuccessorFixture = XTrieSuccessorFixture<Group2, Structure::XFASTTRIE, true>;

BENCHMARK_F(Group2InsertionFixture, Group2, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	insert();
}
/*
BENCHMARK_F(Group2GetWarpFixture, GetWarpGroup2, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	get_warp();
}
*/
BENCHMARK_F(Group2PredecessorFixture, PredecessorGroup2, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	predecessor();
}
/*
BENCHMARK_F(Group2SuccessorFixture, SuccessorGroup2, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	successor();
}
*/

using Group3 = ConcurrentXFastTrieGroupParallel<key_type, mapped_type, HEIGHT, 3u>;
using Group3InsertionFixture = XTrieInsertionFixture<Group3, Structure::XFASTTRIE>;
using Group3GetWarpFixture = XTrieGetWarpFixture<Group3, Structure::XFASTTRIE>;
using Group3PredecessorFixture = XTriePredecessorFixture<Group3, Structure::XFASTTRIE, true>;
using Group3SuccessorFixture = XTrieSuccessorFixture<Group3, Structure::XFASTTRIE, true>;

BENCHMARK_F(Group3InsertionFixture, Group3, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	insert();
}
/*
BENCHMARK_F(Group3GetWarpFixture, GetWarpGroup3, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
get_warp();
}
*/
BENCHMARK_F(Group3PredecessorFixture, PredecessorGroup3, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	predecessor();
}
/*
BENCHMARK_F(Group3SuccessorFixture, SuccessorGroup3, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
successor();
}
*/
