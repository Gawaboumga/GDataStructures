#include <hayai/hayai.hpp>

#include "concurrent/containers/hash_tables/fixed_fast_integer.cuh"

#include "hash_map-fixture.cu"

using FastIntegerLinear = gpu::concurrent::fixed_fast_integer<key_type, mapped_type, gpu::hash<key_type>, gpu::concurrent::linear_probing<gpu::hash<key_type>>>;
using FastIntegerLinearInsertionFixture = HashMapInsertionFixture<FastIntegerLinear>;
using FastIntegerLinearGetFixture = HashMapGetFixture<FastIntegerLinear>;
using FastIntegerLinearGetUnsuccessfulFixture = HashMapGetUnsuccessfulFixture<FastIntegerLinear>;

BENCHMARK_F(FastIntegerLinearInsertionFixture, FastIntegerLinear, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	insert();
}

BENCHMARK_F(FastIntegerLinearGetFixture, FastIntegerLinear, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	get();
}

BENCHMARK_F(FastIntegerLinearGetUnsuccessfulFixture, FastIntegerLinear, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	unsuccessful_get();
}
/*
using FastIntegerQuadratic = gpu::concurrent::fixed_fast_integer<key_type, mapped_type, gpu::hash<key_type>, gpu::concurrent::quadratic_probing<gpu::hash<key_type>>>;
using FastIntegerQuadraticInsertionFixture = HashMapInsertionFixture<FastIntegerQuadratic>;
using FastIntegerQuadraticGetFixture = HashMapGetFixture<FastIntegerQuadratic>;
using FastIntegerQuadraticGetUnsuccessfulFixture = HashMapGetUnsuccessfulFixture<FastIntegerQuadratic>;

BENCHMARK_F(FastIntegerQuadraticInsertionFixture, FastIntegerQuadratic, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	insert();
}

BENCHMARK_F(FastIntegerQuadraticGetFixture, FastIntegerQuadratic, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	get();
}

BENCHMARK_F(FastIntegerQuadraticGetUnsuccessfulFixture, FastIntegerQuadratic, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	unsuccessful_get();
}

using FastIntegerDoubleHashing = gpu::concurrent::fixed_fast_integer<key_type, mapped_type, gpu::hash<key_type>, gpu::concurrent::double_hashing_probing<gpu::hash<key_type>>>;
using FastIntegerDoubleHashingInsertionFixture = HashMapInsertionFixture<FastIntegerDoubleHashing>;
using FastIntegerDoubleHashingGetFixture = HashMapGetFixture<FastIntegerDoubleHashing>;
using FastIntegerDoubleHashingGetUnsuccessfulFixture = HashMapGetUnsuccessfulFixture<FastIntegerDoubleHashing>;

BENCHMARK_F(FastIntegerDoubleHashingInsertionFixture, FastIntegerDoubleHashing, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	insert();
}

BENCHMARK_F(FastIntegerDoubleHashingGetFixture, FastIntegerDoubleHashing, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	get();
}

BENCHMARK_F(FastIntegerDoubleHashingGetUnsuccessfulFixture, FastIntegerDoubleHashing, NUMBER_OF_RUNS, NUMBER_OF_ITERATIONS)
{
	unsuccessful_get();
}
*/
