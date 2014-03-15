#include <stdint.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "bitecoin_miner_wide_maths.cu.h"
#include "bitecoin_minerFNV.cu.h"

#define BIGINT_SIZE_BYTES 32
#define BIGINT_SIZE BIGINT_SIZE_BYTES/4
#define SALT_SIZE BIGINT_SIZE

__device__ 	void PoolHashStep(uint32_t *x, const uint32_t *randomSalt)
{	
	uint32_t tmp[BIGINT_SIZE];
	// tmp=lo(x)*c;
	cuda_wide_mul(4, &tmp[4], &tmp[0], &x[0], randomSalt);
	// [carry,lo(x)] = lo(tmp)+hi(x)
	uint32_t carry=cuda_wide_add(4, &x[0], &tmp[0], &x[4]);
	// hi(x) = hi(tmp) + carry
	cuda_wide_add(4, &x[0]+4, &tmp[4], carry);
	
	// overall:  tmp=lo(x)*c; x=tmp>hi(x)
}

__device__ void PoolHash(const uint64_t roundId, const uint64_t roundSalt, const uint8_t *chainData, const size_t chainDataCount, const uint32_t maxIndices, const uint32_t *randomSalt, const uint32_t hashSteps, const uint32_t index, uint32_t *result)
{

	fnv<64> hasher;
	uint64_t chainHash=hasher((const char*)chainData, chainDataCount);
	
	// The value x is 8 words long (8*32 bits in total)
	// We build (MSB to LSB) as  [ chainHash ; roundSalt ; roundId ; index ]
	uint32_t x[BIGINT_SIZE];
	cuda_wide_zero(BIGINT_SIZE, &x[0]);
	cuda_wide_add(BIGINT_SIZE, &x[0], &x[0], index);	//chosen index goes in at two low limbs
	cuda_wide_add(6, &x[2], &x[2], roundId);	// Round goes in at limbs 3 and 2
	cuda_wide_add(4, &x[4], &x[4], roundSalt);	// Salt goes in at limbs 5 and 4
	cuda_wide_add(2, &x[6], &x[6], chainHash);	// chainHash at limbs 7 and 6
	
	// Now step forward by the number specified by the server
	for(unsigned j=0; j< hashSteps; j++){
		PoolHashStep(&x[0], randomSalt);
	}

	cuda_wide_copy(BIGINT_SIZE, result, &x[0]);
}

__device__ void HashReference(const uint64_t roundId, const uint64_t roundSalt, const uint8_t *chainData, const size_t chainDataCount, const uint32_t maxIndices, const uint32_t *randomSalt, const uint32_t hashSteps, const uint32_t *solution, uint32_t *proof)
{
	uint32_t acc[BIGINT_SIZE];
	cuda_wide_zero(BIGINT_SIZE, &acc[0]);

	for(unsigned i=0;i<maxIndices;i++)
	{	
		uint32_t point[BIGINT_SIZE];
		// Calculate the hash for this specific point
		PoolHash(roundId, roundSalt, chainData, chainDataCount, maxIndices, randomSalt, hashSteps, solution[i], &point[0]);
		
		// Combine the hashes of the points together using xor
		cuda_wide_xor(BIGINT_SIZE, &acc[0], &acc[0], &point[0]);
	}

	cuda_wide_copy(BIGINT_SIZE, proof, &acc[0]);
}