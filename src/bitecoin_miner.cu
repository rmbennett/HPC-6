#include <stdint.h>
#include <stdlib.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <curand_kernel.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "bitecoin_miner.cu.h"

// __global__ void setup_kernel (curandState *state, unsigned long seed)
// {
// 	int i = blockIdx.x * blockDim.x + threadIdx.x;
// 	curand_init (seed, i , i, &state[i]);
// }

__global__ void computeTrials(const uint64_t roundId, const uint64_t roundSalt, const uint8_t *chainData, const size_t chainDataCount, const uint32_t maxIndices, const uint32_t *randomSalt, const uint32_t hashSteps, uint32_t *trialSolutions, uint32_t *trialProofs)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	// curandState localState = globalState[i];

	// uint32_t curr = 0;
	// for (int index = 0; index < maxIndices; index++)
	// {
	// 	curr += 1 + ((int)curand_uniform(&localState) % 10);
	// 	trialSolutions[(i * maxIndices) + index] = curr;
	// }

	uint32_t proof[BIGINT_SIZE];

	HashReference(roundId, roundSalt, chainData, chainDataCount, maxIndices, randomSalt, hashSteps, trialSolutions + (i*maxIndices), &proof[0]);

	cuda_wide_copy(BIGINT_SIZE, trialProofs + (BIGINT_SIZE * i), &proof[0]);

	// globalState[i] = localState;

}

namespace bitecoin
{
bool runBitecoinMiningTrials(const size_t trialCount, const uint64_t roundId, const uint64_t roundSalt, const size_t chainDataCount, const uint32_t maxIndices, const uint32_t hashSteps, uint32_t *bestSolution, uint32_t *bestProof, size_t solutionSize, size_t trialProofSize, uint32_t *trialProofs, uint32_t *d_trialSolutions, uint32_t *d_randomSalt, uint32_t *d_trialProofs, uint8_t *d_chainData)
{

	//Populate trialSolutions

	// for (int trial = 0; trial < trialCount; trial++)
	// {
	// 	srand (time(NULL));
	// 	uint32_t curr = 0;
	// 	for (int index = 0; index < maxIndices; index++)
	// 	{
	// 		curr += 1 + (rand() % 100);
    //      trialSolutions[(trial * maxIndices) + index] = curr;
	// 	}
	// }

	//Define work size
	dim3 grid(1);
	dim3 threads(trialCount);

	//Run Computation of Hashes
	// setup_kernel<<< grid, trialCount >>>(d_curand, time(NULL));

	computeTrials<<< grid, trialCount >>>(roundId, roundSalt, d_chainData, chainDataCount, maxIndices, d_randomSalt, hashSteps, d_trialSolutions, d_trialProofs);

	//Check there are no problems
	getLastCudaError("Kernel execution failed");

	//Fetch Results
	checkCudaErrors(cudaMemcpy(trialProofs, d_trialProofs, trialProofSize, cudaMemcpyDeviceToHost));

	bool newBest = false;

	int bestOne = 0;
	//Check the results for a new best!
	for (int trial = 0; trial < trialCount; trial++)
	{
		if(cuda_wide_compare(BIGINT_SIZE, trialProofs + (BIGINT_SIZE * trial), bestProof) < 0)
		{
			// cuda_wide_copy(maxIndices, bestSolution, trialSolutions + (maxIndices * trial));
			bestOne = trial;
			cuda_wide_copy(BIGINT_SIZE + 1, bestProof, trialProofs + (BIGINT_SIZE * trial));
			printf("    Found new best, score=%lg.\n", cuda_wide_as_double(BIGINT_SIZE, bestProof));
			newBest = true;
		}
	}

	checkCudaErrors(cudaMemcpy(bestSolution, d_trialSolutions + (maxIndices * bestOne), solutionSize, cudaMemcpyDeviceToHost));

	return newBest;
}
}; //End of namespace