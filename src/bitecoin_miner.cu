#include <stdint.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "bitecoin_miner.cu.h"

__global__ void computeTrials(const uint64_t roundId, const uint64_t roundSalt, const uint8_t *chainData, const size_t chainDataCount, const uint32_t maxIndices, const uint32_t *randomSalt, const uint32_t hashSteps, const uint32_t *trialSolutions, uint32_t *trialProofs)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//Make local copy of the solution
	uint32_t *solution = new uint32_t[maxIndices];
	uint32_t proof[BIGINT_SIZE];

	cuda_wide_copy(maxIndices, solution, trialSolutions + (maxIndices * i));

	HashReference(roundId, roundSalt, chainData, chainDataCount, maxIndices, randomSalt, hashSteps, solution, proof);

	cuda_wide_copy(BIGINT_SIZE, trialProofs + (BIGINT_SIZE * i), &proof[0]);

	delete[] solution;

}

namespace bitecoin
{
bool runBitecoinMiningTrials(size_t trialCount, uint64_t roundId, uint64_t roundSalt, uint8_t *chainData, size_t chainDataCount, uint32_t maxIndices, uint32_t *randomSalt, uint32_t hashSteps, uint32_t *bestSolution, uint32_t *bestProof)
{
	//Define some sizes for arrays
	size_t solutionSize = sizeof(uint32_t) * maxIndices;
	size_t trialSolutionsSize = sizeof(uint32_t) * maxIndices * trialCount;
	size_t proofSize = sizeof(uint32_t) * BIGINT_SIZE;
	size_t trialProofSize = sizeof(uint32_t) * BIGINT_SIZE * trialCount;
	size_t randomSaltSize = sizeof(uint32_t) * SALT_SIZE;
	size_t chainDataSize = sizeof(uint8_t) * chainDataCount;

	//Create CPU Buffers
	uint32_t *trialSolutions = (uint32_t *)malloc(trialSolutionsSize);
	uint32_t *trialProofs = (uint32_t *)malloc(trialProofSize);

	//Populate trialSolutions

	for (int trial = 0; trial < trialCount; trial++)
	{
		uint32_t curr = 0;
		for (int limb = 0; limb < BIGINT_SIZE; limb++)
		{
			curr = curr + 1 + (rand() % 10);
            trialSolutions[(trial * BIGINT_SIZE) + limb] = curr;
		}
	}

	//Allocate GPU Buffers
	uint32_t *d_trialSolutions, *d_randomSalt, *d_trialProofs;
	uint8_t *d_chainData;

	checkCudaErrors(cudaMalloc((void **) &d_trialSolutions, trialSolutionsSize));
	checkCudaErrors(cudaMalloc((void **) &d_randomSalt, randomSaltSize));
	checkCudaErrors(cudaMalloc((void **) &d_chainData, chainDataSize));

	//Push buffers to GPU
	checkCudaErrors(cudaMemcpy(d_trialSolutions, trialSolutions, trialSolutionsSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_randomSalt, randomSalt, randomSaltSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_chainData, chainData, chainDataSize, cudaMemcpyHostToDevice));

	//Define work size
	dim3 grid(1);
	dim3 threads(trialCount);

	//Run Computation of Hashes
	computeTrials<<< grid, trialCount >>>(roundId, roundSalt, d_chainData, chainDataCount, maxIndices, d_randomSalt, hashSteps, d_trialSolutions, d_trialProofs);

	//Check there are no problems
	getLastCudaError("Kernel execution failed");

	//Fetch Results
	checkCudaErrors(cudaMemcpy(trialProofs, d_trialProofs, trialProofSize, cudaMemcpyDeviceToHost));

	bool newBest = false;
	//Check the results for a new best!
	for (int trial = 0; trial < trialCount; trial++)
	{
		if(cuda_wide_compare(BIGINT_SIZE, trialProofs + (BIGINT_SIZE * trial), bestProof) < 0)
		{
			cuda_wide_copy(maxIndices, bestSolution, trialSolutions + (maxIndices * trial));
			cuda_wide_copy(BIGINT_SIZE, bestProof, trialProofs + (BIGINT_SIZE * trial));
			printf("    Found new best, score=%lg.\n", cuda_wide_as_double(BIGINT_SIZE, bestProof));
			newBest = true;
		}
	}

	return newBest;
}
}; //End of namespace