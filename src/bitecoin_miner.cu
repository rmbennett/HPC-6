#include <stdint.h>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <csignal>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "bitecoin_miner.cu.h"

__global__ void computeTrials(const uint64_t roundId, const uint64_t roundSalt, const uint8_t *chainData, const size_t chainDataCount, const uint32_t maxIndices, const uint32_t *randomSalt, const uint32_t hashSteps, const uint32_t *trialSolutions, uint32_t *trialProofs)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	uint32_t proof[BIGINT_SIZE];

	HashReference(roundId, roundSalt, chainData, chainDataCount, maxIndices, randomSalt, hashSteps, trialSolutions + (i*maxIndices), &proof[0]);

	cuda_wide_copy(BIGINT_SIZE, trialProofs + (BIGINT_SIZE * i), &proof[0]);

}

namespace bitecoin
{
bool runBitecoinMiningTrials(const size_t trialCount, const uint64_t roundId, const uint64_t roundSalt, const uint8_t *chainData, const size_t chainDataCount, const uint32_t maxIndices, const uint32_t *randomSalt, const uint32_t hashSteps, uint32_t *bestSolution, uint32_t *bestProof)
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
		for (int index = 0; index < maxIndices; index++)
		{
			curr += 1 + (rand() % 10);
            trialSolutions[(trial * maxIndices) + index] = curr;
		}
	}

	//Allocate GPU Buffers
	uint32_t *d_trialSolutions, *d_randomSalt, *d_trialProofs;
	uint8_t *d_chainData;

	checkCudaErrors(cudaMalloc((void **) &d_trialSolutions, trialSolutionsSize));
	checkCudaErrors(cudaMalloc((void **) &d_randomSalt, randomSaltSize));
	checkCudaErrors(cudaMalloc((void **) &d_chainData, chainDataSize));
	checkCudaErrors(cudaMalloc((void **) &d_trialProofs, trialProofSize));

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
			cuda_wide_copy(BIGINT_SIZE + 1, bestProof, trialProofs + (BIGINT_SIZE * trial));
			printf("    Found new best, score=%lg.\n", cuda_wide_as_double(BIGINT_SIZE, bestProof));
			newBest = true;
		}
	}

    checkCudaErrors(cudaFree(d_trialSolutions));
    checkCudaErrors(cudaFree(d_randomSalt));
    checkCudaErrors(cudaFree(d_trialProofs));
	checkCudaErrors(cudaFree(d_chainData));

	free(trialSolutions);
	free(trialProofs);

	// printf("bestProof %d %d %d %d %d %d %d %d\n", bestProof[0], bestProof[1],bestProof[2],bestProof[3],bestProof[4],bestProof[5],bestProof[6],bestProof[7]);

	// printf("BestSolution %d %d %d %d %d %d %d %d\n", bestSolution[0], bestSolution[1],bestSolution[2],bestSolution[3],bestSolution[4],bestSolution[5],bestSolution[6],bestSolution[7]);

	return newBest;
}
}; //End of namespace