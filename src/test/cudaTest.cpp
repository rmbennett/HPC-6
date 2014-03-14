#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <vector>

#include "../contrib/fnv.hpp"

// Required to include CUDA vector types
#include <cuda_runtime.h>

extern bool
runTest(const int argc, const char **argv);

extern uint64_t
gpuHasherTest(uint8_t *chainData, size_t _len);

int
main(int argc, char **argv)
{

	bool testResult;

	testResult = runTest(argc, (const char **)argv);

	assert(testResult == true);

	std::vector<uint8_t> chainData;
	for (int i = 0; i < 8; i++)
	{
		chainData.push_back((uint8_t) (1 + (rand() % 10)));
	}

	uint64_t gpuHash = gpuHasherTest(&chainData[0], chainData.size());

	hash::fnv<64> hasher;
	uint64_t chainHash=hasher((const char*)&chainData[0], chainData.size());

	assert(gpuHash = chainHash);

	printf("%d\n",chainHash);

	cudaDeviceReset();

	return 0;

}