#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Required to include CUDA vector types
#include <cuda_runtime.h>

extern bool
runTest(const int argc, const char **argv);


int
main(int argc, char **argv)
{

	bool testResult;

	testResult = runTest(argc, (const char **)argv);

	assert(testResult == true);

}