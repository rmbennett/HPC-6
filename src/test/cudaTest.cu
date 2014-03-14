#include <stdint.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

__device__ int wide_compare(unsigned n, const uint32_t *a, const uint32_t *b)
{
	if(a==b)
		return 0;

	for(int i=n-1;i>=0;i--){
		if(a[i]<b[i])
			return -1;
		if(a[i]>b[i])
			return +1;
	}
	return 0;
}

__global__ void cudaKernelTest(unsigned n, uint32_t *cmp0, uint32_t *cmp1, int *res)
{
	int i;
}

bool
runTest(const int argc, const char **argv)
{
	return false;
}