#include <stdint.h>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_cuda.h>
#include <helper_functions.h>

#include "cudaTestFNV.cu.h"

__device__ __host__ int wide_compare(unsigned n, const uint32_t *a, const uint32_t *b)
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

__device__ __host__ void wide_copy(unsigned n, uint32_t *res, const uint32_t *a)
{
	for(unsigned i=0;i<n;i++){
		res[i]=a[i];
	}
}

__device__ __host__ void wide_zero(unsigned n, uint32_t *res)
{
	for(unsigned i=0;i<n;i++){
		res[i]=0;
	}
}

__device__ __host__ void wide_ones(unsigned n, uint32_t *res)
{
	for(unsigned i=0;i<n;i++){
		res[i]=0xFFFFFFFFul;
	}
}

__device__ __host__ void wide_xor(unsigned n, uint32_t *res, const uint32_t *a, const uint32_t *b)
{
	for(unsigned i=0;i<n;i++){
		res[i]=a[i]^b[i];
	}
}

__device__ __host__ uint32_t wide_add(unsigned n, uint32_t *res, const uint32_t *a, const uint32_t *b)
{
	uint64_t carry=0;
	for(unsigned i=0;i<n;i++){
		uint64_t tmp=uint64_t(a[i])+b[i]+carry;
		res[i]=uint32_t(tmp&0xFFFFFFFFULL);
		carry=tmp>>32;
	}
	return carry;
}

__device__ __host__ uint32_t wide_add(unsigned n, uint32_t *res, const uint32_t *a, uint32_t b)
{
	uint64_t carry=b;
	for(unsigned i=0;i<n;i++){
		uint64_t tmp=a[i]+carry;
		res[i]=uint32_t(tmp&0xFFFFFFFFULL);
		carry=tmp>>32;
	}
	return carry;
}

__device__ __host__ uint32_t wide_add(unsigned n, uint32_t *res, const uint32_t *a, uint64_t b)
{
	uint64_t acc=a[0]+(b&0xFFFFFFFFULL);
	res[0]=uint32_t(acc&0xFFFFFFFFULL);
	uint64_t carry=acc>>32;
	
	acc=a[1]+(b&0xFFFFFFFFULL)+carry;
	res[1]=uint32_t(acc&0xFFFFFFFFULL);
	carry=acc>>32;
	
	for(unsigned i=2;i<n;i++){
		uint64_t tmp=a[i]+carry;
		res[i]=uint32_t(tmp&0xFFFFFFFFULL);
		carry=tmp>>32;
	}
	return carry;
}

__device__ __host__ void wide_mul(unsigned n, uint32_t *res_hi, uint32_t *res_lo, const uint32_t *a, const uint32_t *b)
{	
	uint64_t carry=0, acc=0;
	for(unsigned i=0; i<n; i++){
		for(unsigned j=0; j<=i; j++){
			uint64_t tmp=uint64_t(a[j])*b[i-j];
			acc+=tmp;
			if(acc < tmp)
				carry++;
			//fprintf(stderr, " (%d,%d)", j,i-j);
		}
		res_lo[i]=uint32_t(acc&0xFFFFFFFFull);
		//fprintf(stderr, "\n  %d : %u\n", i, res_lo[i]);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
	}
	
	for(unsigned i=1; i<n; i++){
		for(unsigned j=i; j<n; j++){
			uint64_t tmp=uint64_t(a[j])*b[n-j+i-1];
			acc+=tmp;
			if(acc < tmp)
				carry++;
			//fprintf(stderr, " (%d,%d)", j,n-j+i-1);
		}
		res_hi[i-1]=uint32_t(acc&0xFFFFFFFFull);
		//fprintf(stderr, "\n  %d : %u\n", i+n-1, res_hi[i-1]);
		acc= (carry<<32) | (acc>>32);
		carry=carry>>32;
	}
	res_hi[n-1]=acc;
}

__device__ __host__ double wide_as_double(unsigned n, const uint32_t *x)
{
	double acc=0;
	for(unsigned i=0;i<n;i++){
		acc+=ldexp((double)x[i], i*32);
	}
	return acc;
}

__global__ void cudaWideCompareTest(unsigned n, uint32_t *cmp0, uint32_t *cmp1, int *res)
{
	int result = 101;
	result = wide_compare(n, cmp0, cmp1);
	*res = result;
}

__global__ void cudaWideCopyTest(unsigned n, uint32_t *cmp0, uint32_t *cmp1)
{
	wide_zero(n, cmp1);
	wide_copy(n, cmp1, cmp0);
}

__global__ void cudaWideXorTest(unsigned n, uint32_t *cmp0, uint32_t *cmp1, uint32_t *res)
{
	wide_xor(n, res, cmp0, cmp1);
}

__global__ void cudaWideAddTest(unsigned n, uint32_t *cmp0, uint32_t *cmp1, uint32_t *res, uint32_t *carry)
{
	*carry = wide_add(n, res, cmp0, cmp1);
}

__global__ void cudaWideAdd64Test(unsigned n, uint32_t *cmp0, const uint64_t cmp1, uint32_t *res, uint32_t *carry)
{
	*carry = wide_add(n, res, cmp0, cmp1);
}

__global__ void cudaHasher(uint8_t *chainData, size_t chainLen, uint64_t *result)
{
	fnv<64> hasher;
	*result = hasher((const char *)chainData, chainLen);
}

bool
runTest(const int argc, const char **argv)
{

	findCudaDevice(argc, (const char **)argv);

	//Initialise data for tests and copy that necessary to the CPU
	unsigned numLimbs = 8;
	const unsigned int memSize = sizeof(uint32_t) * numLimbs;

	uint32_t *h_data0 = (uint32_t *)malloc(memSize);
	uint32_t *h_data1 = (uint32_t *)malloc(memSize);
	uint32_t *h_result = (uint32_t *)malloc(memSize);
	uint32_t *h_CPUResult = (uint32_t *)malloc(memSize);

	uint32_t curr = 0;
    for (unsigned i = 0; i < numLimbs; i++)
    {
        curr = curr + 1 + (rand() % 10);
        h_data0[i] = curr;
    }

    curr = 0;
    for (unsigned i = 0; i < numLimbs; i++)
    {
        curr = curr + 1 + (rand() % 10);
        h_data1[i] = curr;
    }

    uint32_t *d_data0, *d_data1, *d_res;
    int *d_intRes;
    checkCudaErrors(cudaMalloc((void **) &d_data0, memSize));
    checkCudaErrors(cudaMalloc((void **) &d_data1, memSize));
    checkCudaErrors(cudaMalloc((void **) &d_res, memSize));

	checkCudaErrors(cudaMemcpy(d_data0, h_data0, memSize,
                               cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_data1, h_data1, memSize,
                               cudaMemcpyHostToDevice));

	dim3 grid(1);
	dim3 threads(1);

	//Test wide compare on GPU

	checkCudaErrors(cudaMalloc((void **) &d_intRes, sizeof(int)));

	int result = 50;

	cudaWideCompareTest<<< grid, threads >>>(numLimbs, d_data0, d_data0, d_intRes);

	checkCudaErrors(cudaMemcpy(&result, d_intRes, sizeof(int),
                               cudaMemcpyDeviceToHost));

	assert(result == 0);

	checkCudaErrors(cudaFree(d_intRes));

	//Test wide copy on GPU

	cudaWideCopyTest<<< grid, threads >>>(numLimbs, d_data0, d_res);

	checkCudaErrors(cudaMemcpy(h_result, d_res, memSize,
                               cudaMemcpyDeviceToHost));

	assert(wide_compare(numLimbs, h_result, h_data0) == 0);

	//Test wide xor on GPU

	cudaWideXorTest<<< grid, threads >>>(numLimbs, d_data0, d_data1, d_res);

	checkCudaErrors(cudaMemcpy(h_result, d_res, memSize,
                               cudaMemcpyDeviceToHost));

	wide_xor(numLimbs, h_CPUResult, h_data0, h_data1);

	assert(wide_compare(numLimbs, h_result, h_CPUResult) == 0);

	//Test wide add uint32_t*s on GPU
	uint32_t gpuCarry, *d_gpuCarry;
	checkCudaErrors(cudaMalloc((void **)&d_gpuCarry, sizeof(uint32_t)));

	cudaWideAddTest<<< grid, threads >>>(numLimbs, d_data0, d_data1, d_res, d_gpuCarry);

	getLastCudaError("Kernel execution failed");

	checkCudaErrors(cudaMemcpy(h_result, d_res, memSize,
                               cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&gpuCarry, d_gpuCarry, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

	uint32_t cpuCarry = wide_add(numLimbs, h_CPUResult, h_data0, h_data1);

	assert(cpuCarry == gpuCarry);
	assert(wide_compare(numLimbs, h_result, h_CPUResult) == 0);

	//Test wide add uint32_t* + uint64_t on GPU

	uint64_t randomAdd = 1 + (rand() % 10);

	cudaWideAdd64Test<<< grid, threads>>>(numLimbs, d_data0, randomAdd, d_res, d_gpuCarry);

	getLastCudaError("Kernel execution failed");

	checkCudaErrors(cudaMemcpy(h_result, d_res, memSize,
                               cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&gpuCarry, d_gpuCarry, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));

	cpuCarry = wide_add(numLimbs, h_CPUResult, h_data0, randomAdd);

	assert(cpuCarry == gpuCarry);
	assert(wide_compare(numLimbs, h_result, h_CPUResult) == 0);

	checkCudaErrors(cudaFree(d_gpuCarry));

	//

	checkCudaErrors(cudaFree(d_data0));
    checkCudaErrors(cudaFree(d_data1));
    checkCudaErrors(cudaFree(d_res));

	free(h_data0);
	free(h_data1);
	free(h_result);
	free(h_CPUResult);

	return true;
}

uint64_t gpuHasherTest(uint8_t *chainData, size_t _len)
{

	dim3 grid(1);
	dim3 threads(1);

	uint8_t *d_chainData;
	uint64_t *d_result;
	checkCudaErrors(cudaMalloc((void **) &d_chainData, sizeof(uint8_t) * _len));
	checkCudaErrors(cudaMalloc((void **) &d_result, sizeof(uint64_t)));
	checkCudaErrors(cudaMemcpy(d_chainData, chainData, sizeof(uint8_t) * _len, cudaMemcpyHostToDevice));

	cudaHasher<<< grid, threads >>>(d_chainData, _len, d_result);

	uint64_t result;
	checkCudaErrors(cudaMemcpy(&result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));

	return result;
}