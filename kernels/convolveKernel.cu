#include "convolveKernelConstants.h"
#include "kernelHelpers.cu.h" // IMUL etc.

extern "C"
__global__ void CUDA_ComplexMAdd( float2 *result, float2 *a, float2 b )
{
	unsigned int i = IMUL(blockIdx.x, CU_CONVOLVE_BLOCKSIZE) + threadIdx.x;
		// res_t is the result for thread i, like one step in the for(i < n) loop
		// a_t and b_t are the same
		// this is to get basic "caching" of .x and .y values for global mem R/W
	float2 res_t = result[blockIdx.x]; // to get += functionality as in "r_real[i] += ..."
	float2 a_t = a[i];
	float2 b_t = b[i];
	res_t.x += a_t.x * b_t.x  -  a_t.y * b_t.y;
	res_t.y += a_t.y * b_t.x  +  a_t.x * b_t.y;
	result[i] = res_t;
}

extern "C"
__global__ void CUDA_ComplexMAdd_tZero( float2 *result, float2 *a, float2 b )
{
	unsigned int i = IMUL(blockIdx.x, CU_CONVOLVE_BLOCKSIZE) + threadIdx.x;
		// res_t is the result for thread i, like one step in the for(i < n) loop
		// a_t and b_t are the same
		// this is to get basic "caching" of .x and .y values for global mem R/W
	float2 res_t = result[blockIdx.x]; // to get += functionality as in "r_real[i] += ..."
	float2 a_t = a[i];
	float2 b_t = b[i];
	if (i == 0)
	{
		res_t.x += a_t.x * b_t.x;
		res_t.y += a_t.y * b_t.y;
	} else {
		res_t.x += a_t.x * b_t.x  -  a_t.y * b_t.y;
		res_t.y += a_t.y * b_t.x  +  a_t.x * b_t.y;
	}
	result[i] = res_t;
}

extern "C"
__global__ void CUDA_ComplexM( float2 *result, float2 *a, float2 b )
{
	unsigned int i = IMUL(blockIdx.x, CU_CONVOLVE_BLOCKSIZE) + threadIdx.x;
		// res_t is the result for thread i, like one step in the for(i < n) loop
		// a_t and b_t are the same
		// this is to get basic "caching" of .x and .y values for global mem R/W
	float2 res_t;
	float2 a_t = a[i];
	float2 b_t = b[i];
	res_t.x = a_t.x * b_t.x  -  a_t.y * b_t.y;
	res_t.y = a_t.y * b_t.x  +  a_t.x * b_t.y;
	result[i] = res_t;
}

extern "C"
__global__ void CUDA_ComplexM_tZero( float2 *result, float2 *a, float2 b )
{
	unsigned int i = IMUL(blockIdx.x, CU_CONVOLVE_BLOCKSIZE) + threadIdx.x;
		// res_t is the result for thread i, like one step in the for(i < n) loop
		// a_t and b_t are the same
		// this is to get basic "caching" of .x and .y values for global mem R/W
	float2 res_t;
	float2 a_t = a[i];
	float2 b_t = b[i];
	if (i == 0)
	{
		res_t.x = a_t.x * b_t.x;
		res_t.y = a_t.y * b_t.y;
	} else {
		res_t.x = a_t.x * b_t.x  -  a_t.y * b_t.y;
		res_t.y = a_t.y * b_t.x  +  a_t.x * b_t.y;
	}
	result[i] = res_t;
}

