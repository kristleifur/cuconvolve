#include "complexM_plusAdd.cuh"
#include "gpu_settings.h"
#include <assert.h>

// CUDA kernel, does not run on x86 host
extern "C"
__global__ void CUDA_ComplexMAdd( float2 *result, float2 *a, float2 *b )
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

// CUDA kernel, does not run on x86 host
// "stupid" does not do special case calc for array[0], may be faster but step 0 must be done on CPU afterwards
extern "C"
__global__ void CUDA_ComplexMAdd_stupid( float2 *result, float2 *a, float2 *b )
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

// CUDA kernel, does not run on x86 host
extern "C"
__global__ void CUDA_ComplexM( float2 *result, float2 *a, float2 *b )
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

// CUDA kernel, does not run on x86 host
// "stupid" does not do special case calc for array[0], may be faster but step 0 must be done on CPU afterwards
extern "C"
__global__ void CUDA_ComplexM_stupid( float2 *result, float2 *a, float2 *b )
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


// host-callable kernel driver, implicitly async as there are no memcpy's or such
cudaError_t gpu_complexMAdd(float2 *out_result, float2 *in_a, float2 *in_b, int size)
{
	// todo: assert arrays are OK and size is good
	
	// initialise thread blocks and block grid
	dim3 blocksInGrid(size / CU_CONVOLVE_BLOCKSIZE, 1, 1);
	dim3 threadsInBlock(CU_CONVOLVE_BLOCKSIZE, 1, 1);

	CUDA_ComplexMAdd<<<blocksInGrid, threadsInBlock>>>(out_result, in_a, in_b);

	if (!g_gpu_error_checking)
	{
		return cudaSuccess;
	}
	else
	{
		cudaThreadSynchronize();
		return cudaGetLastError();
	}
}


// host-callable kernel driver, implicitly async as there are no memcpy's or such
// "stupid" does not do special case calc for array[0], may be faster but step 0 must be done on CPU afterwards
cudaError_t gpu_complexMAdd_stupid(float2 *out_result, float2 *in_a, float2 *in_b, int size)
{
	// todo: assert arrays are OK and size is good
	
	// initialise thread blocks and block grid
	dim3 blocksInGrid(size / CU_CONVOLVE_BLOCKSIZE, 1, 1);
	dim3 threadsInBlock(CU_CONVOLVE_BLOCKSIZE, 1, 1);

	CUDA_ComplexMAdd_stupid<<<blocksInGrid, threadsInBlock>>>(out_result, in_a, in_b);

	if (!g_gpu_error_checking)
	{
		return cudaSuccess;
	}
	else
	{
		cudaThreadSynchronize();
		return cudaGetLastError();
	}
}


// host-callable kernel driver, implicitly async as there are no memcpy's or such
cudaError_t gpu_complexM(float2 *out_result, float2 *in_a, float2 *in_b, int size)
{
	// todo: assert arrays are OK and size is good
	
	// initialise thread blocks and block grid
	dim3 blocksInGrid(size / CU_CONVOLVE_BLOCKSIZE, 1, 1);
	dim3 threadsInBlock(CU_CONVOLVE_BLOCKSIZE, 1, 1);

	CUDA_ComplexM<<<blocksInGrid, threadsInBlock>>>(out_result, in_a, in_b);

	if (!g_gpu_error_checking)
	{
		return cudaSuccess;
	}
	else
	{
		cudaThreadSynchronize();
		return cudaGetLastError();
	}
}


// host-callable kernel driver, implicitly async as there are no memcpy's or such
// "stupid" does not do special case calc for array[0], may be faster but step 0 must be done on CPU afterwards
cudaError_t gpu_complexM_stupid(float2 *out_result, float2 *in_a, float2 *in_b, int size)
{
	// todo: assert arrays are OK and size is good
	
	// initialise thread blocks and block grid
	dim3 blocksInGrid(size / CU_CONVOLVE_BLOCKSIZE, 1, 1);
	dim3 threadsInBlock(CU_CONVOLVE_BLOCKSIZE, 1, 1);

	CUDA_ComplexM_stupid<<<blocksInGrid, threadsInBlock>>>(out_result, in_a, in_b);

	if (!g_gpu_error_checking)
	{
		return cudaSuccess;
	}
	else
	{
		cudaThreadSynchronize();
		return cudaGetLastError();
	}
}


