#ifndef CONVOLVE_KERNEL_CUH
#define CONVOLVE_KERNEL_CUH

// #include <assert.h>

#include "kernelHelpers.cu.h" // IMUL etc.


#define CU_CONVOLVE_BLOCKSIZE 128
	// what is best for speed ?? test


//! Kernel drivers
cudaError_t gpu_complexMAdd(float2 * out_result_D, float2 * in_a_D, float2 * in_b_D, int size);

cudaError_t gpu_complexMAdd_stupid(float2 * out_result_D, float2 * in_a_D, float2 * in_b_D, int size);

cudaError_t gpu_complexM(float2 * out_result_D, float2 * in_a_D, float2 * in_b_D, int size);

cudaError_t gpu_complexM_stupid(float2 * out_result_D, float2 * in_a_D, float2 * in_b_D, int size);


#endif

