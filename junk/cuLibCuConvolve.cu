#include "cuLibCuConvolve.cu.decl"

#include "kernels/convolveKernelConstants.h"

// extern "C"
int convolve(float2 *out_r_D, float2 *in_a_D, in_b_D, unsigned int size)
{
	if (size % CU_CONVOLVE_BLOCKSIZE)
	{
		printf("size %d is incompatible with CU_CONVOLVE_BLOCKSIZE: %d\n\t- some data will be missing\n", size, CU_CONVOLVE_BLOCKSIZE);
	}
	dim3 blocksInGrid(size / CU_CONVOLVE_BLOCKSIZE);
	dim3 threadsInBlock(CU_CONVOLVE_BLOCKSIZE);
	
	CUDA_ComplexM_tZero<<<blocksInGrid, threadsInBlock>>>(out_r_D, in_a_D, in_b_D);
}

