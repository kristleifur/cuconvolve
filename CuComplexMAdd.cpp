#include "CuCOmplexMAdd.hpp"

using namespace std;
using namespace boost;

CuComplexMAdder::CuComplexMAdder(size_t size)
{
	this.size = size;

	out_r_D, in_a_D, in_b_D = NULL;
		// in_a_D, in_b_D;

	cudaError_t mallocErrorReturn;

	cudaMalloc((void**) &out_r_D, size * sizeof(float2));
	cudaThreadSynchronize();
	mallocErrorReturn = cudaGetLastError();
	if (mallocErrorReturn != cudaSuccess)
	{
		cerr << format("CuComplexMAdder.out_r_D cudaMalloc error - requested %1% size array") % size << endl;
		//TODO: translate error code to sth. meaningful - see nvidia headers 
	}
	
	cudaMalloc((void**) &in_a_D, size * sizeof(float2));
	cudaThreadSynchronize();
	mallocErrorReturn = cudaGetLastError();
	if (mallocErrorReturn != cudaSuccess)
	{
		cerr << format("CuComplexMAdder.in_a_D cudaMalloc error - requested %1% size array") % size << endl;
	}
	cudaMalloc((void**) &in_b_D, size * sizeof(float2));
	cudaThreadSynchronize();
	mallocErrorReturn = cudaGetLastError();
	if (mallocErrorReturn != cudaSuccess)
	{
		cerr << format("CuComplexMAdder.in_b_D cudaMalloc error - requested %1% size array") % size << endl;
	}		
}

int CuComplexMAdder::complexMAdd(float *out_r_RI, float* in_a_RI, float *in_b_RI, size_t size)
{
	//TODO: assert this.size >= size
	//TODO: assert buffers are at least not 0
	cudaMemcpy(&out_r_D, out_r_RI, size * sizeof(float2), cudaMemcpyHostToDevice) //TODO: async memcpy ! -> fit to GPUWorker approach
	cudaMemcpy(&in_a_D, in_a_RI, size * sizeof(float2), cudaMemcpyHostToDevice)
	cudaMemcpy(&in_b_D, in_b_RI, size * sizeof(float2), cudaMemcpyHostToDevice)
	
	cudaError_t mAddErrorReturn = gpu_complexMAdd(float2 * out_r_D, float2 * in_a_D, float2 * in_b_D, int size);

	return 0;
}

int CuComplexMAdder::complexM(float *out_r_RI, float* in_a_RI, float *in_b_RI, size_t size)
{
	//TODO: assert this.size >= size
	//TODO: assert buffers are at least not 0
	cudaMemcpy(&out_r_D, out_r_RI, size * sizeof(float2), cudaMemcpyHostToDevice) //TODO: async memcpy ! -> fit to GPUWorker approach
	cudaMemcpy(&in_a_D, in_a_RI, size * sizeof(float2), cudaMemcpyHostToDevice)
	cudaMemcpy(&in_b_D, in_b_RI, size * sizeof(float2), cudaMemcpyHostToDevice)
	
	cudaError_t mAddErrorReturn = gpu_complexM(float2 * out_r_D, float2 * in_a_D, float2 * in_b_D, int size);

	return 0;
}

