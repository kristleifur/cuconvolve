#ifndef CU_COMPLEX_M_ADD_HPP
#define CU_COMPLEX_M_ADD_HPP

#include <cuda_runtime_api.h> // defines the very convenient float2 type

#include "complexM_plusAdd.cuh"
#include "gpu_settings.h" // remember to link to gpu_settings.cu

#include <string>
#include <stdio>
#include <boost/format.hpp> // super nice printf-style formatting for << string stuff

class CuComplexMAdder
{
private:
	bool flop;
	size_t size;
	float2 * out_r_D, in_a_D, in_b_D; // device memory
public:
		//TODO: ? determine: format as compatible "doubled" float arrays ([r][i][r][i]...) or convenient float2 arrays ([ri][ri]...) ?
		//TODO: ? determine how to handle multiple devices and streams
	int complexMAdd(float *out_r_RI, float* in_a_RI, float *in_b_RI, size_t size);
	int complexM(float *out_r_RI, float* in_a_RI, float *in_b_RI, size_t size);
	public CuComplexMAdder(size_t size);
}

#endif // CU_COMPLEX_M_ADD_HPP

