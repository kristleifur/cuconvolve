#ifndef LIB_CU_CONVOLVE
#define LIB_CU_CONVOLVE

#include <cuda_runtime_api.h>

#include "cuLibCuConvolve.cu.decl"

using namespace std;

class CuConvolver
{
private:
	bool flop;
public:
	int convolve(float *out_r_RI, float* in_a_RI, float *in_b_RI);
}

#endif // LIB_CU_CONVOLVE
