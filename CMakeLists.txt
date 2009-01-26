# Note set CUDA_BUILD_TYPE to either "Emulation" or "Device" to switch between generation 
# modes. 

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

PROJECT(cuConvolve)

INCLUDE(${CMAKE_SOURCE_DIR}/CMake/cuda/FindCuda.cmake)

SUBDIRS(
#  includes
#   kernels
  )

CUDA_INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}
	kernels
)

CUDA_ADD_LIBRARY(libCuConvolve
	libCuConvolve.cpp
)

ADD_EXECUTABLE(cuConvolve
	${CUDA_FILES}
	cuConvolve.cpp
)

TARGET_LINK_LIBRARIES(cuConvolve
	libCuConvolve
)

CUDA_COMPILE(CUDA_FILES kernels/convolveKernel.cu)

TARGET_LINK_LIBRARIES(cuConvolve
	${CUDA_TARGET_LINK}
)

