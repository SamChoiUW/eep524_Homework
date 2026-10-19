#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <assert.h>

inline static cudaError_t checkCuda(cudaError_t result, char* srcStr) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error (src: %s): %s\n", srcStr, cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
	return result;
}