#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef PROFILE_MODE
#define PROFILE_MODE 0
#endif

#if PROFILE_MODE
#define HOST_PRINTF(...) do { if (0) printf(__VA_ARGS__); } while (0)
#else
#define HOST_PRINTF(...) printf(__VA_ARGS__)
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,   \
                    __LINE__, cudaGetErrorString(err_));                       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
