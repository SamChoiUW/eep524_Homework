#include "common.h"
#include "divergence.h"
#include <vector>

__global__ void single_branch_kernel(const int* key, float* out, int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = (float)(i & 1023) * 0.001f;

    if (key[i] == 0) {
        for (int k = 0; k < iters; ++k)
            v = __sinf(v) * 1.0001f + 0.5f;
        out[i] = v;
    } else {
        for (int k = 0; k < iters; ++k)
            v = __cosf(v) * 0.9999f - 0.25f;
        out[i] = v + 1.0f;
    }
}

__global__ void nested_branch_kernel(const int* key, float* out, int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int   path = key[i];
    float v    = (float)(i & 1023) * 0.001f;

    if (path < 2) {
        if (path == 0) {
            for (int k = 0; k < iters; ++k)
                v = __sinf(v) * 1.0001f + 0.5f;
            out[i] = v;
        } else {
            for (int k = 0; k < iters; ++k)
                v = __cosf(v) * 0.9999f - 0.25f;
            out[i] = v + 1.0f;
        }
    } else {
        if (path == 2) {
            for (int k = 0; k < iters; ++k)
                v = sqrtf(fabsf(v) + 1.0f) * 0.75f;
            out[i] = v + 2.0f;
        } else {
            for (int k = 0; k < iters; ++k)
                v = __expf(-fabsf(v)) + 0.125f;
            out[i] = v + 3.0f;
        }
    }
}

void runDivergence(bool nested, bool divergent, int iters)
{
    const int n        = 1 << 22;
    const int numPaths = nested ? 4 : 2;
    dim3 threadsPerBlock(256, 1, 1);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    std::vector<int> h_key(n);
    for (int i = 0; i < n; ++i)
        h_key[i] = divergent ? (i % numPaths)
                             : ((i / WARP_SIZE) % numPaths);

    int*   d_key = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_key, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_key, h_key.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto launch = [&] {
        if (nested)
            nested_branch_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_key, d_out, n, iters);
        else
            single_branch_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_key, d_out, n, iters);
    };

    if (!PROFILE_MODE) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float sample = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sample, d_out + n / 2, sizeof(float), cudaMemcpyDeviceToHost));

    HOST_PRINTF("%s / %s : %.3f ms (sample = %f)\n",
                nested ? "nested" : "single",
                divergent ? "divergent" : "non-divergent", ms, sample);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_key));
    CUDA_CHECK(cudaFree(d_out));
}
