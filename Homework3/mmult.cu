#include "common.h"
#include "mmult.h"
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <functional>

void mmult_cpu(const float* h_M, const float* h_N, float* h_P, int J, int K, int L)
{
    std::vector<double> pval(L);
    for (int row = 0; row < J; ++row)
    {
        std::fill(pval.begin(), pval.end(), 0.0);
        for (int k = 0; k < K; ++k)
        {
            double m = h_M[(size_t)row * K + k];
            for (int col = 0; col < L; ++col)
                pval[col] += m * h_N[(size_t)k * L + col];
        }
        for (int col = 0; col < L; ++col)
            h_P[(size_t)row * L + col] = (float)pval[col];
    }
}

__global__ void mmult_gmem(const float* M, const float* N, float* P, int J, int K, int L)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < J) && (col < L))
    {
        float pval = 0;
        for (int k = 0; k < K; ++k) {
            pval += M[row * K + k] * N[k * L + col];
        }
        P[row * L + col] = pval;
    }
}

__global__ void mmult_tiled(const float* M, const float* N, float* P, int J, int K, int L)
{
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Pvalue = 0;

    for (int p = 0; p < (K - 1) / TILE_WIDTH + 1; ++p) {
        if (Row < J && p * TILE_WIDTH + tx < K) {
            ds_M[ty][tx] = M[Row * K + p * TILE_WIDTH + tx];
        } else {
            ds_M[ty][tx] = 0.0f;
        }
        if (p * TILE_WIDTH + ty < K && Col < L) {
            ds_N[ty][tx] = N[(p * TILE_WIDTH + ty) * L + Col];
        } else {
            ds_N[ty][tx] = 0.0f;
        }
        __syncthreads();

        if (Row < J && Col < L) {
            for (int i = 0; i < TILE_WIDTH; ++i)
                Pvalue += ds_M[ty][i] * ds_N[i][tx];
        }
        __syncthreads();
    }

    if (Row < J && Col < L)
        P[Row * L + Col] = Pvalue;
}

__global__ void transpose_smem(const float* in, float* out, int rows, int cols)
{
    __shared__ float ds_T[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];

    int tx = threadIdx.x; int ty = threadIdx.y;

    int col = blockIdx.x * TRANSPOSE_TILE + tx;
    int row = blockIdx.y * TRANSPOSE_TILE + ty;
    if (row < rows && col < cols)
        ds_T[ty][tx] = in[row * cols + col];
    __syncthreads();

    col = blockIdx.y * TRANSPOSE_TILE + tx;
    row = blockIdx.x * TRANSPOSE_TILE + ty;
    if (row < cols && col < rows)
        out[row * rows + col] = ds_T[tx][ty];
}

__global__ void mmult_gmem_NT(const float* M, const float* Nt, float* P, int J, int K, int L)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((row < J) && (col < L))
    {
        float pval = 0;
        for (int k = 0; k < K; ++k) {
            pval += M[row * K + k] * Nt[col * K + k];
        }
        P[row * L + col] = pval;
    }
}

__global__ void mmult_tiled_NT(const float* M, const float* Nt, float* P, int J, int K, int L)
{
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH + 1];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row   = by * blockDim.y + ty;
    int Col   = bx * blockDim.x + tx;
    int NtRow = bx * blockDim.x + ty;
    float Pvalue = 0;

    for (int p = 0; p < (K - 1) / TILE_WIDTH + 1; ++p) {
        if (Row < J && p * TILE_WIDTH + tx < K) {
            ds_M[ty][tx] = M[Row * K + p * TILE_WIDTH + tx];
        } else {
            ds_M[ty][tx] = 0.0f;
        }
        if (NtRow < L && p * TILE_WIDTH + tx < K) {
            ds_N[tx][ty] = Nt[NtRow * K + p * TILE_WIDTH + tx];
        } else {
            ds_N[tx][ty] = 0.0f;
        }
        __syncthreads();

        if (Row < J && Col < L) {
            for (int i = 0; i < TILE_WIDTH; ++i)
                Pvalue += ds_M[ty][i] * ds_N[i][tx];
        }
        __syncthreads();
    }

    if (Row < J && Col < L)
        P[Row * L + Col] = Pvalue;
}

double mmultFlops(int J, int K, int L)
{
    return 2.0 * J * (double)K * L;
}

double aiNaive(int J, int K, int L)
{
    double bytes = 4.0 * (2.0 * J * (double)K * L + (double)J * L);
    return mmultFlops(J, K, L) / bytes;
}

double aiTiled(int J, int K, int L, int tileWidth)
{
    double blockCols = (L + tileWidth - 1) / tileWidth;
    double blockRows = (J + tileWidth - 1) / tileWidth;
    double bytes = 4.0 * (blockCols * J * (double)K
                        + blockRows * K * (double)L
                        + (double)J * L);
    return mmultFlops(J, K, L) / bytes;
}

struct Opts {
    bool   warm          = !PROFILE_MODE;
    bool   doNT          = true;
    double tol           = 1e-3;
};

double maxErrFraction(const float* h_Pcpu, const float* h_Pgpu, size_t n, size_t* where)
{
    double worst = 0.0;
    size_t w = 0;
    for (size_t i = 0; i < n; ++i) {
        double ref   = h_Pcpu[i];
        double denom = fabs(ref) > 1e-20 ? fabs(ref) : 1.0;
        double e     = fabs(ref - h_Pgpu[i]) / denom;
        if (!(e <= worst)) {
            worst = std::isnan(e) ? INFINITY : e;
            w = i;
            if (std::isinf(worst)) break;
        }
    }
    if (where) *where = w;
    return worst;
}

int theoreticalOccupancyPct(const void* kernel, int threadsPerBlock, const cudaDeviceProp& prop)
{
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, threadsPerBlock, 0));
    int warpsPerBlock = (threadsPerBlock + prop.warpSize - 1) / prop.warpSize;
    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    return (int)lround(100.0 * blocksPerSM * warpsPerBlock / maxWarpsPerSM);
}

struct Ctx {
    int J, K, L;
    const float *d_M, *d_N, *d_Nt;
    float* d_P;
    std::vector<float>* h_P;
    const std::vector<float>* h_Pcpu;
    const Opts* o;
    const cudaDeviceProp* prop;
    cudaEvent_t start = nullptr, stop = nullptr;
    int failures = 0;
};

float timeLaunch(Ctx& c, const std::function<void()>& launch)
{
    if (c.o->warm) {
        launch();
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemset(c.d_P, 0, (size_t)c.J * c.L * sizeof(float)));
    CUDA_CHECK(cudaEventRecord(c.start));
    launch();
    CUDA_CHECK(cudaEventRecord(c.stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(c.stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, c.start, c.stop));
    return ms;
}

void report(Ctx& c, const char* name, dim3 block, float ms, double ai, int occ, float transposeMs = 0.f)
{
    if (PROFILE_MODE) return;

    size_t n = (size_t)c.J * c.L;
    CUDA_CHECK(cudaMemcpy(c.h_P->data(), c.d_P, n * sizeof(float), cudaMemcpyDeviceToHost));
    size_t where;
    double err = maxErrFraction(c.h_Pcpu->data(), c.h_P->data(), n, &where);
    bool ok = err <= c.o->tol;
    if (!ok) ++c.failures;
    double gflops = mmultFlops(c.J, c.K, c.L) / (ms * 1e-3) / 1e9;

    HOST_PRINTF("  %-17s (%2u,%2u,%u) %9.3f ms %8.1f GFLOP/s  AI %5.2f FLOP/B  occ %3d%%  maxErrFrac %.3e  %s",
                name, block.x, block.y, block.z, ms, gflops, ai, occ, err, ok ? "PASS" : "FAIL");
    if (!ok)
        HOST_PRINTF(" @(%zu,%zu)", where / c.L, where % c.L);
    if (transposeMs > 0.f)
        HOST_PRINTF("   [+transpose %.3f ms -> total %.3f ms]", transposeMs, ms + transposeMs);
    HOST_PRINTF("\n");
}

void runTiled(Ctx& c, float transposeMs)
{
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 blocksPerGrid((c.L + TILE_WIDTH - 1) / TILE_WIDTH, (c.J + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    int  nThreads = TILE_WIDTH * TILE_WIDTH;
    double ai = aiTiled(c.J, c.K, c.L, TILE_WIDTH);
    char name[32];

    float ms = timeLaunch(c, [&] {
        mmult_tiled<<<blocksPerGrid, threadsPerBlock>>>(c.d_M, c.d_N, c.d_P, c.J, c.K, c.L);
    });
    snprintf(name, sizeof name, "tiled TW=%d", TILE_WIDTH);
    report(c, name, threadsPerBlock, ms, ai,
           theoreticalOccupancyPct((const void*)mmult_tiled, nThreads, *c.prop));

    if (c.o->doNT) {
        ms = timeLaunch(c, [&] {
            mmult_tiled_NT<<<blocksPerGrid, threadsPerBlock>>>(c.d_M, c.d_Nt, c.d_P, c.J, c.K, c.L);
        });
        snprintf(name, sizeof name, "tiled_NT TW=%d", TILE_WIDTH);
        report(c, name, threadsPerBlock, ms, ai,
               theoreticalOccupancyPct((const void*)mmult_tiled_NT, nThreads, *c.prop),
               transposeMs);
    }
}

int runCase(int J, int K, int L, const Opts& o, const cudaDeviceProp& prop)
{
    HOST_PRINTF("\n=== P: %d x %d   M: %d x %d   N: %d x %d   (k = %d) ===\n", J, L, J, K, K, L, K);

    std::vector<float> h_M((size_t)J * K), h_N((size_t)K * L);
    std::vector<float> h_Pcpu((size_t)J * L), h_P((size_t)J * L);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : h_M) v = dist(rng);
    for (auto& v : h_N) v = dist(rng);

    if (!PROFILE_MODE) {
        auto t0 = std::chrono::high_resolution_clock::now();
        mmult_cpu(h_M.data(), h_N.data(), h_Pcpu.data(), J, K, L);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        HOST_PRINTF("  %-17s %19.3f ms %8.1f GFLOP/s\n", "CPU serial", cpuMs,
                    mmultFlops(J, K, L) / (cpuMs * 1e-3) / 1e9);
    }

    float *d_M, *d_N, *d_Nt = nullptr, *d_P;
    CUDA_CHECK(cudaMalloc(&d_M, h_M.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_N, h_N.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P, h_P.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), h_M.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N.data(), h_N.size() * sizeof(float), cudaMemcpyHostToDevice));

    Ctx c{J, K, L, d_M, d_N, nullptr, d_P, &h_P, &h_Pcpu, &o, &prop};
    CUDA_CHECK(cudaEventCreate(&c.start));
    CUDA_CHECK(cudaEventCreate(&c.stop));

    dim3 naiveBlock(16, 16, 1);
    dim3 naiveGrid((L + naiveBlock.x - 1) / naiveBlock.x, (J + naiveBlock.y - 1) / naiveBlock.y, 1);
    {
        float ms = timeLaunch(c, [&] { mmult_gmem<<<naiveGrid, naiveBlock>>>(d_M, d_N, d_P, J, K, L); });
        report(c, "naive v0 (gmem)", naiveBlock, ms, aiNaive(J, K, L),
               theoreticalOccupancyPct((const void*)mmult_gmem, naiveBlock.x * naiveBlock.y, prop));
    }

    float transposeMs = 0.f;
    if (o.doNT) {
        CUDA_CHECK(cudaMalloc(&d_Nt, h_N.size() * sizeof(float)));
        c.d_Nt = d_Nt;
        dim3 tBlock(TRANSPOSE_TILE, TRANSPOSE_TILE, 1);
        dim3 tGrid((L + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE, (K + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE, 1);
        if (o.warm) transpose_smem<<<tGrid, tBlock>>>(d_N, d_Nt, K, L);
        CUDA_CHECK(cudaEventRecord(c.start));
        transpose_smem<<<tGrid, tBlock>>>(d_N, d_Nt, K, L);
        CUDA_CHECK(cudaEventRecord(c.stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(c.stop));
        CUDA_CHECK(cudaEventElapsedTime(&transposeMs, c.start, c.stop));

        float ms = timeLaunch(c, [&] { mmult_gmem_NT<<<naiveGrid, naiveBlock>>>(d_M, d_Nt, d_P, J, K, L); });
        report(c, "naive_NT (gmem)", naiveBlock, ms, aiNaive(J, K, L),
               theoreticalOccupancyPct((const void*)mmult_gmem_NT, naiveBlock.x * naiveBlock.y, prop),
               transposeMs);
    }

    runTiled(c, transposeMs);

    CUDA_CHECK(cudaEventDestroy(c.start));
    CUDA_CHECK(cudaEventDestroy(c.stop));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));
    if (d_Nt) CUDA_CHECK(cudaFree(d_Nt));
    return c.failures;
}

int runMmult()
{
    Opts o;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    HOST_PRINTF("Device: %s  (sm_%d%d, %d SMs, max %d threads/SM, max %d threads/block, "
                "%zu KB smem/SM, %d regs/SM)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerBlock,
                prop.sharedMemPerMultiprocessor / 1024, prop.regsPerMultiprocessor);
    HOST_PRINTF("Tolerance on max abs[Pcpu - Pgpu] / Pcpu: %.1e\n", o.tol);
    HOST_PRINTF("AI = FLOPs / GMEM bytes, no-cache model (Lecture 3)\n");

    int failures = 0;
    const int cases[3][3] = { {100, 100, 100}, {1000, 1000, 1000}, {2000, 2500, 3000} };
    for (int i = 0; i < 3; ++i)
        failures += runCase(cases[i][0], cases[i][1], cases[i][2], o, prop);

    HOST_PRINTF("\n%s (%d failing kernel run%s)\n",
                failures ? "VERIFICATION FAILED" : "ALL KERNELS VERIFIED",
                failures, failures == 1 ? "" : "s");
    return failures ? 2 : 0;
}
