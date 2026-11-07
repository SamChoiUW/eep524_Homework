// divergence.cu
// Control-divergence exercise: single-branch and nested-branch kernels.
//
// The kernel code is IDENTICAL for the divergent and non-divergent runs.
// Only the input "key" grid changes, so any difference NCU reports comes
// from divergence alone (same SASS, same memory traffic).
//
//   Non-divergent key:  key[i] = (i / 32) % P   -> all 32 threads of a warp
//                                                  take the same path
//   Divergent key:      key[i] =  i       % P   -> threads in one warp take
//                                                  P different paths
//
// Usage:   divergence <single|nested> <div|nodiv> [iters]
//   e.g.   divergence single div
//          divergence nested nodiv 512
//
// Build (command line):  nvcc -O3 -lineinfo -o divergence divergence.cu
// Build (Visual Studio): Release | x64 configuration (see notes).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Each branch body is a loop with a RUNTIME trip count and a different
// instruction mix. That keeps the compiler from:
//   (a) turning the if/else into a predicated select (if-conversion), and
//   (b) merging the branches into one common code path.
// So the hardware really has to serialize the paths when a warp diverges.

// ---------------------------------------------------------------------------
// 1. Single branch (control-flow bifurcation): 2 paths
// ---------------------------------------------------------------------------
__global__ void singleBranchKernel(const int* __restrict__ key,
    float* __restrict__ out,
    int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float v = (float)(i & 1023) * 0.001f;

    if (key[i] == 0) {                       // path A
        for (int k = 0; k < iters; ++k)
            v = __sinf(v) * 1.0001f + 0.5f;
        out[i] = v;                          // GMEM write in branch A
    }
    else {                                 // path B
        for (int k = 0; k < iters; ++k)
            v = __cosf(v) * 0.9999f - 0.25f;
        out[i] = v + 1.0f;                   // GMEM write in branch B
    }
}

// ---------------------------------------------------------------------------
// 2. Nested branches: 4 distinct control paths per warp
// ---------------------------------------------------------------------------
__global__ void nestedBranchKernel(const int* __restrict__ key,
    float* __restrict__ out,
    int n, int iters)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int   k0 = key[i];
    float v = (float)(i & 1023) * 0.001f;

    if (k0 < 2) {
        if (k0 == 0) {                       // path A
            for (int k = 0; k < iters; ++k)
                v = __sinf(v) * 1.0001f + 0.5f;
            out[i] = v;
        }
        else {                             // path B
            for (int k = 0; k < iters; ++k)
                v = __cosf(v) * 0.9999f - 0.25f;
            out[i] = v + 1.0f;
        }
    }
    else {
        if (k0 == 2) {                       // path C
            for (int k = 0; k < iters; ++k)
                v = sqrtf(fabsf(v) + 1.0f) * 0.75f;
            out[i] = v + 2.0f;
        }
        else {                             // path D
            for (int k = 0; k < iters; ++k)
                v = __expf(-fabsf(v)) + 0.125f;
            out[i] = v + 3.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------
static void usage(const char* prog)
{
    fprintf(stderr, "Usage: %s <single|nested> <div|nodiv> [iters]\n", prog);
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
    if (argc < 3) usage(argv[0]);

    bool nested = (strcmp(argv[1], "nested") == 0);
    bool divergent = (strcmp(argv[2], "div") == 0);
    if (!nested && strcmp(argv[1], "single") != 0) usage(argv[0]);
    if (!divergent && strcmp(argv[2], "nodiv") != 0) usage(argv[0]);
    int iters = (argc > 3) ? atoi(argv[3]) : 256;

    const int N = 1 << 22;          // ~4M threads
    const int BLOCK = 256;              // 8 warps per block
    const int GRID = (N + BLOCK - 1) / BLOCK;   // 16384 blocks
    const int WARP = 32;
    const int numPaths = nested ? 4 : 2;

    // Build the input test grid that forces the desired behaviour.
    std::vector<int> hKey(N);
    for (int i = 0; i < N; ++i)
        hKey[i] = divergent ? (i % numPaths)            // paths mixed in a warp
        : ((i / WARP) % numPaths);  // one path per warp

    int* dKey = nullptr;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dKey, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dOut, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dKey, hKey.data(), N * sizeof(int),
        cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // One launch only, so each NCU report holds exactly one kernel.
    CUDA_CHECK(cudaEventRecord(start));
    if (nested)
        nestedBranchKernel << <GRID, BLOCK >> > (dKey, dOut, N, iters);
    else
        singleBranchKernel << <GRID, BLOCK >> > (dKey, dOut, N, iters);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copy back a value so the work is observably used.
    float sample = 0.0f;
    CUDA_CHECK(cudaMemcpy(&sample, dOut + (N / 2), sizeof(float),
        cudaMemcpyDeviceToHost));

    // Host-side output only (there are no device printfs). Comment this
    // line out if your instructor wants zero output during NCU runs.
    fprintf(stdout, "%s / %s : %.3f ms (sample=%f)\n",
        nested ? "nested" : "single",
        divergent ? "divergent" : "non-divergent", ms, sample);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dKey));
    CUDA_CHECK(cudaFree(dOut));
    return 0;
}
// mmult.cu  --  MMULT variants: CPU reference, naive GPU (v0), shared-memory tiled GPU
//
//   P = M * N      M: J x K     N: K x L     P: J x L     (all row-major, float)
//
// Kernels
//   mmNaive          : v0, every operand read straight from global memory
//   mmTiled<TW>      : PMPP 5.4-5.5 phased/tiled kernel with boundary checks.
//                      Works for any J,K,L (non-square, not multiples of TW).
//                      Tile width == thread-block width (TW x TW x 1).
//   (additional question: N transposed to Nt (L x K) so both operands are row-major)
//   transposeKernel  : coalesced shared-memory transpose N -> Nt
//   mmNaiveNT        : naive kernel reading Nt
//   mmTiledNT<TW>    : tiled kernel reading Nt
//
// Usage
//   mmult                      run all 3 assignment cases, all tile widths
//   mmult -case 3              only case 3 (P 2000x3000, k=2500)
//   mmult J K L                custom sizes  (P is J x L, inner dim K)
//   options:  -t TW            only one tile width (8, 12, 16 or 32)
//             -nowarm          no warm-up launch (one launch per kernel -> cleaner Nsight list)
//             -nont            skip the transposed-N variants
//             -tol x           error-fraction tolerance (default 1e-3)
//
// Build (VS: Release config). For the FMA experiment add  --fmad=false  in
//   Project Properties -> CUDA C/C++ -> Command Line -> Additional Options.
// Command line equivalent:  nvcc -O3 -arch=native mmult.cu -o mmult   [--fmad=false]

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
                    __LINE__, cudaGetErrorString(err__));                         \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// CPU serial reference.  i-k-j loop order (streams through rows of N, cache
// friendly) and double accumulation so it is a trustworthy "ground truth".
// ---------------------------------------------------------------------------
void mmCPU(const float* M, const float* N, float* P, int J, int K, int L)
{
    std::vector<double> acc(L);
    for (int i = 0; i < J; ++i) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int m = 0; m < K; ++m) {
            const double a = M[(size_t)i * K + m];
            const float* nrow = N + (size_t)m * L;
            for (int c = 0; c < L; ++c) acc[c] += a * nrow[c];
        }
        for (int c = 0; c < L; ++c) P[(size_t)i * L + c] = (float)acc[c];
    }
}

// ---------------------------------------------------------------------------
// GPU v0: naive, all global memory
// ---------------------------------------------------------------------------
__global__ void mmNaive(const float* __restrict__ M, const float* __restrict__ N,
                        float* __restrict__ P, int J, int K, int L)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < J && col < L) {
        float sum = 0.0f;
        for (int m = 0; m < K; ++m)
            sum += M[row * K + m] * N[m * L + col];
        P[row * L + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// GPU tiled / phased kernel with boundary checks (PMPP Fig. 5.13 style).
// Out-of-range tile elements are loaded as 0 so they add nothing to the dot
// product; threads outside P still help load tiles but do not write.
// ---------------------------------------------------------------------------
template <int TW>
__global__ void mmTiled(const float* __restrict__ M, const float* __restrict__ N,
                        float* __restrict__ P, int J, int K, int L)
{
    __shared__ float Ms[TW][TW];
    __shared__ float Ns[TW][TW];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TW + ty;
    const int col = blockIdx.x * TW + tx;

    float acc = 0.0f;
    const int phases = (K + TW - 1) / TW;

    for (int ph = 0; ph < phases; ++ph) {
        const int mCol = ph * TW + tx;   // column of M this thread loads
        const int nRow = ph * TW + ty;   // row of N this thread loads

        Ms[ty][tx] = (row < J && mCol < K) ? M[row * K + mCol] : 0.0f;
        Ns[ty][tx] = (nRow < K && col < L) ? N[nRow * L + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TW; ++i)
            acc += Ms[ty][i] * Ns[i][tx];
        __syncthreads();
    }

    if (row < J && col < L) P[row * L + col] = acc;
}

// ---------------------------------------------------------------------------
// Additional question: transpose N (K x L) -> Nt (L x K)
// ---------------------------------------------------------------------------
constexpr int TD = 32;   // transpose tile
__global__ void transposeKernel(const float* __restrict__ in, float* __restrict__ out,
                                int rows, int cols)
{
    __shared__ float t[TD][TD + 1];          // +1 pad avoids bank conflicts
    int x = blockIdx.x * TD + threadIdx.x;   // column in "in"
    int y = blockIdx.y * TD + threadIdx.y;   // row in "in"
    if (x < cols && y < rows) t[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    x = blockIdx.y * TD + threadIdx.x;       // column in "out" (= row in "in")
    y = blockIdx.x * TD + threadIdx.y;       // row in "out"    (= col in "in")
    if (x < rows && y < cols) out[y * rows + x] = t[threadIdx.x][threadIdx.y];
}

// Naive with Nt: thread (row,col) walks row 'col' of Nt. Adjacent threads in a
// warp now read addresses K floats apart -> uncoalesced (expected to be slower).
__global__ void mmNaiveNT(const float* __restrict__ M, const float* __restrict__ Nt,
                          float* __restrict__ P, int J, int K, int L)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < J && col < L) {
        float sum = 0.0f;
        for (int m = 0; m < K; ++m)
            sum += M[row * K + m] * Nt[col * K + m];
        P[row * L + col] = sum;
    }
}

// Tiled with Nt: both tile loads are row-wise and coalesced. The Nt tile is
// written transposed into shared memory (padded to avoid bank conflicts) so the
// inner loop is identical to mmTiled.
template <int TW>
__global__ void mmTiledNT(const float* __restrict__ M, const float* __restrict__ Nt,
                          float* __restrict__ P, int J, int K, int L)
{
    __shared__ float Ms[TW][TW];
    __shared__ float Ns[TW][TW + 1];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row  = blockIdx.y * TW + ty;
    const int col  = blockIdx.x * TW + tx;
    const int nCol = blockIdx.x * TW + ty;   // row of Nt (= column of N) this thread loads

    float acc = 0.0f;
    const int phases = (K + TW - 1) / TW;

    for (int ph = 0; ph < phases; ++ph) {
        const int k = ph * TW + tx;
        Ms[ty][tx] = (row  < J && k < K) ? M[row * K + k]   : 0.0f;
        Ns[tx][ty] = (nCol < L && k < K) ? Nt[nCol * K + k] : 0.0f;  // Ns[i][c] = N[ph*TW+i][bx*TW+c]
        __syncthreads();

#pragma unroll
        for (int i = 0; i < TW; ++i)
            acc += Ms[ty][i] * Ns[i][tx];
        __syncthreads();
    }

    if (row < J && col < L) P[row * L + col] = acc;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
struct Opts {
    int  onlyTW  = 0;
    bool warm    = true;
    bool doNT    = true;
    double tol   = 1e-3;
};

// max over elements of |Pcpu - Pgpu| / |Pcpu|
double maxErrFraction(const float* ref, const float* gpu, size_t n, size_t* where)
{
    double worst = 0.0; size_t w = 0;
    for (size_t i = 0; i < n; ++i) {
        double r = ref[i], g = gpu[i];
        double denom = fabs(r) > 1e-20 ? fabs(r) : 1.0;   // guard divide-by-zero
        double e = fabs(r - g) / denom;
        if (!(e <= worst)) {                               // also catches NaN
            worst = std::isnan(e) ? INFINITY : e; w = i;
            if (std::isinf(worst)) break;
        }
    }
    if (where) *where = w;
    return worst;
}

int theoreticalOccupancyPct(const void* kernel, int threadsPerBlock, const cudaDeviceProp& p)
{
    int blocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel, threadsPerBlock, 0));
    int warpsPerBlock = (threadsPerBlock + p.warpSize - 1) / p.warpSize;
    int maxWarps = p.maxThreadsPerMultiProcessor / p.warpSize;
    return (int)lround(100.0 * blocksPerSM * warpsPerBlock / maxWarps);
}

struct Ctx {
    int J, K, L;
    const float *dM, *dN, *dNt;
    float* dP;
    std::vector<float>* hP;
    const std::vector<float>* ref;
    const Opts* o;
    const cudaDeviceProp* prop;
    cudaEvent_t e0, e1;
    int failures = 0;
};

template <typename Launch>
float timeLaunch(Ctx& c, Launch launch)
{
    if (c.o->warm) { launch(); CUDA_CHECK(cudaGetLastError()); }
    CUDA_CHECK(cudaMemset(c.dP, 0, (size_t)c.J * c.L * sizeof(float)));   // stale results can't pass
    CUDA_CHECK(cudaEventRecord(c.e0));
    launch();
    CUDA_CHECK(cudaEventRecord(c.e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(c.e1));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, c.e0, c.e1));
    return ms;
}

void report(Ctx& c, const char* name, int bx, int by, float ms, int occ, float extraMs = 0.f)
{
    size_t n = (size_t)c.J * c.L;
    CUDA_CHECK(cudaMemcpy(c.hP->data(), c.dP, n * sizeof(float), cudaMemcpyDeviceToHost));
    size_t where;
    double err = maxErrFraction(c.ref->data(), c.hP->data(), n, &where);
    bool ok = err <= c.o->tol;
    if (!ok) ++c.failures;
    double gflops = 2.0 * c.J * (double)c.K * c.L / (ms * 1e-3) / 1e9;
    printf("  %-16s (%2d,%2d,1) %10.3f ms %9.1f GFLOP/s  occ %3d%%  maxErrFrac %.3e @(%zu,%zu)  %s",
           name, bx, by, ms, gflops, occ, err, where / c.L, where % c.L, ok ? "PASS" : "FAIL");
    if (extraMs > 0.f) printf("   [+transpose %.3f ms -> total %.3f ms]", extraMs, ms + extraMs);
    printf("\n");
}

template <int TW>
void runTiled(Ctx& c, float transposeMs)
{
    dim3 block(TW, TW, 1);
    dim3 grid((c.L + TW - 1) / TW, (c.J + TW - 1) / TW, 1);
    char name[32];

    float ms = timeLaunch(c, [&] { mmTiled<TW><<<grid, block>>>(c.dM, c.dN, c.dP, c.J, c.K, c.L); });
    snprintf(name, sizeof name, "tiled TW=%d", TW);
    report(c, name, TW, TW, ms, theoreticalOccupancyPct((const void*)mmTiled<TW>, TW * TW, *c.prop));

    if (c.o->doNT) {
        ms = timeLaunch(c, [&] { mmTiledNT<TW><<<grid, block>>>(c.dM, c.dNt, c.dP, c.J, c.K, c.L); });
        snprintf(name, sizeof name, "tiledNT TW=%d", TW);
        report(c, name, TW, TW, ms, theoreticalOccupancyPct((const void*)mmTiledNT<TW>, TW * TW, *c.prop),
               transposeMs);
    }
}

int runCase(int J, int K, int L, const Opts& o, const cudaDeviceProp& prop)
{
    printf("\n=== P: %d x %d   M: %d x %d   N: %d x %d   (k = %d) ===\n", J, L, J, K, K, L, K);

    std::vector<float> hM((size_t)J * K), hN((size_t)K * L), ref((size_t)J * L), hP((size_t)J * L);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);   // positive -> Pcpu never 0
    for (auto& v : hM) v = dist(rng);
    for (auto& v : hN) v = dist(rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    mmCPU(hM.data(), hN.data(), ref.data(), J, K, L);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  %-16s %10s %10.3f ms %9.1f GFLOP/s\n", "CPU serial", "", cpuMs,
           2.0 * J * (double)K * L / (cpuMs * 1e-3) / 1e9);

    float *dM, *dN, *dNt = nullptr, *dP;
    CUDA_CHECK(cudaMalloc(&dM, hM.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dN, hN.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dP, hP.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dM, hM.data(), hM.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dN, hN.data(), hN.size() * sizeof(float), cudaMemcpyHostToDevice));

    Ctx c{J, K, L, dM, dN, nullptr, dP, &hP, &ref, &o, &prop};
    CUDA_CHECK(cudaEventCreate(&c.e0));
    CUDA_CHECK(cudaEventCreate(&c.e1));

    // naive v0, 16x16 blocks
    {
        dim3 block(16, 16, 1), grid((L + 15) / 16, (J + 15) / 16, 1);
        float ms = timeLaunch(c, [&] { mmNaive<<<grid, block>>>(dM, dN, dP, J, K, L); });
        report(c, "naive v0", 16, 16, ms, theoreticalOccupancyPct((const void*)mmNaive, 256, prop));
    }

    // transpose N once (time reported alongside the NT kernels)
    float transposeMs = 0.f;
    if (o.doNT) {
        CUDA_CHECK(cudaMalloc(&dNt, hN.size() * sizeof(float)));
        c.dNt = dNt;
        dim3 block(TD, TD, 1), grid((L + TD - 1) / TD, (K + TD - 1) / TD, 1);
        if (o.warm) transposeKernel<<<grid, block>>>(dN, dNt, K, L);
        CUDA_CHECK(cudaEventRecord(c.e0));
        transposeKernel<<<grid, block>>>(dN, dNt, K, L);
        CUDA_CHECK(cudaEventRecord(c.e1));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(c.e1));
        CUDA_CHECK(cudaEventElapsedTime(&transposeMs, c.e0, c.e1));

        dim3 nb(16, 16, 1), ng((L + 15) / 16, (J + 15) / 16, 1);
        float ms = timeLaunch(c, [&] { mmNaiveNT<<<ng, nb>>>(dM, dNt, dP, J, K, L); });
        report(c, "naiveNT", 16, 16, ms, theoreticalOccupancyPct((const void*)mmNaiveNT, 256, prop), transposeMs);
    }

    auto want = [&](int tw) {
        return (o.onlyTW == 0 || o.onlyTW == tw) && tw * tw <= prop.maxThreadsPerBlock;
    };
    if (want(8))  runTiled<8>(c, transposeMs);
    if (want(12)) runTiled<12>(c, transposeMs);
    if (want(16)) runTiled<16>(c, transposeMs);
    if (want(32)) runTiled<32>(c, transposeMs);

    CUDA_CHECK(cudaEventDestroy(c.e0));
    CUDA_CHECK(cudaEventDestroy(c.e1));
    CUDA_CHECK(cudaFree(dM)); CUDA_CHECK(cudaFree(dN)); CUDA_CHECK(cudaFree(dP));
    if (dNt) CUDA_CHECK(cudaFree(dNt));
    return c.failures;
}

int main(int argc, char** argv)
{
    Opts o;
    int dims[3], nd = 0, onlyCase = 0;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "-t")    && i + 1 < argc) o.onlyTW = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-case") && i + 1 < argc) onlyCase = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-tol")  && i + 1 < argc) o.tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "-nowarm")) o.warm = false;
        else if (!strcmp(argv[i], "-nont"))   o.doNT = false;
        else if (nd < 3) dims[nd++] = atoi(argv[i]);
        else { fprintf(stderr, "unrecognised argument %s\n", argv[i]); return 1; }
    }
    if (o.onlyTW && o.onlyTW != 8 && o.onlyTW != 12 && o.onlyTW != 16 && o.onlyTW != 32) {
        fprintf(stderr, "-t must be 8, 12, 16 or 32\n"); return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s  (sm_%d%d, %d SMs, max %d threads/SM, max %d threads/block, %zu KB smem/SM, %d regs/SM)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
           prop.maxThreadsPerBlock, prop.sharedMemPerMultiprocessor / 1024, prop.regsPerMultiprocessor);
    printf("Tolerance on max |Pcpu-Pgpu|/|Pcpu|: %.1e\n", o.tol);

    int failures = 0;
    if (nd == 3) {
        failures += runCase(dims[0], dims[1], dims[2], o, prop);   // J K L
    } else if (nd == 0) {
        const int cases[3][3] = { {100, 100, 100}, {1000, 1000, 1000}, {2000, 2500, 3000} };  // J, K, L
        for (int i = 0; i < 3; ++i)
            if (onlyCase == 0 || onlyCase == i + 1)
                failures += runCase(cases[i][0], cases[i][1], cases[i][2], o, prop);
    } else {
        fprintf(stderr, "give either no sizes or all three: J K L\n"); return 1;
    }

    printf("\n%s (%d failing kernel run%s)\n", failures ? "VERIFICATION FAILED" : "ALL KERNELS VERIFIED",
           failures, failures == 1 ? "" : "s");
    return failures ? 2 : 0;
}
