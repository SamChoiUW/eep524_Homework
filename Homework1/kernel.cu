
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "errorCheck.h"
#include "homework1.h"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
} 

int main() { 
     
    //printfKernelChecker();


    /*
    //the SAXPY testing
    const float a = 3.0;
    const int sizes[3] = { 10,1e3,1e7 };
    int error = 0;
    for (int i = 0; i < 3; i++) {
        error += testSAXPY(sizes[i], a);
    }
    printf("any fails %d", error);
    */


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    checkCuda(cudaDeviceReset(),"cudaDeviceReset");

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size){
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = checkCuda(cudaSetDevice(0),"cudaSetDevice");

    // Allocate GPU buffers for three vectors (two input, one output)    .
    if ((cudaStatus = checkCuda(cudaMalloc((void**)&dev_c, size * sizeof(int)), "cudaMalloc")) != cudaSuccess) {
        goto Error;
    }
    if ((cudaStatus = checkCuda(cudaMalloc((void**)&dev_a, size * sizeof(int)), "cudaMalloc")) != cudaSuccess) {
        goto Error;
    }
    if ((cudaStatus = checkCuda(cudaMalloc((void**)&dev_b, size * sizeof(int)), "cudaMalloc")) != cudaSuccess) {
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.

    if ((cudaStatus = checkCuda(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy")) != cudaSuccess) {
        goto Error;
    }
    if ((cudaStatus = checkCuda(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy")) != cudaSuccess) {
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    if ((cudaStatus = checkCuda(cudaGetLastError(), "addKernel")) != cudaSuccess) {
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    if ((cudaStatus = checkCuda(cudaDeviceSynchronize(), "addKernel")) != cudaSuccess) {
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    if ((cudaStatus = checkCuda(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost), "addKernel")) != cudaSuccess) {
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


/*
printf kernel
1. Inside the kernel function, obtain the thread ID, block ID using the CUDA built-in
variables, and use them to compute the Globally unique Thread IDentifier (GTID).
2. Use printf( ) inside the kernel function body to print out the values of the thread ID (.x),
block ID (.x), and GTID on a single line (use the “\n” end-line character to include a
carriage return).
3. in the host application, experiment with various values of the execution configuration
(exec cfg) grid and block sizes, such as
a. <<<1,32>>>
b. <<<4,8>>>
c. <<<32,1>>>
d. Run each case several times (like 10 times) and observe the result behavior.
4. Some questions to consider:
a. What do you observe about the result? Is it consistent? Is it predictable?
b. Does the behavior of block and thread indices seem to differ?
c. What does this imply about computations you perform in the kernels?

*/
__global__ void printfKernel() {
    int threadID = threadIdx.x;
    int blockID = blockIdx.x;
    int GTID = blockID * blockDim.x + threadID;

    printf("the thread IDx.x is: %d\nthe block IDx.x is: %d\nthe GTID is: %d\n", threadID, blockID, GTID);
}
 
void printfKernelChecker() {
    int testblockcases[4] = {1, 4,8,32};
    int testthreadcases[4] = {32,8,8,1};
    for (int j = 0; j < 4; j++) {
        printf("********case: %d, thread %d************\n", testblockcases[j], testthreadcases[j]);
        for (int i = 0; i < 10; i++) {
            printf("***Run %d***\n", i);
            printfKernel<<<testblockcases[j], testthreadcases[j]>>>();
            cudaDeviceSynchronize();
        }
        printf("\n\n");

     }
}
bool testTolerance(float cpuResult, float gpuResult) {
    float rTol = 1e-5;
    float aTol = 1e-6;
    if (abs(cpuResult - gpuResult) <= 
        max(aTol, rTol * max(abs(cpuResult), abs(gpuResult)))) {
        return true;
    }
    else {
        return false;
    }
}
/*
SAXPY kernel
SAXPY stands for Single-Precision A times X plus Y.
Thus all the terms are single-precision (32-bit) floats where
● a is a scalar float multiplier value
● x is a vector of length 1xN
● y is a vector of length 1xN
The expression to be computed is
y = ax + y
SAXPY is one of the BLAS (Basic Linear Algebra Subroutines) Level-1 routines, and is known
to take linear time O(n) [We’ll cover algorithmic complexity and asymptotic notation officially in a
coming lecture soon).
Include the integer vector dimension N in the kernel input arguments passed from the host so
that the kernel can perform bounds-checking.
● Test cases:
○ N = 10 // this is a sanity check case to ensure it’s working right
○ N = 1e3 // scaling things up
○ N = 1e7 // everything still passing?
*/
__global__ void SAXPYkernel(int n, float a, const float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = a * x[index] + y[index];
    }
}
void SAXPYkernelCPU(int n, float a, const float* x, float* y) {
    for (int i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}
int testSAXPY(int n, float a) {
    int bytes = n * sizeof(float);

    float* vectorX = (float*)malloc(bytes);
    if (vectorX == NULL) {
        fprintf(stderr, "memory allocation failure");
    }
    float* vectorY = (float*)malloc(bytes);
    if (vectorY == NULL) {
        fprintf(stderr, "memory allocation failure");
    }
    float* vectorYCPU = (float*)malloc(bytes);
    if (vectorYCPU == NULL) {
        fprintf(stderr, "memory allocation failure");
    }
    float* vectorYGPU = (float*)malloc(bytes);
    if (vectorYGPU == NULL) {
        fprintf(stderr, "memory allocation failure");
    }

    //initialize to known vals
    for (int i = 0; i < n; i++) {
        vectorX[i] = (float)(i % 100) * 0.5f;
        vectorY[i] = (float)(i % 50) * 0.25f;
    }
    memcpy(vectorYCPU, vectorY, bytes);
    memcpy(vectorYGPU, vectorY, bytes);

    //cpu test
    SAXPYkernelCPU(n,  a, vectorX, vectorYCPU);

    //gpu test
    float* gpuX = nullptr;
    float* gpuY = nullptr;
    checkCuda(cudaMalloc(&gpuX, bytes), "cudaMalloc X");
    checkCuda(cudaMalloc(&gpuY, bytes), "cudaMalloc Y");
    checkCuda(cudaMemcpy(gpuX, vectorX, bytes, cudaMemcpyHostToDevice), "cudaMemcpy X");
    checkCuda(cudaMemcpy(gpuY, vectorYGPU, bytes, cudaMemcpyHostToDevice), "cudaMemcpy Y");

    checkCuda(cudaGetLastError(), "check");
    int threadsPerBlock = 128;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("n=%d, blocks=%d,threadsPerBlock=%d\n", n, blocks, threadsPerBlock);
    SAXPYkernel<<<blocks, threadsPerBlock >>>(n, a, gpuX, gpuY);
    checkCuda(cudaGetLastError(), "SAXPYkernel");
    checkCuda(cudaDeviceSynchronize(), "sync");

    checkCuda(cudaMemcpy(vectorYGPU, gpuY, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy Y rtn");

    int numberOfDifferences = 0;

    for (int i = 0; i < n; i++) {
        if (testTolerance(vectorYCPU[i], vectorYGPU[i]) == false) {
            if (numberOfDifferences++ < 10) {
                fprintf(stderr, "difference at %d\nCPU Value: %f\nGPU Value\n", i, vectorYCPU[i], vectorYGPU[i]);
            }
        }
    }
    if (numberOfDifferences != 0) {
        printf("Number of Differences = %d\n", numberOfDifferences);
    }
    checkCuda(cudaFree(gpuX), "cudaFree gpuX");
    checkCuda(cudaFree(gpuY), "cudaFree gpuY");
    free(vectorX);
    free(vectorY);
    free(vectorYCPU);
    free(vectorYGPU);

    return numberOfDifferences;
}
/*
2D matrix addition kernel
In this kernel you’ll implement a 2D matrix addition using single-precision (32-bit) floats. The
expression to be computed is
C = A + B
where A, B are input MxN matrices and C is an output MxN matrix, all floats.
Include the integer matrix dimensions M and N in the kernel arguments so that the kernel can
perform the 2D bounds-checking on each thread of execution.
● Test cases:
○ MxN = 4x4 // this is a sanity check case to ensure it’s working right
○ 500x500 // this is a small matrix
○ 3024x4032 // like a 12MP iPhone image resolution!
*/
__global__ void  matrixAdditionKernel(int m,int n, float* a, float* b, float* c) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int index = row * n + col;
    if (row < m && col < n) {
        c[index] = a[index] + b[index];
    }
}
void matrixAdditionCPU(int m, int n, float* a, float* b, float* c) {
    int index = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            index = row * n + col;
            c[index] = a[index] + b[index];
        }
    }
}
int testSAXPY(int m, int n) {
    int bytes = n * sizeof(float);

    float* matrixA = (float*)malloc(bytes);
    if (matrixA == NULL) {
        fprintf(stderr, "memory allocation failure");
    }
    float* matrixB = (float*)malloc(bytes);
    if (matrixB == NULL) {
        fprintf(stderr, "memory allocation failure");
    }
    float* resultMatrixCPU = (float*)malloc(bytes);
    if (resultMatrixCPU == NULL) {
        fprintf(stderr, "memory allocation failure");
    }
    float* resultMatrixGPU = (float*)malloc(bytes);
    if (resultMatrixGPU == NULL) {
        fprintf(stderr, "memory allocation failure");
    }

    //initialize to known vals
    int index = 0;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            index = row * n + col;
            matrixA[index] = (float)(index % 100) * 0.5f;
            matrixB[index] = (float)(index % 100) * 0.25f;
        }
    }

    //cpu test
    matrixAdditionCPU(m, n, matrixA, matrixB, resultMatrixCPU);

    //gpu test
    float* gpuA = nullptr;
    float* gpuB = nullptr;
    checkCuda(cudaMalloc(&gpuA, bytes), "cudaMalloc A");
    checkCuda(cudaMalloc(&gpuB, bytes), "cudaMalloc B");
    checkCuda(cudaMemcpy(gpuA, matrixA, bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCuda(cudaMemcpy(gpuB, matrixB, bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

    checkCuda(cudaGetLastError(), "check");
    int threadsPerBlock = 128;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("n=%d, blocks=%d,threadsPerBlock=%d\n", n, blocks, threadsPerBlock);
    SAXPYkernel << <blocks, threadsPerBlock >> > (n, a, gpuX, gpuY);
    checkCuda(cudaGetLastError(), "SAXPYkernel");
    checkCuda(cudaDeviceSynchronize(), "sync");

    checkCuda(cudaMemcpy(vectorYGPU, gpuY, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy Y rtn");

    int numberOfDifferences = 0;

    for (int i = 0; i < n; i++) {
        float diff = fabsf(vectorYGPU[i] - vectorYCPU[i]);
        if (diff > tol) {
            if (numberOfDifferences++ < 10) {
                fprintf(stderr, "difference at %d\nCPU Value: %f\nGPU Value\n", i, vectorYCPU[i], vectorYGPU[i]);
            }
        }
    }
    if (numberOfDifferences != 0) {
        printf("Number of Differences = %d\n", numberOfDifferences);
    }
    checkCuda(cudaFree(gpuX), "cudaFree gpuX");
    checkCuda(cudaFree(gpuY), "cudaFree gpuY");
    free(vectorX);
    free(vectorY);
    free(vectorYCPU);
    free(vectorYGPU);

    return numberOfDifferences;
}

/*
3D volume grid addition kernel
In this kernel you’ll implement a 3D grid addition. The expression to be computed is
C = A + B
where A, B are input MxNxP float data grids and C is an output MxNxP data grid
Include the integer grid dimensions M, N, P in the kernel input arguments so that the kernel can
perform the 3D bounds-checking on each thread of execution. Use the dim3 struct type to
define the 3D block and thread sizes
● Test cases:
○ MxNxP = 3x3x3 // this is a sanity check case to ensure it’s working right
○ 1003 // total data points = 1e6
○ 1e33 ?!? // total data points = 1e9
■ for this large test case be sure to compile a RELEASE build!
Host-Side Result Verification
1. As discussed in Lecture #2, include result verification routines in the host application
code to quantitatively VERIFY results of same the computations on CPU vs GPU.
a. A sequential CPU host routine will need to be implemented for each calculation
being performed by the GPU kernels implemented in this HW1.
b. Use a tolerance threshold for result agreement to account for numerical precision
differences, as discussed in Lecture 2.
c. Use RELEASE builds for result verifications:
i. On the Visual studio top toolbar, switch from Debug to RELEASE build mode
else you may not be able to see different output at different tolerances since
no optimisations will be added.
d. HW Q: At what tolerance value do verifications start to fail?
e. HW Q: which of the 3 kernels exhibits the most severe result disagreements
between CPU and GPU versions? Why?
i. submit results of any result discrepancies observed. A screenshot is fine.
*/

