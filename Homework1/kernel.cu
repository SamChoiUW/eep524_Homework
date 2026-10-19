
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "errorCheck.h"
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
__global__ void printfKernel();
    void printfKernelChecker();
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
} 

int main() { 
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
     
    printfKernelChecker();

    checkCuda(addWithCuda(c, a, b, arraySize), "addWithCuda");

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
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