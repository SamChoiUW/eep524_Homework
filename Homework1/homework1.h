#pragma once
void SAXPYkernelCPU(int n, float a, const float* x, float* y);
int testSAXPY(int n, float a);
void printfKernelChecker();
void matrixAdditionCPU(int m, int n, float* a, float* b, float* c);
int testMatrixAddition(int m, int n);
void volumeGridAdditionCPU(int m, int n, int p, float* a, float* b, float* c);
int testVolumeGridAddition(int m, int n, int p);
#define ROW_MAJOR_INDEX row * n + col
#define INDEX_3D (slice * m * n) + row * n + col