#pragma once
void SAXPYkernelCPU(int n, float a, const float* x, float* y);
int testSAXPY(int n, float a, float tol);
void printfKernelChecker();