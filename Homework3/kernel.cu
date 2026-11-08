#include "common.h"
#include "divergence.h"
#include "mmult.h"

int main()
{
    const int iters = 256;

    runDivergence(false, false, iters);
    runDivergence(false, true, iters);
    runDivergence(true, false, iters);
    runDivergence(true, true, iters);
    //runMmult();
    return 0;
}
