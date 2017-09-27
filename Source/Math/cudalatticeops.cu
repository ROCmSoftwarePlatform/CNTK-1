// .cu file --#includes all actual .cu files which we store as .cu.h so we get syntax highlighting (VS does not recognize .cu files)
//
// F. Seide, V-hansu

#include <stdexcept>
#include "Basics.h"
#include "BestGpu.h"
#ifdef HIP_COMPILE
#include "hip/hip_runtime.h"
#endif

#ifndef CPUONLY

namespace msra { namespace cuda {

// call this after all kernel launches
// This is non-blocking. It catches launch failures, but not crashes during execution.
static void checklaunch(const char* fn)
{
    #ifdef CUDA_COMPILE
    cudaError_t rc = cudaGetLastError();
    if (rc != cudaSuccess)
	RuntimeError("%s: launch failure: %s (cuda error %d)", fn, cudaGetErrorString(rc), (int) rc);
    #elif defined HIP_COMPILE
    hipError_t rc = hipGetLastError();
    if (rc != hipSuccess)
        RuntimeError("%s: launch failure: %s (cuda error %d)", fn, hipGetErrorString(rc), (int) rc);
    #endif
}
};
};

// now include actual code which is in those files to allow for code highlighting etc.
#include "cudalatticeops.cu.h"

#endif // CPUONLY
