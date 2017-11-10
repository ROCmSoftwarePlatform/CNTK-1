// cudalib.cpp -- all CUDA calls (but not hipblas) are encapsulated here
// All actual CUDA API calls go here, to keep the header out of our other headers.
//
// F. Seide, V-hansu

#define _CRT_SECURE_NO_WARNINGS 1 // so we can use getenv()...

#include "Basics.h"
#ifdef CUDA_COMPILE
#include <cuda_runtime_api.h> // for CUDA API
#include <cuda.h> // for device API
#elif defined HIP_COMPILE
#include <hip/hip_runtime_api.h> // for CUDA API
#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>             // for device API
#endif //NV platform
#endif //HIP compile
#include "cudalib.h"
#include "cudadevice.h"
#include <string>
#include <assert.h>
#ifdef CUDA_COMPILE
#include <cublas_v2.h>
#elif defined HIP_COMPILE
#include <hipblas.h>
#endif

#undef NOMULTIDEVICE // define this to disable any context/driver stuff

#ifndef NOMULTIDEVICE
#pragma comment(lib, "cuda.lib") // link CUDA device API
#endif
#pragma comment(lib, "cudart.lib") // link CUDA runtime
#pragma comment(lib, "cublas.lib")

namespace msra { namespace cuda {

static int devicesallocated = -1; // -1 means not initialized

// allows to write cudaFunction() || "error"   (CUDA runtime)
#ifdef CUDA_COMPILE
static void operator||(cudaError_t rc, const char *msg)
{
    if (rc != cudaSuccess)
        RuntimeError("%s: %s (cuda error %d)", msg, cudaGetErrorString(rc), (int) rc);
}

cudaStream_t GetCurrentStream()
{
    return cudaStreamDefault;
}

// synchronize with ongoing thread
void join()
{
    cudaDeviceSynchronize() || "cudaDeviceSynchronize failed";
}
#elif defined HIP_COMPILE
static void operator||(hipError_t rc, const char *msg)
{
    if (rc != hipSuccess)
        RuntimeError("%s: %s (hip error %d)", msg, hipGetErrorString(rc), (int) rc);
}

hipStream_t GetCurrentStream()
{
    return hipStreamDefault;
}

// synchronize with ongoing thread
void join()
{
    hipDeviceSynchronize() || "hipDeviceSynchronize failed";
}
#endif

// allocate a stack to store the devices that have been pushed
const int stackSize = 20;
static int curStack = 0;
static size_t deviceStack[stackSize] = {0};

// memory allocation
void *mallocbytes(size_t nelem, size_t sz)
{
    for (size_t retry = 0;; retry++)
    {
        try
        {
            // fprintf (stderr, "mallocbytes: allocating %d elements of size %d, %d bytes\n", (int) nelem, (int) sz, (int) (nelem * sz));        // comment out by [v-hansu] to get rid out annoying output
            void *p;
#ifdef CUDA_COMPILE
	    cudaMalloc(&p, nelem * sz) || "cudaMalloc failed";
#elif defined HIP_COMPILE
            hipMalloc(&p, nelem * sz) || "hipMalloc failed";
#endif
            return p;
        }
        catch (const std::exception &e)
        {
            fprintf(stderr, "mallocbytes: failed with error %s\n", e.what());
            if (retry >= 5)
                throw;
        }
    }
}

void freebytes(void *p)
{
#ifdef CUDA_COMPILE
    cudaFree(p) || "cudaFree failed";
#elif defined HIP_COMPILE
    hipFree(p) || "hipFree failed";
#endif
}

void memcpyh2d(void *dst, size_t byteoffset, const void *src, size_t nbytes)
{
#ifdef CUDA_COMPILE
    cudaMemcpy(byteoffset + (char *) dst, src, nbytes, cudaMemcpyHostToDevice) || "cudaMemcpy failed";
#elif defined HIP_COMPILE
    hipMemcpy(byteoffset + (char *) dst, src, nbytes, hipMemcpyHostToDevice) || "hipMemcpy failed";
#endif
}

void memcpyd2h(void *dst, const void *src, size_t byteoffset, size_t nbytes)
{
#ifdef CUDA_COMPILE
    cudaMemcpy(dst, byteoffset + (const char *) src, nbytes, cudaMemcpyDeviceToHost) || "cudaMemcpy failed";
#elif defined HIP_COMPILE
    hipMemcpy(dst, byteoffset + (const char *) src, nbytes, hipMemcpyDeviceToHost) || "hipMemcpy failed";
#endif
}
};
};
