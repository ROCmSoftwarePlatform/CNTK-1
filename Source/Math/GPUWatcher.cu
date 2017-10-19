//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "BestGpu.h"

#ifndef CPUONLY

#include "GPUWatcher.h"
#ifdef CUDA_COMPILE
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined HIP_COMPILE
#include <hip/hip_runtime.h>
#endif

int GPUWatcher::GetGPUIdWithTheMostFreeMemory()
{
    int deviceCount = 0;
#ifdef CUDA_COMPILE
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess || deviceCount == 0)
    {
        return -1;
    }
#elif defined HIP_COMPILE
    hipError_t error_id = hipGetDeviceCount(&deviceCount);
    if (error_id != hipSuccess || deviceCount == 0)
    {
        return -1;
    }
#endif
    int curDev = 0;
    size_t curMemory = 0;
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        size_t freeMem = GetFreeMemoryOnCUDADevice(dev);
        if (freeMem > curMemory)
        {
            curMemory = freeMem;
            curDev = dev;
        }
    }
    return curDev;
}

size_t GPUWatcher::GetFreeMemoryOnCUDADevice(int devId)
{
#ifdef CUDA_COMPILE
    cudaError_t result = cudaSetDevice(devId);
    if (result != cudaSuccess)
    {
        return 0;
    }
#elif defined HIP_COMPILE
    hipError_t result = hipSetDevice(devId);
    if (result != hipSuccess)
    {
        return 0;
    }
#endif
    // get the amount of free memory on the graphics card
    size_t free = 0;
    size_t total = 0;
#ifdef CUDA_COMPILE
    result = cudaMemGetInfo(&free, &total);
    if (result != cudaSuccess)
    {
        return 0;
    }
#elif defined HIP_COMPILE
    result = hipMemGetInfo(&free, &total);
    if (result != hipSuccess)
    {
        return 0;
    }
#endif
    else
        return free;
}

GPUWatcher::GPUWatcher(void)
{
}

GPUWatcher::~GPUWatcher(void)
{
}

#endif // CPUONLY
