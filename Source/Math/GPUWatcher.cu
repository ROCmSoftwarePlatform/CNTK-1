//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "BestGpu.h"

#ifndef CPUONLY

#include "GPUWatcher.h"
#ifdef __HIP_PLATFORM_NVCC__
    #include <cuda.h>
#endif
#include <hip/hip_runtime.h>

int GPUWatcher::GetGPUIdWithTheMostFreeMemory()
{
    int deviceCount = 0;
    hipError_t error_id = hipGetDeviceCount(&deviceCount);
    if (error_id != hipSuccess || deviceCount == 0)
    {
        return -1;
    }
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
    hipError_t result = hipSetDevice(devId);
    if (result != hipSuccess)
    {
        return 0;
    }
    // get the amount of free memory on the graphics card
    size_t free = 0;
    size_t total = 0;
    result = hipMemGetInfo(&free, &total);
    if (result != hipSuccess)
    {
        return 0;
    }
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
