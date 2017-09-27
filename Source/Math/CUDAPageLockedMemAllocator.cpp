#include "stdafx.h"
#include "CUDAPageLockedMemAllocator.h"
#include "BestGpu.h" // for CPUONLY
#ifndef CPUONLY
#ifdef CUDA_COMPILE
#include <cuda_runtime_api.h>
#elif defined HIP_COMPILE
#include <hip/hip_runtime_api.h>
#endif // cuda-hip compile
#endif // cpuonly

namespace Microsoft { namespace MSR { namespace CNTK {

#ifndef CPUONLY

#ifdef CUDA_COMPILE
inline static void CheckCudaReturnCode(cudaError_t rc, const char* msg)
{
    if (rc != cudaSuccess)
        RuntimeError("%s: %s (cuda error %d)", msg, cudaGetErrorString(rc), (int)rc);
}
#elif defined HIP_COMPILE
inline static void CheckCudaReturnCode(hipError_t rc, const char* msg)
{
    if (rc != hipSuccess)
        RuntimeError("%s: %s (hip error %d)", msg, hipGetErrorString(rc), (int)rc);
}
#endif

CUDAPageLockedMemAllocator::CUDAPageLockedMemAllocator(int deviceID)
    : m_deviceID(deviceID)
{
}

void* CUDAPageLockedMemAllocator::Malloc(size_t size, int deviceId)
{
    void* p = nullptr;
#ifdef CUDA_COMPILE
    CheckCudaReturnCode(cudaSetDevice(deviceId), "Cannot set cuda device");

    // Note: I ask for cudaHostAllocDefault but cudaHostGetFlags() shows that it is allocated as 'cudaHostAllocMapped'
    CheckCudaReturnCode(cudaHostAlloc(&p, size, cudaHostAllocDefault), "Malloc in CUDAPageLockedMemAllocator failed");
#elif defined HIP_COMPILE
    CheckCudaReturnCode(hipSetDevice(deviceId), "Cannot set hip device");

    // Note: I ask for hipHostAllocDefault but hipHostGetFlags() shows that it is allocated as 'hipHostAllocMapped'
    CheckCudaReturnCode(hipHostMalloc(&p, size, hipHostMallocDefault), "Malloc in CUDAPageLockedMemAllocator failed");
#endif
    return p;
}

void CUDAPageLockedMemAllocator::Free(void* p, int deviceId)
{
#ifdef CUDA_COMPILE
    CheckCudaReturnCode(cudaSetDevice(deviceId), "Cannot set cuda device");
    CheckCudaReturnCode(cudaFreeHost(p), "Free in CUDAPageLockedMemAllocator failed");
#elif defined HIP_COMPILE
    CheckCudaReturnCode(hipSetDevice(deviceId), "Cannot set hip device");
    CheckCudaReturnCode(hipHostFree(p), "Free in CUDAPageLockedMemAllocator failed");
#endif
}

void* CUDAPageLockedMemAllocator::Malloc(size_t size)
{
    return Malloc(size, m_deviceID);
}

void CUDAPageLockedMemAllocator::Free(void* p)
{
    Free(p, m_deviceID);
}

int CUDAPageLockedMemAllocator::GetDeviceId() const
{
    return m_deviceID;
}
#else
// Dummy definitions when compiling for CPUONLY
CUDAPageLockedMemAllocator::CUDAPageLockedMemAllocator(int)
{
}

int CUDAPageLockedMemAllocator::GetDeviceId() const
{
    return -1;
}

void* CUDAPageLockedMemAllocator::Malloc(size_t)
{
    return nullptr;
}

void* CUDAPageLockedMemAllocator::Malloc(size_t, int)
{
    return nullptr;
}

void CUDAPageLockedMemAllocator::Free(void*)
{
}

void CUDAPageLockedMemAllocator::Free(void*, int)
{
}
#endif
} } }
