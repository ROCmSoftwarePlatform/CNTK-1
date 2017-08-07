#include "stdafx.h"
#include "CUDAPageLockedMemAllocator.h"
#include "BestGpu.h" // for CPUONLY
#ifndef CPUONLY
#include <hip/hip_runtime_api.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

#ifndef CPUONLY

inline static void CheckCudaReturnCode(hipError_t rc, const char* msg)
{
    if (rc != hipSuccess)
        RuntimeError("%s: %s (hip error %d)", msg, hipGetErrorString(rc), (int)rc);
}

CUDAPageLockedMemAllocator::CUDAPageLockedMemAllocator(int deviceID)
    : m_deviceID(deviceID)
{
}

void* CUDAPageLockedMemAllocator::Malloc(size_t size, int deviceId)
{
    void* p = nullptr;
    CheckCudaReturnCode(hipSetDevice(deviceId), "Cannot set hip device");

    // Note: I ask for hipHostAllocDefault but hipHostGetFlags() shows that it is allocated as 'hipHostAllocMapped'
    CheckCudaReturnCode(hipHostMalloc(&p, size, hipHostMallocDefault), "Malloc in CUDAPageLockedMemAllocator failed");
    return p;
}

void CUDAPageLockedMemAllocator::Free(void* p, int deviceId)
{
    CheckCudaReturnCode(hipSetDevice(deviceId), "Cannot set hip device");
    CheckCudaReturnCode(hipHostFree(p), "Free in CUDAPageLockedMemAllocator failed");
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
