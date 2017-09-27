#pragma once

#ifndef CPUONLY
#ifdef CUDA_COMPILE
#include <cuda_runtime_api.h>
#include <cuda.h>
#elif defined HIP_COMPILE
#include <hip/hip_runtime_api.h>
#endif
#endif // !CPUONLY

#include "Basics.h"

#ifdef _WIN32
#ifndef MATH_API
#ifdef MATH_EXPORTS
#define MATH_API __declspec(dllexport)
#else
#define MATH_API __declspec(dllimport)
#endif
#endif /* MATH_API */
#else  // no DLLs in Linux
#define MATH_API
#endif

#include "DataTransferer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class MATH_API GranularGPUDataTransferer : public DataTransferer
{
public:
#ifndef CPUONLY
#ifdef CUDA_COMPILE
    GranularGPUDataTransferer(int deviceId, const cudaStream_t& fetchStream, const cudaStream_t& assignStream, bool blocking = false);
#elif defined HIP_COMPILE
    GranularGPUDataTransferer(int deviceId, const hipStream_t& fetchStream, const hipStream_t& assignStream, bool blocking = false);
#endif
#else
    GranularGPUDataTransferer() {}
#endif // !CPUONLY

    ~GranularGPUDataTransferer();

    void CopyGPUToCPUAsync(const void* gpuBuffer, size_t numElements, size_t elementSize, void* cpuBuffer) override;
    void RecordGPUToCPUCopy() override;
    void WaitForCopyGPUToCPU() override;

    void CopyCPUToGPUAsync(const void* cpuBuffer, size_t numElements, size_t elementSize, void* gpuBuffer) override;
    void RecordCPUToGPUCopy() override;
    void WaitForCopyCPUToGPU() override;

    void RecordComputeStreamSyncPoint() override;
    void WaitForSyncPointOnFetchStreamAsync() override;
    void WaitForSyncPointOnAssignStreamAsync() override;

#ifndef CPUONLY
private:
    // Not owned by this class, are always injected.
#ifdef CUDA_COMPILE
    const cudaStream_t& m_fetchStream;
    const cudaStream_t& m_assignStream;
#elif defined HIP_COMPILE
    const hipStream_t& m_fetchStream;
    const hipStream_t& m_assignStream;
#endif

protected:

#ifdef CUDA_COMPILE
    virtual const cudaStream_t& GetAssignStream() const
    {
        return m_assignStream;
    }

    virtual const cudaStream_t& GetFetchStream() const
    {
        return m_fetchStream;
    }

    mutable cudaEvent_t m_fetchCompleteEvent;
    mutable cudaEvent_t m_assignCompleteEvent;
    mutable cudaEvent_t m_syncEvent;
#elif defined HIP_COMPILE
    virtual const hipStream_t& GetAssignStream() const
    {
        return m_assignStream;
    }

    virtual const hipStream_t& GetFetchStream() const
    {
        return m_fetchStream;
    }

    mutable hipEvent_t m_fetchCompleteEvent;
    mutable hipEvent_t m_assignCompleteEvent;
    mutable hipEvent_t m_syncEvent;
#endif
#endif // !CPUONLY

protected:
    int m_deviceId;

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(GranularGPUDataTransferer);

    friend class GPUDataTransferer;
};

class MATH_API GPUDataTransferer
{
#pragma warning(push)
#pragma warning(disable : 4251) // Using std::unique pointer on the dll boundary.
    std::unique_ptr<GranularGPUDataTransferer> m_inner;
#pragma warning(pop)

public:
    GPUDataTransferer(int deviceId, bool useConcurrentStreams);
    ~GPUDataTransferer();

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(GPUDataTransferer);

    // GPU to CPU
    void CopyGPUToCPUAsync(void* gpuBuffer, size_t totalSize, void* cpuBuffer);

    template <class ElemType>
    void CopyGPUToCPUAsync(ElemType* gpuBuffer, size_t numElements, ElemType* cpuBuffer)
    {
        CopyGPUToCPUAsync(static_cast<void*>(gpuBuffer), numElements * sizeof(ElemType), cpuBuffer);
    }

    void WaitForCopyGPUToCPUAsync();

    // CPU to GPU
    void CopyCPUToGPUAsync(void* cpuBuffer, size_t totalSize, void* gpuBuffer);

    template <class ElemType>
    void CopyCPUToGPUAsync(ElemType* cpuBuffer, size_t numElements, ElemType* gpuBuffer)
    {
        CopyCPUToGPUAsync(static_cast<void*>(cpuBuffer), numElements * sizeof(ElemType), gpuBuffer);
    }

    void WaitForCopyCPUToGPUAsync();

#ifndef CPUONLY
#ifdef CUDA_COMPILE
    static cudaStream_t GetFetchStream();
#elif defined HIP_COMPILE
    static hipStream_t GetFetchStream();
#endif
#endif // !CPUONLY

private:
#ifndef CPUONLY

    // TODO: this needs to be refactored to get rid of all statics
#ifdef CUDA_COMPILE
    static void SyncEvent(cudaEvent_t ev);

    static cudaStream_t s_fetchStream;
    static cudaStream_t s_assignStream;
#elif defined HIP_COMPILE
    static void SyncEvent(hipEvent_t ev);

    static hipStream_t s_fetchStream;
    static hipStream_t s_assignStream;
#endif
#endif // !CPUONLY
};

class PrefetchGPUDataTransferer : public GranularGPUDataTransferer
{
public:
    PrefetchGPUDataTransferer(int deviceId);
    ~PrefetchGPUDataTransferer();

private:
#ifndef CPUONLY
#ifdef CUDA_COMPILE
    cudaStream_t m_stream;

    virtual const cudaStream_t& GetAssignStream() const override
    {
        return m_stream;
    }
#elif defined HIP_COMPILE
    hipStream_t m_stream;

    virtual const hipStream_t& GetAssignStream() const override
    {
        return m_stream;
    }
#endif
#endif

    DISABLE_COPY_AND_MOVE(PrefetchGPUDataTransferer);
};

}}}
