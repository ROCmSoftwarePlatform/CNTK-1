#include "stdafx.h"
#include "Basics.h"
#include "GPUDataTransferer.h"
#include "GPUMatrix.h"

#pragma comment(lib, "cudart.lib")

#pragma warning(disable : 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4702) // unreachable code; triggered for unknown reasons

namespace Microsoft { namespace MSR { namespace CNTK {

// CUDA failed
// Since the outer code sometimes does not recover properly, as an option we log and die right away.
// This is needed for our GCD farm which has intermittent CUDA errors that sometimes cause the DBN tool, when running with MPI, to hang instead of terminating.
#ifdef CUDA_COMPILE
static void cudafail(const char* msg)
{
    // TODO: get from an env variable
    bool dieoncudafailure = false;
    if (!dieoncudafailure)
    {
        RuntimeError("%s", msg);
    }
    fprintf(stderr, "%s\n", msg);
    fprintf(stderr, "cudafail: terminating\n"), fflush(stderr);
#ifdef WIN32
    TerminateProcess(GetCurrentProcess(), EXIT_FAILURE); // fail the hard way to ensure it won't hang elsewhere
#else
    exit(1);
#endif
}
#elif defined HIP_COMPILE 
static void hipfail(const char* msg)
{
    // TODO: get from an env variable
    bool dieonhipfailure = false;
    if (!dieonhipfailure)
    {
        RuntimeError("%s", msg);
    }
    fprintf(stderr, "%s\n", msg);
    fprintf(stderr, "hipfail: terminating\n"), fflush(stderr);
#ifdef WIN32
    TerminateProcess(GetCurrentProcess(), EXIT_FAILURE); // fail the hard way to ensure it won't hang elsewhere
#else
    exit(1);
#endif
}
#endif

// allows to write hipFunction() || "error"   (CUDA runtime)
#ifdef CUDA_COMPILE
static
#ifdef WIN32
    __declspec(noinline)
#endif
        void
        operator||(cudaError_t rc, const char* msg)
{
    if (rc != cudaSuccess)
    {
        char buf[1000];
        sprintf_s(buf, 1000, "%s: %s (cuda error %d)", msg, cudaGetErrorString(rc), rc);
        cudafail(buf);
    }
}
#elif defined HIP_COMPILE
static
#ifdef WIN32
    __declspec(noinline)
#endif
        void
        operator||(hipError_t rc, const char* msg)
{
    if (rc != hipSuccess)
    {
        char buf[1000];
        sprintf_s(buf, 1000, "%s: %s (hip error %d)", msg, hipGetErrorString(rc), rc);
        hipfail(buf);
    }
}
#endif

//// Base class for different data transferers.
#ifdef CUDA_COMPILE
GranularGPUDataTransferer::GranularGPUDataTransferer(int deviceId, const cudaStream_t& fetchStream, const cudaStream_t& assignStream, bool blocking)
    : m_fetchStream(fetchStream),
      m_assignStream(assignStream),
      m_deviceId(deviceId),
      m_fetchCompleteEvent(nullptr),
      m_assignCompleteEvent(nullptr),
      m_syncEvent(nullptr)
{
    PrepareDevice(m_deviceId);

    // Note: Do NOT use cudaEventBlockingSync (which supposedly yields the process)--it will totally break cudaEventSynchronize(), causing it to take 50 or 100 ms randomly.
    // NOTE: We never saw this in reading prefetch.
    unsigned flags = cudaEventDisableTiming;
    if (blocking)
        flags |= cudaEventBlockingSync;

    // events
    cudaEventCreateWithFlags(&m_fetchCompleteEvent, flags) || "cudaEventCreateWithFlags failed";
    cudaEventCreateWithFlags(&m_assignCompleteEvent, flags) || "cudaEventCreateWithFlags failed";
    cudaEventCreateWithFlags(&m_syncEvent, cudaEventDisableTiming) || "cudaEventCreateWithFlags failed";
}
#elif defined HIP_COMPILE
GranularGPUDataTransferer::GranularGPUDataTransferer(int deviceId, const hipStream_t& fetchStream, const hipStream_t& assignStream, bool blocking)
    : m_fetchStream(fetchStream),
      m_assignStream(assignStream),
      m_deviceId(deviceId),
      m_fetchCompleteEvent(nullptr),
      m_assignCompleteEvent(nullptr),
      m_syncEvent(nullptr)
{
    PrepareDevice(m_deviceId);

    // Note: Do NOT use hipEventBlockingSync (which supposedly yields the process)--it will totally break hipEventSynchronize(), causing it to take 50 or 100 ms randomly.
    // NOTE: We never saw this in reading prefetch.
    unsigned flags = hipEventDisableTiming;
    if (blocking)
        flags |= hipEventBlockingSync;

    // events
    hipEventCreateWithFlags(&m_fetchCompleteEvent, flags) || "hipEventCreateWithFlags failed";
    hipEventCreateWithFlags(&m_assignCompleteEvent, flags) || "hipEventCreateWithFlags failed";
    hipEventCreateWithFlags(&m_syncEvent, hipEventDisableTiming) || "hipEventCreateWithFlags failed";
}
#endif

GranularGPUDataTransferer::~GranularGPUDataTransferer()
{
    // TODO: Check for error code and throw if !std::uncaught_exception()
#ifdef CUDA_COMPILE
    cudaEventDestroy(m_assignCompleteEvent);
    cudaEventDestroy(m_fetchCompleteEvent);
    cudaEventDestroy(m_syncEvent);
#elif defined HIP_COMPILE
    hipEventDestroy(m_assignCompleteEvent);
    hipEventDestroy(m_fetchCompleteEvent);
    hipEventDestroy(m_syncEvent);
#endif
}

void GranularGPUDataTransferer::CopyGPUToCPUAsync(const void* gpuBuffer, size_t numElements, size_t elementSize, void* cpuBuffer)
{
    PrepareDevice(m_deviceId);

#ifdef CUDA_COMPILE
    cudaMemcpyAsync(cpuBuffer, gpuBuffer, numElements * elementSize, cudaMemcpyDeviceToHost, GetFetchStream()) || "cudaMemcpyAsync failed";
#elif defined HIP_COMPILE
    hipMemcpyAsync(cpuBuffer, gpuBuffer, numElements * elementSize, hipMemcpyDeviceToHost, GetFetchStream()) || "hipMemcpyAsync failed";
#endif
}

void GranularGPUDataTransferer::RecordGPUToCPUCopy()
{
#ifdef CUDA_COMPILE
    cudaEventRecord(m_fetchCompleteEvent, GetFetchStream()) || "cudaEventRecord failed";
#elif defined HIP_COMPILE
    hipEventRecord(m_fetchCompleteEvent, GetFetchStream()) || "hipEventRecord failed";
#endif
}

void GranularGPUDataTransferer::WaitForCopyGPUToCPU()
{
    PrepareDevice(m_deviceId);
#ifdef CUDA_COMPILE
    cudaEventSynchronize(m_fetchCompleteEvent) || "cudaEventSynchronize failed";
#elif defined HIP_COMPILE
    hipEventSynchronize(m_fetchCompleteEvent) || "hipEventSynchronize failed";
#endif
}

void GranularGPUDataTransferer::CopyCPUToGPUAsync(const void* cpuBuffer, size_t numElements, size_t elementSize, void* gpuBuffer)
{
    PrepareDevice(m_deviceId);
#ifdef CUDA_COMPILE
    cudaMemcpyAsync(gpuBuffer, cpuBuffer, numElements * elementSize, cudaMemcpyHostToDevice, GetAssignStream()) || "cudaMemcpyAsync failed";
#elif defined HIP_COMPILE
    hipMemcpyAsync(gpuBuffer, cpuBuffer, numElements * elementSize, hipMemcpyHostToDevice, GetAssignStream()) || "hipMemcpyAsync failed";
#endif
}

void GranularGPUDataTransferer::RecordCPUToGPUCopy()
{
#ifdef CUDA_COMPILE
    cudaEventRecord(m_assignCompleteEvent, GetAssignStream()) || "cudaEventRecord failed";
#elif defined HIP_COMPILE
    hipEventRecord(m_assignCompleteEvent, GetAssignStream()) || "hipEventRecord failed";
#endif
}

void GranularGPUDataTransferer::WaitForCopyCPUToGPU()
{
    PrepareDevice(m_deviceId);
#ifdef CUDA_COMPILE
    cudaEventSynchronize(m_assignCompleteEvent) || "cudaEventSynchronize failed";
#elif defined HIP_COMPILE
    hipEventSynchronize(m_assignCompleteEvent) || "hipEventSynchronize failed";
#endif
}

void GranularGPUDataTransferer::RecordComputeStreamSyncPoint()
{
    PrepareDevice(m_deviceId);
#ifdef CUDA_COMPILE
    cudaEventRecord(m_syncEvent, GetStream()) || "cudaEventRecord failed";
#elif defined HIP_COMPILE
    hipEventRecord(m_syncEvent, GetStream()) || "hipEventRecord failed";
#endif
}

void GranularGPUDataTransferer::WaitForSyncPointOnFetchStreamAsync()
{
    PrepareDevice(m_deviceId);
#ifdef CUDA_COMPILE
    cudaStreamWaitEvent(GetFetchStream(), m_syncEvent, 0 /*flags 'must be 0'*/) || "cudaStreamWaitEvent failed";
#elif defined HIP_COMPILE
    hipStreamWaitEvent(GetFetchStream(), m_syncEvent, 0 /*flags 'must be 0'*/) || "hipStreamWaitEvent failed";
#endif
}

void GranularGPUDataTransferer::WaitForSyncPointOnAssignStreamAsync()
{
    PrepareDevice(m_deviceId);
#ifdef CUDA_COMPILE
    cudaStreamWaitEvent(GetAssignStream(), m_syncEvent, 0 /*flags 'must be 0'*/) || "cudaStreamWaitEvent failed";
#elif defined HIP_COMPILE
    hipStreamWaitEvent(GetAssignStream(), m_syncEvent, 0 /*flags 'must be 0'*/) || "hipStreamWaitEvent failed";
#endif
}

//// GPUDataTransferer

// same but for event
#ifdef CUDA_COMPILE
void GPUDataTransferer::SyncEvent(cudaEvent_t ev)
{
    auto rc = cudaEventQuery(ev);
    if (rc != cudaErrorNotReady)
    {
        // if Event is ready then no need to wait
        rc || "cudaEventQuery failed";
        return;
    }
    // we must wait
    cudaEventSynchronize(ev) || "cudaEventSynchronize failed";
}

//streams
cudaStream_t GPUDataTransferer::s_fetchStream = NULL;

cudaStream_t GPUDataTransferer::s_assignStream = NULL;

cudaStream_t GPUDataTransferer::GetFetchStream()
{
    return s_fetchStream;
}
#elif defined HIP_COMPILE
void GPUDataTransferer::SyncEvent(hipEvent_t ev)
{
    auto rc = hipEventQuery(ev);
    if (rc != hipErrorNotReady)
    {
        // if Event is ready then no need to wait
        rc || "hipEventQuery failed";
        return;
    }
    // we must wait
    hipEventSynchronize(ev) || "hipEventSynchronize failed";
}

//streams
hipStream_t GPUDataTransferer::s_fetchStream = NULL;

hipStream_t GPUDataTransferer::s_assignStream = NULL;

hipStream_t GPUDataTransferer::GetFetchStream()
{
    return s_fetchStream;
}
#endif

GPUDataTransferer::GPUDataTransferer(int deviceId, bool useConcurrentStreams) 
{
#pragma warning(disable : 4127)
    if (useConcurrentStreams && (s_fetchStream == NULL))
    {
#ifdef CUDA_COMPILE
	cudaStreamCreateWithFlags(&s_fetchStream, cudaStreamNonBlocking) || "cudaStreamCreateWithFlags failed";
	cudaStreamCreateWithFlags(&s_assignStream, cudaStreamNonBlocking) || "cudaStreamCreateWithFlags failed";
#elif defined HIP_COMPILE
        hipStreamCreateWithFlags(&s_fetchStream, hipStreamNonBlocking) || "hipStreamCreateWithFlags failed";
        hipStreamCreateWithFlags(&s_assignStream, hipStreamNonBlocking) || "hipStreamCreateWithFlags failed";
#endif
    }

    m_inner = make_unique<GranularGPUDataTransferer>(deviceId, s_fetchStream, s_assignStream);
}

GPUDataTransferer::~GPUDataTransferer()
{
    // BUGBUG: we don't destroy our streams (they are static variables); we need a static destructor, I am too lazy now
}

void GPUDataTransferer::CopyGPUToCPUAsync(void* gpuBuffer, size_t totalSize, void* cpuBuffer)
{
    m_inner->CopyGPUToCPUAsync(gpuBuffer, 1, totalSize, cpuBuffer);
    m_inner->RecordGPUToCPUCopy();
}

void GPUDataTransferer::CopyCPUToGPUAsync(void* cpuBuffer, size_t totalSize, void* gpuBuffer)
{
    m_inner->CopyCPUToGPUAsync(cpuBuffer, 1, totalSize, gpuBuffer);
    m_inner->RecordCPUToGPUCopy();
}

void GPUDataTransferer::WaitForCopyGPUToCPUAsync()
{
    PrepareDevice(m_inner->m_deviceId);
    SyncEvent(m_inner->m_fetchCompleteEvent);
}

void GPUDataTransferer::WaitForCopyCPUToGPUAsync()
{
    PrepareDevice(m_inner->m_deviceId);
    SyncEvent(m_inner->m_assignCompleteEvent);
}

/// PrefetchGPUDataTransferer

PrefetchGPUDataTransferer::PrefetchGPUDataTransferer(int deviceId) : GranularGPUDataTransferer(deviceId, nullptr, nullptr, true)
{
#ifdef CUDA_COMPILE
     cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking) || "cudaStreamCreateWithFlags failed (PrefetchGPUDataTransferer ctor)";
#elif defined HIP_COMPILE
     hipStreamCreateWithFlags(&m_stream, hipStreamNonBlocking) || "hipStreamCreateWithFlags failed (PrefetchGPUDataTransferer ctor)";
#endif
}

PrefetchGPUDataTransferer::~PrefetchGPUDataTransferer()
{
    try
    {
        PrepareDevice(m_deviceId);
    }
    catch (...)
    {
        // the error is already logged
        return;
    }

#ifdef CUDA_COMPILE
    auto code = cudaStreamDestroy(m_stream);
    if (code != cudaSuccess)
    {
        std::cerr << "cudaStreamDestroy failed (PrefetchGPUDataTransferer dtor): "
            << cudaGetErrorString(code) << " (cuda error " <<  code << ")"<< std::endl;
    }
#elif defined HIP_COMPILE
    auto code = hipStreamDestroy(m_stream);
    if (code != hipSuccess)
    {
        std::cerr << "hipStreamDestroy failed (PrefetchGPUDataTransferer dtor): "
            << hipGetErrorString(code) << " (hip error " <<  code << ")"<< std::endl;
    }
#endif
}

}}}
