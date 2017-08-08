#include "stdafx.h"
#include "Basics.h"
#include "GPUDataTransferer.h"
#include "GPUMatrix.h"

#pragma comment(lib, "hiprt.lib")

#pragma warning(disable : 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4702) // unreachable code; triggered for unknown reasons

namespace Microsoft { namespace MSR { namespace CNTK {

// CUDA failed
// Since the outer code sometimes does not recover properly, as an option we log and die right away.
// This is needed for our GCD farm which has intermittent CUDA errors that sometimes cause the DBN tool, when running with MPI, to hang instead of terminating.
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

// allows to write hipFunction() || "error"   (CUDA runtime)
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

//// Base class for different data transferers.
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

GranularGPUDataTransferer::~GranularGPUDataTransferer()
{
    // TODO: Check for error code and throw if !std::uncaught_exception()
    hipEventDestroy(m_assignCompleteEvent);
    hipEventDestroy(m_fetchCompleteEvent);
    hipEventDestroy(m_syncEvent);
}

void GranularGPUDataTransferer::CopyGPUToCPUAsync(const void* gpuBuffer, size_t numElements, size_t elementSize, void* cpuBuffer)
{
    PrepareDevice(m_deviceId);

    hipMemcpyAsync(cpuBuffer, gpuBuffer, numElements * elementSize, hipMemcpyDeviceToHost, GetFetchStream()) || "hipMemcpyAsync failed";
}

void GranularGPUDataTransferer::RecordGPUToCPUCopy()
{
    hipEventRecord(m_fetchCompleteEvent, GetFetchStream()) || "hipEventRecord failed";
}

void GranularGPUDataTransferer::WaitForCopyGPUToCPU()
{
    PrepareDevice(m_deviceId);
    hipEventSynchronize(m_fetchCompleteEvent) || "hipEventSynchronize failed";
}

void GranularGPUDataTransferer::CopyCPUToGPUAsync(const void* cpuBuffer, size_t numElements, size_t elementSize, void* gpuBuffer)
{
    PrepareDevice(m_deviceId);
    hipMemcpyAsync(gpuBuffer, cpuBuffer, numElements * elementSize, hipMemcpyHostToDevice, GetAssignStream()) || "hipMemcpyAsync failed";
}

void GranularGPUDataTransferer::RecordCPUToGPUCopy()
{
    hipEventRecord(m_assignCompleteEvent, GetAssignStream()) || "hipEventRecord failed";
}

void GranularGPUDataTransferer::WaitForCopyCPUToGPU()
{
    PrepareDevice(m_deviceId);
    hipEventSynchronize(m_assignCompleteEvent) || "hipEventSynchronize failed";
}

void GranularGPUDataTransferer::RecordComputeStreamSyncPoint()
{
    PrepareDevice(m_deviceId);
    hipEventRecord(m_syncEvent, GetStream()) || "cudeEventRecord failed";
}

void GranularGPUDataTransferer::WaitForSyncPointOnFetchStreamAsync()
{
    PrepareDevice(m_deviceId);
    hipStreamWaitEvent(GetFetchStream(), m_syncEvent, 0 /*flags 'must be 0'*/) || "hipStreamWaitEvent failed";
}

void GranularGPUDataTransferer::WaitForSyncPointOnAssignStreamAsync()
{
    PrepareDevice(m_deviceId);
    hipStreamWaitEvent(GetAssignStream(), m_syncEvent, 0 /*flags 'must be 0'*/) || "hipStreamWaitEvent failed";
}

//// GPUDataTransferer

// same but for event
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

GPUDataTransferer::GPUDataTransferer(int deviceId, bool useConcurrentStreams) 
{
#pragma warning(disable : 4127)
    if (useConcurrentStreams && (s_fetchStream == NULL))
    {
        hipStreamCreateWithFlags(&s_fetchStream, hipStreamNonBlocking) || "hipStreamCreateWithFlags failed";
        hipStreamCreateWithFlags(&s_assignStream, hipStreamNonBlocking) || "hipStreamCreateWithFlags failed";
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
     hipStreamCreateWithFlags(&m_stream, hipStreamNonBlocking) || "hipStreamCreateWithFlags failed (PrefetchGPUDataTransferer ctor)";
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

    auto code = hipStreamDestroy(m_stream);
    if (code != hipSuccess)
    {
        std::cerr << "hipStreamDestroy failed (PrefetchGPUDataTransferer dtor): "
            << hipGetErrorString(code) << " (hip error " <<  code << ")"<< std::endl;
    }
}

}}}
