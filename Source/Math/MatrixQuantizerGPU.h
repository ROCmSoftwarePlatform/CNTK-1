#pragma once

#include "QuantizedMatrix.h" // TODO: strangely, this must be included first, although it is the first thing MatrixQuantizer.h includes. Without, nvcc fails.
#include "MatrixQuantizerImpl.h"
#include "ColumnQuantizer.h"
#include "GPUMatrix.h"
#ifndef CPUONLY
#ifdef CUDA_COMPILE
#include <cuda_runtime_api.h>
#include <cuda.h>
#elif defined HIP_COMPILE
#include <hip/hip_runtime_api.h>
#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
#endif
#endif
#endif // !CPUONLY
#include <vector>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class MatrixQuantizerGPU : public MatrixQuantizerImpl<ElemType>
{
public:
    MatrixQuantizerGPU(int deviceId, bool useDedicatedComputeStream, bool forceSync = false);
    ~MatrixQuantizerGPU();

    // Disallow copy and move construction and assignment
    DISABLE_COPY_AND_MOVE(MatrixQuantizerGPU);

    void QuantizeAsync(const Matrix<ElemType>& inMatrix, const Matrix<ElemType>& inResidual, QuantizedMatrix<ElemType>& outQMatrix, Matrix<ElemType>& outResidual, bool zeroThresholdFor1Bit) override;
    void WaitQuantizeAsyncDone() override;

    void UnquantizeAsync(QuantizedMatrix<ElemType>& inQMatrix, Matrix<ElemType>& outMatrix, bool add = false) override;
    void WaitUnquantizeAsyncDone() override;

private:
    // Helper function to get a temporary intermediate matrix on the GPU to store quantization results
    QuantizedMatrix<ElemType>& GetTempGPUQuantizedMatrix(size_t numRows, size_t numCols, size_t nBits, bool& newlyAllocated);

#ifndef CPUONLY
#ifdef CUDA_COMPILE
    // Record a event to flag the completion of quantization/unquantization kernel on the compute stream
    void RecordQuantizeCompleteEvent(cudaStream_t computestream) const;
#elif defined HIP_COMPILE
    // Record a event to flag the completion of quantization/unquantization kernel on the compute stream
    void RecordQuantizeCompleteEvent(hipStream_t computestream) const;
#endif

    // Synchronize the fetch stream to the quantization completion event and record an event on the fetch
    // stream to flag the completion of fetching the quantization results from the GPU
    void SyncQuantizeCompleEventAndFetchAndRecordFetchCompleteEvent(char* cpuBuffer, char* gpuBuffer, size_t size) const;

#ifdef CUDA_COMPILE
    // Synchronize the compute stream to the assign completion event to ensure that subsequent compute stream operations
    // wait for the assign stream operations, scheduled so far, to finish
    void SyncAssignCompleteEvent(cudaStream_t computestream) const;
#elif defined HIP_COMPILE
    // Synchronize the compute stream to the assign completion event to ensure that subsequent compute stream operations
    // wait for the assign stream operations, scheduled so far, to finish
    void SyncAssignCompleteEvent(hipStream_t computestream) const;
#endif

    // for concurrent computation and memcpy
    //  - assign to GPU : CPU-to-GPU,started by CPU when data read; flags assigncomplete
    //  - GPU-side operation        --waits for assigncomplete; flags quantizecomplete
    //  - fetch from GPU            --waits for quantizecomplete; flags fetchcomplete
    //  - CPU-side access of buffer --read: waits for fetchcomplete, write: waits for assigncomplete

public:
#ifdef CUDA_COMPILE
    static cudaStream_t GetComputeStream(); // get the compute stream
    static cudaStream_t GetFetchStream();   // and the copy streams
    static cudaStream_t GetAssignStream();
#elif defined HIP_COMPILE
    static hipStream_t GetComputeStream(); // get the compute stream
    static hipStream_t GetFetchStream();   // and the copy streams
    static hipStream_t GetAssignStream();
#endif

private:
    // helper functions for gpus
    static void Sync();
#ifdef CUDA_COMPILE
    static void SyncStream(cudaStream_t stream);
    static void SyncEvent(cudaEvent_t ev);
#elif defined HIP_COMPILE
    static void SyncStream(hipStream_t stream);
    static void SyncEvent(hipEvent_t ev);
#endif

private:
#ifdef CUDA_COMPILE
    static cudaStream_t m_computeStream;
    static cudaStream_t m_fetchStream;
    static cudaStream_t m_assignStream;

    mutable cudaEvent_t m_tempMatrixZeroingCompleteEvent;
    mutable cudaEvent_t m_quantizeCompleteEvent;
    mutable cudaEvent_t m_fetchCompleteEvent;
    mutable cudaEvent_t m_assignCompleteEvent;
#elif defined HIP_COMPILE
    static hipStream_t m_computeStream;
    static hipStream_t m_fetchStream;
    static hipStream_t m_assignStream;

    mutable hipEvent_t m_tempMatrixZeroingCompleteEvent;
    mutable hipEvent_t m_quantizeCompleteEvent;
    mutable hipEvent_t m_fetchCompleteEvent;
    mutable hipEvent_t m_assignCompleteEvent;
#endif
#endif // !CPUONLY

private:
    bool m_forceSync;
    bool m_quantizeOpIncludedFetch;

    // A temporary intermediate QuantizedMatrix buffer on the GPU
    QuantizedMatrix<ElemType>* m_tempGPUQuantizedMatrix;
};

// This type records and synchronizes events on the main
// GPU matrix computation work stream
class MATH_API GPUMatrixComputeStreamEvent : public MatrixComputeStreamEvent
{
public:
    GPUMatrixComputeStreamEvent(int deviceId);
    ~GPUMatrixComputeStreamEvent();

    void SynchronizeEvent() override;

    template <typename ElemType>
    void SynchronizeQuantizationComputeStreamWithEvent();

    template <typename ElemType>
    void SynchronizeDataTransferFetchStreamWithEvent();

private:
#ifndef CPUONLY
#ifdef CUDA_COMPILE
    cudaEvent_t m_mainGPUComputeStreamCUDAEvent;
#elif defined HIP_COMPILE
    hipEvent_t m_mainGPUComputeStreamCUDAEvent;
#endif
#endif
};
} } }
