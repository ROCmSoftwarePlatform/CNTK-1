//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Basics.h"
#include "BestGpu.h"

#ifndef CPUONLY

#include "GPUMatrix.h"
#include "GPUMatrixCUDAKernels.cuh"
//#include "GPUSparseMatrix.h"
#include "GPUTensor.h"
#include "CommonMatrix.h"
#define TENSOR_OPS_DECL __device__ __host__
#include "TensorOps.h"
#ifdef __HIP_PLATFORM_NVCC__
#include "device_launch_parameters.h"
#endif // nv platform check
#include <hip/hip_runtime.h>
#include <hiprand.h>
#include "hipblas.h"
#include <assert.h>
#include <memory>
#include "CntkBatchNormalization.cuh"
#include "Convolution.cuh"
#include "CuDnnRNN.h"

#pragma comment(lib, "cudart.lib") // instruct linker to reference these libs
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cusparse.lib")
#pragma comment(lib, "curand.lib")

#pragma warning(disable : 4267) // conversion from 'size_t' to 'unsigned int'; happens in CUDA <<<a,b>>> syntax if a and b are size_t
#pragma warning(disable : 4127) // conditional expression is constant; "if (sizeof(ElemType)==sizeof(float))" triggers this
#pragma warning(disable : 4702) // unreachable code; triggered for unknown reasons

#define DEFAULT_THREAD_PER_DIM 16

#define UNCONST(t, c, uc) GPUMatrix<t>& uc = const_cast<GPUMatrix<t>&>(c);

#ifdef _WIN32
// thread local storage to access the current stream, initialize to default stream
__declspec(thread)
#endif
    hipStream_t t_stream = hipStreamDefault;


#define DEFAULT_THREAD_PER_DIM 16

extern int _ConvertSMVer2Cores(int major, int minor); // forward declaration

// SetStream - set the stream that will be used by the GPU routines
void MATH_API SetStream(hipStream_t stream)
{
    t_stream = stream;
}

// GetStream - get the stream that will be used by the GPU routines
hipStream_t MATH_API GetStream()
{
    return t_stream;
}

// Helper macro patterns for elementwise methods
#define DEF_ELEMWISE_INPLACE_FUNC(f)                                      \
    template <class ElemType>                                             \
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::Inplace##f()                \
    {                                                                     \
        performElementWiseFunction(ElementWiseOperator::op##f, Data());   \
        return *this;                                                     \
    }
#define DEF_ELEMWISE_ASSIGN_FUNC(f)                                                       \
    template <class ElemType>                                                             \
    GPUMatrix<ElemType>& GPUMatrix<ElemType>::Assign##f##Of(const GPUMatrix<ElemType>& a) \
    {                                                                                     \
        if (a.IsEmpty())                                                                  \
            LogicError("Assign##f##Of: Matrix a is empty.");                              \
        if (this != &a)                                                                   \
            RequireSize(a.GetNumRows(), a.GetNumCols());                                  \
        performElementWiseFunction(ElementWiseOperator::op##f, a.Data());                 \
        return *this;                                                                     \
    }

template <>
const char* CudaErrString<hipError_t>(hipError_t x)
{
    hipDeviceSynchronize();
    return hipGetErrorString(x);
}
template <>
const char* CudaErrString<hipblasStatus_t>(hipblasStatus_t e)
{
    hipDeviceSynchronize();
    switch (e)
    {
    case HIPBLAS_STATUS_SUCCESS:          return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:  return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:     return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:    return "HIPBLAS_STATUS_INVALID_VALUE";
    //case HIPBLAS_STATUS_ARCH_MISMATCH:    return "HIPBLAS_STATUS_ARCH_MISMATCH"; //TODO: __revert__
    case HIPBLAS_STATUS_MAPPING_ERROR:    return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED: return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:   return "HIPBLAS_STATUS_INTERNAL_ERROR";
    case HIPBLAS_STATUS_NOT_SUPPORTED:    return "HIPBLAS_STATUS_NOT_SUPPORTED/HIPBLAS_STATUS_LICENSE_ERROR/HIPBLAS_STATUS_ARCH_MISMATCH";
    //case HIPBLAS_STATUS_LICENSE_ERROR:    return "HIPBLAS_STATUS_LICENSE_ERROR";
    default:                             return "(look for HIPBLAS_STATUS_xxx in cublas_api.h)";
    }
}
template <>
const char* CudaErrString<hiprandStatus_t>(hiprandStatus_t)
{
    hipDeviceSynchronize();
    return "(see hiprand.h & look for hiprandStatus or HIPRAND_STATUS_xxx)";
}

namespace Microsoft { namespace MSR { namespace CNTK {

/*static*/ std::vector<hipDeviceProp_t> GridDim::s_cachedDeviceProps;
/*static*/ std::once_flag GridDim::s_cachedDevicePropsInitFlag;

/*static*/ bool SyncGuard::s_isSyncEnabled = false;

/*static*/ void SyncGuard::EnableSync()
{
    s_isSyncEnabled = true;
}

/*static*/ bool SyncGuard::IsSyncEnabled()
{
    return s_isSyncEnabled;
}

SyncGuard::SyncGuard(bool forceSync /*= false*/)
    : m_forceSync(forceSync)
{
    m_done = nullptr;
    if (m_forceSync || s_isSyncEnabled)
    {
        CUDA_CALL(hipGetLastError());
        CUDA_CALL(hipEventCreate(&m_done));
    }
}

SyncGuard::~SyncGuard()
{
    if (m_forceSync || s_isSyncEnabled)
    {
        // The regular use of this destructor is to synchronize the GPU, but also
        // to check for errors. So this destructor is where CUDA errors would be thrown.
        // If this destructor runs during stack unwinding, then a different error has
        // already happened that should be reported; so we only clean up the resource.
        if (std::uncaught_exception())
            hipEventDestroy(m_done);
        else
        {
            // failures in a prior launch might be reported here
            CUDA_CALL(hipEventRecord(m_done));
            CUDA_CALL(hipEventSynchronize(m_done));
            CUDA_CALL(hipEventDestroy(m_done));
        }
    }
}

template <typename AllocatedElemType>
AllocatedElemType* TracingGPUMemoryAllocator::Allocate(int deviceId, size_t numRows, size_t numCols)
{
    if (IsTraceEnabled())
    {
        auto freeAndTotalMemory = GetFreeAndTotalMemoryInMBs(deviceId);
        fprintf(stderr, "Allocating Matrix<%s> (Rows = %d, Cols = %d) buffer on DeviceId = %d; GPU Memory Free = %d MB of %d MB\n", typeid(AllocatedElemType).name(), (int)numRows, (int)numCols, (int)deviceId, (int)freeAndTotalMemory.first, (int)freeAndTotalMemory.second);
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
    }

    AllocatedElemType* deviceBufferPtr = AllocateNoTrace<AllocatedElemType>(deviceId, numRows * numCols);

    if (IsTraceEnabled())
    {
        fprintf(stderr, "Allocated DeviceData = %p\n", (void*) deviceBufferPtr);
    }

    return deviceBufferPtr;
}

template <typename AllocatedElemType>
AllocatedElemType* TracingGPUMemoryAllocator::Allocate(int deviceId, size_t numElements)
{
    if (IsTraceEnabled())
    {
        auto freeAndTotalMemory = GetFreeAndTotalMemoryInMBs(deviceId);
        fprintf(stderr, "Allocating array<%s> (NumElements = %d) on DeviceId = %d; GPU Memory Free = %d MB of %d MB\n", typeid(AllocatedElemType).name(), (int)numElements, (int)deviceId, (int)freeAndTotalMemory.first, (int)freeAndTotalMemory.second);
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
    }

    AllocatedElemType* deviceBufferPtr = AllocateNoTrace<AllocatedElemType>(deviceId, numElements);

    if (IsTraceEnabled())
    {
        fprintf(stderr, "Allocated DeviceData = %p\n", (void*)deviceBufferPtr);
    }

    return deviceBufferPtr;
}

template <typename AllocatedElemType>
void TracingGPUMemoryAllocator::Free(int deviceId, AllocatedElemType* bufferPtr, bool ignoreCUDARetCode /*= false*/)
{
    PrepareDevice(deviceId);
    if (ignoreCUDARetCode)
        hipFree((void*) bufferPtr);
    else
        CUDA_CALL(hipFree((void*) bufferPtr));

    if (IsTraceEnabled())
    {
        auto freeAndTotalMemory = GetFreeAndTotalMemoryInMBs(deviceId);
        fprintf(stderr, "Freed buffer<%s> DeviceData = %p on DeviceId = %d; GPU Memory Free = %d MB of %d MB\n", typeid(AllocatedElemType).name(), (void*) bufferPtr, (int) deviceId, (int) freeAndTotalMemory.first, (int) freeAndTotalMemory.second);
        Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
    }
}


template <typename AllocatedElemType>
AllocatedElemType* TracingGPUMemoryAllocator::AllocateNoTrace(int deviceId, size_t numElements)
{
    AllocatedElemType* deviceBufferPtr;

    PrepareDevice(deviceId);
    // In case numElements is odd we allocate a buffer with one more element. The reason is
    // we might call curandGenerateNormal (e.g. for Gaussian noise injection) which would fail
    // if the number of elements it needs to generate is odd.
    CUDA_CALL(hipMalloc((void**) &deviceBufferPtr, sizeof(AllocatedElemType) * AsMultipleOf(numElements, 2)));

    return deviceBufferPtr;
}

std::pair<size_t, size_t> TracingGPUMemoryAllocator::GetFreeAndTotalMemoryInMBs(int deviceId)
{
    PrepareDevice(deviceId);

    size_t free, total;
    CUDA_CALL(hipMemGetInfo(&free, &total));

    size_t numBytesPerMB = 1 << 20;
    return {free / numBytesPerMB, total / numBytesPerMB};
}

// PrepareDevice - Setup the correct cuda context for an operation
// deviceId - the device on which the operation will take place
void PrepareDevice(DEVICEID_TYPE deviceId)
{
    THREAD_LOCAL static DEVICEID_TYPE currentDevice = DEVICEID_NOTYETDETERMINED;
    // and if we last set the device to be this device we are good
    if (deviceId == currentDevice)
        return;
    CUDA_CALL(hipSetDevice(deviceId));
    currentDevice = deviceId;
}

#pragma region DeviceBoundNumber class

template <class ElemType>
DeviceBoundNumber<ElemType>::DeviceBoundNumber(const DeviceBoundNumber<ElemType>& /*deepCopy*/)
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
DeviceBoundNumber<ElemType>::DeviceBoundNumber(DeviceBoundNumber<ElemType>&& shallowCopy)
{
    ShallowCopyFrom(shallowCopy.m_data, shallowCopy.m_computeDevice);
    shallowCopy.m_data = NULL;
}

template <class ElemType>
void DeviceBoundNumber<ElemType>::ShallowCopyFrom(ElemType* newVal, int newValsDevceId)
{
    m_computeDevice = newValsDevceId;
    m_data = newVal;
}

template <class ElemType>
DeviceBoundNumber<ElemType>::~DeviceBoundNumber()
{
    if (m_data != NULL)
    {
        if (m_computeDevice < 0)
        {
            delete m_data;
            m_data = NULL;
        }
        else
        {
            TracingGPUMemoryAllocator::Free<ElemType>(m_computeDevice, m_data);
        }
    }
}

#pragma endregion DeviceBoundNumber class

#pragma region Helper functions
template <class ElemType>
hipblasHandle_t _initHIPBLAS(int devId)
{
    PrepareDevice((DEVICEID_TYPE) devId);
    hipblasHandle_t cuHandle;
    HIPBLAS_CALL(hipblasCreate(&cuHandle));
    return cuHandle;
}

template <class ElemType>
void GPUMatrix<ElemType>::SetDevice(DEVICEID_TYPE deviceId)
{
#if defined( __HIP_ENABLE_ASSERT__ )
    assert(deviceId >= 0);
#endif

    CUDA_CALL(hipSetDevice(deviceId));
}

// PrepareDevice - Setup the correct cuda context for an operation
// deviceId - the device on which the operation will take place
//            defaults to -1, which means use matrices current device
template <class ElemType>
DEVICEID_TYPE GPUMatrix<ElemType>::PrepareDevice(DEVICEID_TYPE deviceId /*=-1*/) const
{
    // if default value use current compute device
    DEVICEID_TYPE newId = deviceId >= 0 ? deviceId : GetComputeDeviceId();

    Microsoft::MSR::CNTK::PrepareDevice(newId);
    return newId;
}

template <class ElemType>
ElemType* GPUMatrix<ElemType>::CopyToArray() const
{
    size_t numElements = GetNumElements();
    if (numElements != 0)
    {
        PrepareDevice();
        ElemType* pArray = new ElemType[numElements];
        CUDA_CALL(hipMemcpy(pArray, Data(), sizeof(ElemType) * m_numRows * m_numCols, hipMemcpyDeviceToHost));
        return pArray;
    }
    else
    {
        return NULL;
    }
}

//memory will be allocated by the callee if not enough but need to be deleted by the caller after it's done
//return number of elements copied
template <class ElemType>
size_t GPUMatrix<ElemType>::CopyToArray(ElemType*& arrayCopyTo, size_t& currentArraySize) const
{
    size_t numElements = GetNumElements();

    if (numElements > currentArraySize)
    {
        delete arrayCopyTo;
        arrayCopyTo = new ElemType[numElements];
        currentArraySize = numElements;
    }

    if (numElements != 0)
    {
        PrepareDevice();
        CUDA_CALL(hipMemcpy(arrayCopyTo, Data(), sizeof(ElemType) * numElements, hipMemcpyDeviceToHost));
    }

    return numElements;
}

template <class ElemType>
void GPUMatrix<ElemType>::CopySection(size_t numRows, size_t numCols, ElemType* dst, size_t colStride) const
{
    HIPBLAS_CALL(hipblasGetMatrix((int) numRows, (int) numCols, sizeof(ElemType),
                                Data(), (int) GetNumRows(), dst, (int) colStride));
}
template <class ElemType>
void GPUMatrix<ElemType>::ChangeDeviceTo(DEVICEID_TYPE to_id)
{
    if (to_id == CPUDEVICE)
        LogicError("to_id must be valid GPU");
    if (GetComputeDeviceId() == to_id)
        return;

    ElemType* d_dst = TracingGPUMemoryAllocator::Allocate<ElemType>(to_id, m_numRows, m_numCols);

    SetSizeAllocated(m_numRows * m_numCols);

    // check to make sure we have something to copy (on init we often have zero sized allocations)
    if (GetSizeAllocated() > 0)
    {
#if 0 // see the backlog item # 1220
        // IOMMU DMAR needs to be disabled for CUDA P2P, otherwise it will silently hang.
        // Unfortunately, hipDeviceCanAccessPeer returns true irrespective of the IOMMU settings.
        // More details: https://bugzilla.kernel.org/show_bug.cgi?id=188271
        // http://docs.nvidia.com/cuda/gpudirect-rdma/#supported-systems
        // TODO: enable UVA p2p access once this is fixed.

        // first try peer access
        int canAccessPeer = false;
        CUDA_CALL(hipDeviceCanAccessPeer(&canAccessPeer, to_id, GetComputeDeviceId()));
        if (canAccessPeer)
        {
            hipError_t hipStatus = hipDeviceEnablePeerAccess(GetComputeDeviceId(), 0);
            if (hipStatus != hipErrorPeerAccessAlreadyEnabled)
            {
                CUDA_CALL(hipStatus);
            }
            CUDA_CALL(hipMemcpyPeer(d_dst, to_id, Data(), GetComputeDeviceId(), sizeof(ElemType) * m_numRows * m_numCols));
        }
        else
#endif
        {
            // peer access didn't work, just copy normal
            // make this more efficient by keeping some buffers available for each copy
            ElemType* h_dst = NULL;
            PrepareDevice();
            CUDA_CALL(hipHostMalloc((void**) &h_dst, sizeof(ElemType) * m_numRows * m_numCols));
            CUDA_CALL(hipMemcpy(h_dst, Data(), sizeof(ElemType) * m_numRows * m_numCols, hipMemcpyDeviceToHost));
            PrepareDevice((DEVICEID_TYPE) to_id);
            CUDA_CALL(hipMemcpy(d_dst, h_dst, sizeof(ElemType) * m_numRows * m_numCols, hipMemcpyHostToDevice));
            CUDA_CALL(hipHostFree(h_dst));
        }
    }

    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
    SetBuffer(d_dst, m_numRows * m_numCols * sizeof(ElemType));

    PrepareDevice((DEVICEID_TYPE) to_id);
    SetComputeDeviceId(to_id);
}

template <class ElemType>
template <class ElemType2>
void GPUMatrix<ElemType>::CastAssignValuesOf(const GPUMatrix<ElemType2>* other)
{
    PrepareDevice();
    CUDA_LONG N = (CUDA_LONG)GetNumElements();
    int blocksPerGrid = (int)ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_castValue<ElemType, ElemType2>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, other->Data(), Data(), N);
#else
    const ElemType2* hostA = other->Data();
    ElemType* hostC = Data();
    const CUDA_LONG hostN = N;
    hipLaunchKernelGGL((_castValue<ElemType, ElemType2>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostC, hostN);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::performElementWiseFunction(ElementWiseOperator kind, const ElemType* src)
{
    PrepareDevice();
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    const ElemType* hostA;
    ElemType* hostRes;
    const CUDA_LONG hostN = N;
    switch (kind)
    {
    case ElementWiseOperator::opSigmoid:
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL(_elementWiseSigmoidOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hostA = src;
        hostRes = Data();
        //hostN = N;
        hipLaunchKernelGGL(_elementWiseSigmoidOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opTanh:
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWiseTanhOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hostA = src;
        hostRes = Data();
        //hostN = N;
        hipLaunchKernelGGL((_elementWiseTanhOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opSqrt:
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWiseSqrtOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hostA = src;
        hostRes = Data();
        //hostN = N;
        hipLaunchKernelGGL((_elementWiseSqrtOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opExp:
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWiseExpOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hostA = src;
        hostRes = Data();
        //hostN = N;
        hipLaunchKernelGGL((_elementWiseExpOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opLog:
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWiseLogOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hostA = src;
        hostRes = Data();
        //hostN = N;
        hipLaunchKernelGGL((_elementWiseLogOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opAbs:
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWiseAbsOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hostA = src;
        hostRes = Data();
        //hostN = N;
        hipLaunchKernelGGL((_elementWiseAbsOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opLinearRectifierDerivative:
        hostA = src;
        hostRes = Data();
        //hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWiseLinRectDerivativeOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hipLaunchKernelGGL((_elementWiseLinRectDerivativeOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opCosine:
        hostA = src;
        hostRes = Data();
        //hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWiseCosineOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hipLaunchKernelGGL((_elementWiseCosineOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opNegativeSine:
        hostA = src;
        hostRes = Data();
        //hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWiseNegativeSineOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hipLaunchKernelGGL((_elementWiseNegativeSineOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opAcos:
        hostA = src;
        hostRes = Data();
       // hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL(_elementWiseAcosOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hipLaunchKernelGGL(_elementWiseAcosOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return;
    case ElementWiseOperator::opAsin:
        hostA = src;
        hostRes = Data();
        //hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL(_elementWiseAsinOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hipLaunchKernelGGL(_elementWiseAsinOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return;
    case ElementWiseOperator::opCosh:
        hostA = src;
        hostRes = Data();
        //hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL(_elementWiseCoshOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hipLaunchKernelGGL(_elementWiseCoshOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opSinh:
        hostA = src;
        hostRes = Data();
        //hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL(_elementWiseSinhOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hipLaunchKernelGGL(_elementWiseSinhOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opAsinh:
        hostA = src;
        hostRes = Data();
        //hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL(_elementWiseAsinhOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hipLaunchKernelGGL(_elementWiseAsinhOnCuda<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    case ElementWiseOperator::opSigmoidDerivative:
        hostA = src;
        hostRes = Data();
        //hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWiseSigmoidDerivativeOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, src, Data(), N);
#else
        hipLaunchKernelGGL((_elementWiseSigmoidDerivativeOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
        return ;
    default: LogicError("performElementWiseFunction: unexpected op code %d", (int)kind);
    }
}

#pragma endregion Helper functions

#pragma region Constructors and Destructor

// should only be used by constructors
template <class ElemType>
void GPUMatrix<ElemType>::ZeroInit(int deviceId)
{
    BaseMatrix<ElemType>::ZeroInit();
    SetComputeDeviceId(deviceId);
}

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(int deviceId)
{
    ZeroInit(deviceId);
};

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId)
{
    ZeroInit(deviceId);
    m_numRows = numRows;
    m_numCols = numCols;
    SetSizeAllocated(GetNumElements());

    if (GetNumElements() != 0)
    {
        SetBuffer(TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), m_numRows, m_numCols), GetNumElements() * sizeof(ElemType));
        CUDA_CALL(hipMemset(Buffer(), 0, sizeof(ElemType) * GetSizeAllocated()));
    }
};

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId, ElemType* pArray, const size_t matrixFlags)
{
    ZeroInit(deviceId);
    SetValue(numRows, numCols, deviceId, pArray, matrixFlags);
};

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(const GPUMatrix<ElemType>& deepCopyFrom)
{
    ZeroInit();
    SetValue(deepCopyFrom);
}

template <class ElemType>
GPUMatrix<ElemType>::GPUMatrix(GPUMatrix<ElemType>&& moveFrom)
{
    ShallowCopyFrom(moveFrom);
    moveFrom.ZeroValues();
}

//assignment operator, deep copy
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator=(const GPUMatrix<ElemType>& deepCopyFrom)
{
    if (this != &deepCopyFrom)
    {
        SetValue(deepCopyFrom);
    }
    return *this;
}

//move assignment operator, shallow copy
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator=(GPUMatrix<ElemType>&& moveFrom)
{
    if (this != &moveFrom)
    {
        ShallowCopyFrom(moveFrom);
        moveFrom.ZeroValues();
    }
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>::~GPUMatrix(void)
{
}

// TODO: This should be in the storage object.
// Clear will clear your storage, zeroinit just drops it on the ground.
template <class ElemType>
void GPUMatrix<ElemType>::Clear()
{
    VerifyWritable(__FUNCTION__);
    //if (OwnBuffer() && m_pArray != NULL)
    if (m_sob != nullptr)
    {
        if (GetComputeDeviceId()>= 0)
        {
            // BUG: We do not check the CUDA return code for cudaFree here since this may get called
            // during processExit when cudaFree will fail. The destruction of CUDA objects during
            // process exit must be avoided
            ReleaseStorageMemory();
        }
    }

    ZeroInit(GetComputeDeviceId());
}
#pragma endregion Constructors and Destructor

template <class ElemType>
std::unique_ptr<GPUMatrix<ElemType>> GPUMatrix<ElemType>::GetOrCreateWorkspace() const
{
    // REVIEW alexeyk: not thread-safe, fine for now.
    if (m_workspace == nullptr)
        m_workspace = std::make_unique<conc_stack<std::unique_ptr<GPUMatrix<ElemType>>>>();
#if defined( __HIP_ENABLE_ASSERT__ )
    assert(m_workspace != nullptr);
#endif
    auto deviceId = GetComputeDeviceId();
    return m_workspace->pop_or_create([deviceId]()
                                      {
                                          return std::make_unique<GPUMatrix<ElemType>>(deviceId);
                                      });
}

template <class ElemType>
void GPUMatrix<ElemType>::ReleaseWorkspace(std::unique_ptr<GPUMatrix<ElemType>> src) const
{
#if defined( __HIP_ENABLE_ASSERT__ )
    assert(m_workspace != nullptr);
#endif
    m_workspace->push(std::move(src));
}

#pragma region Basic Operators
template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::ColumnSlice(size_t startColumn, size_t numCols) const
{
    if (startColumn + numCols > GetNumCols())
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) GetNumCols());

    GPUMatrix<ElemType> slice(GetComputeDeviceId());

    slice.ShallowCopyFrom(*this);
    slice.m_numCols = numCols;
    slice.m_sliceViewOffset = m_sliceViewOffset + startColumn * GetNumRows();

    return slice;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignColumnSlice(const GPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
{
    if (numCols == 0)
        LogicError("The slice cannot have 0 columns.");

    if (startColumn + numCols > fromMatrix.GetNumCols())
        InvalidArgument("The slice (%d+%d) is out of range of the source matrix (%d).", (int) startColumn, (int) numCols, (int) fromMatrix.GetNumCols());

    Clear();

    ShallowCopyFrom(fromMatrix);
    m_numCols = numCols;
    m_sliceViewOffset = fromMatrix.m_sliceViewOffset + startColumn * GetNumRows();

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::SetColumnSlice(const GPUMatrix<ElemType>& fromMatrix, size_t startColumn, size_t numCols)
{
    if (startColumn + numCols > GetNumCols())
        LogicError("The slice is out of range of the destination matrix.");
    if (numCols > fromMatrix.GetNumCols())
        InvalidArgument("The slice (%d) is out of range of the source matrix (%d).", (int) numCols, (int) fromMatrix.GetNumCols());
    if (m_numRows != fromMatrix.m_numRows)
        LogicError("The number of rows in source and destination matrices do not match");

    if (m_numRows * numCols > 0) // TODO: remove if unnecessary
        CUDA_CALL(hipMemcpy(Data() + LocateColumn(startColumn), fromMatrix.Data(), sizeof(ElemType) * m_numRows * numCols, hipMemcpyDeviceToDevice));

    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::CopyColumnsStrided(const GPUMatrix<ElemType>& fromMatrix, size_t numCols, size_t srcNumColsStride, size_t destNumColsStride)
{
    if ((((numCols - 1) * srcNumColsStride) + 1) > fromMatrix.m_numCols)
        LogicError("The numCols to copy and srcNumColsStride specified is out of range of the source matrix.");
    if ((((numCols - 1) * destNumColsStride) + 1) > m_numCols)
        LogicError("The numCols to copy and srcNumColsStride specified is out of range of the destination matrix.");
    if (m_numRows != fromMatrix.m_numRows)
        LogicError("The number of rows in source and destination matrices do not match");

    if ((m_numRows * numCols) > 0)
    {
        // Launch a kernel to do the strided copy
        CUDA_LONG N = (CUDA_LONG)(m_numRows * numCols);
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        PrepareDevice();
        SyncGuard syncGuard;
        ElemType* hostDest = Data();
        ElemType* hostSrc = fromMatrix.Data();
        CUDA_LONG hostN = N;
        CUDA_LONG hostNumRows = m_numRows;
        CUDA_LONG hostDestNumColsStride = destNumColsStride;
        CUDA_LONG hostSrcNumColsStride = srcNumColsStride;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_copyColumnsStrided<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), fromMatrix.Data(), N, (CUDA_LONG) m_numRows, (CUDA_LONG) destNumColsStride, (CUDA_LONG) srcNumColsStride);
#else
        hipLaunchKernelGGL((_copyColumnsStrided<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostDest, hostSrc, hostN, hostNumRows, hostDestNumColsStride, hostSrcNumColsStride);
#endif
    }
}

//for each column of a, we assign all rows of a to this starting from startIndex
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignToRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AddToRowSliceValuesOf: input matrix a is empty.");

    if (a.GetNumRows() != numRows)
        LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows.");

    if (startIndex + numRows > GetNumRows())
        LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows().");

    if (a.GetNumCols() != GetNumCols())
        LogicError("AddToRowSliceValuesOf: columns does not match.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostDest = Data();
    ElemType* hostSrc = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostStartIndex = startIndex;
    const CUDA_LONG hostDestRows = GetNumRows();
    const CUDA_LONG hostSrcRows = a.GetNumRows();
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignToRowSliceValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, (CUDA_LONG) startIndex, (CUDA_LONG) GetNumRows(), (CUDA_LONG) a.GetNumRows());
#else
    hipLaunchKernelGGL((_assignToRowSliceValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostDest, hostSrc, hostN, hostStartIndex, hostDestRows, hostSrcRows);
#endif
    return *this;
}

//for each column of a, we assign numRows starting from startIndex to this
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AssignRowSliceValuesOf: input matrix a is empty.");

    if (startIndex + numRows > a.GetNumRows())
        LogicError("AssignRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows().");

    RequireSize(numRows, a.GetNumCols());

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostDest = Data();
    ElemType* hostSrc = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostStartIndex = startIndex;
    const CUDA_LONG hostDestRows = numRows;
    const CUDA_LONG hostSrcRows = a.GetNumRows();
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignRowSliceValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, (CUDA_LONG) startIndex, (CUDA_LONG) numRows, (CUDA_LONG) a.GetNumRows());
#else
    hipLaunchKernelGGL((_assignRowSliceValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostDest, hostSrc, hostN, hostStartIndex, hostDestRows, hostSrcRows);
#endif
    return *this;
}

//for the row slice of this starting from startIndex we add a to it.
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddToRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AddToRowSliceValuesOf: input matrix a is empty.");

    if (a.GetNumRows() != numRows)
        LogicError("AddToRowSliceValuesOf: a.GetNumRows() != numRows.");

    if (startIndex + numRows > GetNumRows())
        LogicError("AddToRowSliceValuesOf: startIndex + numRows exceeds GetNumRows().");

    if (a.GetNumCols() != GetNumCols())
        LogicError("AddToRowSliceValuesOf: columns does not match.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostDest = Data();
    ElemType* hostSrc = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostStartIndex = startIndex;
    const CUDA_LONG hostDestRows = GetNumRows();
    const CUDA_LONG hostSrcRows = a.GetNumRows();
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_addToRowSliceValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, (CUDA_LONG) startIndex, (CUDA_LONG) GetNumRows(), (CUDA_LONG) a.GetNumRows());
#else
    hipLaunchKernelGGL((_addToRowSliceValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostDest, hostSrc, hostN, hostStartIndex, hostDestRows, hostSrcRows);
#endif
    return *this;
}

//for each column of this, we add row slice of a starting from startIndex
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddWithRowSliceValuesOf(const GPUMatrix<ElemType>& a, const size_t startIndex, const size_t numRows)
{
    if (a.IsEmpty())
        LogicError("AddWithRowSliceValuesOf: input matrix a is empty.");

    if (GetNumRows() != numRows)
        LogicError("AddWithRowSliceValuesOf: GetNumRows() != numRows.");

    if (startIndex + numRows > a.GetNumRows())
        LogicError("AddWithRowSliceValuesOf: startIndex + numRows exceeds a.GetNumRows().");

    if (a.GetNumCols() != GetNumCols())
        LogicError("AddWithRowSliceValuesOf: columns does not match.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostDest = Data();
    ElemType* hostSrc = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostStartIndex = startIndex;
    const CUDA_LONG hostDestRows = GetNumRows();
    const CUDA_LONG hostSrcRows = a.GetNumRows();
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_addWithRowSliceValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, (CUDA_LONG) startIndex, (CUDA_LONG) GetNumRows(), (CUDA_LONG) a.GetNumRows());
#else
    hipLaunchKernelGGL((_addWithRowSliceValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostDest, hostSrc, hostN, hostStartIndex, hostDestRows, hostSrcRows);
#endif
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Diagonal() const
{
    size_t m = GetNumRows();
    size_t n = GetNumCols();
    if (m != n)
        LogicError("Diagonal can be called only for square matrix. (rows=%d, cols=%d)", (int) m, (int) n);

    GPUMatrix<ElemType> diag(1, n, GetComputeDeviceId());

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostDest = diag.Data();
    ElemType* hostSrc = Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostSrcCols = n;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignToDiagonalValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, diag.Data(), Data(), N, (CUDA_LONG) n);
#else
    hipLaunchKernelGGL((_assignToDiagonalValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostDest, hostSrc, hostN, hostSrcCols);
#endif
    return diag;
}

// c = c - 1.0 for a specific position
template <class ElemType>
void GPUMatrix<ElemType>::MinusOneAt(GPUMatrix<ElemType>& c, const size_t position)
{
#if defined( __HIP_ENABLE_ASSERT__ )
    assert(position < c.GetNumElements());
#endif

    CUDA_LONG n = (CUDA_LONG) c.GetNumElements();
    CUDA_LONG p = (CUDA_LONG) position;

    int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
    // BUGBUG: PrepareDevice() missing?
    SyncGuard syncGuard;
    ElemType* hostC = c.Data();
    CUDA_LONG hostPosition = p;
    CUDA_LONG hostN = n;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_minusOneAt<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, c.Data(), p, n);
#else
    hipLaunchKernelGGL((_minusOneAt<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostC, hostPosition, hostN);
#endif
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignRepeatOf(const GPUMatrix<ElemType>& a, const size_t numRowRepeats, const size_t numColRepeats)
{
    if (this == &a)
        LogicError("AssignRepeatOf: a is the same as [this]. Does not support inplace repeat.");

    if (a.IsEmpty())
        LogicError("AssignRepeatOf: Matrix a is empty.");

    RequireSize(a.GetNumRows() * numRowRepeats, a.GetNumCols() * numColRepeats);

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    CUDA_LONG n = (CUDA_LONG) a.GetNumCols(), m = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostDest = Data();
    ElemType* hostSrc = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostSrcRows = m;
    const CUDA_LONG hostSrcCols = n;
    const CUDA_LONG hostDestRows = GetNumRows();
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignRepeatOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, m, n, (CUDA_LONG) GetNumRows());
#else
   hipLaunchKernelGGL((_assignRepeatOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostDest, hostSrc, hostN, hostSrcRows, hostSrcCols, hostDestRows);
#endif
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddToRowRepeatValuesOf(const GPUMatrix<ElemType>& a, const size_t numRepeats)
{
    if (a.IsEmpty())
        LogicError("AddToRowRepeatValuesOf: input matrix a is empty.");

    if (a.GetNumRows() != GetNumRows() * numRepeats)
        LogicError("AddToRowSliceValuesOf: a.GetNumRows() != GetNumRows() * numRepeats.");

    RequireSize(a.GetNumRows() / numRepeats, a.GetNumCols());

    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostDest = Data();
    ElemType* hostSrc = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostSrcRows = a.GetNumRows();
    const CUDA_LONG hostSrcCols = a.GetNumCols();
    const CUDA_LONG hostDestRows = GetNumRows();
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_addToRowRepeatValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, (CUDA_LONG) a.GetNumRows(), (CUDA_LONG) a.GetNumCols(), (CUDA_LONG) GetNumRows());
#else
    hipLaunchKernelGGL((_addToRowRepeatValuesOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostDest, hostSrc, hostN, hostSrcRows, hostSrcCols, hostDestRows);
#endif
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignPositiveAndShiftedNegSample(const GPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
{
    if (this == &a)
        LogicError("AssignPositiveAndShiftedNegSample: a is the same as [this]. Does not support inplace assignment.");

    if (a.IsEmpty())
        LogicError("AssignPositiveAndShiftedNegSample: Matrix a is empty.");

    RequireSize(a.GetNumRows() * (posNumber + negNumber), a.GetNumCols());

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    CUDA_LONG n = (CUDA_LONG) a.GetNumCols(), m = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostDest = Data();
    const ElemType* hostSrc = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostSrcRows = m;
    const CUDA_LONG hostSrcCols = n;
    const CUDA_LONG hostDestRows = GetNumRows();
    const CUDA_LONG hostPosNumber = posNumber;
    const CUDA_LONG hostShiftNumber = shiftNumber;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignPositiveAndShiftedNegSample<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, m, n, (CUDA_LONG) GetNumRows(), posNumber, shiftNumber);
#else
    hipLaunchKernelGGL((_assignPositiveAndShiftedNegSample<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostDest, hostSrc, hostN, hostSrcRows, hostSrcCols, hostDestRows, hostPosNumber, hostShiftNumber);
#endif
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddFoldedPositiveAndShiftedNegSample(const GPUMatrix<ElemType>& a, const size_t posNumber, const size_t negNumber, const size_t shiftNumber)
{
    if (this == &a)
        LogicError("AddFoldedPositiveAndShiftedNegSample: a is the same as [this]. Does not support inplace assignment.");

    if (a.IsEmpty())
        LogicError("AddFoldedPositiveAndShiftedNegSample: Matrix a is empty.");

    if (a.GetNumRows() != GetNumRows() * (posNumber + negNumber) || a.GetNumCols() != GetNumCols())
        LogicError("AddFoldedPositiveAndShiftedNegSample: dimensions mismatch.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    CUDA_LONG n = (CUDA_LONG) a.GetNumCols(), m = (CUDA_LONG) a.GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostFolded = Data();
    const ElemType* hostUnfolded = a.Data();
    const CUDA_LONG hostUnfoldedN = N;
    const CUDA_LONG hostUnfoldedRows = m;
    const CUDA_LONG hostUnfoldedCols = n;
    const CUDA_LONG hostFoldedRows = GetNumRows();
    const CUDA_LONG hostPosNumber = posNumber;
    const CUDA_LONG hostShiftNumber = shiftNumber;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_addFoldedPositiveAndShiftedNegSample<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, m, n, (CUDA_LONG) GetNumRows(), posNumber, shiftNumber);
#else
    hipLaunchKernelGGL((_addFoldedPositiveAndShiftedNegSample<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostFolded, hostUnfolded, hostUnfoldedN, hostUnfoldedRows, hostUnfoldedCols, hostFoldedRows, hostPosNumber, hostShiftNumber);
#endif
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Transpose() const
{
    if (IsEmpty())
        LogicError("Transpose: Matrix is empty.");

    GPUMatrix<ElemType> c(GetComputeDeviceId());
    c.AssignTransposeOf(*this);
    return c;
}

// GetCublasHandle - get a cublas.handle for the given GPU, should only need one per GPU
// computeDevice - The compute device for which the hipblas.handle is desired
// returns: hipblas.handle
// NOTE: we currently don't bother to ever free the HIPBLAS handle, it will be freed automatically by CUDA when the process ends
template <class ElemType>
hipblasHandle_t GPUMatrix<ElemType>::GetCublasHandle(int computeDevice /*=-1*/)
{
    // if the compute device is not passed, get the current device from CUDA
    if (computeDevice < 0)
        hipGetDevice(&computeDevice);

    if (computeDevice < 0 || computeDevice >= MaxGpus)
        LogicError("GetCublasHandle: Maximum GPU exceeded");
    hipblasHandle_t cuHandle = s_cuHandle[computeDevice];
    if (cuHandle == NULL)
    {
        s_cuHandle[computeDevice] = cuHandle = _initHIPBLAS<ElemType>(computeDevice);
    }
    HIPBLAS_CALL(hipblasSetStream(cuHandle, t_stream));

    return cuHandle;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTransposeOf(const GPUMatrix<ElemType>& a)
{
    if (this == &a)
        LogicError("AssignTransposeOf: a is the same as [this]. Does not support inplace transpose.");

    if (a.IsEmpty())
        LogicError("AssignTransposeOf: Matrix a is empty.");

    if (GetNumRows() != a.GetNumCols() || GetNumCols() != a.GetNumRows())
        RequireSize(a.GetNumCols(), a.GetNumRows());

    hipblasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    hipblasOperation_t transA = HIPBLAS_OP_T;
    hipblasOperation_t transB = HIPBLAS_OP_T;
    int m = (int) a.m_numCols;
    int n = (int) a.m_numRows;
    ElemType alpha = 1;
    ElemType beta = 0;
    hipblasStatus_t st;
    st = hipblasTransposeHelper(cuHandle, transA, transB, m, n, &alpha, a.Data(), n, &beta, a.Data(), n, Data(), (int) m_numRows);
    if (st != HIPBLAS_STATUS_SUCCESS)
        RuntimeError("AssignTransposeOf failed");
    m_numRows = a.m_numCols;
    m_numCols = a.m_numRows;
    return *this;
}

template <class ElemType>
__global__ void _doGatherColumnsOf(ElemType* us, size_t usStride, const ElemType beta, const ElemType* idx, size_t idxStride, const ElemType* a, size_t aStride, size_t aCols, const ElemType alpha, CUDA_LONG numElements)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    CUDA_LONG id = GridDim::GetLinearThreadId();
    if (id >= numElements) // note: there are no __syncthread() calls inside
        return;

    // id = i + jOut * usStride;
    // Each thread processes one element of the output matrix.
    CUDA_LONG i    = id % usStride; // row index into 'us' and 'a'
    CUDA_LONG jOut = id / usStride; // col index into 'us' and 'idx'

    comp_t jInF = idx[jOut * idxStride]; // this is the column we need to get
    if (isnan_(jInF) || jInF < 0)     // negative index means gap
        return;
    size_t jIn = (size_t)jInF; // TODO_NV:bad idea to store idx in ElemType matrix
    //if (jIn >= aCols)
    //    return; // actually a failure

    const ElemType&  ra = a[    i + jIn  *  aStride  ];
    ElemType&       rus = us[id/*i + jOut * usStride*/];

    comp_t res = (comp_t)ra * (comp_t)alpha;
    if (beta != 0)
        res += (comp_t)rus * (comp_t)beta;
    rus = res;
}

// *this[:,j] = a[:,idx[j]] * alpha + *this[:,j] * beta
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::DoGatherColumnsOf(ElemType beta, const GPUMatrix<ElemType>& idx, const GPUMatrix<ElemType>& a, ElemType alpha)
{
    if (idx.GetNumRows() != 1) // index is 1-dimensional only
        InvalidArgument("DoGatherColumnsOf: Map must be a row vector.");

    if (beta == 0)
        RequireSize(a.GetNumRows(), idx.GetNumCols()); // output has same column format as a, but number of columns comes from idx
    else
        VerifySize(a.GetNumRows(), idx.GetNumCols());

    if (idx.GetComputeDeviceId() != a.GetComputeDeviceId() || GetComputeDeviceId() != a.GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");
    a.PrepareDevice();

    // launch the kernel
    CUDA_LONG NN = (CUDA_LONG)GetNumElements(); // linear space identifying each individual input element
    SyncGuard syncGuard;
    GridDim grid(NN);
    ElemType* hostUs = Data();
    size_t hostUsStride = GetNumRows();
    const ElemType hostBeta = beta;
    const ElemType* hostIdx = idx.Data();
    size_t hostIdxStride = idx.GetNumRows();
    const ElemType* hostA = a.Data();
    size_t hostAStride = a.GetNumRows();
    size_t hostACols = a.GetNumCols();
    const ElemType hostAlpha = alpha;
    CUDA_LONG hostNumElements = grid.m_N;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_doGatherColumnsOf<ElemType>), dim3(grid.m_blocksPerGrid), dim3(grid.m_threadsPerBlock), 0, t_stream, Data(), GetNumRows(), beta, idx.Data(), idx.GetNumRows(), a.Data(), a.GetNumRows(), a.GetNumCols(), alpha, grid.m_N);
#else
    hipLaunchKernelGGL((_doGatherColumnsOf<ElemType>), dim3(grid.m_blocksPerGrid), dim3(grid.m_threadsPerBlock), 0, t_stream, hostUs, hostUsStride, hostBeta, hostIdx, hostIdxStride, hostA, hostAStride, hostACols, hostAlpha, hostNumElements);
#endif

    // Note: The following fails silently (no error, immediate or delayed) for numcols = 10000 under CUDA 7.0.
    //hipLaunchKernelGGL((_doGatherColumnsOf<ElemType>), dim3(GetNumCols()), dim3(GetNumRows()), 0, t_stream, Data(), GetNumRows(), beta, idx.Data(), idx.GetNumRows(), a.Data(), a.GetNumRows(), a.GetNumCols(), alpha);

    return *this;
}

// little helper for debugging
template <class ElemType>
static void Peek(const GPUMatrix<ElemType>& m, const char* which)
{
    size_t rows = m.GetNumRows();
    size_t cols = m.GetNumCols();
    ElemType buf[10000] = { 0 };
    size_t n = min(rows * cols, _countof(buf));
    CUDA_CALL(hipMemcpy(buf, m.Data(), sizeof(ElemType) * n, hipMemcpyDeviceToHost));
    UNUSED(which); UNUSED(rows); UNUSED(cols); sin(1.0f); // set breakpoint here
    //CUDA_CALL(hipMemcpy(const_cast<ElemType*>(m.Data()), buf, sizeof(ElemType) * n, hipMemcpyHostToDevice));
}

#define ALLOW_ATOMIC_SCATTER // allow to disable this, until we know atomicAdd() works properly here

template <class ElemType>
__global__ void _doScatterColumnsOf(ElemType* us, size_t usStride, size_t usCols, const ElemType* idx, size_t idxStride, const ElemType* a, size_t aStride, const ElemType alpha, CUDA_LONG numElements)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;

    CUDA_LONG id = GridDim::GetLinearThreadId();
    if (id >= numElements) // note: there are no __syncthread() calls inside
        return;

    // id = i + jIn  *  aStride
    // Each thread processes one element of a
    CUDA_LONG i   = id % aStride; // row index into 'a' and 'us'
    CUDA_LONG jIn = id / aStride; // col index into 'a' and 'idx'

    comp_t jOutF = idx[jIn * idxStride];  // this is the column we copy/add into
    if (isnan_(jOutF) || jOutF < 0)    // negative index means gap
        return;
    size_t jOut = (size_t)jOutF; // TODO_NV:bad idea to store idx in ElemType matrix
    //if (jOut >= usCols)
    //    return; // actually a failure  --TODO: This should not be necessary. Why is it?

    const ElemType&  ra =  a[id/*i + jIn  *  aStride*/];
    ElemType&       rus = us[    i + jOut * usStride  ];

    ElemType res = (comp_t)ra * (comp_t)alpha; // TODO_NV: investigate atomicAdd
    if (res != 0)             // avoid memory conflict if e.g. an entire column has no gradient
#ifdef ALLOW_ATOMIC_SCATTER
        atomicAdd(&rus, res); // rus += res;
#else
        rus += res;
#endif
    // Note: atomicAdd() is supposed to be fast in case of no conflict (the simple case of Scatter())
}

// *this[:,idx[j]] = a[:,j] * alpha + *this[:,idx[j]] * beta
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::DoScatterColumnsOf(ElemType beta, const GPUMatrix<ElemType>& idx, const GPUMatrix<ElemType>& a, ElemType alpha)
{
    if (idx.GetNumRows() != 1) // index is 1-dimensional only
        InvalidArgument("DoScatterColumnsOf: Map must be a row vector.");
    if (idx.GetNumCols() != a.GetNumCols())
        InvalidArgument("DoScatterColumnsOf: Map must have width of input vector.");
    if (a.GetNumRows() != GetNumRows())
        InvalidArgument("DoScatterColumnsOf: Output must have same height as input vector.");

    if (idx.GetComputeDeviceId() != a.GetComputeDeviceId() || GetComputeDeviceId() != a.GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");
    a.PrepareDevice();

    auto& us = *this;

#ifndef ALLOW_ATOMIC_SCATTER // verify that atomicAdd is not needed  --this is not efficient
    {
        vector<ElemType> buf(idx.GetNumRows() * idx.GetNumCols()); // idx(,)are the column(s) we copy/add into
        CUDA_CALL(hipMemcpy(buf.data(), idx.Data(), sizeof(ElemType) * buf.size(), hipMemcpyDeviceToHost));
        vector<bool> writtenTo(GetNumCols(), false); // remember whether an output column is in fact a target
        for (size_t i = 0; i < buf.size(); i++)
        {
            auto colF = buf[i];
            if (std::isnan(colF) || colF < 0)
                continue;
            size_t col = (size_t)colF;
            if (col >= GetNumCols())
                LogicError("DoScatterColumnsOf: Index value out of bounds.");
            if (writtenTo[col])
                LogicError("DoScatterColumnsOf: #ifndef ALLOW_ATOMIC_SCATTER then columns must be unique. Column idx(%d,%d)=%d is used twice.", (int)(i % idx.GetNumCols()), (int)(i / idx.GetNumCols()), (int)col);
            else
                writtenTo[col] = true;
        }
    }
#endif

    // pre-scale with beta upfront
    // Scatter may add more than one source column to the same target, so we must pre-scale with beta, and then just keep adding.
    Scale(beta, us); // if beta is 0, then this will be a memset()

    // launch the kernel
    CUDA_LONG NN = (CUDA_LONG)(a.GetNumElements()); // linear space identifying each individual input element
    SyncGuard syncGuard;
    GridDim grid(NN);
    ElemType* hostUs = Data();
    size_t hostUsStride = GetNumRows();
    size_t hostUsCols = GetNumCols();
    const ElemType* hostIdx = idx.Data();
    size_t hostIdxStride = idx.GetNumRows();
    const ElemType* hostA = a.Data();
    size_t hostAStride = a.GetNumRows();
    const ElemType hostAlpha = alpha;
    CUDA_LONG hostNumElements = NN;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_doScatterColumnsOf<ElemType>), dim3(grid.m_blocksPerGrid), dim3(grid.m_threadsPerBlock), 0, t_stream, Data(), GetNumRows(), GetNumCols(), idx.Data(), idx.GetNumRows(), a.Data(), a.GetNumRows(), alpha, NN);
#else
    hipLaunchKernelGGL((_doScatterColumnsOf<ElemType>), dim3(grid.m_blocksPerGrid), dim3(grid.m_threadsPerBlock), 0, t_stream, hostUs, hostUsStride, hostUsCols, hostIdx, hostIdxStride, hostA, hostAStride, hostAlpha, hostNumElements);
#endif

    //SyncGuard syncGuard;
    //hipLaunchKernelGGL((_doScatterColumnsOf<ElemType>), dim3(a.GetNumCols()), dim3(a.GetNumRows()), 0, t_stream, Data(), GetNumRows(), GetNumCols(), idx.Data(), idx.GetNumRows(), a.Data(), a.GetNumRows(), alpha, NN);

    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const ElemType v)
{
    if (IsEmpty())
        return;

    CUDA_LONG N = (CUDA_LONG) GetNumElements();

    // Check if value is zero, which can be set using hipMemset
    bool isZero = true;
    const char* valArray = reinterpret_cast<const char*>(&v);

    for (int i = 0; i < sizeof(ElemType); i++)
    {
        if (valArray[i] != 0)
        {
            isZero = false;
            break;
        }
    }

    if (isZero)
    {
        CUDA_CALL(hipMemset(Data(), 0, N * sizeof(ElemType)));
    }
    else
    {
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        PrepareDevice();
        SyncGuard syncGuard;
        ElemType* hostA = Data();
        const ElemType hostV = v; 
        const CUDA_LONG hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_setValue<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), v, N);
#else
        hipLaunchKernelGGL((_setValue<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostV, hostN);
#endif
    }
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const ElemType* d_v) // d_v is pointer to the value in GPU memory
{
    if (IsEmpty())
        LogicError("SetValue: Matrix is empty.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* a = Data();
    const ElemType* ad_v = d_v;
    const CUDA_LONG NN = N;
    hipLaunchKernelGGL((_setValue<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, a, ad_v, NN);
}

template <class ElemType>
void GPUMatrix<ElemType>::MaskColumnsValue(const GPUMatrix<char>& columnsMask, ElemType val, size_t numColsPerMaskEntry)
{
    if (GetNumCols() != (columnsMask.GetNumCols() * numColsPerMaskEntry))
        RuntimeError("Matrix number of columns must equal 'number of columns in column mask * numColsPerMaskEntry'.");

    if (GetComputeDeviceId() != columnsMask.GetComputeDeviceId())
        RuntimeError("Matrix and column mask must be on the same device");

    int blocksPerGrid = (int)columnsMask.GetNumCols();
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostA = Data();
    const char* hostColumnsMask = columnsMask.Data();
    CUDA_LONG hostNumCols = GetNumCols();
    CUDA_LONG hostNumRows = GetNumRows();
    ElemType hostval = val;
    CUDA_LONG hostNumColsPerMaskEntry = numColsPerMaskEntry;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_maskColumnsValue<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), columnsMask.Data(), (CUDA_LONG) GetNumCols(), (CUDA_LONG) GetNumRows(), val, numColsPerMaskEntry);
#else
    hipLaunchKernelGGL((_maskColumnsValue<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostColumnsMask, hostNumCols, hostNumRows, hostval, hostNumColsPerMaskEntry);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::SetColumn(const ElemType* colPointer, size_t colInd)
{
    if (IsEmpty())
        LogicError("SetValue: Matrix is empty.");
     if (colPointer == NULL)
        return;
    CUDA_CALL(hipMemcpy(Data() + LocateColumn(colInd), colPointer, sizeof(ElemType) * m_numRows, hipMemcpyHostToDevice));
}

template <class ElemType>
void GPUMatrix<ElemType>::SetColumn(const GPUMatrix<ElemType>& valMat, size_t colInd)
{
    if (IsEmpty())
        LogicError("SetColumn: Matrix is empty.");
    if (valMat.GetNumCols() != 1)
        LogicError("SetColumn: only support one column matrix now.");
    CUDA_CALL(hipMemcpy(Data() + LocateColumn(colInd), valMat.Data(), sizeof(ElemType) * m_numRows, hipMemcpyDeviceToDevice));
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const GPUMatrix<ElemType>& deepCopyFrom)
{
    if (this == &deepCopyFrom)
        return;

    SetValue(deepCopyFrom.GetNumRows(), deepCopyFrom.GetNumCols(), deepCopyFrom.GetComputeDeviceId(), deepCopyFrom.Data(), matrixFlagSetValueOnDevice);
}

#if 0
template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const CPUMatrix<ElemType>& /*deepCopyFrom*/)
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const CPUSparseMatrix<ElemType>& /*deepCopyFrom*/)
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const GPUSparseMatrix<ElemType>& deepCopyFrom)
{
    deepCopyFrom.CopyToDenseMatrix(*this);
}
#endif

template <class ElemType>
void GPUMatrix<ElemType>::SetValue(const size_t numRows, const size_t numCols, int deviceId, ElemType* pArray, size_t matrixFlags, DataTransferer* transferer)
{
    // handle externally managed case
    // BUGBUG: This is super super ugly, and needs to be fixed, but if matrixFlags has the right value, then we can't free anything,
    // and everything gets wonky. This should be fixed, and would go away if it is made a shared_ptr.
    if (matrixFlags & matrixFlagDontOwnBuffer)
    {
        // free the existing array if it used to be an owned array
        if ( Buffer() != NULL)
        {
            TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
        }
        m_numRows = numRows;
        m_numCols = numCols;
        SetBuffer(pArray, GetNumElements() * sizeof(ElemType), true);
        SetSizeAllocated(GetNumElements());
        SetFormat(matrixFormatDense);
        SetComputeDeviceId(deviceId);
    }
    else
    {
        if (transferer && (matrixFlags & matrixFlagSetValueOnDevice))
            RuntimeError("Asynchronous data copy from device to device is currently not supported.");

        // if the devices are different move it now
        if (GetComputeDeviceId() != deviceId && deviceId >= 0)
        {
            Clear();
            ZeroInit(deviceId);
        }

        // now RequireSize/allocate as necessary
        RequireSize(numRows, numCols);

        // copy over the content to the buffer
        PrepareDevice();
        if (pArray != NULL)
        {
            if (!(matrixFlags & matrixFormatRowMajor))
            {
                if (transferer)
                    transferer->CopyCPUToGPUAsync(pArray, GetNumElements(), sizeof(ElemType), Data());
                else
                    CUDA_CALL(hipMemcpy(Data(), pArray, sizeof(ElemType) * GetNumElements(), (matrixFlags & matrixFlagSetValueOnDevice) ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice));
            }
            else // row major: must transpose (this is not meant to be efficient, but very useful for defining inline matrices for test code)
            {
                vector<ElemType> transposed(GetNumElements());
                for (size_t i = 0; i < numRows; i++)
                    for (size_t j = 0; j < numCols; j++)
                        transposed[i + numRows * j] = pArray[j + numCols * i];

                if (transferer)
                    transferer->CopyCPUToGPUAsync(transposed.data(), GetNumElements(), sizeof(ElemType), Data());
                else
                    CUDA_CALL(hipMemcpy(Data(), transposed.data(), sizeof(ElemType) * GetNumElements(), (matrixFlags & matrixFlagSetValueOnDevice) ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice));
            }
        }
    }
    SetFormat(matrixFormatDense);
}

template <class ElemType>
void GPUMatrix<ElemType>::SetDiagonalValue(const ElemType v)
{
    CUDA_LONG N = (CUDA_LONG) GetNumRows();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostA = Data();
    const ElemType hostV  = v;
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostLd = GetNumRows();
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_setDiagonalValue<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), v, N, (CUDA_LONG) GetNumRows());
#else
    hipLaunchKernelGGL((_setDiagonalValue<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostV, hostN, hostLd);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::SetDiagonalValue(const GPUMatrix<ElemType>& vector)
{
    if (IsEmpty() || vector.IsEmpty())
        LogicError("SetDiagonalValue: Matrix is empty.");

    if (GetNumRows() != GetNumCols())
        LogicError("SetDiagonalValue: NumRows and NumCols do not agree.");

    if (vector.GetNumRows() != 1 && vector.GetNumCols() != 1)
        LogicError("SetDiagonalValue: input vector must be a vector.");

    if (vector.GetNumElements() == 1) // reduce to simple form
        SetDiagonalValue(vector.Data()[0]);

    else if (vector.GetNumRows() != GetNumRows() && vector.GetNumCols() != GetNumRows())
        LogicError("SetDiagonalValue: input vector's dimension does not agree with [this].");
    else
    {
        CUDA_LONG N = (CUDA_LONG) GetNumRows();
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
        PrepareDevice();
        SyncGuard syncGuard;
        ElemType* hostA = Data();
        const ElemType* hostB = vector.Data();
        const CUDA_LONG hostN = N;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_setDiagonalValueFromVector<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), vector.Data(), N);
#else
        hipLaunchKernelGGL((_setDiagonalValueFromVector<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostB, hostN);
#endif
    }
}

template <class ElemType>
void RescaleToRange(const GPUMatrix<ElemType>& matrix, const ElemType low, const ElemType high)
{
	size_t N = matrix.GetNumElements();
    size_t blocksPerGrid = (size_t)ceil(N / (double)GridDim::maxThreadsPerBlock);

    //Nobody is ever calling SetStream so all work is done one the same stream
    //Therefore we don't need to sync
    //SyncGuard syncGuard;
    // Copying to local variables to avoid typecast issue
    ElemType* tempA = matrix.Data();
    const CUDA_LONG tempN = N;
    const ElemType tempLow = low;
    const ElemType tempHigh = high;
    ElemType* hostA = tempA;
    const CUDA_LONG hostN = tempN;
    const ElemType hostLow = tempLow;
    const ElemType hostHigh = tempHigh;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_rescaleToRange<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, tempA, tempN, tempLow, tempHigh);
#else
    hipLaunchKernelGGL((_rescaleToRange<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostN, hostLow, hostHigh);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::SetUniformRandomValue(const ElemType low, const ElemType high, unsigned long seed)
{
    PrepareDevice();
    CreateCurandObject(seed, __FUNCTION__); // TODO call ResetCurandObject() instead?

    {
        //Nobody is ever calling SetStream so all work is done one the same stream
        //Therefore we don't need to sync
        //SyncGuard syncGuard;
        HIPRAND_CALL(hiprandGenerateUniformHelper(((hiprandGenerator_t*) s_hiprandGenerator)[0], Data(), GetNumElements()));
    }
    RescaleToRange(*this, low, high);
}

template <class ElemType>
void GPUMatrix<ElemType>::SetUniformRandomValue(RNGHandle& rngHandle, const ElemType low, const ElemType high)
{
    PrepareDevice();

    GPURNGHandle* gpuRNGHandle = dynamic_cast<GPURNGHandle*>(&rngHandle);
    assert(gpuRNGHandle != nullptr);

    {
        //Nobody is ever calling SetStream so all work is done one the same stream
        //Therefore we don't need to sync
        //SyncGuard syncGuard;
        HIPRAND_CALL(hiprandGenerateUniformHelper(gpuRNGHandle->Generator(), Data(), GetNumElements()));
    }
    RescaleToRange(*this, low, high);
}

template <class ElemType>
void SetNormalRandomValue(const GPUMatrix<ElemType>& matrix, const hiprandGenerator_t& generator, const ElemType mean, const ElemType stdev)
{
    //Nobody is ever calling SetStream so all work is done one the same stream
    //Therefore we don't need to sync
    //SyncGuard syncGuard;

    // hiprandGenerateNormal can return the error HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if GetNumElements() is odd.
    // To avoid this we always allocate a buffer of even size and potentially generate one more random element.
    auto n = AsMultipleOf(matrix.GetNumElements(), 2);
    HIPRAND_CALL(hiprandGenerateNormalHelper(generator, matrix.Data(), n, mean, stdev));
}

template <class ElemType>
void GPUMatrix<ElemType>::SetGaussianRandomValue(RNGHandle& rngHandle, const ElemType mean, const ElemType stdev)
{
    PrepareDevice();
    GPURNGHandle* gpuRNGHandle = dynamic_cast<GPURNGHandle*>(&rngHandle);
    assert(gpuRNGHandle != nullptr);
    SetNormalRandomValue(*this, gpuRNGHandle->Generator(), mean, stdev);
}

template <class ElemType>
void GPUMatrix<ElemType>::SetGumbelRandomValue(RNGHandle& rngHandle, const ElemType loc, const ElemType scale)
{
    PrepareDevice();

    GPURNGHandle* gpuRNGHandle = dynamic_cast<GPURNGHandle*>(&rngHandle);
    assert(gpuRNGHandle != nullptr);

    {
        //Nobody is ever calling SetStream so all work is done one the same stream
        //Therefore we don't need to sync
        //SyncGuard syncGuard;
        HIPRAND_CALL(hiprandGenerateUniformHelper(gpuRNGHandle->Generator(), Data(), GetNumElements()));
    }

    CUDA_LONG N = GetNumElements();
    size_t blocksPerGrid = (size_t)ceil(N / (double)GridDim::maxThreadsPerBlock);

    {
        //Nobody is ever calling SetStream so all work is done one the same stream
        //Therefore we don't need to sync
        //SyncGuard syncGuard;
        ElemType* hostA = Data();
        const CUDA_LONG hostN = N;
        const ElemType hostLoc = loc;
        const ElemType hostScale = scale;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_gumbelFromUniform<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), N, loc, scale);
#else
        hipLaunchKernelGGL((_gumbelFromUniform<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostN, hostLoc, hostScale);
#endif
    }
}

template <class ElemType>
void GPUMatrix<ElemType>::SetGaussianRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed)
{
    PrepareDevice();
    CreateCurandObject(seed, __FUNCTION__); // TODO call ResetCurandObject() instead?
    SetNormalRandomValue(*this, ((hiprandGenerator_t*)s_hiprandGenerator)[0], mean, sigma);
}

template <class ElemType>
void GPUMatrix<ElemType>::SetTruncatedNormalRandomValue(const ElemType mean, const ElemType sigma, unsigned long seed)
{
    // We use the method described in https://en.wikipedia.org/wiki/Truncated_normal_distribution
    // i.e. generate uniform, scale it to the right range, pass it through the inverse cdf, scale by sigma, and add the mean
    PrepareDevice();
    CreateCurandObject(seed, __FUNCTION__); // TODO call ResetCurandObject() instead?

    {
        //Nobody is ever calling SetStream so all work is done one the same stream
        //Therefore we don't need to sync
        //SyncGuard syncGuard;
        HIPRAND_CALL(hiprandGenerateUniformHelper(((hiprandGenerator_t*)s_hiprandGenerator)[0], Data(), GetNumElements()));
    }

    CUDA_LONG N = GetNumElements();
    size_t blocksPerGrid = (size_t)ceil(N / (double)GridDim::maxThreadsPerBlock);

    {
        //Nobody is ever calling SetStream so all work is done one the same stream
        //Therefore we don't need to sync
        //SyncGuard syncGuard;
        ElemType* hostA = Data();
        const CUDA_LONG hostN = N;
        const ElemType hostMean = mean;
        const ElemType hostSigma = sigma;
        //hipLaunchKernelGGL((_truncated_normal_transform<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), N, mean, sigma);
        hipLaunchKernelGGL((_truncated_normal_transform<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostN, hostMean, hostSigma);
    }
}

//maskRate: percentage of values masked out (similar to dropout rate)
//scaleValue: which scale value to set to the left ones (unmasked items).
template <class ElemType>
void GPUMatrix<ElemType>::SetUniformRandomMask(const ElemType maskRate, const ElemType scaleValue, RNGHandle& rngHandle)
{
    PrepareDevice();

    GPURNGHandle* gpuRNGHandle = dynamic_cast<GPURNGHandle*>(&rngHandle);
    assert(gpuRNGHandle != nullptr);

    hipEvent_t done = nullptr;
    CUDA_CALL(hipEventCreate(&done)); // TODO: why not condition on do_sync, so that we can use SyncGuard?
    HIPRAND_CALL(hiprandGenerateUniformHelper(gpuRNGHandle->Generator(), Data(), GetNumElements()));
    CUDA_CALL(hipEventRecord(done));
    CUDA_CALL(hipEventSynchronize(done));
    CUDA_CALL(hipEventDestroy(done));

    CUDA_LONG N = GetNumElements();
    size_t blocksPerGrid = (size_t) ceil(N / (double) GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    ElemType* hostA = Data();
    const CUDA_LONG hostN = N;
    const ElemType hostMaskRate = maskRate;
    const ElemType hostScaleValue = scaleValue;
    //hipLaunchKernelGGL((_setMaskAndScale<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, a, N, maskRate, scaleValue);
    hipLaunchKernelGGL((_setMaskAndScale<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostN, hostMaskRate, hostScaleValue);
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::Adagrad(GPUMatrix<ElemType>& gradients, const bool needAveMultiplier)
{
    size_t numColsNeeded = gradients.GetNumCols();
    if (needAveMultiplier)
        numColsNeeded += gradients.GetNumCols();

    if (IsEmpty() || GetNumCols() < numColsNeeded)
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);
    }

    assert(GetNumRows() == gradients.GetNumRows() && GetNumCols() == numColsNeeded);

    size_t n = gradients.GetNumElements();

    ElemType* multipliers = nullptr;
    if (needAveMultiplier)
        multipliers = Data() + n; // temp memory used to store multipliers,

    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    ElemType* hostA = Data();
    ElemType* hostD_v = gradients.Data();
    const CUDA_LONG hostN = n;
    ElemType* hostMultipliers = multipliers;
    //hipLaunchKernelGGL((_adagrad<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, Data(), gradients.Data(), n, multipliers);
    hipLaunchKernelGGL((_adagrad<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, hostA, hostD_v, hostN, hostMultipliers);

    if (!needAveMultiplier)
        return 1;

    hipblasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
    ElemType aveMultiplier = 0;
    HIPBLAS_CALL(hipblasasumHelper(cuHandle, (CUDA_LONG) n, multipliers, 1, &aveMultiplier));
    return aveMultiplier / n;
}

template <class ElemType>
void GPUMatrix<ElemType>::FSAdagrad(GPUMatrix<ElemType>& gradients,
                                    GPUMatrix<ElemType>& functionValues,
                                    ElemType learnRatePerSample,
                                    ElemType momentum,
                                    ElemType adaWeight,
                                    ElemType adaMul,
                                    ElemType unitGainFactor)
{
    size_t numColsNeeded = 2 * gradients.GetNumCols();

    if (IsEmpty() || (GetNumCols() < numColsNeeded))
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);
    }

    assert((GetNumRows() == gradients.GetNumRows()) && (GetNumCols() == numColsNeeded));

    size_t n = gradients.GetNumElements();
    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    // Copying to local variables to avoid typecast issue
    CUDA_LONG tempSize = n;
    ElemType* tempGrad = gradients.Data();
    ElemType* tempSmoothAda = Data();
    ElemType* tempSmoothMom = Data()+n;
    ElemType* tempVal = functionValues.Data();
    ElemType tempLr = learnRatePerSample;
    ElemType tempMom = momentum;
    ElemType tempAdaWeight = adaWeight;
    ElemType tempAdaMul = adaMul;
    ElemType tempTypedUnitGainFactor = unitGainFactor;
    hipLaunchKernelGGL((_fsadagrad<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, tempSize, tempGrad, tempSmoothAda, tempSmoothMom, tempVal,
																			 tempLr, tempMom, tempAdaWeight, tempAdaMul, tempTypedUnitGainFactor);
}

template <class ElemType>
void GPUMatrix<ElemType>::Adam(GPUMatrix<ElemType>& gradients,
    GPUMatrix<ElemType>& functionValues,
    ElemType learnRatePerSample,
    ElemType momentum,
    ElemType adaWeight,
    ElemType adaMul,
    ElemType epsilon,
    ElemType unitGainFactor,
    bool adamax)
{
    size_t numColsNeeded = 2 * gradients.GetNumCols();

    if (IsEmpty() || (GetNumCols() < numColsNeeded))
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);
    }

    assert((GetNumRows() == gradients.GetNumRows()) && (GetNumCols() == numColsNeeded));

    size_t n = gradients.GetNumElements();
    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    CUDA_LONG hostSize = n;
    ElemType* hostGrad = gradients.Data();
    ElemType* hostSmoothAda = Data();
    ElemType* hostSmoothMom = Data() + n;
    ElemType* hostVal = functionValues.Data();
    ElemType hostLr = learnRatePerSample;
    ElemType hostMom = momentum;
    ElemType hostAdaWeight = adaWeight;
    ElemType hostAdaMul = adaMul;
    ElemType hostEpsilon = epsilon;
    ElemType hostTypedUnitGainFactor = unitGainFactor;
    bool hostAdamax = adamax;
    //hipLaunchKernelGGL((_adam<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, n, gradients.Data(), Data(), Data() + n, functionValues.Data(),
      //  learnRatePerSample, momentum, adaWeight, adaMul, epsilon, unitGainFactor, adamax);
    hipLaunchKernelGGL((_adam<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, hostSize, hostGrad, hostSmoothAda, hostSmoothMom, hostVal,
        hostLr, hostMom, hostAdaWeight, hostAdaMul, hostEpsilon, hostTypedUnitGainFactor, hostAdamax);
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::RmsProp(GPUMatrix<ElemType>& gradients,
                                      ElemType RMS_GAMMA,
                                      ElemType RMS_WGT_INC,
                                      ElemType RMS_WGT_MAX,
                                      ElemType RMS_WGT_DEC,
                                      ElemType RMS_WGT_MIN,
                                      const bool needAveMultiplier,
                                      const bool initialized)
{
    const ElemType floor = 1e-6f;
    static ElemType* upd_gpu = (ElemType*) 0;

    size_t n = gradients.GetNumElements();
    int blocksPerGrid = (GetNumElements() + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;

    size_t numColsNeeded = gradients.GetNumCols() * 3;
    if (needAveMultiplier)
        numColsNeeded += gradients.GetNumCols();

    if (IsEmpty() || GetNumCols() < numColsNeeded || !initialized)
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);

        ElemType* avars = Data();         // accumulated variances for RMS scaling
        ElemType* signs = Data() + n;     // sign of previous gradient
        ElemType* steps = Data() + 2 * n; // current step size
        // Data()+3*n is temp memory used to store multipliers, no need to initialize
        // Copying to local variables to avoid typecast issue
        ElemType* tempAvars = avars;
        ElemType* tempSigns = signs;
        ElemType* tempSteps = steps;
        ElemType* tempCurrgrad = gradients.Data();
        const CUDA_LONG tempN = n;
        hipLaunchKernelGGL((_rmsprop_init<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, tempAvars, tempSigns, tempSteps, tempCurrgrad, tempN);
    }
    assert(GetNumRows() == gradients.GetNumRows() && GetNumCols() == numColsNeeded);

    ElemType* avars = Data();         // accumulated variances for RMS scaling
    ElemType* signs = Data() + n;     // sign of previous gradient
    ElemType* steps = Data() + 2 * n; // current step size

    ElemType* multipliers = nullptr;
    if (needAveMultiplier)
        multipliers = Data() + 3 * n; // temp memory used to store multipliers,

    if (!upd_gpu)
    {
        const ElemType upd[] = {
            2, 2, 0,
            2, 2, 0,
            1, 1, 1,
            2, 2, 0,
            1, 2, 1,
            0, 2, 2,
            1, 1, 1,
            0, 2, 2,
            0, 2, 2,
        };

        upd_gpu = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 27);
        CUDA_CALL(hipMemcpy(upd_gpu, upd, sizeof(ElemType) * _countof(upd), hipMemcpyHostToDevice));
    }
    // Copying to local variables to avoid typecast issue
    ElemType* tempAvars = avars;
    ElemType* tempSigns = signs;
    ElemType* tempSteps = steps;
    ElemType* tempCurrgrad = gradients.Data();
    const CUDA_LONG tempN = n;
    ElemType tempRMS_GAMMA = RMS_GAMMA;
    ElemType tempRMS_WGT_INC = RMS_WGT_INC;
    ElemType tempRMS_WGT_MAX = RMS_WGT_MAX;
    ElemType tempRMS_WGT_DEC = RMS_WGT_DEC;
    ElemType tempRMS_WGT_MIN = RMS_WGT_MIN;
    ElemType tempFloor = floor;
    ElemType* tempUpd_gpu = upd_gpu;
    ElemType* tempMultipliers = multipliers;
    hipLaunchKernelGGL((_rmsprop<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, tempAvars, tempSigns, tempSteps, tempCurrgrad, tempN,
                                                                           tempRMS_GAMMA, tempRMS_WGT_INC, tempRMS_WGT_MAX, tempRMS_WGT_DEC, tempRMS_WGT_MIN,
                                                                           tempFloor, tempUpd_gpu, tempMultipliers);
    if (!needAveMultiplier)
        return 1;

    hipblasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
    ElemType aveMultiplier = 0;
    HIPBLAS_CALL(hipblasasumHelper(cuHandle, (CUDA_LONG) n, multipliers, 1, &aveMultiplier));
    return aveMultiplier / n;
}

template <class ElemType>
template <class GradType>
void GPUMatrix<ElemType>::AdaDelta(GPUMatrix<GradType>& gradients, GPUMatrix<ElemType>& functionValues, ElemType learningRate, ElemType rho, ElemType epsilon)
{
    size_t numColsNeeded = 2 * gradients.GetNumCols();

    if (IsEmpty() || (GetNumCols() < numColsNeeded))
    {
        RequireSize(gradients.GetNumRows(), numColsNeeded);
        SetValue(0.0);
    }

    //assert((GetNumRows() == gradients.GetNumRows()) && (GetNumCols() == numColsNeeded));

    size_t n = gradients.GetNumElements();
    int blocksPerGrid = (n + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    CUDA_LONG hostSize = n;
    GradType* hostGrad = gradients.Data();
    ElemType* hostSmoothAda = Data();
    ElemType* hostSmoothX2 = Data() + n;
    ElemType* hostVal = functionValues.Data();
    ElemType hostLearningRate = learningRate;
    ElemType hostRho = rho;
    ElemType hostEpsilon = epsilon;
    //hipLaunchKernelGGL((_adadelta<ElemType, GradType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, n, gradients.Data(), Data(), Data() + n, functionValues.Data(), learningRate, rho, epsilon);
    hipLaunchKernelGGL((_adadelta<ElemType, GradType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, hostSize, hostGrad, hostSmoothAda, hostSmoothX2, hostVal, hostLearningRate,hostRho, hostEpsilon);
}

template <class ElemType>
void GPUMatrix<ElemType>::AdaDeltaFlushTimestamps(size_t cols, ElemType rho, int* timestamps, int currentTimestamp)
{
    // Sets all timestamps to 0 and updates the two logical buffers that this object holds
    // so that their values are the same as if a dense implementation of adadelta had been used.
    // This basically means that the values of these buffers are set to decay * original value 
    // where decay is rho ** (currentTimestamp - timestamp for that column)
    size_t rows = GetNumRows();
    int blocksPerGrid = (cols + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
    CUDA_LONG hostN = cols;
    size_t hostRows = rows;
    ElemType* hostSmoothAda = Data();
    ElemType* hostSmoothX2 = Data() + cols * rows;
    ElemType hostRho = rho;
    int* hostTimestamps = timestamps;
    int hostCurrentTimestamp = currentTimestamp;
    //hipLaunchKernelGGL(_adadeltaFlush<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, cols, rows, Data(), Data() + cols * rows, rho, timestamps, currentTimestamp);
    hipLaunchKernelGGL(_adadeltaFlush<ElemType>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, hostN, hostRows, hostSmoothAda, hostSmoothX2, hostRho, hostTimestamps, hostCurrentTimestamp);
}

template <class ElemType>
void GPUMatrix<ElemType>::Reshape(const size_t numRows, const size_t numCols)
{
    assert(numRows * numCols == GetNumElements());
    if (numRows * numCols != GetNumElements())
        InvalidArgument("Reshape: total number of elements does not match.");

    m_numRows = numRows;
    m_numCols = numCols;
}

template <class ElemType>
void GPUMatrix<ElemType>::RequireSize(const size_t numRows, const size_t numCols, bool growOnly)
{
    if (GetNumRows() != numRows || GetNumCols() != numCols)
        Resize(numRows, numCols, growOnly);
}

template <class ElemType>
void GPUMatrix<ElemType>::Resize(const size_t numRows, const size_t numCols, bool growOnly)
{
    if (GetNumRows() == numRows && GetNumCols() == numCols)
        return;

    VerifyResizable(__FUNCTION__);

    size_t numElements = numRows * numCols;
    if (numElements > GetSizeAllocated() ||                     // grow allocation
        (!growOnly && numElements != GetSizeAllocated()))   // shrink allocation if not growOnly
    {
        // If the buffer exists, free it before allocate
        if (Buffer())
        {
            TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), Buffer());
        }

        // reallocate buffer if numElements > 0
        ElemType* pArray = nullptr;
        if (numElements > 0)
        {
            pArray = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), numRows, numCols);
        }

        SetBuffer(pArray, numElements * sizeof(ElemType));
        SetSizeAllocated(numElements);
    }

    // success
    m_sliceViewOffset = 0;
    m_numRows = numRows;
    m_numCols = numCols;
}

template <class ElemType>
size_t GPUMatrix<ElemType>::LocateElement(const size_t row, const size_t col) const
{
    assert(row < m_numRows && col < m_numCols);
    return LocateColumn(col) + row; // matrix in column-wise storage
}

template <class ElemType>
size_t GPUMatrix<ElemType>::LocateColumn(const size_t col) const
{
    assert(col < GetNumCols());
    return col * m_numRows; // matrix in column-wise storage
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::Get00Element() const
{
    ElemType res = 0;
    CUDA_CALL(hipMemcpy(&res, Data(), sizeof(ElemType), hipMemcpyDeviceToHost));
    return res;
}
#pragma endregion Basic Operators

#pragma region Member BLAS Functions
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator+=(ElemType alpha)
{
    if (IsEmpty())
        LogicError("operator+=: Matrix is empty.");
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    ElemType* hostA = Data();
    const ElemType hostV = alpha;
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_addValue<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), alpha, N);
    hipLaunchKernelGGL((_addValue<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostV, hostN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator+(ElemType alpha) const
{
    if (IsEmpty())
        LogicError("operator+: Matrix is empty.");

    GPUMatrix<ElemType> c(*this);
    c += alpha;
    return c;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSumOf(const ElemType alpha, const GPUMatrix<ElemType>& a)
{
    SetValue(a);
    (*this) += alpha;
    return (*this);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator+=(const GPUMatrix<ElemType>& a)
{
    ScaleAndAdd(1, a, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator+(const GPUMatrix<ElemType>& a) const
{
    if (GetNumElements() == 1)
    {
        GPUMatrix<ElemType> c(a);
        c += Get00Element();
        return c;
    }
    else if (a.GetNumElements() == 1)
    {
        GPUMatrix<ElemType> c(*this);
        c += a.Get00Element();
        return c;
    }
    else
    {
        GPUMatrix<ElemType> c(*this); // this implementation will introduce a copy overhead. but make resue of the code
        c += a;
        return c;
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSumOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    SetValue(a);
    (*this) += b;
    return (*this);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator-=(ElemType alpha)
{
    if (IsEmpty())
        LogicError("operato-=: Matrix is empty.");
    return operator+=(-1 * alpha);
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator-(ElemType alpha) const
{
    if (IsEmpty())
        LogicError("operator-: Matrix is empty.");
    return operator+(-1 * alpha);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignDifferenceOf(const ElemType alpha, const GPUMatrix<ElemType>& a)
{
    RequireSize(a.m_numRows, a.m_numCols);
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType hostAlpha = alpha;
    const ElemType* hostA = a.Data();
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_assignDifferenceOf1<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), alpha, a.Data(), N);
    hipLaunchKernelGGL((_assignDifferenceOf1<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostAlpha, hostA, hostN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignDifferenceOf(const GPUMatrix<ElemType>& a, const ElemType alpha)
{
    RequireSize(a.m_numRows, a.m_numCols);
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType hostAlpha = alpha;
    const ElemType* hostA = a.Data();
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_assignDifferenceOf2<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), alpha, a.Data(), N);
    hipLaunchKernelGGL((_assignDifferenceOf2<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostAlpha, hostA, hostN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator-=(const GPUMatrix<ElemType>& a)
{
    ScaleAndAdd(-1, a, *this);

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator-(const GPUMatrix<ElemType>& a) const
{
    GPUMatrix<ElemType> c(*this); // this implementation will introduce a copy overhead. but make resue of the code
    c -= a;
    return c;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignDifferenceOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (this != &a)
    {
        RequireSize(a.GetNumRows(), a.GetNumCols());
        SetValue(a);
    }
    (*this) -= b;
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator*=(ElemType alpha)
{
    Scale(alpha, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator*(ElemType alpha) const
{
    GPUMatrix<ElemType> c(GetNumRows(), GetNumCols(), GetComputeDeviceId());
    Scale(alpha, *this, c);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignProductOf(const ElemType alpha, const GPUMatrix<ElemType>& a)
{
    Scale(alpha, a, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignProductOf(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB)
{
    if (a.GetNumElements() == 1)
    {
        if (transposeB)
            AssignTransposeOf(b);
        (*this) *= a.Get00Element();
    }
    else if (b.GetNumElements() == 1)
    {
        if (transposeA)
            AssignTransposeOf(a);
        (*this) *= b.Get00Element();
    }
    else
        Multiply(a, transposeA, b, transposeB, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator*(const GPUMatrix<ElemType>& a) const
{
    const GPUMatrix<ElemType>& us = *this;
    if (GetNumElements() == 1)
    {
        GPUMatrix<ElemType> c(GetComputeDeviceId());
        c.AssignProductOf(Get00Element(), a);
        return c;
    }
    else if (a.GetNumElements() == 1)
    {
        GPUMatrix<ElemType> c(GetComputeDeviceId());
        c.AssignProductOf(a.Get00Element(), us);
        return c;
    }
    else
    {
        GPUMatrix<ElemType> c(GetNumRows(), a.GetNumCols(), GetComputeDeviceId());
        Multiply(*this, a, c);
        return c;
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator/=(ElemType alpha)
{
    (*this) *= 1 / alpha;
    return (*this);
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator/(ElemType alpha) const
{
    return ((*this) * (1 / alpha));
}

//element-wise power
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::operator^=(ElemType alpha)
{
    GPUMatrix<ElemType>& us = *this;
    ElementWisePower(alpha, us, us);
    return us;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::operator^(ElemType alpha) const
{
    GPUMatrix<ElemType> c(GetNumRows(), GetNumCols(), GetComputeDeviceId());
    ElementWisePower(alpha, *this, c);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementPowerOf(const GPUMatrix<ElemType>& a, const ElemType power)
{
    ElementWisePower(power, a, *this);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddElementProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AddElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match [this].");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const ElemType* hostB = b.Data();
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_addElementProductOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), b.Data(), N);
    hipLaunchKernelGGL((_addElementProductOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostB, hostN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ColumnElementMultiplyWith(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("ColumnElementMultiplyWith: Matrix is empty.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
        InvalidArgument("ColumnElementMultiplyWith: The input matrix should be a col vector and match [this]'s rows.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumRows();
    CUDA_LONG M = (CUDA_LONG) GetNumCols();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostM = M;
    //hipLaunchKernelGGL((_columnElementMultiplyWith<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, M);
    hipLaunchKernelGGL((_columnElementMultiplyWith<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostN, hostM);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::RowElementMultiplyWith(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("RowElementMultiplyWith: Matrix is empty.");

    if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
        InvalidArgument("RowElementMultiplyWith: The input matrix should be a row vector and match [this]'s columns.");

    CUDA_LONG N = (CUDA_LONG) GetNumRows();
    CUDA_LONG M = (CUDA_LONG) a.GetNumCols();
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostM = M;
    //hipLaunchKernelGGL((_rowElementMultiplyWith<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, Data(), a.Data(), N, M);
    hipLaunchKernelGGL((_rowElementMultiplyWith<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, hostUs, hostA, hostN, hostM);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::RowElementDivideBy(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("RowElementDivideBy: Matrix is empty.");

    if (!(a.GetNumRows() == 1 && a.GetNumCols() == GetNumCols()))
        InvalidArgument("RowElementDivideBy: The input matrix should be a row vector and match [this]'s columns.");

    CUDA_LONG N = (CUDA_LONG) GetNumRows();
    CUDA_LONG M = (CUDA_LONG) a.GetNumCols();
    int blocksPerGrid = (int) ceil(1.0 * M / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostM = M;
    //hipLaunchKernelGGL((_rowElementDivideBy<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, Data(), a.Data(), N, M);
    hipLaunchKernelGGL((_rowElementDivideBy<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, hostUs, hostA, hostN, hostM);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ColumnElementDivideBy(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty() || IsEmpty())
        LogicError("ColumnElementDivideBy: Matrix is empty.");

    if (!(a.GetNumRows() == GetNumRows() && a.GetNumCols() == 1))
        InvalidArgument("ColumnElementDivideBy: The input matrix should be a col vector and match [this]'s rows.");

    CUDA_LONG N = (CUDA_LONG) a.GetNumRows();
    CUDA_LONG M = (CUDA_LONG) GetNumCols();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const CUDA_LONG hostN = N;
    const CUDA_LONG hostM = M;
    //hipLaunchKernelGGL((_ColumnElementDivideBy<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N, M);
    hipLaunchKernelGGL((_ColumnElementDivideBy<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostN, hostM);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ElementInverse()
{
    if (IsEmpty())
        LogicError("ElementInverse: Matrix is empty.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_elemInverse<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), N);
    hipLaunchKernelGGL((_elemInverse<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementInverseOf(const GPUMatrix<ElemType>& a)
{
    SetValue(a);
    return ElementInverse();
}

DEF_ELEMWISE_INPLACE_FUNC(Sigmoid)

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSigmoidOf(const GPUMatrix<ElemType>& a)
{
    RequireSize(a.GetNumRows(), a.GetNumCols());
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    // _elementWIseSigmoidOnCuda has an implementation that avoids possible overflow errors, but has a slight accuracy regression.
#if 0
    const ElemType* hostA = a.data();
    ElemType* hostRes = Data();
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_elementWiseSigmoidOnCuda), dim3(blocksPerGrid), dim3(threadsPerBlock), 0, t_stream, a.Data(), Data(), N);
    hipLaunchKernelGGL((_elementWiseSigmoidOnCuda), dim3(blocksPerGrid), dim3(threadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#else
    const ElemType* hostA = a.Data();
    ElemType* hostRes = Data();
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_assignSigmoidOf), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, static_cast<const ElemType*>(a.Data()), static_cast<ElemType*>(Data()), static_cast<const CUDA_LONG>(N));
    hipLaunchKernelGGL((_assignSigmoidOf), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostRes, hostN);
#endif
    return *this;
}

DEF_ELEMWISE_INPLACE_FUNC(SigmoidDerivative)
DEF_ELEMWISE_ASSIGN_FUNC(SigmoidDerivative)

template <class ElemType>
void GPUMatrix<ElemType>::AssignNoiseContrastiveEstimation(const GPUMatrix<ElemType>& a,
                                                           const GPUMatrix<ElemType>& b, const GPUMatrix<ElemType>& bias, size_t sampleCount, GPUMatrix<ElemType>& tmp, GPUMatrix<ElemType>& c)
//this:   samples+probs
// a  :   hidden
// b  :   embedding
// tmp:   softmax
// c  :   loglikelihood
{
    UNCONST(ElemType, a, my_a);
    UNCONST(ElemType, b, my_b);
    UNCONST(ElemType, bias, my_bias);
    SyncGuard syncGuard;
    // a: dim * minibatch
    // b: dim * |vocab|
    int p = 512;
    int width = a.GetNumRows(); // dimension of hidden vector

    while (p / 2 > width)
        p = p / 2; 
 
    const ElemType* hostCol = Data();
    int hostNumRows = sampleCount;
    int hostSampleCount = m_numRows/2;
    const ElemType* hostA = my_a.Data();
    int hostNumCols_a = a.GetNumRows();
    const ElemType* hostB = my_b.Data();
    const ElemType* hostBias = my_bias.Data();
    ElemType* hostRes = tmp.Data();

    // note: kernel has hard-coded dimension of 512
    /*hipLaunchKernelGGL((_computeNceOutputMax512Threads<ElemType>), dim3(GetNumElements() / 2), dim3(p), 0, 0, 
        Data(),
        sampleCount,
        m_numRows / 2,
        my_a.Data(), // a
        a.GetNumRows(),
        my_b.Data(), // b
        my_bias.Data(),
        tmp.Data()); // tmp*/
    hipLaunchKernelGGL((_computeNceOutputMax512Threads<ElemType>), dim3(GetNumElements() / 2), dim3(p), 0, 0, 
        hostCol,
        hostNumRows,
        hostSampleCount,
        hostA, // a
        hostNumCols_a,
        hostB, // b
        hostBias,
        hostRes); // tmp

    p = 512;
    while (p / 2 > GetNumElements() / 2)
        p = p / 2;
    const ElemType* hostVal = Data();
    int hostNumRows1 = sampleCount;
    int hostSampleCount1 = m_numRows/2;
    const ElemType* hostA1 = my_a.Data();
    int hostWidth = a.GetNumCols(); // number of columns in a
    const ElemType* hostB1 = my_b.Data();
    ElemType* hostTmp = tmp.Data();
    ElemType* hostC = c.Data(); // run on 512 threads per block
    // summing up objective must be done in one block
    // note: kernel has hard-coded dimension of 512
    /*hipLaunchKernelGGL((_assignNoiseContrastiveEstimationMax512Threads<ElemType>), dim3(1), dim3(p), 0, 0, 
        Data(),
        sampleCount,
        m_numRows / 2,
        my_a.Data(),
        a.GetNumCols(),
        my_b.Data(),
        tmp.Data(),
        c.Data());*/
    hipLaunchKernelGGL((_assignNoiseContrastiveEstimationMax512Threads<ElemType>), dim3(1), dim3(p), 0, 0, 
        hostVal,
        hostNumRows1,
        hostSampleCount1,
        hostA1,
        hostWidth,
        hostB1,
        hostTmp,
        hostC);
}

template <class ElemType>
void GPUMatrix<ElemType>::AssignNCEDerivative(GPUMatrix<ElemType>& tmp, const GPUMatrix<ElemType>& a,
                                              const GPUMatrix<ElemType>& b, size_t inputIndex, GPUMatrix<ElemType>& c)
{
    UNCONST(ElemType, a, my_a);
    UNCONST(ElemType, b, my_b);
    SyncGuard syncGuard;
    int p = 512;
    int width = a.GetNumRows();
    while (p / 2 > width)
        p = p / 2;

    const ElemType* hostVal = Data();
    int hostNumRows = tmp.GetNumCols();
    int hostSampleCount = m_numRows / 2;
    const ElemType* hostA = my_a.Data();
    int hostWidth = a.GetNumRows(); // number of columns in a
    const ElemType* hostB = my_b.Data();
    const ElemType* hostTmp = tmp.Data();
    ElemType* hostC = c.Data();
    size_t hostInputIndex = inputIndex;
/*    hipLaunchKernelGGL((_assignNceDerivativeNew<ElemType>), dim3((tmp.GetNumElements() + p - 1) / p), dim3(p), 0, 0, 
        Data(),
        tmp.GetNumCols(),
        m_numRows / 2,
        my_a.Data(),
        a.GetNumRows(),
        my_b.Data(),
        tmp.Data(),
        c.Data(),
      	inputIndex);*/
    hipLaunchKernelGGL((_assignNceDerivativeNew<ElemType>), dim3((tmp.GetNumElements() + p - 1) / p), dim3(p), 0, 0, 
        hostVal,
        hostNumRows,
        hostSampleCount,
        hostA,
        hostWidth,
        hostB,
        hostTmp,
        hostC,
      	hostInputIndex);

}

template <class ElemType>
void GPUMatrix<ElemType>::AssignSoftmaxSum(const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
{
    UNCONST(ElemType, a, my_a);
    SyncGuard syncGuard;
    int p = 512;
    int width = a.GetNumRows();
    while (p / 2 > width)
        p = p / 2;
    const ElemType* hostSoftmax = my_a.Data();
    int hostSampleCount = width;
    const ElemType* hostA = Data();
    ElemType* hostC = c.Data(); // run on 512 threads per block

    // note: kernel has hard-coded dimension of 512
/*    hipLaunchKernelGGL((_assignSoftmaxSumMax512Threads<ElemType>), dim3(1), dim3(p), 0, 0, 
        my_a.Data(),
        width,
        Data(),
        c.Data());*/
    hipLaunchKernelGGL((_assignSoftmaxSumMax512Threads<ElemType>), dim3(1), dim3(p), 0, 0, 
        hostSoftmax,
        hostSampleCount,
        hostA,
        hostC);
}

template <class ElemType>
void GPUMatrix<ElemType>::AssignNCEUnnormalizedEval(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    assert(a.GetComputeDeviceId() == b.GetComputeDeviceId());
    assert(GetNumRows() == a.GetNumRows());
    assert(GetNumCols() == b.GetNumRows());
    assert(a.GetNumCols() == b.GetNumRows());
    UNUSED(a);
    UNUSED(b);
    UNUSED(c); // TODO: this function seems like a stub
    /*
        EnsureAuxMemory();
        int p = 512;
        int width = a.GetNumCols();
        while (p / 2 > width) p = p / 2;

        // this kernel need be launched in nnz blocks
        hipLaunchKernelGGL((_sparseInnerProductDenseTimesDense<ElemType>), dim3(m_nz), dim3(p), 0, 0, 
        m_dVal,
        m_buf,
        m_dCol,
        m_nz,
        GetNumRows(),
        a.Buffer(),
        b.Buffer(),
        b.GetNumRows(),
        m_res);

        // sum up the results
        hipLaunchKernelGGL((_reductionSum32<ElemType>), dim3(1), dim3(32), 0, 0, m_res, c.Buffer(), m_nz);
*/
}

DEF_ELEMWISE_INPLACE_FUNC(Tanh)
DEF_ELEMWISE_ASSIGN_FUNC(Tanh)

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceLogSoftmax(const bool isColWise)
{
    if (IsEmpty())
        LogicError("InplaceLogSoftmax: Matrix is empty.");

    PrepareDevice();
    if (isColWise)
    {
        CUDA_LONG N = (CUDA_LONG) GetNumCols(); // one kernel per column
        int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        ElemType* hostA = Data();
        const CUDA_LONG hostM_numCols = m_numCols;
        const CUDA_LONG hostM_numRows = m_numRows; // ld
        //hipLaunchKernelGGL((_logSoftMaxColWise), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), (CUDA_LONG) m_numCols, (CUDA_LONG) m_numRows);
        hipLaunchKernelGGL((_logSoftMaxColWise), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostM_numCols, hostM_numRows);
    }
    else
    {
        CUDA_LONG N = (CUDA_LONG) GetNumRows(); // one kernel per column
        int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
        ElemType* hostA = Data();
        const CUDA_LONG hostM_numCols = m_numCols;
        const CUDA_LONG hostM_numRows = m_numRows; // ld
        //hipLaunchKernelGGL((_logSoftMaxRowWise), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), (CUDA_LONG) m_numCols, (CUDA_LONG) m_numRows);
        hipLaunchKernelGGL((_logSoftMaxRowWise), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostM_numCols, hostM_numRows);
    }
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignLogSoftmaxOf(const GPUMatrix<ElemType>& a, const bool isColWise)
{
    RequireSize(a.GetNumRows(), a.GetNumCols());
    if (isColWise)
    {
        PrepareDevice();
        CUDA_LONG N = (CUDA_LONG) GetNumCols();
        CUDA_LONG M = (CUDA_LONG) GetNumRows();
        SyncGuard syncGuard;
        const ElemType* hostA = a.Data();
        ElemType* hostUs = Data();
        const CUDA_LONG hostM_numCols = N;
        const CUDA_LONG hostM_numRows = M; // ld
        // note: kernel uses hard-coded thread dimension
        //hipLaunchKernelGGL((_assignColumnwiseLogSoftmaxOf512Threads), dim3(N), dim3(512), 0, t_stream, static_cast<const ElemType*>(a.Data()), static_cast<ElemType*>(Data()), static_cast<const CUDA_LONG>(N), static_cast<const CUDA_LONG>(M));
        hipLaunchKernelGGL((_assignColumnwiseLogSoftmaxOf512Threads), dim3(N), dim3(512), 0, t_stream, hostA, hostUs, hostM_numCols, hostM_numRows);
    }
    else
    {
        NOT_IMPLEMENTED;
    }

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceHardmax(const bool isColWise)
{
    return AssignHardmaxOf(*this, isColWise);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignHardmaxOf(const GPUMatrix<ElemType>& a, const bool isColWise)
{
    RequireSize(a.GetNumRows(), a.GetNumCols());
    if (isColWise)
    {
        PrepareDevice();
        CUDA_LONG N = (CUDA_LONG) GetNumCols();
        CUDA_LONG M = (CUDA_LONG) GetNumRows();
        SyncGuard syncGuard;
        const ElemType* hostA = a.Data();
        ElemType* hostUs = Data();
        const CUDA_LONG hostM_numCols = N;
        const CUDA_LONG hostM_numRows = M; // ld
        // note: kernel uses hard-coded thread dimension
        //hipLaunchKernelGGL((_assignColumnwiseHardmaxOf512Threads), dim3(N), dim3(512), 0, t_stream, static_cast<const ElemType*>(a.Data()), static_cast<ElemType*>(Data()), static_cast<const CUDA_LONG>(N), static_cast<const CUDA_LONG>(M));
        hipLaunchKernelGGL((_assignColumnwiseHardmaxOf512Threads), dim3(N), dim3(512), 0, t_stream, hostA, hostUs, hostM_numCols, hostM_numRows);
    }
    else
    {
        NOT_IMPLEMENTED;
    }

    return *this;
}

DEF_ELEMWISE_INPLACE_FUNC(Sqrt)
DEF_ELEMWISE_ASSIGN_FUNC(Sqrt)

DEF_ELEMWISE_INPLACE_FUNC(Exp)
DEF_ELEMWISE_ASSIGN_FUNC(Exp)

DEF_ELEMWISE_INPLACE_FUNC(Log)
DEF_ELEMWISE_ASSIGN_FUNC(Log)

DEF_ELEMWISE_INPLACE_FUNC(Abs)
DEF_ELEMWISE_ASSIGN_FUNC(Abs)

DEF_ELEMWISE_INPLACE_FUNC(LinearRectifierDerivative)
DEF_ELEMWISE_ASSIGN_FUNC(LinearRectifierDerivative)

DEF_ELEMWISE_INPLACE_FUNC(Cosine)
DEF_ELEMWISE_ASSIGN_FUNC(Cosine)

DEF_ELEMWISE_INPLACE_FUNC(NegativeSine)
DEF_ELEMWISE_ASSIGN_FUNC(NegativeSine)

DEF_ELEMWISE_INPLACE_FUNC(Acos)
DEF_ELEMWISE_ASSIGN_FUNC(Acos)

DEF_ELEMWISE_INPLACE_FUNC(Asin)
DEF_ELEMWISE_ASSIGN_FUNC(Asin)

DEF_ELEMWISE_INPLACE_FUNC(Cosh)
DEF_ELEMWISE_ASSIGN_FUNC(Cosh)

DEF_ELEMWISE_INPLACE_FUNC(Sinh)
DEF_ELEMWISE_ASSIGN_FUNC(Sinh)

DEF_ELEMWISE_INPLACE_FUNC(Asinh)
DEF_ELEMWISE_ASSIGN_FUNC(Asinh)

DEF_ELEMWISE_INPLACE_FUNC(Atanh)
DEF_ELEMWISE_ASSIGN_FUNC(Atanh)

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTruncateBottom(const ElemType threshold)
{
    return AssignTruncateBottomOf(*this, threshold);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTruncateBottomOf(const GPUMatrix<ElemType>& a, const ElemType threshold)
{
    if (a.IsEmpty())
        LogicError("AssignTruncateBottomOf: Matrix a is empty.");

    if (this != &a)
    {
        RequireSize(a.GetNumRows(), a.GetNumCols());
    }

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    // Copying to local variables to avoid typecast issue
    ElemType* tempUs = Data();
    const ElemType* tempA = a.Data();
    const ElemType tempThreshold = threshold;
    const CUDA_LONG tempN = N;
    hipLaunchKernelGGL((_assignTruncateBottom<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, tempUs, tempA, tempThreshold, tempN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTruncateTop(const ElemType threshold)
{
    return AssignTruncateTopOf(*this, threshold);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignTruncateTopOf(const GPUMatrix<ElemType>& a, const ElemType threshold)
{
    if (a.IsEmpty())
        LogicError("AssignTruncateTopOf: Matrix a is empty.");

    if (this != &a)
    {
        RequireSize(a.GetNumRows(), a.GetNumCols());
    }

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const ElemType hostThreshold = threshold;
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_assignTruncateTop<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), threshold, N);
    hipLaunchKernelGGL((_assignTruncateTop<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostThreshold, hostN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceTruncate(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceTruncate: Matrix is empty.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostA = Data();
    const ElemType hostThreshold = threshold;
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_inplaceTruncate<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), threshold, N);
    hipLaunchKernelGGL((_inplaceTruncate<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostThreshold, hostN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::InplaceSoftThreshold(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("InplaceSoftThreshold: Matrix is empty.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostA = Data();
    const ElemType hostThreshold = threshold;
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_inplaceSoftThreshold<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), threshold, N);
    hipLaunchKernelGGL((_inplaceSoftThreshold<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostThreshold, hostN);
    return *this;
}
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::SetToZeroIfAbsLessThan(const ElemType threshold)
{
    if (IsEmpty())
        LogicError("SetToZeroIfAbsLessThan: Matrix is empty.");
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostA = Data();
    const ElemType hostThreshold = threshold;
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_setToZeroIfAbsLessThan<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), threshold, N);
    hipLaunchKernelGGL((_setToZeroIfAbsLessThan<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostThreshold, hostN);
    return *this;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::SumOfAbsElements() const
{
    if (IsEmpty())
        LogicError("SumOfAbsElements: Matrix is empty");

    hipblasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
    ElemType res = 0;
    HIPBLAS_CALL(hipblasasumHelper(cuHandle, (CUDA_LONG) GetNumElements(), Data(), 1, &res));
    return res;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::SumOfElements() const
{
    if (IsEmpty())
        LogicError("SumOfElements: Matrix is empty");

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);
    ElemType h_sum;

    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    const ElemType* hostData = Data();
    ElemType* hostSum = d_sum;
    CUDA_LONG hostN = GetNumElements();
    //hipLaunchKernelGGL((_reductionSum1024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, Data(), d_sum, (CUDA_LONG)GetNumElements());
    hipLaunchKernelGGL((_reductionSum1024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, hostData, hostSum, hostN);
    CUDA_CALL(hipMemcpy(&h_sum, d_sum, sizeof(ElemType), hipMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_sum);
    return h_sum;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSumOfElements(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSumOfElements: Matrix a is empty");

    RequireSize(1, 1);

    PrepareDevice();
    SyncGuard syncGuard;
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    ElemType* hostToAssign = Data();
    const ElemType* hostData = a.Data();
    CUDA_LONG hostN = a.GetNumElements(); // length of data
    CUDA_LONG hostM = GetNumElements(); 
    //hipLaunchKernelGGL((_reductionSumAndAssign1024Threads<ElemType>), dim3(1), dim3(1024), 0, 0, Data(), a.Data(), (CUDA_LONG)a.GetNumElements(), (CUDA_LONG)GetNumElements());
    hipLaunchKernelGGL((_reductionSumAndAssign1024Threads<ElemType>), dim3(1), dim3(1024), 0, 0, hostToAssign, hostData, hostN, hostM);
    return (*this);
}

template <class ElemType>
DeviceBoundNumber<ElemType> GPUMatrix<ElemType>::Sum_AsDeviceBoundNum() const
{
    if (IsEmpty())
        LogicError("Matrix is empty");
    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);

    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    const ElemType* hostData = Data();
    ElemType* hostSum = d_sum;
    CUDA_LONG hostN = GetNumElements();
    //hipLaunchKernelGGL((_reductionSum1024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, Data(), d_sum, (CUDA_LONG)GetNumElements());
    hipLaunchKernelGGL((_reductionSum1024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, hostData, hostSum, hostN);
    DeviceBoundNumber<ElemType> result;
    result.ShallowCopyFrom(d_sum, GetComputeDeviceId());
    return result;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::AbsoluteMax() const
{
    hipblasHandle_t cuHandle = GetCublasHandle(GetComputeDeviceId());
    ElemType res;
    int resInd = 0;
    hipblasamaxHelper(cuHandle, (CUDA_LONG)GetNumElements(), Data(), 1, &resInd);
    resInd--;
    CUDA_CALL(hipMemcpy(&res, Data() + resInd, sizeof(ElemType), hipMemcpyDeviceToHost));
    return res;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ElementMultiplyWith(const GPUMatrix<ElemType>& a)
{
    if (IsEmpty() || a.IsEmpty())
        LogicError("ElementMultiplyWith: Matrix is empty.");

    GPUMatrix<ElemType>& us = *this;
    assert(us.GetNumRows() == a.GetNumRows() && us.GetNumCols() == a.GetNumCols());
    if (us.GetNumRows() != a.GetNumRows() || us.GetNumCols() != a.GetNumCols())
        InvalidArgument("The matrix dimensions do not match.");

    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostA = Data();
    const ElemType* hostB = a.Data();
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_elemMul<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), N);
    hipLaunchKernelGGL((_elemMul<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostB, hostN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    RequireSize(a.GetNumRows(), a.GetNumCols());
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const ElemType* hostB = b.Data();
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_assignElementProductOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), b.Data(), N);
    hipLaunchKernelGGL((_assignElementProductOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostB, hostN);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ElementDivideBy(const GPUMatrix<ElemType>& a)
{
    return AssignElementDivisionOf(*this, a);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementDivisionOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementDivisionOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    RequireSize(a.GetNumRows(), a.GetNumCols());
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const ElemType* hostB = b.Data();
    const CUDA_LONG hostN = N;
    //hipLaunchKernelGGL((_assignElementDivisionOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), b.Data(), N);
    hipLaunchKernelGGL((_assignElementDivisionOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostB, hostN);
    return *this;
}

template <class ElemType>
bool GPUMatrix<ElemType>::IsEqualTo(const GPUMatrix<ElemType>& a, const ElemType threshold /*= 1e-8*/) const
{
    return AreEqual(*this, a, threshold);
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorSum(const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c, const bool isColWise)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }

    a.PrepareDevice();

    if (a.IsEmpty())
        LogicError("VectorSum:  Input matrix is empty.");

    const CUDA_LONG n = (CUDA_LONG) a.GetNumRows();
    const CUDA_LONG m = (CUDA_LONG) a.GetNumCols();
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    int blocksPerGrid = 0;
    if (isColWise) // col-wise
    {
        c.RequireSize(1, m);
        blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
    }
    else
    {
        c.RequireSize(n, 1);
        blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
    }

    SyncGuard syncGuard;
    ElemType* hostC = c.Data();
    const ElemType* hostA = a.Data();
    const CUDA_LONG hostN = n;
    const CUDA_LONG hostM = m;
    const bool hostIsColWise = isColWise;
    //hipLaunchKernelGGL((_vectorSum<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, c.Data(), a.Data(), n, m, isColWise);
    hipLaunchKernelGGL((_vectorSum<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostC, hostA, hostN, hostM, hostIsColWise);
}
template <class ElemType>
void GPUMatrix<ElemType>::VectorNorm1(GPUMatrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNorm1: Matrix is empty.");

    const CUDA_LONG n = (CUDA_LONG) GetNumRows();
    const CUDA_LONG m = (CUDA_LONG) GetNumCols();
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    PrepareDevice();
    c.ChangeDeviceTo(GetComputeDeviceId());

    int blocksPerGrid = 0;
    if (isColWise) // col-wise
    {
        c.RequireSize(1, m);
        blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
    }
    else
    {
        c.RequireSize(n, 1);
        blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
    }

    SyncGuard syncGuard;
    ElemType* hostC = c.Data();
    const ElemType* hostA = Data();
    const CUDA_LONG hostN = n;
    const CUDA_LONG hostM = m;
    const bool hostIsColWise = isColWise;
    //hipLaunchKernelGGL((_vectorNorm1<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, c.Data(), Data(), n, m, isColWise);
    hipLaunchKernelGGL((_vectorNorm1<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostC, hostA, hostN, hostM, hostIsColWise);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignVectorNorm1Of(GPUMatrix<ElemType>& a, const bool isColWise)
{
    a.VectorNorm1(*this, isColWise);
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorNorm2(GPUMatrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorNorm2: Matrix is empty.");

    const CUDA_LONG n = (CUDA_LONG) GetNumRows();
    const CUDA_LONG m = (CUDA_LONG) GetNumCols();
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    PrepareDevice();
    c.ChangeDeviceTo(GetComputeDeviceId());

    int blocksPerGrid = 0;
    if (isColWise) // col-wise
    {
        c.RequireSize(1, m);
        blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
    }
    else
    {
        c.RequireSize(n, 1);
        c.ChangeDeviceTo(GetComputeDeviceId());
        blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
    }

    SyncGuard syncGuard;
    ElemType* hostC = c.Data();
    const ElemType* hostA = Data();
    const CUDA_LONG hostN = n;
    const CUDA_LONG hostM = m;
    const bool hostIsColWise = isColWise;
    //hipLaunchKernelGGL((_vectorNorm2<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, c.Data(), Data(), n, m, isColWise);
    hipLaunchKernelGGL((_vectorNorm2<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostC, hostA, hostN, hostM, hostIsColWise);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignVectorNorm2Of(GPUMatrix<ElemType>& a, const bool isColWise)
{
    a.VectorNorm2(*this, isColWise);
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorNormInf(GPUMatrix<ElemType>& c, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    // this implementation is not efficient
    GPUMatrix<ElemType> tmp(GetComputeDeviceId());
    GPUMatrix<ElemType> tmp1(GetComputeDeviceId());
    tmp.AssignAbsOf((*this));
    tmp.VectorMax(tmp1, c, isColWise);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignVectorNormInfOf(GPUMatrix<ElemType>& a, const bool isColWise)
{
    a.VectorNormInf(*this, isColWise);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignInnerProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const bool isColWise)
{
    InnerProduct(a, b, *this, isColWise);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignKhatriRaoProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignKhatriRaoProductOf: Matrix is empty.");

    CUDA_LONG cols = a.GetNumCols();
    assert(cols == b.GetNumCols());
    if (!(cols == b.GetNumCols()))
        InvalidArgument("AssignKhatriRaoProductOf: The input matrix dimensions do not match.");

    CUDA_LONG rowsA = (CUDA_LONG) a.GetNumRows();
    CUDA_LONG rowsB = (CUDA_LONG) b.GetNumRows();
    RequireSize(rowsA * rowsB, cols);
    float N = (float) GetNumElements();
    int blocksPerGrid = (int) ceil(N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const ElemType* hostB = b.Data();
    const CUDA_LONG hostRowsA = rowsA;
    const CUDA_LONG hostRowsB = rowsB;
    const CUDA_LONG hostCols = cols;
    //hipLaunchKernelGGL((_assignKhatriRaoProductOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), b.Data(), rowsA, rowsB, cols);
    hipLaunchKernelGGL((_assignKhatriRaoProductOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostB, hostRowsA, hostRowsB, hostCols);
    return *this;
}

//column-wise reshaped product. Used to compute KhatriRaoProduct Gradient
//   this = reshape each column of a from (K1xK2,1) to (K1, K2)
//   if each column of a is not transposed, each (K1, K2) times each column of b (K2, frames).
//   the output is a (K1, frames) matrix
//   if each column of a is tranposed, each (K1, K2)^T times each column of b(K1, frames) and output is (K2, frames)
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddColumnReshapeProductOf(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const bool transposeAColumn)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AddColumnReshapeProductOf: Matrix is empty.");

    CUDA_LONG cols = a.GetNumCols();
    assert(cols == b.GetNumCols());
    if (!(cols == b.GetNumCols()))
        InvalidArgument("AddColumnReshapeProductOf: The input matrix dimensions do not match.");

    CUDA_LONG rowsA = (CUDA_LONG) a.GetNumRows();
    CUDA_LONG rowsB = (CUDA_LONG) b.GetNumRows();
    if (rowsA % rowsB != 0)
        InvalidArgument("AddColumnReshapeProductOf: number of rows in a should be multiples of that in b.");

    CUDA_LONG rowsC = rowsA / rowsB;
    if (rowsC != GetNumRows() || cols != GetNumCols())
        InvalidArgument("AddColumnReshapeProductOf: This matrix does not have the right size.");

    float N = (float) GetNumElements();
    int blocksPerGrid = (int) ceil(N / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const ElemType* hostB = b.Data();
    const CUDA_LONG hostRowsA = rowsB;
    const CUDA_LONG hostRowsB = rowsC;
    const CUDA_LONG hostCols = cols;
    const bool hostTransposeAColumn = transposeAColumn;
    hipLaunchKernelGGL((_addColumnReshapeProductOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostB, hostRowsA, hostRowsB, hostCols, hostTransposeAColumn);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddWithScaleOf(ElemType alpha, const GPUMatrix<ElemType>& a)
{
    ScaleAndAdd(alpha, a, *this);
    return *this;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::FrobeniusNorm() const
{
    if (IsEmpty())
        LogicError("FrobeniusNorm: Matrix is empty.");

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);

    ElemType h_sum = 0;
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    const ElemType* hostData = Data();
    ElemType* hostSum = d_sum;
    CUDA_LONG hostN = GetNumElements();
    //hipLaunchKernelGGL((_reductionSum21024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, Data(), d_sum, (CUDA_LONG)GetNumElements(), true);
    hipLaunchKernelGGL((_reductionSum21024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, hostData, hostSum, hostN, true);
    CUDA_CALL(hipMemcpy(&h_sum, d_sum, sizeof(ElemType), hipMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_sum);

    return (h_sum);
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignFrobeniusNormOf(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignFrobeniusNormOf: Matrix a is empty.");

    RequireSize(1, 1);

    PrepareDevice();
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
    const ElemType* hostData = a.Data();
    ElemType* hostSum = Data();
    CUDA_LONG hostN = GetNumElements();
    //hipLaunchKernelGGL((_reductionSum21024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, a.Data(), Data(), (CUDA_LONG)a.GetNumElements(), true);
    hipLaunchKernelGGL((_reductionSum21024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, hostData, hostSum, hostN, true);

    return *this;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::MatrixNormInf() const
{
    if (IsEmpty())
        LogicError("MatrixNormInf: Matrix is empty.");

    ElemType* d_maxAbs = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);

    ElemType h_maxAbs = 0;
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_reductionMatrixNormInf1024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, Data(), d_maxAbs, (CUDA_LONG)GetNumElements());
#else
    const ElemType* hostData = Data();
    CUDA_LONG hostN = GetNumElements();
    hipLaunchKernelGGL((_reductionMatrixNormInf1024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, hostData, d_maxAbs, hostN);
#endif
    CUDA_CALL(hipMemcpy(&h_maxAbs, d_maxAbs, sizeof(ElemType), hipMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_maxAbs);
    return h_maxAbs;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::MatrixNorm1() const
{
    if (IsEmpty())
        LogicError("MatrixNorm1: Matrix is empty.");
    return SumOfAbsElements();
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::MatrixNorm0() const
{
    if (IsEmpty())
        LogicError("MatrixNorm0: Matrix is empty.");

    ElemType* d_nz = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);
    ElemType h_nz = 0;
    // WARNING: THIS kernel is not the most efficient way!
    // note: kernel has hard-coded dimension of 1024
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_reductionMatrixNorm01024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, Data(), d_nz, (CUDA_LONG)GetNumElements());
#else
    const ElemType* hostData = Data();
    CUDA_LONG hostN = GetNumElements();
    hipLaunchKernelGGL((_reductionMatrixNorm01024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, hostData, d_nz, hostN);
#endif
    CUDA_CALL(hipMemcpy(&h_nz, d_nz, sizeof(ElemType), hipMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_nz);
    return h_nz;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSignOf(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AssignSignOf: Matrix a is empty.");

    if (this != &a)
        RequireSize(a.GetNumRows(), a.GetNumCols());

    PrepareDevice();
    int blocksPerGrid = (int) ceil(1.0 * GetNumElements() / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignSignOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), (CUDA_LONG) GetNumElements());
#else
    ElemType* hostData = Data();
    const ElemType* hostAData = a.Data();
    const CUDA_LONG hostN = GetNumElements();
    hipLaunchKernelGGL((_assignSignOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostData, hostAData, hostN);
#endif
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddSignOf(const GPUMatrix<ElemType>& a)
{
    if (a.IsEmpty())
        LogicError("AddSignOf: Matrix a is empty.");

    if (this != &a)
        RequireSize(a.GetNumRows(), a.GetNumCols());

    PrepareDevice();
    int blocksPerGrid = (int) ceil(1.0 * GetNumElements() / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_addSignOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), (CUDA_LONG) GetNumElements());
#else
    ElemType* hostData = Data();
    const ElemType* hostAData = a.Data();
    const CUDA_LONG hostN = GetNumElements();
    hipLaunchKernelGGL((_addSignOf<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostData, hostAData, hostN);
#endif
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorMax(GPUMatrix<ElemType>& maxIndexes, GPUMatrix<ElemType>& maxValues, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    const GPUMatrix<ElemType>& us = *this;
    const CUDA_LONG m = (CUDA_LONG) GetNumRows();
    const CUDA_LONG n = (CUDA_LONG) GetNumCols();
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    PrepareDevice();
    SyncGuard syncGuard;
    if (isColWise)
    {
        maxValues.RequireSize(1, n);
        maxIndexes.RequireSize(1, n);

        int blocksPerGrid = n; // we'll have 1 block processing 1 column
        // note: kernel has hard-coded dimension of 512
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_vectorMaxMinReduce512Threads<ElemType, true>), dim3(blocksPerGrid), dim3(512), 0, t_stream, us.Data(), maxIndexes.Data(), maxValues.Data(), m, n);
#else
        const ElemType* hostUsData = us.Data();
        ElemType* hostMIData = maxIndexes.Data();
        ElemType* hostMVData = maxValues.Data();
        const CUDA_LONG hostM = m;
        const CUDA_LONG hostN = n;
        hipLaunchKernelGGL((_vectorMaxMinReduce512Threads<ElemType, true>), dim3(blocksPerGrid), dim3(512), 0, t_stream, hostUsData, hostMIData, hostMVData, hostM, hostN);
#endif

        /*int blocksPerGrid=(int)ceil(1.0*n/GridDim::maxThreadsPerBlock);
            hipLaunchKernelGGL((_vectorMax<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, us.Data(),maxIndexes.Data(),maxValues.Data(),m,n,isColWise);*/
    }
    else
    {
        maxValues.RequireSize(m, 1);
        maxIndexes.RequireSize(m, 1);
        int blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_vectorMax<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, us.Data(), maxIndexes.Data(), maxValues.Data(), m, n, isColWise);
#else
        const ElemType* hostUsData = us.Data();
        ElemType* hostMIData = maxIndexes.Data();
        ElemType* hostMVData = maxValues.Data();
        const CUDA_LONG hostM = m;
        const CUDA_LONG hostN = n;
        const bool hostICW = isColWise;
        hipLaunchKernelGGL((_vectorMax<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUsData, hostMIData, hostMVData, hostM, hostN, hostICW);
#endif
    }
}

__global__ void _initIndicesForSort(uint64_t* indexes, CUDA_LONG crow, CUDA_LONG ccol)
{
    CUDA_LONG id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (id >= crow * ccol)
        return;
    uint32_t irow = id % crow;
    uint32_t icol = id / crow;
    indexes[id] = (static_cast<uint64_t>(irow) << 32) | icol;
}

template <class ElemType>
void GPUMatrix<ElemType>::VectorMax(GPUMatrix<ElemType>& maxIndexes, GPUMatrix<ElemType>& maxValues, const bool isColWise, int topK) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    if (topK == 1)
    {
        VectorMax(maxIndexes, maxValues, isColWise);
        return;
    }

    if (!isColWise)
        RuntimeError("Row-wise TopK max is not supported.");

    const GPUMatrix<ElemType>& us = *this;
    const CUDA_LONG m = (CUDA_LONG) GetNumRows();
    const CUDA_LONG n = (CUDA_LONG) GetNumCols();
    assert(topK <= m);
    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow

    PrepareDevice();
    SyncGuard syncGuard;
    maxValues.RequireSize(topK, n);
    maxIndexes.RequireSize(topK, n);

    // To sort matrix columns we use 2-pass _stable_ sort algorithm:
    // 1. Sort by values (descending) with corresponding row/col indexes.
    // 2. Sort by col indices (ascending) with corresponding values/row indices.
    // Indices are stored as 64-bit ints where low 32 bits represent column and high 32 bits - row index.
    // On the second pass only first 32 bits of the index are used in sorting, so SortPairs has
    // begin_bit and end_bit set accordingly.

    CUDA_LONG celt = static_cast<CUDA_LONG>(GetNumElements());
    ElemType* inVal = us.Data();
    ElemType* outVal1 = nullptr;
    ElemType* outVal2 = nullptr;
    uint64_t* inIdx = nullptr;
    uint64_t* outIdx = nullptr;
    // Determine temp buffer size needed for SortPairsDescending to sort values on the first pass.
    size_t cbtemp = 0;
    // If first param is nullptr then no actual work is done except writing result to cbtemp.
#ifdef __HIP_ENABLE_CUB__
    CUDA_CALL(SortPairsDescending(nullptr, cbtemp, inVal, outVal1, inIdx, outIdx, celt, 0, sizeof(ElemType) * 8, t_stream));
#endif /*__HIP_ENABLE_CUB__*/
    size_t ctemp1 = (cbtemp + sizeof(ElemType) - 1) / sizeof(ElemType);
    cbtemp = 0;
#ifdef __HIP_ENABLE_CUB__
    CUDA_CALL(hipcub::DeviceRadixSort::SortPairs(nullptr, cbtemp, outIdx, inIdx, outVal1, outVal2, celt, 0, 32, t_stream));
#endif /*__HIP_ENABLE_CUB__*/
    size_t ctemp2 = (cbtemp + sizeof(ElemType) - 1) / sizeof(ElemType);
    size_t ctemp = std::max(ctemp1, ctemp2);
    cbtemp = ctemp * sizeof(ElemType);
    // ElemType count needed to store indices, accounting for natural alignment for uint64_t type.
    size_t cidx = ((celt + 1) * sizeof(uint64_t) - 1 + sizeof(ElemType) - 1) / sizeof(ElemType);
    // Get temp workspace.
    auto workspace = GetOrCreateWorkspace();
    // RequireSize to store: output values for the 1st and 2nd passes, input indices, output indices, and temp storage.
    workspace->RequireSize(m, 2 * n + (2 * cidx + ctemp + m - 1) / m);
    outVal1 = workspace->Data();
    outVal2 = outVal1 + celt;
    inIdx = reinterpret_cast<uint64_t*>(outVal2 + celt);
    // Align indices pointer if needed.
    size_t cbAlign = reinterpret_cast<size_t>(inIdx) % sizeof(uint64_t);
    if (cbAlign != 0)
        reinterpret_cast<uint8_t*&>(inIdx) += sizeof(uint64_t) - cbAlign;
    outIdx = inIdx + celt;
    void* ptmp = outIdx + celt;
    assert(reinterpret_cast<ElemType*>(reinterpret_cast<uint8_t*>(ptmp) + cbtemp) <= workspace->Data() + workspace->GetNumElements());

    // Initialize indices.
    const int ThreadsPerBlock = 128;
    int cblock = (celt + ThreadsPerBlock - 1) / ThreadsPerBlock;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_initIndicesForSort), dim3(cblock), dim3(ThreadsPerBlock), 0, t_stream, inIdx, m, n);
#else
    CUDA_LONG hostM = m;
    CUDA_LONG hostN = n;
    hipLaunchKernelGGL((_initIndicesForSort), dim3(cblock), dim3(ThreadsPerBlock), 0, t_stream, inIdx, hostM, hostN);
#endif
    // Sort by values.
#ifdef __HIP_ENABLE_CUB__
    CUDA_CALL(SortPairsDescending(ptmp, cbtemp, inVal, outVal1, inIdx, outIdx, celt, 0, sizeof(ElemType) * 8, t_stream));
#endif /*__HIP_ENABLE_CUB__*/
    // Sort by column indices. outIdx contains indices after the first pass so it's used as an input.
#ifdef __HIP_ENABLE_CUB__
    CUDA_CALL(hipcub::DeviceRadixSort::SortPairs(ptmp, cbtemp, outIdx, inIdx, outVal1, outVal2, celt, 0, 32, t_stream));
#endif /*__HIP_ENABLE_CUB__*/
    // Copy results.
    cblock = (topK * n + ThreadsPerBlock - 1) / ThreadsPerBlock;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_copyTopKResults), dim3(cblock), dim3(ThreadsPerBlock), 0, t_stream, static_cast<const uint64_t*>(inIdx), static_cast<const ElemType*>(outVal2), static_cast<ElemType*>(maxIndexes.Data()), static_cast<ElemType*>(maxValues.Data()), static_cast<CUDA_LONG>(m), static_cast<CUDA_LONG>(n), static_cast<int>(topK));
#else
    const uint64_t* hostIndexes = static_cast<const uint64_t*>(inIdx);
    const ElemType* hostOV2 = static_cast<const ElemType*>(outVal2);
    ElemType* hostMI = static_cast<ElemType*>(maxIndexes.Data());
    ElemType* hostMV = static_cast<ElemType*>(maxValues.Data());
    CUDA_LONG hostM1 = static_cast<CUDA_LONG>(m);
    CUDA_LONG hostN1 = static_cast<CUDA_LONG>(n);
    int hostTopK = static_cast<int>(topK);
    hipLaunchKernelGGL((_copyTopKResults), dim3(cblock), dim3(ThreadsPerBlock), 0, t_stream, hostIndexes, hostOV2, hostMI, hostMV, hostM1, hostN1, hostTopK);
#endif

    ReleaseWorkspace(std::move(workspace));

}

template <class ElemType>
void GPUMatrix<ElemType>::VectorMin(GPUMatrix<ElemType>& minIndexes, GPUMatrix<ElemType>& minValues, const bool isColWise) const
{
    if (IsEmpty())
        LogicError("VectorMax: Matrix is empty.");

    const GPUMatrix<ElemType>& us = *this;
    const CUDA_LONG m = (CUDA_LONG) GetNumRows();
    const CUDA_LONG n = (CUDA_LONG) GetNumCols();

    assert(m > 0 && n > 0); // converting from size_t to int may cause overflow
    PrepareDevice();
    SyncGuard syncGuard;
    if (isColWise)
    {
        minValues.RequireSize(1, n);
        minIndexes.RequireSize(1, n);

        int blocksPerGrid = n; // we'll have 1 block processing 1 column
        // note: kernel has hard-coded dimension of 512
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_vectorMaxMinReduce512Threads<ElemType, false>), dim3(blocksPerGrid), dim3(512), 0, t_stream, us.Data(), minIndexes.Data(), minValues.Data(), m, n);
#else
        const ElemType* hostUsData = us.Data();
        ElemType* hostMIData = minIndexes.Data();
        ElemType* hostMVData = minValues.Data();
        const CUDA_LONG hostM = m;
        const CUDA_LONG hostN = n;
        hipLaunchKernelGGL((_vectorMaxMinReduce512Threads<ElemType, false>), dim3(blocksPerGrid), dim3(512), 0, t_stream, hostUsData, hostMIData, hostMVData, hostM, hostN);
#endif

        /*
            int blocksPerGrid=(int)ceil(1.0*n/GridDim::maxThreadsPerBlock);
            hipLaunchKernelGGL((_vectorMin<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, us.Data(),minIndexes.Data(),minValues.Data(),m,n,isColWise);*/
    }
    else
    {
        minValues.RequireSize(m, 1);
        minIndexes.RequireSize(m, 1);
        int blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_vectorMin<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, us.Data(), minIndexes.Data(), minValues.Data(), m, n, isColWise);
#else
        const ElemType* hostUsData = us.Data();
        ElemType* hostMIData = minIndexes.Data();
        ElemType* hostMVData = minValues.Data();
        const CUDA_LONG hostM = m;
        const CUDA_LONG hostN = n;
        const bool hostICW = isColWise;
        hipLaunchKernelGGL((_vectorMin<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUsData, hostMIData, hostMVData, hostM, hostN, hostICW);
#endif
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignNumOfDiff(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, bool searchInCol)
{
    if (a.GetNumCols() != b.GetNumCols())
        InvalidArgument("AssignNumOfDiff: a and b must have the same number of columns.");
    if (!searchInCol && a.GetNumRows() != b.GetNumRows())
        InvalidArgument("AssignNumOfDiff: a and b must have the same number of rows.");

    RequireSize(1, 1); // result should be one element

    PrepareDevice();
    SyncGuard syncGuard;
    if (!searchInCol)
    {
        // int blocksPerGrid=(int)ceil(1.0*a.GetNumElements()/GridDim::maxThreadsPerBlock);
        // hipLaunchKernelGGL((_assignNumOfDiff1024Threads<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, a.Data(), b.Data(), Data(), a.GetNumElements());
        // note: kernel has hard-coded dimension of 1024
        CUDA_LONG N = a.GetNumElements();
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_assignNumOfDiff1024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, a.Data(), b.Data(), Data(), N);
#else
        const ElemType* hostAData = a.Data();
        const ElemType* hostBData = b.Data();
        ElemType* hostData = Data();
        CUDA_LONG hostN = N;
        hipLaunchKernelGGL((_assignNumOfDiff1024Threads<ElemType>), dim3(1), dim3(1024), 0, t_stream, hostAData, hostBData, hostData, hostN);
#endif
    }
    else
    {
        const int blockSize = 1024;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_assignNumOfDiffCol<blockSize>), dim3(1), dim3(blockSize), 0, t_stream, static_cast<const ElemType*>(a.Data()), static_cast<const ElemType*>(b.Data()), static_cast<ElemType*>(Data()),
                                                                      static_cast<CUDA_LONG>(b.GetNumRows()), static_cast<CUDA_LONG>(a.GetNumCols()));
#else
        const ElemType* hostAData = static_cast<const ElemType*>(a.Data());
        const ElemType* hostBData = static_cast<const ElemType*>(b.Data());
        ElemType* hostData = static_cast<ElemType*>(Data());
        CUDA_LONG hostGNR = static_cast<CUDA_LONG>(b.GetNumRows());
        CUDA_LONG hostGNC = static_cast<CUDA_LONG>(a.GetNumCols());
        hipLaunchKernelGGL((_assignNumOfDiffCol<blockSize>), dim3(1), dim3(blockSize), 0, t_stream, hostAData, hostBData, hostData, hostGNR, hostGNC);
#endif
    }
    return *this;
}

#pragma endregion Member BLAS Functions

#pragma region Other helper functions
template <class ElemType>
void GPUMatrix<ElemType>::Print(const char* /*matrixName*/, size_t /*rowStart*/, size_t /*rowEnd*/, size_t /*colStart*/, size_t /*colEnd*/) const
{
    NOT_IMPLEMENTED;
}

template <class ElemType>
void GPUMatrix<ElemType>::Print(const char* matrixName /*=nullptr*/) const
{
    size_t elemCount = GetNumRows() * GetNumCols();
    vector<ElemType> localCopy(elemCount);
    hipMemcpy(localCopy.data(), Data(), elemCount * sizeof(ElemType), hipMemcpyDeviceToHost);

    fprintf(stderr, "\n###### ");
    if (matrixName != nullptr)
        fprintf(stderr, "%s ", matrixName);
    fprintf(stderr, "(%lu, %lu) ######\n\n", (unsigned long)GetNumRows(), (unsigned long)GetNumCols());

    if (IsEmpty())
    {
        fprintf(stderr, "(empty)\n");
        return;
    }

    // CNTK is using column-major storage
    for (size_t i = 0; i < GetNumRows(); i++)
    {
        for (size_t j = 0; j < GetNumCols(); j++)
        {
            fprintf(stderr, "%.10f\t", (float)localCopy[i + j * GetNumRows()]);
        }
        fprintf(stderr, "\n");
    }
}

//helpfer function used for convolution neural network
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignPackedConvolutionInput(const GPUMatrix<ElemType>& inputSubBatch,
                                                                       const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                                       const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                                       const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                                       const bool zeroPadding)
{
    assert(verticalSubsample <= kernelHeight && horizontalSubsample <= kernelWidth);

    size_t packedInputRows = kernelWidth * kernelHeight * inputChannels;
    size_t packedInputColsPerSample = outputWidth * outputHeight;
    size_t smallBatchSize = inputSubBatch.GetNumCols();
    RequireSize(packedInputRows, packedInputColsPerSample * smallBatchSize);
    if (zeroPadding)
        SetValue((ElemType) 0);

    PrepareDevice();
    int numThreadPerBlock = GridDim::maxThreadsPerBlock;
#if 1
    int blocksPerGrid = (smallBatchSize * inputWidth * inputHeight * inputChannels + numThreadPerBlock - 1) / numThreadPerBlock;
#else
    dim3 blocksPerGrid((inputWidth * inputHeight * inputChannels + numThreadPerBlock - 1) / numThreadPerBlock, smallBatchSize);
#endif
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignPackedConvolutionInput), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, static_cast<ElemType*>(Data()),
                                                                                     static_cast<const ElemType*>(inputSubBatch.Data()),
                                                                                     static_cast<const CUDA_LONG>(smallBatchSize),
                                                                                     static_cast<const CUDA_LONG>(inputWidth), static_cast<const CUDA_LONG>(inputHeight), static_cast<const CUDA_LONG>(inputChannels),
                                                                                     static_cast<const CUDA_LONG>(outputWidth), static_cast<const CUDA_LONG>(outputHeight), static_cast<const CUDA_LONG>(outputChannels),
                                                                                     static_cast<const CUDA_LONG>(kernelWidth), static_cast<const CUDA_LONG>(kernelHeight), static_cast<const CUDA_LONG>(horizontalSubsample), static_cast<const CUDA_LONG>(verticalSubsample), static_cast<const bool>(zeroPadding));
#else
    ElemType* hostData = Data();
    const ElemType* hostISB = inputSubBatch.Data();
    const CUDA_LONG hostSBS = smallBatchSize;
    const CUDA_LONG hostIW = inputWidth;
    const CUDA_LONG hostIH = inputHeight;
    const CUDA_LONG hostIC = inputChannels;
    const CUDA_LONG hostOW = outputWidth;
    const CUDA_LONG hostOH = outputHeight;
    const CUDA_LONG hostOC = outputChannels;
    const CUDA_LONG hostKW = kernelWidth;
    const CUDA_LONG hostKH = kernelHeight;
    const CUDA_LONG hostHSS = horizontalSubsample;
    const CUDA_LONG hostVSS = verticalSubsample;
    const bool hostZP = zeroPadding;
    hipLaunchKernelGGL((_assignPackedConvolutionInput), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, hostData, hostISB, hostSBS, hostIW, hostIH, hostIC, hostOW, hostOH, hostOC, hostKW, hostKH, hostHSS, hostVSS, hostZP);
#endif

    return *this;
}

//helpfer function used for convolution neural network
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::UnpackConvolutionInput(GPUMatrix<ElemType>& inputSubBatch,
                                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputChannels,
                                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputChannels,
                                                                 const size_t kernelWidth, const size_t kernelHeight, const size_t horizontalSubsample, const size_t verticalSubsample,
                                                                 const bool zeroPadding) const
{
    assert(verticalSubsample <= kernelHeight && horizontalSubsample <= kernelWidth);

    size_t smallBatchSize = inputSubBatch.GetNumCols();

    PrepareDevice();
    int numThreadPerBlock = GridDim::maxThreadsPerBlock;
#if 1
    int blocksPerGrid = (smallBatchSize * inputWidth * inputHeight * inputChannels + numThreadPerBlock - 1) / numThreadPerBlock;
#else
    dim3 blocksPerGrid((inputWidth * inputHeight * inputChannels + numThreadPerBlock - 1) / numThreadPerBlock, smallBatchSize);
#endif
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_unpackConvolutionInput), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, static_cast<const ElemType*>(Data()),
                                                                               static_cast<ElemType*>(inputSubBatch.Data()),
                                                                               static_cast<const CUDA_LONG>(smallBatchSize),
                                                                               static_cast<const CUDA_LONG>(inputWidth), static_cast<const CUDA_LONG>(inputHeight), static_cast<const CUDA_LONG>(inputChannels),
                                                                               static_cast<const CUDA_LONG>(outputWidth), static_cast<const CUDA_LONG>(outputHeight), static_cast<const CUDA_LONG>(outputChannels),
                                                                               static_cast<const CUDA_LONG>(kernelWidth), static_cast<const CUDA_LONG>(kernelHeight), static_cast<const CUDA_LONG>(horizontalSubsample), static_cast<const CUDA_LONG>(verticalSubsample), static_cast<const bool>(zeroPadding));
#else
    const ElemType* hostData = Data();
    ElemType* hostISB = inputSubBatch.Data();
    const CUDA_LONG hostSBS = smallBatchSize;
    const CUDA_LONG hostIW = inputWidth;
    const CUDA_LONG hostIH = inputHeight;
    const CUDA_LONG hostIC = inputChannels;
    const CUDA_LONG hostOW = outputWidth;
    const CUDA_LONG hostOH = outputHeight;
    const CUDA_LONG hostOC = outputChannels;
    const CUDA_LONG hostKW = kernelWidth;
    const CUDA_LONG hostKH = kernelHeight;
    const CUDA_LONG hostHSS = horizontalSubsample;
    const CUDA_LONG hostVSS = verticalSubsample;
    const bool hostZP = zeroPadding;
    hipLaunchKernelGGL((_unpackConvolutionInput), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, hostData, hostISB, hostSBS, hostIW, hostIH, hostIC, hostOW, hostOH, hostOC, hostKW, hostKH, hostHSS, hostVSS, hostZP);
#endif

    return inputSubBatch;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignMaxPoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                                 const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                                 const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                                 const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    assert(verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

    unsigned int batchSize = inputBatch.GetNumCols();
    RequireSize(outputSizePerSample, batchSize);

    int numThreadPerBlock = GridDim::maxThreadsPerBlock;
    int blocksPerGrid = (batchSize * outputSizePerSample + numThreadPerBlock - 1) / numThreadPerBlock;

    PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignMaxPoolingResult), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, static_cast<ElemType*>(Data()), static_cast<const ElemType*>(inputBatch.Data()), static_cast<const CUDA_LONG>(batchSize), static_cast<const CUDA_LONG>(channels),
                                                                               static_cast<const CUDA_LONG>(inputWidth), static_cast<const CUDA_LONG>(inputHeight), static_cast<const CUDA_LONG>(inputSizePerSample),
                                                                               static_cast<const CUDA_LONG>(outputWidth), static_cast<const CUDA_LONG>(outputHeight), static_cast<const CUDA_LONG>(outputSizePerSample),
                                                                               static_cast<const CUDA_LONG>(windowWidth), static_cast<const CUDA_LONG>(windowHeight), static_cast<const CUDA_LONG>(horizontalSubsample), static_cast<const CUDA_LONG>(verticalSubsample));
#else
    ElemType* hostData = Data();
    const ElemType* hostIB = inputBatch.Data();
    const CUDA_LONG hostBS = batchSize;
    const CUDA_LONG hostC = channels;
    const CUDA_LONG hostIW = inputWidth;
    const CUDA_LONG hostIH = inputHeight;
    const CUDA_LONG hostISPS = inputSizePerSample;
    const CUDA_LONG hostOW = outputWidth;
    const CUDA_LONG hostOH = outputHeight;
    const CUDA_LONG hostOSPS = outputSizePerSample;
    const CUDA_LONG hostWW = windowWidth;
    const CUDA_LONG hostWH = windowHeight;
    const CUDA_LONG hostHSS = horizontalSubsample;
    const CUDA_LONG hostVSS = verticalSubsample;
    hipLaunchKernelGGL((_assignMaxPoolingResult), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, hostData, hostIB, hostBS, hostC, hostIW, hostIH, hostISPS, hostOW, hostOH, hostOSPS, hostWW, hostWH, hostHSS, hostVSS);
#endif

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddMaxPoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch, const GPUMatrix<ElemType>& inputBatch, const GPUMatrix<ElemType>& outputBatch,
                                                                const size_t channels,
                                                                const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                                const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    assert(verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

    unsigned int batchSize = outputGradientBatch.GetNumCols();
    int numThreadPerBlock = GridDim::maxThreadsPerBlock;

    PrepareDevice();
    SyncGuard syncGuard;

    int blocksPerGrid = (batchSize * inputSizePerSample + numThreadPerBlock - 1) / numThreadPerBlock;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_addMaxPoolingGradient), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, static_cast<ElemType*>(Data()), static_cast<const ElemType*>(outputGradientBatch.Data()), static_cast<const ElemType*>(inputBatch.Data()), static_cast<const ElemType*>(outputBatch.Data()), static_cast<const CUDA_LONG>(batchSize), static_cast<const CUDA_LONG>(channels),
                                                                              static_cast<const CUDA_LONG>(inputWidth), static_cast<const CUDA_LONG>(inputHeight), static_cast<const CUDA_LONG>(inputSizePerSample),
                                                                              static_cast<const CUDA_LONG>(outputWidth), static_cast<const CUDA_LONG>(outputHeight), static_cast<const CUDA_LONG>(outputSizePerSample),
                                                                              static_cast<const CUDA_LONG>(windowWidth), static_cast<const CUDA_LONG>(windowHeight), static_cast<const CUDA_LONG>(horizontalSubsample), static_cast<const CUDA_LONG>(verticalSubsample));
#else
    ElemType* hostData = Data();
    const ElemType* hostOGB = outputGradientBatch.Data();
    const ElemType* hostIB = inputBatch.Data();
    const ElemType* hostOB = outputBatch.Data();
    const CUDA_LONG hostBS = batchSize;
    const CUDA_LONG hostC = channels;
    const CUDA_LONG hostIW = inputWidth;
    const CUDA_LONG hostIH = inputHeight;
    const CUDA_LONG hostISPS = inputSizePerSample;
    const CUDA_LONG hostOW = outputWidth;
    const CUDA_LONG hostOH = outputHeight;
    const CUDA_LONG hostOSPS = outputSizePerSample;
    const CUDA_LONG hostWW = windowWidth;
    const CUDA_LONG hostWH = windowHeight;
    const CUDA_LONG hostHSS = horizontalSubsample;
    const CUDA_LONG hostVSS = verticalSubsample;
    hipLaunchKernelGGL((_addMaxPoolingGradient), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, hostData, hostOGB, hostIB, hostOB, hostBS, hostC, hostIW, hostIH, hostISPS, hostOW, hostOH, hostOSPS, hostWW, hostWH, hostHSS, hostVSS);
#endif 

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignAveragePoolingResult(const GPUMatrix<ElemType>& inputBatch, const size_t channels,
                                                                     const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                                     const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                                     const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    assert(verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

    unsigned int batchSize = inputBatch.GetNumCols();
    RequireSize(outputSizePerSample, batchSize);

    int numThreadPerBlock = GridDim::maxThreadsPerBlock;
    int blocksPerGrid = (batchSize * outputSizePerSample + numThreadPerBlock - 1) / numThreadPerBlock;

    PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignAveragePoolingResult), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, static_cast<ElemType*>(Data()), static_cast<const ElemType*>(inputBatch.Data()), static_cast<const CUDA_LONG>(batchSize), static_cast<const CUDA_LONG>(channels),
                                                                                   static_cast<const CUDA_LONG>(inputWidth), static_cast<const CUDA_LONG>(inputHeight), static_cast<const CUDA_LONG>(inputSizePerSample),
                                                                                   static_cast<const CUDA_LONG>(outputWidth), static_cast<const CUDA_LONG>(outputHeight), static_cast<const CUDA_LONG>(outputSizePerSample),
                                                                                   static_cast<const CUDA_LONG>(windowWidth), static_cast<const CUDA_LONG>(windowHeight), static_cast<const CUDA_LONG>(horizontalSubsample), static_cast<const CUDA_LONG>(verticalSubsample));
#else
    ElemType* hostData = Data();
    const ElemType* hostIB = inputBatch.Data();
    const CUDA_LONG hostBS = batchSize;
    const CUDA_LONG hostC = channels;
    const CUDA_LONG hostIW = inputWidth;
    const CUDA_LONG hostIH = inputHeight;
    const CUDA_LONG hostISPS = inputSizePerSample;
    const CUDA_LONG hostOW = outputWidth;
    const CUDA_LONG hostOH = outputHeight;
    const CUDA_LONG hostOSPS = outputSizePerSample;
    const CUDA_LONG hostWW = windowWidth;
    const CUDA_LONG hostWH = windowHeight;
    const CUDA_LONG hostHSS = horizontalSubsample;
    const CUDA_LONG hostVSS = verticalSubsample;
    hipLaunchKernelGGL((_assignAveragePoolingResult), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, hostData, hostIB, hostBS, hostC, hostIW, hostIH, hostISPS, hostOW, hostOH, hostOSPS, hostWW, hostWH, hostHSS, hostVSS);
#endif

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AddAveragePoolingGradient(const GPUMatrix<ElemType>& outputGradientBatch,
                                                                    const size_t channels,
                                                                    const size_t inputWidth, const size_t inputHeight, const size_t inputSizePerSample,
                                                                    const size_t outputWidth, const size_t outputHeight, const size_t outputSizePerSample,
                                                                    const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample)
{
    assert(verticalSubsample <= windowHeight && horizontalSubsample <= windowWidth);

    size_t batchSize = outputGradientBatch.GetNumCols();
    int numThreadPerBlock = GridDim::maxThreadsPerBlock;

    PrepareDevice();
    SyncGuard syncGuard;
    size_t blocksPerGrid = (batchSize * inputSizePerSample + numThreadPerBlock - 1) / numThreadPerBlock;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_addAveragePoolingGradient), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, static_cast<ElemType*>(Data()), static_cast<const ElemType*>(outputGradientBatch.Data()), (CUDA_LONG) batchSize, static_cast<const CUDA_LONG>(channels),
                                                                                  static_cast<const CUDA_LONG>(inputWidth), static_cast<const CUDA_LONG>(inputHeight), static_cast<const CUDA_LONG>(inputSizePerSample),
                                                                                  static_cast<const CUDA_LONG>(outputWidth), static_cast<const CUDA_LONG>(outputHeight), static_cast<const CUDA_LONG>(outputSizePerSample),
                                                                                  static_cast<const CUDA_LONG>(windowWidth), static_cast<const CUDA_LONG>(windowHeight), static_cast<const CUDA_LONG>(horizontalSubsample), static_cast<const CUDA_LONG>(verticalSubsample));
#else
    ElemType* hostData = Data();
    const ElemType* hostOGB = outputGradientBatch.Data();
    const CUDA_LONG hostBS = batchSize;
    const CUDA_LONG hostC = channels;
    const CUDA_LONG hostIW = inputWidth;
    const CUDA_LONG hostIH = inputHeight;
    const CUDA_LONG hostISPS = inputSizePerSample;
    const CUDA_LONG hostOW = outputWidth;
    const CUDA_LONG hostOH = outputHeight;
    const CUDA_LONG hostOSPS = outputSizePerSample;
    const CUDA_LONG hostWW = windowWidth;
    const CUDA_LONG hostWH = windowHeight;
    const CUDA_LONG hostHSS = horizontalSubsample;
    const CUDA_LONG hostVSS = verticalSubsample;
    hipLaunchKernelGGL((_addAveragePoolingGradient), dim3(blocksPerGrid), dim3(numThreadPerBlock), 0, t_stream, hostData, hostOGB, hostBS, hostC, hostIW, hostIH, hostISPS, hostOW, hostOH, hostOSPS, hostWW, hostWH, hostHSS, hostVSS);
#endif

    return *this;
}

#pragma endregion Other helper functions

template <class ElemType>
void GPUMatrix<ElemType>::ConvolutionForward(const GPUMatrix<ElemType>& kernel, const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIwht,
                                             const GPUMatrix<int>& mpRowRun, const GPUMatrix<int>& runs, GPUMatrix<ElemType>& output) const
{
    const int BlockSize = 128;
    auto gdim = dim3((output.GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kConvolutionForward), dim3(gdim), dim3(BlockSize), 0, t_stream, (int) GetNumCols(), static_cast<const ElemType*>(kernel.Data()), static_cast<const int*>(mpRowCol.Data()), static_cast<const int*>(mpRowIwht.Data()), static_cast<const int*>(mpRowRun.Data()),
                                                            static_cast<const int*>(runs.Data()), static_cast<const ElemType*>(Data()), (int) GetNumRows(), static_cast<ElemType*>(output.Data()), (int)output.GetNumRows());
#else
    int hostGNC = (int) GetNumCols();
    const ElemType* hostKD = kernel.Data();
    const int* hostRCD = mpRowCol.Data();
    const int* hostIWD = mpRowIwht.Data();
    const int* hostRRD = mpRowRun.Data();
    const int* hostRD = runs.Data();
    const ElemType* hostData = Data();
    int hostGNR = GetNumRows();
    ElemType* hostOD = output.Data();
    int hostOGNR = output.GetNumRows();
    hipLaunchKernelGGL((kConvolutionForward), dim3(gdim), dim3(BlockSize), 0, t_stream, hostGNC, hostKD, hostRCD, hostIWD, hostRRD, hostRD, hostData, hostGNR, hostOD, hostOGNR);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::ConvolutionBackwardData(const GPUMatrix<ElemType>& kernel, const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIwht,
                                                  const GPUMatrix<int>& mpRowRun, const GPUMatrix<int>& runs, GPUMatrix<ElemType>& grad) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kConvolutionBackwardData), dim3(gdim), dim3(BlockSize), 0, t_stream, (int)GetNumCols(), static_cast<const ElemType*>(kernel.Data()), static_cast<const int*>(mpRowCol.Data()), static_cast<const int*>(mpRowIwht.Data()), static_cast<const int*>(mpRowRun.Data()),
                                                                 static_cast<const int*>(runs.Data()), static_cast<const ElemType*>(Data()), (int)GetNumRows(), static_cast<ElemType*>(grad.Data()), (int)grad.GetNumRows());
#else
    int hostGNC = (int) GetNumCols();
    const ElemType* hostKD = kernel.Data();
    const int* hostRCD = mpRowCol.Data();
    const int* hostIWD = mpRowIwht.Data();
    const int* hostRRD = mpRowRun.Data();
    const int* hostRD = runs.Data();
    const ElemType* hostData = Data();
    int hostGNR = GetNumRows();
    ElemType* hostGD = grad.Data();
    int hostGGNR = grad.GetNumRows();
    hipLaunchKernelGGL((kConvolutionBackwardData), dim3(gdim), dim3(BlockSize), 0, t_stream, hostGNC, hostKD, hostRCD, hostIWD, hostRRD, hostRD, hostData, hostGNR, hostGD, hostGGNR);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::ConvolutionBackwardKernel(const GPUMatrix<ElemType>& in, const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIwht,
                                                    const GPUMatrix<int>& mpRowRun, const GPUMatrix<int>& runs, GPUMatrix<ElemType>& kernelGrad) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kConvolutionBackwardKernel), dim3(gdim), dim3(BlockSize), 0, t_stream, (int)GetNumCols(), (int)in.GetNumRows(), (int)GetNumRows(),
                                                                   static_cast<const ElemType*>(in.Data()), static_cast<const int*>(mpRowCol.Data()), static_cast<const int*>(mpRowIwht.Data()), static_cast<const int*>(mpRowRun.Data()),
                                                                   static_cast<const int*>(runs.Data()), static_cast<const ElemType*>(Data()), static_cast<ElemType*>(kernelGrad.Data()));
#else
    int hostGNC = (int)GetNumCols();
    int hostIGNR = (int)in.GetNumRows();
    int hostGNR = (int)GetNumRows();
    const ElemType* hostID = in.Data();
    const int* hostRCD = mpRowCol.Data();
    const int* hostIWD = mpRowIwht.Data();
    const int* hostRRD = mpRowRun.Data();
    const int* hostRD = runs.Data();
    const ElemType* hostData = Data();
    ElemType* hostKGD = kernelGrad.Data();
    hipLaunchKernelGGL((kConvolutionBackwardKernel), dim3(gdim), dim3(BlockSize), 0, t_stream, hostGNC, hostIGNR, hostGNR, hostID, hostRCD, hostIWD, hostRRD, hostRD, hostData, hostKGD);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::MaxPoolingForward(const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices, GPUMatrix<ElemType>& output) const
{
    const int BlockSize = 128;
    auto gdim = dim3((output.GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kMaxPoolingForward), dim3(gdim), dim3(BlockSize), 0, t_stream, (int)GetNumCols(), static_cast<const int*>(mpRowCol.Data()), static_cast<const int*>(mpRowIndices.Data()), static_cast<const int*>(indices.Data()),
                                                           static_cast<const ElemType*>(Data()), (int)GetNumRows(), static_cast<ElemType*>(output.Data()), (int)output.GetNumRows());
#else
    int hostBatchSize = GetNumCols();
    const int* hostMpRowCol = mpRowCol.Data();
    const int* hostMpRowIndices = mpRowIndices.Data();
    const int* hostIndices = indices.Data();
    const ElemType* __restrict__ hostSrc = Data();
    int hostSrcVecSize = GetNumRows();
    ElemType* hostDst = output.Data();
    int hostDstVecSize = output.GetNumRows();
    hipLaunchKernelGGL((kMaxPoolingForward), dim3(gdim), dim3(BlockSize), 0, t_stream, hostBatchSize, hostMpRowCol, hostMpRowIndices, hostIndices, hostSrc, hostSrcVecSize, hostDst, hostDstVecSize);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::MaxPoolingBackward(const GPUMatrix<ElemType>& out, const GPUMatrix<ElemType>& in,
                                             const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices,
                                             GPUMatrix<ElemType>& grad, bool accumulateGradient) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();

    if (!accumulateGradient)
        grad.SetValue((ElemType)0);

    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kMaxPoolingBackward), dim3(gdim), dim3(BlockSize), 0, t_stream, (int)GetNumCols(), static_cast<const ElemType*>(out.Data()), static_cast<const ElemType*>(in.Data()),
                                                            static_cast<const int*>(mpRowCol.Data()), static_cast<const int*>(mpRowIndices.Data()), static_cast<const int*>(indices.Data()),
                                                            static_cast<const ElemType*>(Data()), (int)GetNumRows(), static_cast<ElemType*>(grad.Data()), (int)grad.GetNumRows());
#else
    int hostBatchSize = GetNumCols();
    const ElemType* hostOut = out.Data();
    const ElemType* hostIn = in.Data();
    const int* hostMpRowCol = mpRowCol.Data();
    const int* hostMpRowIndices = mpRowIndices.Data();
    const int* hostIndices = indices.Data();
    const ElemType* __restrict__ hostSrcgrad = Data();
    int hostSrcVecSize = GetNumRows();
    ElemType* hostGrad = grad.Data();
    int hostDstVecSize = grad.GetNumRows();
    hipLaunchKernelGGL((kMaxPoolingBackward), dim3(gdim), dim3(BlockSize), 0, t_stream, (int)GetNumCols(), hostOut, hostIn,hostMpRowCol, hostMpRowIndices, hostIndices, hostSrcgrad, hostSrcVecSize,
                                                                                        hostGrad, hostDstVecSize);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::MaxROIPoolingForward(const size_t numRois, const size_t numImg, const size_t channels, const size_t width, const size_t height,
                                               const size_t pooledWidth, const size_t pooledHeight, const GPUMatrix<ElemType>& roiData, GPUMatrix<ElemType>& output,
                                               GPUMatrix<ElemType>& argmax, double spatialScale) const
{
    PrepareDevice();
    SyncGuard syncGuard;

    int count = numRois * numImg * channels * pooledHeight * pooledWidth;
    const int blockSize = GridDim::maxThreadsPerBlock;
    auto numThreads = dim3((int)floor((double)(count + blockSize - 1) / blockSize));
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kMaxROIPoolingForward), dim3(numThreads), dim3(blockSize), 0, t_stream, static_cast<const int>(count), static_cast<const int>(numRois), static_cast<const int>(numImg), static_cast<const int>(channels), static_cast<const int>(width), static_cast<const int>(height), 
                                                                  static_cast<const int>(pooledWidth), static_cast<const int>(pooledHeight), static_cast<const ElemType*>(Data()), static_cast<const ElemType*>(roiData.Data()), static_cast<ElemType*>(output.Data()), static_cast<ElemType*>(argmax.Data()), static_cast<double>(spatialScale));
#else
    const int hostTotalIterations = count;
    const int hostNumROIs = numRois;
    const int hostNumImg = numImg;
    const int hostChannels = channels;
    const int hostWidth = width;
    const int hostHeight = height;
    const int hostPooledWidth = pooledWidth;
    const int hostPooledHeight = pooledHeight;
    const ElemType* hostSrc = Data();
    const ElemType* hostRoiData = roiData.Data();
    ElemType* hostDst = output.Data();
    ElemType* hostArgmax = argmax.Data();
    double hostSpatialScale = spatialScale;
    hipLaunchKernelGGL((kMaxROIPoolingForward), dim3(numThreads), dim3(blockSize), 0, t_stream, hostTotalIterations, hostNumROIs, hostNumImg, hostChannels, hostWidth, hostHeight, hostPooledWidth,
                                                                                                hostPooledHeight, hostSrc, hostRoiData, hostDst, hostArgmax, hostSpatialScale);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::MaxROIPoolingBackward(const size_t numRois, const size_t numImg, const size_t channels, const size_t width, const size_t height,
                                                const size_t pooledWidth, const size_t pooledHeight, const GPUMatrix<ElemType>& roiData, GPUMatrix<ElemType>& grad,
                                                GPUMatrix<ElemType>& argmax, double spatialScale) const
{
    PrepareDevice();
    SyncGuard syncGuard;

    int count = numImg * channels * height * width;
    const int blockSize = GridDim::maxThreadsPerBlock;
    auto numThreads = dim3((int)floor((double)(count + blockSize - 1) / blockSize));
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kMaxROIPoolingBackward), dim3(numThreads), dim3(blockSize), 0, t_stream, static_cast<const int>(count), static_cast<const int>(numRois), static_cast<const int>(numImg), static_cast<const int>(channels), static_cast<const int>(width), static_cast<const int>(height), 
                                                                   static_cast<const int>(pooledWidth), static_cast<const int>(pooledHeight), static_cast<const ElemType*>(Data()), static_cast<const ElemType*>(roiData.Data()), static_cast<ElemType*>(grad.Data()), static_cast<const ElemType*>(argmax.Data()), static_cast<double>(spatialScale));
#else
    const int hostTotalIterations = count;
    const int hostNumROIs = numRois;
    const int hostNumImg = numImg;
    const int hostChannels = channels;
    const int hostWidth = width;
    const int hostHeight = height;
    const int hostPooledWidth = pooledWidth;
    const int hostPooledHeight = pooledHeight;
    const ElemType* hostPooledGrad = Data();
    const ElemType* hostRoiData = roiData.Data();
    ElemType* hostGrad = grad.Data();
    const ElemType* hostArgmax = argmax.Data();
    double hostSpatialScale = spatialScale;

    /*const int totalIterations,
              const int numROIs, const int numImg,
                  const int channels, const int width, const int height,
                      const int pooledWidth, const int pooledHeight, const ElemType* pooledGrad,
                          const ElemType* roiData, ElemType* grad, const ElemType* argmax, double spatialScale*/
    hipLaunchKernelGGL((kMaxROIPoolingBackward), dim3(numThreads), dim3(blockSize), 0, t_stream, hostTotalIterations, hostNumROIs, hostNumImg, hostChannels, hostWidth, hostHeight, hostPooledWidth, hostPooledHeight, hostPooledGrad, hostRoiData, hostGrad, hostArgmax, hostSpatialScale);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::MaxUnpooling(const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices, const GPUMatrix<ElemType>& poolInput, GPUMatrix<ElemType>& input) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kMaxUnpooling), dim3(gdim), dim3(BlockSize), 0, t_stream, (int)GetNumCols(), static_cast<const int*>(mpRowCol.Data()), static_cast<const int*>(mpRowIndices.Data()), static_cast<const int*>(indices.Data()),
                                                     static_cast<const ElemType*>(Data()), static_cast<const ElemType*>(poolInput.Data()), (int)GetNumRows(), static_cast<ElemType*>(input.Data()), (int)input.GetNumRows());
#else
    int hostBatchSize = GetNumCols();
    const int* hostMpRowCol = mpRowCol.Data();
    const int* hostMpRowIndices = mpRowIndices.Data();
    const int* hostIndices = indices.Data();
    const ElemType* __restrict__ hostSrc = Data();
    const ElemType* hostPoolIn = poolInput.Data();
    int hostSrcVecSize = GetNumRows();
    ElemType* hostDst = input.Data();
    int hostDstVecSize = input.GetNumRows();
    hipLaunchKernelGGL((kMaxUnpooling), dim3(gdim), dim3(BlockSize), 0, t_stream, hostBatchSize, hostMpRowCol, hostMpRowIndices, hostIndices, hostSrc, hostPoolIn, hostSrcVecSize, hostDst, hostDstVecSize);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::AveragePoolingForward(const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices, GPUMatrix<ElemType>& output) const
{
    const int BlockSize = 128;
    auto gdim = dim3((output.GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kAveragePoolingForward), dim3(gdim), dim3(BlockSize), 0, t_stream, (int)GetNumCols(), static_cast<const int*>(mpRowCol.Data()), static_cast<const int*>(mpRowIndices.Data()), static_cast<const int*>(indices.Data()),
                                                               static_cast<const ElemType*>(Data()), (int)GetNumRows(), static_cast<ElemType*>(output.Data()), (int)output.GetNumRows());
#else
    int hostBatchSize = GetNumCols();
    const int* hostMpRowCol = mpRowCol.Data();
    const int* hostMpRowIndices = mpRowIndices.Data();
    const int* hostIndices = indices.Data();
    const ElemType* __restrict__ hostSrc = Data();
    int hostSrcVecSize = GetNumRows();
    ElemType* hostDst = output.Data();
    int hostDstVecSize = output.GetNumRows();
    hipLaunchKernelGGL((kAveragePoolingForward), dim3(gdim), dim3(BlockSize), 0, t_stream, hostBatchSize, hostMpRowCol, hostMpRowIndices, hostIndices, hostSrc, hostSrcVecSize, hostDst, hostDstVecSize);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::AveragePoolingBackward(const GPUMatrix<int>& mpRowCol, const GPUMatrix<int>& mpRowIndices, const GPUMatrix<int>& indices, GPUMatrix<ElemType>& grad, bool accumulateGradient) const
{
    const int BlockSize = 128;
    auto gdim = dim3((GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)GetNumCols(), 65535));
    PrepareDevice();

    if (!accumulateGradient)
        grad.SetValue((ElemType)0);

    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((kAveragePoolingBackward), dim3(gdim), dim3(BlockSize), 0, t_stream, (int)GetNumCols(), static_cast<const int*>(mpRowCol.Data()), static_cast<const int*>(mpRowIndices.Data()), static_cast<const int*>(indices.Data()),
                                                                static_cast<const ElemType*>(Data()), (int)GetNumRows(), static_cast<ElemType*>(grad.Data()), (int)grad.GetNumRows());
#else
    int hostBatchSize = GetNumCols();
    const int* hostMpRowCol = mpRowCol.Data();
    const int* hostMpRowIndices = mpRowIndices.Data();
    const int* hostIndices = indices.Data();
    const ElemType* __restrict__ hostSrc = Data();
    int hostSrcVecSize = GetNumRows();
    ElemType* hostGrad = grad.Data();
    int hostDstVecSize = grad.GetNumRows();
    hipLaunchKernelGGL((kAveragePoolingBackward), dim3(gdim), dim3(BlockSize), 0, t_stream, hostBatchSize, hostMpRowCol, hostMpRowIndices, hostIndices, hostSrc, hostSrcVecSize, hostGrad, hostDstVecSize);
#endif
}

// returns savedMean/savedInvStdDev which are the actual values used to perform the normalization, except for blendFactor 1, in which case they are unused and set to empty
template <class ElemType>
template <class StatType>
void GPUMatrix<ElemType>::BatchNormalizationForward(const GPUMatrix<StatType>& scale, const GPUMatrix<StatType>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor,
                                                    GPUMatrix<StatType>& runMean, GPUMatrix<StatType>& runVariance, GPUMatrix<ElemType>& out, double epsilon,
                                                    GPUMatrix<StatType>& savedMean, GPUMatrix<StatType>& savedInvStdDev) const
{
    assert((GetNumRows() % scale.GetNumRows()) == 0);

    bool spatial = GetNumRows() != scale.GetNumRows();
    size_t vectorSize = GetNumRows();
    size_t spatialSize = spatial ? (GetNumRows() / scale.GetNumRows()) : 1;
    size_t batchSize = GetNumCols();
    bool normalizeRunningStats;

    assert(0 < vectorSize && vectorSize <= std::numeric_limits<int>::max());
    assert(0 < batchSize  && batchSize  <= std::numeric_limits<int>::max());

    SyncGuard syncGuard;
    if (inferenceOnly)
    {
        // Pick running statistics for normalizing. No update reuqired, and
        // saved statistics do not need to be produced.
        assert(expAvgFactor == 0 && blendFactor == 1);
        normalizeRunningStats = true;
        savedMean.RequireSize(0, 0);
        savedInvStdDev.RequireSize(0, 0);
    }
    else
    {
        // Compute data mean and inverse standard deviation (into savedMean and
        // savedInvStdDev), and update running mean and variance.
        // TODO expAvgFactor == 0 && blendFactor == 1 can be optimized (no need for update).
        normalizeRunningStats = false;
        savedMean.RequireSize(runMean);
        savedInvStdDev.RequireSize(runMean);
        if (spatial)
        {
            Call2<ComputeSpatialBatchMeanAndInvStdDev, ElemType, StatType>(spatialSize, vectorSize, spatialSize, batchSize, Data(),
                                                                expAvgFactor, blendFactor,
                                                                runMean.Data(), runVariance.Data(), epsilon,
                                                                savedMean.Data(), savedInvStdDev.Data(), GetStream());
        }
        else
        {
            Call2<ComputeBatchMeanAndInvStdDev, ElemType, StatType>(vectorSize, vectorSize, batchSize, Data(),
                                                         expAvgFactor, blendFactor,
                                                         runMean.Data(), runVariance.Data(), epsilon,
                                                         savedMean.Data(), savedInvStdDev.Data(), GetStream());
        }
    }

    Call2<NormalizeBatchTraining, ElemType, StatType>(spatial ? spatialSize : vectorSize, vectorSize, spatialSize, batchSize, spatial,
                                           normalizeRunningStats, epsilon,
                                           Data(), out.Data(),
                                           scale.Data(), bias.Data(),
                                           runMean.Data(), runVariance.Data(),
                                           savedMean.Data(), savedInvStdDev.Data(),
                                           GetStream());
}

// savedMean/savedInvStdDev are the interpolated mean/inverse standard deviation as used in ForwardProp().
// For blendFactor=1, they are not used and can be uninitialized or empty.
template <class ElemType>
template <class StatType>
void GPUMatrix<ElemType>::BatchNormalizationBackward(const GPUMatrix<ElemType>& in, GPUMatrix<ElemType>& grad, const GPUMatrix<StatType>& scale, double blendFactor,
                                                     const GPUMatrix<StatType>& savedMean, const GPUMatrix<StatType>& savedInvStdDev,
                                                     GPUMatrix<StatType>& scaleGrad, GPUMatrix<StatType>& biasGrad) const
{
    assert((GetNumRows() % scale.GetNumRows()) == 0);

    bool spatial = GetNumRows() != scale.GetNumRows();
    size_t vectorSize = GetNumRows();
    size_t spatialSize = spatial ? (GetNumRows() / scale.GetNumRows()) : 1;
    size_t batchSize = GetNumCols();

    assert(0 < vectorSize && vectorSize <= std::numeric_limits<int>::max());
    assert(0 < batchSize  && batchSize  <= std::numeric_limits<int>::max());

    SyncGuard syncGuard;
    if (spatial)
    {
        Call2<ComputeSpatialScaleAndBiasGradients, ElemType, StatType>(spatialSize, vectorSize, spatialSize, batchSize, in.Data(), Data(), scaleGrad.Data(), biasGrad.Data(),
                                                            savedMean.Data(), savedInvStdDev.Data(), GetStream());
    }
    else
    {
        Call2<ComputeScaleAndBiasGradients, ElemType, StatType>(vectorSize, vectorSize, batchSize, in.Data(), Data(), scaleGrad.Data(), biasGrad.Data(),
                                                     savedMean.Data(), savedInvStdDev.Data(), GetStream());
    }

#ifdef _MSC_VER
// half only takes float as input, so suppress the warning about double to float conversion
#pragma warning(push)
#pragma warning(disable : 4244) // warning C4244: conversion from 'double' to 'float', possible loss of data
#endif
    StatType mbStatsWeight = (StatType)(1 - blendFactor); // weight for contribution from actual MB stats (0 if none, e.g. locked BN node)
#ifdef _MSC_VER
#pragma warning(pop)
#endif

    Call2<BackpropagateBatchNormGradients, ElemType, StatType>(spatial ? spatialSize : vectorSize, vectorSize, spatialSize, batchSize, spatial,
                                                    in.Data(), Data(), grad.Data(), scale.Data(), mbStatsWeight, scaleGrad.Data(), biasGrad.Data(), savedMean.Data(), savedInvStdDev.Data(), GetStream());
}

#pragma region RNN Functions

template <class ElemType>
void GPUMatrix<ElemType>::RNNForward(const GPUMatrix<ElemType> &inputX, const GPUMatrix<ElemType> &paramW, size_t xDim, size_t yDim, const vector<size_t>& numSequencesForFrame, const RnnAttributes& rnnAttributes, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace)
{
    // numLayers, hiddenSize are input parameters
    if (!m_rnnExecutor)
        m_rnnExecutor = std::make_unique<CuDnnRNNExecutor<ElemType>>(xDim, yDim, rnnAttributes);
    m_rnnExecutor->ForwardCore(paramW, inputX, *this, numSequencesForFrame, rnnAttributes, reserve, workspace);
}

template <class ElemType>
void GPUMatrix<ElemType>::RNNBackwardData(const GPUMatrix<ElemType>& outputDY, const GPUMatrix<ElemType>& paramW, GPUMatrix<ElemType>& outputDX, const RnnAttributes& rnnAttributes, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace)
{
    if (!m_rnnExecutor)
        LogicError("RNNBackwardData called, but RNNWrapper object is not yet initialized");
    m_rnnExecutor->BackwardDataCore(*this, outputDY, paramW, outputDX, rnnAttributes, reserve, workspace);
}

template <class ElemType>
void GPUMatrix<ElemType>::RNNBackwardWeights(const GPUMatrix<ElemType>& inputX, const GPUMatrix<ElemType>& outputY, GPUMatrix<ElemType>& dw, const RnnAttributes& rnnAttributes, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace)
{
    if (!m_rnnExecutor)
        LogicError("RNNBackwardWeights called, but RNNWrapper object is not yet initialized");
    m_rnnExecutor->BackwardWeightsCore(inputX, outputY, dw, rnnAttributes, reserve, workspace);
}

#pragma region Static BLAS Functions

template <class ElemType>
void GPUMatrix<ElemType>::MultiplyAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB,
                                                 ElemType beta, GPUMatrix<ElemType>& c)
{
    a.PrepareDevice();
    if ((a.GetComputeDeviceId() != b.GetComputeDeviceId()) || (b.GetComputeDeviceId() != c.GetComputeDeviceId())) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    hipblasHandle_t cuHandle = GetCublasHandle(b.GetComputeDeviceId());
    hipblasOperation_t transA = transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t transB = transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    int m = int(transposeA ? a.m_numCols : a.m_numRows);
    int n = int(transposeB ? b.m_numRows : b.m_numCols);
    int k = int(transposeA ? a.m_numRows : a.m_numCols);
    int l = int(transposeB ? b.m_numCols : b.m_numRows);

    if (beta == 0)
        c.RequireSize(m, n);
    else
        c.VerifySize(m, n); // Can't resize if beta != 0

    if (!(m > 0 && k > 0 && l > 0 && n > 0))
        RuntimeError("!(m>0 && k>0 && l>0 && n>0)"); // converting from size_t to int may cause overflow
    if (k != l)
        RuntimeError("matrix dim mismatch in MultiplyAndWeightedAdd");
    HIPBLAS_CALL(hipblasgemmHelper(cuHandle, transA, transB, m, n, k, &alpha, a.Data(), (int) a.m_numRows, b.Data(), (int) b.m_numRows, &beta, c.Data(), (int) c.m_numRows));
}

template <class ElemType>
void GPUMatrix<ElemType>::Multiply1x1AndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, ElemType beta, GPUMatrix<ElemType>& c)
{
    a.PrepareDevice();
    if ((a.GetComputeDeviceId() != b.GetComputeDeviceId()) || (b.GetComputeDeviceId() != c.GetComputeDeviceId())) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");
    CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_multiply1x1AndWeightedAdd<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, alpha, a.Data(), b.Data(), beta, c.Data(), N);
#else
    ElemType hostAlpha = alpha;
    const ElemType* hostA = a.Data();
    const ElemType* hostB = b.Data();
    ElemType hostBeta = beta;
    ElemType* hostC = c.Data();
    CUDA_LONG hostN = N;
    hipLaunchKernelGGL((_multiply1x1AndWeightedAdd<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostAlpha, hostA, hostB, hostBeta, hostC, hostN);
#endif
}

template <class ElemType>
void GPUMatrix<ElemType>::MultiplyAndAdd(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c)
{
    return GPUMatrix<ElemType>::MultiplyAndWeightedAdd(1, a, transposeA, b, transposeB, 1, c);
}

template <class ElemType>
void GPUMatrix<ElemType>::Multiply(const GPUMatrix<ElemType>& a, const bool transposeA, const GPUMatrix<ElemType>& b, const bool transposeB, GPUMatrix<ElemType>& c)
{
    return GPUMatrix<ElemType>::MultiplyAndWeightedAdd(1, a, transposeA, b, transposeB, 0, c);
}

template <class ElemType>
void GPUMatrix<ElemType>::Multiply(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    return GPUMatrix<ElemType>::MultiplyAndWeightedAdd(1, a, false, b, false, 0, c);
}

template <class ElemType>
void GPUMatrix<ElemType>::ColumnwiseScaleAndWeightedAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& v, ElemType beta, GPUMatrix<ElemType>& c)
{
    if (v.GetNumRows() != 1 && v.GetNumCols() != 1)
        InvalidArgument("the argument v must be a vector"); // v is a vector

    if (beta == 0)
        c.RequireSize(a.GetNumRows(), a.GetNumCols());
    else
        c.VerifySize(a.GetNumRows(), a.GetNumCols()); // Can't resize if beta != 0

    int blocksPerGrid = (int)ceil(1.0 * c.GetNumElements() / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    ElemType hostAlpha = alpha;
    ElemType hostBeta = beta;
    ElemType *hostaData;
    ElemType *hostvData;
    ElemType *hostcData;
    hostaData = a.Data();
    hostvData = v.Data();
    hostcData = c.Data();
    int rows, cols;
    rows = a.GetNumRows();
    cols = a.GetNumCols();
    hipLaunchKernelGGL((_columnwiseScaleAndWeightedAdd<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream , hostAlpha, hostaData, hostvData, hostBeta, hostcData, rows, cols);
}

/// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a + c</summary>
/// if a is a column vector, add to all columns of c
/// if a is a row vector, add to all rows of c
/// if a is a scalar, add to all elements of c
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
/*static*/ void GPUMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        if (a.IsEmpty() && c.IsEmpty())
            return;
        a.PrepareDevice();
        if (a.IsEmpty() || c.IsEmpty())
            LogicError("ScaleAndAdd:  one of the input matrices is empty.");
        // if (a.GetNumRows() != 1 && a.GetNumCols() != 1) // a is not a col or row vector
        if (a.GetNumRows() == c.GetNumRows() && a.GetNumCols() == c.GetNumCols()) // dimensions match
        {
            const int m = (int) a.GetNumRows();
            const int n = (int) a.GetNumCols();
            const int len = m * n;
            const int incx = 1;
            const int incy = 1;

            assert(m > 0 && n > 0 && len > 0); // converting from size_t to int may cause overflow
            assert((int) c.GetNumRows() == m && (int) c.GetNumCols() == n);
            if ((int) c.GetNumRows() != m || (int) c.GetNumCols() != n)
                InvalidArgument("dimension of matrix c does not match dimension of matrix a.");

            hipblasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
            HIPBLAS_CALL(hipblasaxpyHelper(cuHandle, len, &alpha, a.Data(), incx, c.Data(), incy));
        }
        else if (a.GetNumElements() == 1)
        {
            CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
            int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
            c.PrepareDevice();
            SyncGuard syncGuard; 
#ifdef __HIP_ENABLE_ORG__
            hipLaunchKernelGGL((_scaleAndAddScalar<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, c.Data(), N, alpha, a.Data(), c.Data());
#else
            ElemType* hostC = c.Data();
            const CUDA_LONG hostN = N;
            const ElemType hostAlpha = alpha;
            const ElemType* hostA = a.Data();
            const ElemType* hostB = c.Data();
            hipLaunchKernelGGL((_scaleAndAddScalar<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostC, hostN, hostAlpha, hostA, hostB);
#endif
        }
        else if (a.GetNumCols() == 1) // col vector, add it to all columns
        {
            CUDA_LONG m = (CUDA_LONG) c.GetNumRows();
            CUDA_LONG n = (CUDA_LONG) c.GetNumCols();
            if (m != (CUDA_LONG) a.GetNumRows())
                InvalidArgument("To add column vector, rows should match.");

            int blocksPerGrid = (int) (ceil(1.0 * m * n / GridDim::maxThreadsPerBlock));
            SyncGuard syncGuard;
#ifdef VALIDATION
            printf(">>>> CUDA compute device is %d\n", a.GetComputeDeviceId());
            printf(">>>> a.Data()= %p, c.Data()= %p, alpha = %f, m = %ld, n = %ld\n", a.Data(), c.Data(), alpha, m, n);
            for (int i = 0; i < 2; i++)
            {
                ElemType buffer[10] = {-1.234f};
                hipError_t error = hipMemcpy(buffer, !i ? a.Data(): c.Data(), sizeof(buffer), hipMemcpyKind::hipMemcpyDeviceToHost);
                if (error == hipError_t::hipSuccess)
                    printf("buffer valid\n");
            }
#endif
            // Copying to local variables to avoid typecast issue
            const ElemType* tempA = a.Data();
            const ElemType* tempB = c.Data();
            ElemType* tempC = c.Data();
            ElemType tempAlpha = alpha;
            const CUDA_LONG tempRows = m;
            const CUDA_LONG tempCols = n;
            hipLaunchKernelGGL((_matrixVectorColumnWiseAddWithThreadPerElem<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, tempA, tempB, tempC, tempAlpha, tempRows, tempCols);
        }
        else if (a.GetNumRows() == 1) // row vector, add it to all rows
        {
            hipblasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
            int m = (int) c.GetNumRows();
            int n = (int) c.GetNumCols();
            assert(n == (int) a.GetNumCols());
            if (n != (int) a.GetNumCols())
                InvalidArgument("To add row vector, cols should match.");
            int num_x = a.GetNumElements(); //TODO: PRAS_2.4
            int num_y = c.GetNumElements();

            foreach_row (i, c)
            {
                HIPBLAS_CALL(hipblasaxpyHelper(cuHandle, n, &alpha, a.Data(), 1, c.Data()+ i, m, num_x, num_y - i));
            }
        }
        else
            InvalidArgument("dimension of matrix c does not match dimension of matrix a.");
    }
}

/// <summary>Matrix-scalar multiply with col-major matrices: c = alpha * a + b</summary>
/// if a is a column vector, add to all columns of b
/// if a is a row vector, add to all rows of b
/// if a is a scalar, add to all elements of b
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
/*static*/ void GPUMatrix<ElemType>::ScaleAndAdd(ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId() || a.GetComputeDeviceId() != b.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        if (a.IsEmpty() && b.IsEmpty())
            return;
        a.PrepareDevice();
        if (a.IsEmpty() || b.IsEmpty())
            LogicError("ScaleAndAdd: One of the input matrices is empty.");
        c.RequireSize(b.GetNumRows(), b.GetNumCols());
        // if (a.GetNumRows() != 1 && a.GetNumCols() != 1) // a is not a col or row vector
        if (a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()) // dimensions match
        {
            /*
                const int m = (int)a.GetNumRows();
                const int n = (int)a.GetNumCols();
                const int len = m * n;
                const int incx = 1;
                const int incy = 1;
                assert (m>0 && n>0 && len>0); // converting from size_t to int may cause overflow
                */
            CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
            int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
            c.PrepareDevice();
            SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
            hipLaunchKernelGGL((_matrixMatrixAddOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, alpha, a.Data(), b.Data(), c.Data(), N);
#else
            const ElemType hostAlpha = alpha;
            const ElemType* hostA = a.Data();
            const ElemType* hostB = b.Data();
            ElemType* hostC = c.Data();
            const CUDA_LONG hostN = N;
            hipLaunchKernelGGL((_matrixMatrixAddOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostAlpha, hostA, hostB, hostC, hostN);
#endif
        }
        else if (a.GetNumElements() == 1)
        {
            CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
            int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
            c.PrepareDevice();
            SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
            hipLaunchKernelGGL((_scaleAndAddScalar<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, c.Data(), N, alpha, a.Data(), b.Data());
#else
            ElemType* hostC = c.Data();
            const CUDA_LONG hostN = N;
            const ElemType hostAlpha = alpha;
            const ElemType* hostA = a.Data();
            const ElemType* hostB = b.Data();
            hipLaunchKernelGGL((_scaleAndAddScalar<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostC, hostN, hostAlpha, hostA, hostB);
#endif
        }
        else if (a.GetNumCols() == 1) // col vector, add it to all columns
        {
            CUDA_LONG m = (CUDA_LONG) c.GetNumRows();
            CUDA_LONG n = (CUDA_LONG) c.GetNumCols();
            if (m != (CUDA_LONG) a.GetNumRows())
                InvalidArgument("To add column vector, rows should match.");

            int blocksPerGrid = (int) (ceil(1.0 * m * n / GridDim::maxThreadsPerBlock));
            SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
            hipLaunchKernelGGL((_matrixVectorColumnWiseAddWithThreadPerElem<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, a.Data(), b.Data(), c.Data(), alpha, m, n);
#else
            const ElemType* tempA = a.Data();
            const ElemType* tempB = c.Data();
            ElemType* tempC = c.Data();
            ElemType tempAlpha = alpha;
            const CUDA_LONG tempRows = m;
            const CUDA_LONG tempCols = n;
            hipLaunchKernelGGL((_matrixVectorColumnWiseAddWithThreadPerElem<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, tempA, tempB, tempC, tempAlpha, tempRows, tempCols);
#endif

        }
        else if (a.GetNumRows() == 1) // row vector, add it to all rows
        {
            CUDA_LONG m = (CUDA_LONG) c.GetNumRows();
            CUDA_LONG n = (CUDA_LONG) c.GetNumCols();
            if (m != (CUDA_LONG) a.GetNumRows())
                InvalidArgument("To add column vector, rows should match.");

            int blocksPerGrid = (int) (ceil(1.0 * m * n / GridDim::maxThreadsPerBlock));
            SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
            hipLaunchKernelGGL((_matrixVectorRowWiseAddWithThreadPerElem<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, a.Data(), b.Data(), c.Data(), alpha, m, n);
#else
            const ElemType* hostA = a.Data();
            const ElemType* hostB = b.Data();
            ElemType* hostUs = c.Data();
            ElemType hostAlpha = alpha;
            const CUDA_LONG hostM = m;
            const CUDA_LONG hostN = n;
            hipLaunchKernelGGL((_matrixVectorRowWiseAddWithThreadPerElem<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostB, hostUs, hostAlpha, hostM, hostN);
#endif
        }
        else
            InvalidArgument("Dimension of matrix c does not match dimension of matrix a.");
    }
}

/// <summary>c += alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void GPUMatrix<ElemType>::AddScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        a.PrepareDevice();

        assert(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
               a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols());

        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
              a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols()))
        {
            InvalidArgument("AddScaledDifference: a, b, and c must have same dimension.");
        }

        if (a.IsEmpty())
            LogicError("AddScaledDifference: Input matrix a is empty.");

        CUDA_LONG n = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_addScaledDifference<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, alpha, a.Data(), b.Data(), c.Data(), n);
#else
        ElemType hostAlpha = alpha;
        ElemType* hostA = a.Data();
        ElemType* hostB = b.Data();
        ElemType* hostC = c.Data();
        CUDA_LONG hostN = n;
        hipLaunchKernelGGL((_addScaledDifference<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostAlpha, hostA, hostB, hostC, hostN);
#endif
    }
}

/// <summary> c = alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void GPUMatrix<ElemType>::AssignScaledDifference(const ElemType alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        a.PrepareDevice();

        assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());

        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
            InvalidArgument("AssignScaledDifference: a, b must have same dimension.");

        if (a.IsEmpty())
            LogicError("AssignScaledDifference: Input matrix a is empty.");

        if (&c != &a && &c != &b)
            c.RequireSize(a.GetNumRows(), a.GetNumCols());

        CUDA_LONG n = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_assignScaledDifference<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, alpha, a.Data(), b.Data(), c.Data(), n);
#else
        ElemType hostAlpha = alpha;
        ElemType* hostA = a.Data();
        ElemType* hostB = b.Data();
        ElemType* hostC = c.Data();
        CUDA_LONG hostN = n;
        hipLaunchKernelGGL((_assignScaledDifference<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostAlpha, hostA, hostB, hostC, hostN);
#endif
    }
}

/// <summary>c += alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">1X1 matrix</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void GPUMatrix<ElemType>::AddScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    assert(alpha.GetNumElements() == 1);
    if (!(alpha.GetNumElements() == 1))
        InvalidArgument("AddScaledDifference: alpha must be a 1X1 matrix.");

    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        a.PrepareDevice();

        assert(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
               a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols());

        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumRows() == c.GetNumRows() &&
              a.GetNumCols() == b.GetNumCols() && a.GetNumCols() == c.GetNumCols()))
        {
            InvalidArgument("AddScaledDifference: a, b, and c must have same dimension.");
        }

        if (a.IsEmpty())
            LogicError("AddScaledDifference: Input matrix a is empty.");

        CUDA_LONG n = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_addScaledDifference<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, alpha.Data(), a.Data(), b.Data(), c.Data(), n);
#else
        ElemType* hostAlpha = alpha.Data();
        ElemType* hostA = a.Data();
        ElemType* hostB = b.Data();
        ElemType* hostC = c.Data();
        CUDA_LONG hostN = n;
        hipLaunchKernelGGL((_addScaledDifference<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostAlpha, hostA, hostB, hostC, hostN);
#endif
    }
}

/// <summary> c = alpha * (a-b)</summary>
/// if a, b, c  must have same dim
/// <param name="alpha">Scalar</param>
/// <param name="a">Input matrix</param>
/// <param name="b">Input matrix</param>
/// <param name="c">Resulting matrix, user is responsible for allocating this</param>
template <class ElemType>
void GPUMatrix<ElemType>::AssignScaledDifference(const GPUMatrix<ElemType>& alpha, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    assert(alpha.GetNumElements() == 1);
    if (!(alpha.GetNumElements() == 1))
        InvalidArgument("AddScaledDifference: alpha must be a 1X1 matrix.");

    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        a.PrepareDevice();

        assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());

        if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        {
            InvalidArgument("AssignScaledDifference: a, b must have same dimension.");
        }

        if (a.IsEmpty())
            LogicError("AssignScaledDifference: Input matrix a is empty.");

        c.RequireSize(a.GetNumRows(), a.GetNumCols());

        CUDA_LONG n = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_assignScaledDifference<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, alpha.Data(), a.Data(), b.Data(), c.Data(), n);
#else
        ElemType* hostAlpha = alpha.Data();
        ElemType* hostA = a.Data();
        ElemType* hostB = b.Data();
        ElemType* hostC = c.Data();
        CUDA_LONG hostN = n;
        hipLaunchKernelGGL((_assignScaledDifference<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostAlpha, hostA, hostB, hostC, hostN);
#endif
    }
}

//c[ci,cj] += a[ai,aj]
template <class ElemType>
void GPUMatrix<ElemType>::AddElementToElement(ElemType beta, const GPUMatrix<ElemType>& a, const size_t ai, const size_t aj, GPUMatrix<ElemType>& c, const size_t ci, const size_t cj)
{
    if (ai >= a.GetNumRows() || aj >= a.GetNumCols() ||
        ci >= c.GetNumRows() || cj >= c.GetNumCols())
        InvalidArgument("AddElementToElement: Index out of range.");

    a.PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_addElementToElement<ElemType>), dim3(1), dim3(1), 0, t_stream, beta, a.Data(), (CUDA_LONG) a.LocateElement(ai, aj), c.Data(), (CUDA_LONG) c.LocateElement(ci, cj));
#else
    ElemType hostBeta = beta;
    const ElemType* hostA = a.Data();
    CUDA_LONG hostIndexA = a.LocateElement(ai, aj); 
    ElemType* hostC = c.Data();
    CUDA_LONG hostIndexC = c.LocateElement(ci, cj);
    hipLaunchKernelGGL((_addElementToElement<ElemType>), dim3(1), dim3(1), 0, t_stream, hostBeta, hostA, hostIndexA, hostC, hostIndexC);
#endif
}

template <class ElemType>
/*static*/ void GPUMatrix<ElemType>::Scale(ElemType alpha, GPUMatrix<ElemType>& a)
{
    if (alpha == 0) // if 0 then do not access the value, so that we can use this to multiply uninitialized matrices with beta=0
    {
        CUDA_CALL(hipMemset(a.Data(), 0, a.m_numRows * a.m_numCols * sizeof(ElemType)));
        return;
    }

    hipblasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    HIPBLAS_CALL(hipblasscalHelper(cuHandle, int(a.m_numRows * a.m_numCols), &alpha, a.Data(), 1));
    return;
}

template <class ElemType>
/*static*/ void GPUMatrix<ElemType>::Scale(GPUMatrix<ElemType>& alpha, GPUMatrix<ElemType>& a)
{
    if (alpha.GetNumElements() != 1)
    {
        RuntimeError("Matrix alpha must be 1x1");
    }
    hipblasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    hipblasSetPointerMode(cuHandle, HIPBLAS_POINTER_MODE_DEVICE);
    HIPBLAS_CALL(hipblasscalHelper(cuHandle, int(a.m_numRows * a.m_numCols), alpha.Data(), a.Data(), 1));
    if (sizeof(ElemType) == sizeof(float))
    {
        HIPBLAS_CALL(hipblasSscal(cuHandle, int(a.m_numRows * a.m_numCols), (float*) alpha.Data(), (float*) a.Data(), 1));
    }
    else if (sizeof(ElemType) == sizeof(double))
    {
        HIPBLAS_CALL(hipblasDscal(cuHandle, int(a.m_numRows * a.m_numCols), (double*) alpha.Data(), (double*) a.Data(), 1));
    }
    else
    {
        hipblasSetPointerMode(cuHandle, HIPBLAS_POINTER_MODE_HOST);
        RuntimeError("Unsupported template argument in GPUMatrix");
    }
    hipblasSetPointerMode(cuHandle, HIPBLAS_POINTER_MODE_HOST);
}

template <class ElemType> // c = alpha * a
/*static*/ void GPUMatrix<ElemType>::Scale(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
{
    c = a;
    Scale(alpha, c);
}

template <class ElemType>
void GPUMatrix<ElemType>::InnerProduct(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const bool isColWise)
{
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId()) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    if (a.IsEmpty() || b.IsEmpty())
        LogicError("Scale:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("Matrices a and b should have same dimension.");

    if (isColWise)
        c.RequireSize(1, n);
    else
        c.RequireSize(m, 1);

    if ((isColWise && m == 1) || (!isColWise && n == 1)) // in this case it's equivalent to element-wise product
    {
        c.AssignElementProductOf(a, b);
    }
    else
    {
        c.PrepareDevice();

        int blocksPerGrid = 0;
        if (isColWise) // col-wise
        {
            c.RequireSize(1, n);
            blocksPerGrid = (int) ceil(1.0 * n / GridDim::maxThreadsPerBlock);
        }
        else
        {
            c.RequireSize(m, 1);
            blocksPerGrid = (int) ceil(1.0 * m / GridDim::maxThreadsPerBlock);
        }

        SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_innerProduct<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, c.Data(), a.Data(), b.Data(), m, n, isColWise);
#else
        ElemType* hostC = c.Data();
        const ElemType* hostA = a.Data();
        const ElemType* hostB = b.Data();
        const CUDA_LONG hostN = m;
        const CUDA_LONG hostM = n;
        const bool hostIsColWise = isColWise;
        hipLaunchKernelGGL((_innerProduct<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostC, hostA, hostB, hostN, hostM, hostIsColWise);
#endif
    }
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::InnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProductOfMatrices:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension.");

    hipblasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    ElemType tmp = 0;
    HIPBLAS_CALL(hipblasdotHelper(cuHandle, m * n, a.Data(), 1, b.Data(), 1, &tmp));
    return tmp;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignInnerProductOfMatrices(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("InnerProductOfMatrices:  one of the input matrices is empty.");

    RequireSize(1, 1);

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension.");

    hipblasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
    hipblasSetPointerMode(cuHandle, HIPBLAS_POINTER_MODE_DEVICE);
    HIPBLAS_CALL(hipblasdotHelper(cuHandle, m * n, a.Data(), 1, b.Data(), 1, Data()));
    hipblasSetPointerMode(cuHandle, HIPBLAS_POINTER_MODE_HOST);
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::ElementWisePower(ElemType alpha, const GPUMatrix<ElemType>& a, GPUMatrix<ElemType>& c)
{
    if (a.GetComputeDeviceId() != c.GetComputeDeviceId())
    {
        InvalidArgument("All matrices must be on the same GPU");
    }
    else
    {
        if (a.IsEmpty())
            LogicError("ElementWisePower:  The input matrix a is empty.");

        c.RequireSize(a.GetNumRows(), a.GetNumCols());

        a.PrepareDevice();
        SyncGuard syncGuard;
        CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
        int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_elementWisePowerOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, alpha, a.Data(), c.Data(), N);
#else
        const ElemType hostAlpha = alpha;
        const ElemType* hostA = a.Data();
        ElemType* hostRes = c.Data();
        const CUDA_LONG hostN = N;
        hipLaunchKernelGGL((_elementWisePowerOnCuda<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostAlpha, hostA, hostRes, hostN);
#endif
    }
}

template <class ElemType>
void GPUMatrix<ElemType>::BatchMatMul(ElemType beta, const GPUMatrix<ElemType>& a, const bool transposeA, const int m, const GPUMatrix<ElemType>& b, const bool transposeB, const int n, GPUMatrix<ElemType>& c, const bool isColWise)
{
    a.PrepareDevice();
    if ((a.GetComputeDeviceId() != b.GetComputeDeviceId()) || (b.GetComputeDeviceId() != c.GetComputeDeviceId())) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    if (!isColWise)
        LogicError("Only column wise is supported.");

    hipblasHandle_t cuHandle = GetCublasHandle(b.GetComputeDeviceId());
    hipblasOperation_t transA = transposeA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t transB = transposeB ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    const int aSampleElemNum = (int)a.GetNumRows();
    const int aBatchSize = (int)a.GetNumCols();
    const int bSampleElemNum = (int)b.GetNumRows();
    const int bBatchSize = (int)b.GetNumCols();

    if (!(aSampleElemNum > 0 && aBatchSize > 0 && bSampleElemNum > 0 && bBatchSize > 0))
        RuntimeError("BatchMatMul: Matrices a and b's cols & rows number should > 0.");
    if (aBatchSize != bBatchSize)
        RuntimeError("BatchMatMul: Matrices a and b should have same batch size.");

    int k = aSampleElemNum / m;
    int kb = bSampleElemNum / n;
    if (k != kb)
        InvalidArgument("BatchMatMul: Matrices a's cols number should match Matrices b's rows number.");
    size_t cSampleElemNum = m * n;

    if (beta == 0)
        c.RequireSize(cSampleElemNum, aBatchSize);
    else
        c.VerifySize(cSampleElemNum, aBatchSize); // Can't resize if beta != 0

    const ElemType alpha = 1.0;

    const int lda = transposeA ? k : m;
    const int ldb = transposeB ? n : k;
    const int ldc = m;
    ElemType* aBufPtr = a.Data();
    ElemType* bBufPtr = b.Data();
    ElemType* cBufPtr = c.Data();
    std::vector<const ElemType*> Aarray;
    std::vector<const ElemType*> Barray;
    std::vector<ElemType*> Carray;
    Aarray.reserve(aBatchSize);
    Barray.reserve(aBatchSize);
    Carray.reserve(aBatchSize);
    for (int i = 0; i < aBatchSize; i++)
    {
        Aarray.push_back(aBufPtr + a.LocateColumn(i));
        Barray.push_back(bBufPtr + b.LocateColumn(i));
        Carray.push_back(cBufPtr + c.LocateColumn(i));
    }
    ElemType** devAList = 0;
    ElemType** devBList = 0;
    ElemType** devCList = 0;
    CUDA_CALL(hipMalloc(&devAList, aBatchSize * sizeof(ElemType*)));
    CUDA_CALL(hipMalloc(&devBList, aBatchSize * sizeof(ElemType*)));
    CUDA_CALL(hipMalloc(&devCList, aBatchSize * sizeof(ElemType*)));
    CUDA_CALL(hipMemcpy(devAList, &Aarray[0], sizeof(ElemType*) * aBatchSize, hipMemcpyHostToDevice));
    CUDA_CALL(hipMemcpy(devBList, &Barray[0], sizeof(ElemType*) * aBatchSize, hipMemcpyHostToDevice));
    CUDA_CALL(hipMemcpy(devCList, &Carray[0], sizeof(ElemType*) * aBatchSize, hipMemcpyHostToDevice));

    HIPBLAS_CALL(hipblasGemmBatchedHelper(cuHandle, transA, transB, m, n, k, &alpha, (const ElemType**)devAList, lda, (const ElemType**)devBList, ldb, &beta, devCList, ldc, aBatchSize));
    CUDA_CALL(hipFree(devAList));
    CUDA_CALL(hipFree(devBList));
    CUDA_CALL(hipFree(devCList));
}

template <class ElemType>
bool GPUMatrix<ElemType>::AreEqual(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const ElemType threshold /*= 1e-8*/)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AreEqual: one of the input matrices is empty.");

    if (a.GetNumRows() != b.GetNumRows() || a.GetNumCols() != b.GetNumCols())
        return false;

    bool bResult = false;

    long* res = new long[1];
    res[0] = 1;
    long* d_res = TracingGPUMemoryAllocator::Allocate<long>(a.GetComputeDeviceId(), 1);
    CUDA_CALL(hipMemcpy(d_res, res, sizeof(long) * 1, hipMemcpyHostToDevice));
    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
    // Copying to local variables to avoid typecast issue
    const ElemType* tempA = a.Data();
    const ElemType* tempB = b.Data();
    const CUDA_LONG tempN = N;
    const ElemType tempThreshold = threshold;
    long* tempD_res = d_res;
    hipLaunchKernelGGL((_areEqual<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, tempA, tempB, tempN, tempThreshold, tempD_res);
    CUDA_CALL(hipMemcpy(res, d_res, sizeof(long) * 1, hipMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<long>(a.GetComputeDeviceId(), d_res);
    if (res[0] != 0)
        bResult = true;
    delete[] res;
    return bResult;
}

// see Matrix<ElemType>::TensorShuffleScaleAndAdd() for comments
template <class ElemType>
void GPUMatrix<ElemType>::TensorShuffleScaleAndAdd(ElemType keepWeight, const GPUMatrix<ElemType>& a, size_t D, size_t S, size_t M, size_t K, size_t T, ElemType scaleFactor, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c)
{
    CUDA_LONG N = (CUDA_LONG) c.GetNumElements();
    assert(N == (CUDA_LONG) a.GetNumElements() && N == (CUDA_LONG) b.GetNumElements());
    assert(a.GetComputeDeviceId() == c.GetComputeDeviceId() && b.GetComputeDeviceId() == c.GetComputeDeviceId());
    a.PrepareDevice();
    SyncGuard syncGuard;
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_tensorShuffleScaleAndAdd<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, keepWeight, a.Data(), D, S, M, K, T, scaleFactor, b.Data(), c.Data());
#else
    ElemType hostKeepWeight = keepWeight;
    const ElemType* hostPa = a.Data();
    size_t hostD = D;
    size_t hostS = S;
    size_t hostM = M;
    size_t hostK = K;
    size_t hostT = T;
    ElemType hostScaleFacto = scaleFactor;
    const ElemType* hostPb = b.Data();
    ElemType* hostPc = c.Data();
    hipLaunchKernelGGL((_tensorShuffleScaleAndAdd<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostKeepWeight, hostPa, hostD, hostS, hostM, hostK, hostT, hostScaleFacto, hostPb, hostPc);
#endif
}

template <class ElemType>
bool GPUMatrix<ElemType>::HasElement(const GPUMatrix<ElemType>& a, const ElemType v)
{
    if (a.IsEmpty())
        LogicError("HasElement: the input matrix is empty.");

    bool bResult = false;
    ElemType* res = new ElemType[2];
    res[0] = v;
    res[1] = 0;
    ElemType* d_res = TracingGPUMemoryAllocator::Allocate<ElemType>(a.GetComputeDeviceId(), 2);
    CUDA_CALL(hipMemcpy(d_res, res, sizeof(ElemType) * 2, hipMemcpyHostToDevice));
    CUDA_LONG N = (CUDA_LONG) a.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_hasElement<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, a.Data(), N, d_res);
#else
    const ElemType* hostA = a.Data();
    const CUDA_LONG hostN = N;
    ElemType* hostD_res = d_res;
    hipLaunchKernelGGL((_hasElement<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostN, hostD_res);
#endif
    CUDA_CALL(hipMemcpy(res, d_res, sizeof(ElemType) * 2, hipMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(a.GetComputeDeviceId(), d_res);
    if (res[1] != 0)
        bResult = true;
    else
        bResult = false;

    delete[] res;
    return bResult;
}

template <class ElemType>
void GPUMatrix<ElemType>::CreateCurandObject(unsigned long seed, const char* caller)
{
    assert(caller != nullptr);

    if (s_hiprandGenerator == NULL)
    {
        unsigned long long hipSeed = (seed == USE_TIME_BASED_SEED) ? time(NULL) : seed;
        if (GetMathLibTraceLevel() > 0)
        {
            fprintf(stderr, "%s (GPU): creating hiprand object with seed %llu, sizeof(ElemType)==%lu\n",
                    caller, hipSeed, (unsigned long)sizeof(ElemType));
        }
        s_hiprandGenerator = new hiprandGenerator_t;
        // Create pseudo-random number generator
        HIPRAND_CALL(hiprandCreateGenerator(&(((hiprandGenerator_t*) s_hiprandGenerator)[0]), HIPRAND_RNG_PSEUDO_XORWOW));
        HIPRAND_CALL(hiprandSetPseudoRandomGeneratorSeed(((hiprandGenerator_t*) s_hiprandGenerator)[0], hipSeed));
        //TODO: __hip__ HIPRAND_CALL(hiprandSetGeneratorOrdering(((hiprandGenerator_t*) s_hiprandGenerator)[0], HIPRAND_ORDERING_PSEUDO_SEEDED));
    }
}

template <class ElemType>
void GPUMatrix<ElemType>::ResetCurandObject(unsigned long seed, const char* caller)
{
    assert(caller != nullptr);

    if (s_hiprandGenerator && (seed != USE_TIME_BASED_SEED))
    {
        // Note: this might be slow.
        HIPRAND_CALL(hiprandSetPseudoRandomGeneratorSeed(((hiprandGenerator_t*) s_hiprandGenerator)[0], seed));
        HIPRAND_CALL(hiprandSetGeneratorOffset(((hiprandGenerator_t*) s_hiprandGenerator)[0], 0));
    }
    else
    {
        CreateCurandObject(seed, caller);
    }
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Ones(const size_t rows, const size_t cols, int deviceId)
{
    GPUMatrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetValue(1);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Zeros(const size_t rows, const size_t cols, int deviceId)
{
    GPUMatrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    // c.SetValue(0);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::Eye(const size_t rows, int deviceId)
{
    GPUMatrix<ElemType> c(rows, rows, deviceId); // will initialize to 0
    c.SetDiagonalValue(1);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::RandomUniform(const size_t rows, const size_t cols, int deviceId, const ElemType low, const ElemType high, unsigned long seed)
{
    GPUMatrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetUniformRandomValue(low, high, seed);
    return c;
}

template <class ElemType>
GPUMatrix<ElemType> GPUMatrix<ElemType>::RandomGaussian(const size_t rows, const size_t cols, int deviceId, const ElemType mean, const ElemType sigma, unsigned long seed)
{
    GPUMatrix<ElemType> c(rows, cols, deviceId); // will initialize to 0
    c.SetGaussianRandomValue(mean, sigma, seed);
    return c;
}

template <class ElemType>
ElemType GPUMatrix<ElemType>::GetLearnRateForBlock_Helper(const GPUMatrix<ElemType>& Gradients, const GPUMatrix<ElemType>& SmoothedGradients)
{
    ElemType* d_res = TracingGPUMemoryAllocator::Allocate<ElemType>(Gradients.GetComputeDeviceId(), 1);

    // Compute inner product of matrices and keep it on device
    const int m = (int) Gradients.GetNumRows();
    const int n = (int) Gradients.GetNumCols();
    const int k = (int) SmoothedGradients.GetNumRows();
    const int l = (int) SmoothedGradients.GetNumCols();
    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("InnerProductOfMatrices: Matrices a and b should have same dimension.");

    hipblasHandle_t cuHandle = GetCublasHandle(Gradients.GetComputeDeviceId());
    hipblasSetPointerMode(cuHandle, HIPBLAS_POINTER_MODE_DEVICE);
    HIPBLAS_CALL(hipblasdotHelper(cuHandle, m * n, Gradients.Data(), 1, SmoothedGradients.Data(), 1, d_res));
    hipblasSetPointerMode(cuHandle, HIPBLAS_POINTER_MODE_HOST);

    // d_res[0] should now contain inner product of matrices
    // Compute squared Frobenius norms (squared sums of elements)
    // note: kernel has hard-coded dimension of 512
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_lrHelper512Threads<ElemType>), dim3(1), dim3(512), 0, t_stream, Gradients.Data(), SmoothedGradients.Data(), (CUDA_LONG)Gradients.GetNumElements(), d_res);
#else
    const ElemType* hostData1 = Gradients.Data();
    const ElemType* hostData2 = SmoothedGradients.Data();
    const CUDA_LONG hostN = Gradients.GetNumElements();
    ElemType* hostD_res = d_res;
    hipLaunchKernelGGL((_lrHelper512Threads<ElemType>), dim3(1), dim3(512), 0, t_stream, hostData1, hostData2, hostN, hostD_res);
#endif
    ElemType res;
    CUDA_CALL(hipMemcpy(&res, d_res, sizeof(ElemType), hipMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(Gradients.GetComputeDeviceId(), d_res);
    return res;
}
// The inputs are two row vectors [a1 a2 a3 a4] [b1 b2 b3 b4]
// The outputs are one matrix of size (nt+1)*4
// The first row is just element multiplication
// The rest rows will be with shift
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementProductOfWithShiftNeg(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const size_t shift, const size_t nt)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOf: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    if (!(a.GetNumRows() == 1))
        InvalidArgument("The input matrix must be a row vector.");

    RequireSize(nt + 1, a.GetNumCols());
    int BS = a.GetNumCols();

    // the output matrix is of size (nt+1, BS)
    dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
    dim3 block_tail((nt + 1 + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (BS + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

    a.PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignElementProductOfWithShiftNeg<ElemType>), dim3(block_tail), dim3(thread_tail), 0, t_stream, Data(), a.Data(), b.Data(), shift, nt + 1, BS);
#else
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const ElemType* hostB = b.Data();
    const int hostShift = shift;
    const int hostNTPlusOne = nt + 1;
    const int hostBS = BS;
    hipLaunchKernelGGL((_assignElementProductOfWithShiftNeg<ElemType>), dim3(block_tail), dim3(thread_tail), 0, t_stream, hostUs, hostA, hostB, hostShift, hostNTPlusOne, hostBS);
#endif
    //      hipLaunchKernelGGL((_assignElementProductOf<ElemType>), dim3(block_tail), dim3(thread_tail), 0, t_stream, Data(), a.Data(), b.Data(), nt);

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignOneHot(const GPUMatrix<ElemType>& a, vector<size_t>& shape, size_t axis)
{
    if (a.IsEmpty())
        LogicError("AssignOneHot: Matrix a is empty.");

    if (axis >= shape.size())
        LogicError("AssignOneHot: axis is not correct");

    size_t item_size = 1;
    for (size_t i = 0; i < shape.size() && i < axis; i++)
        item_size *= shape[i];

    size_t num_class = shape[axis];

    auto nCols = a.GetNumCols();
    auto nRows = num_class * a.GetNumRows();
    this->RequireSize(nRows, nCols);
    this->PrepareDevice();

    CUDA_CALL(hipMemset(Data(), 0, nCols * nRows * sizeof(ElemType)));


    CUDA_LONG N = (CUDA_LONG)a.GetNumElements();
    int blocksPerGrid = (int)ceil(((double)N) / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
    // Copying to local variables to avoid typecast issue
    ElemType *tempIndices = a.Data();
    ElemType *tempTargetBuffer = Data();
    size_t tempNum_class = num_class;
    size_t tempNum_item = item_size;
    size_t tempNum_element = N;
    hipLaunchKernelGGL((_assignOneHot<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, tempIndices, tempTargetBuffer, tempNum_class, tempNum_item, tempNum_element);
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::GatherFromTarget(const GPUMatrix<ElemType>& indices, const GPUMatrix<ElemType>& target, size_t row_elements)
{
    if (indices.IsEmpty() || target.IsEmpty())
        LogicError("GatherFromTarget: input matrix is empty.");

    if (row_elements == 0)
        LogicError("GatherFromTarget: target matrix at least need 1 dim.");

    auto nCols = indices.GetNumCols();
    auto nRows = indices.GetNumRows() * row_elements;
    this->RequireSize(nRows, nCols);
    this->PrepareDevice();

    ElemType* indicesBufPtr = indices.Data();
    ElemType* targetBufPtr = target.Data();
    ElemType* buffer = Data();

    size_t num_indices = indices.GetNumElements();
    CUDA_LONG N = (CUDA_LONG)num_indices * row_elements;
    int blocksPerGrid = (int)ceil(((double)N) / GridDim::maxThreadsPerBlock);
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_gatherFromTarget<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock ), 0, 0, indicesBufPtr, targetBufPtr, buffer, row_elements, num_indices, N);
#else
    ElemType *hostIndices = indicesBufPtr;
    ElemType *hostTarget = targetBufPtr;
    ElemType *hostBuffer = buffer;
    size_t hostNum_row_elements = row_elements;
    size_t hostNum_indices = num_indices;
    CUDA_LONG hostNum_elements = N;
    hipLaunchKernelGGL((_gatherFromTarget<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock ), 0, 0, hostIndices, hostTarget, hostBuffer, hostNum_row_elements, hostNum_indices, hostNum_elements);
#endif

    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::ScatterToIndices(const GPUMatrix<ElemType>& values, const GPUMatrix<ElemType>& indices, size_t row_elements)
{
    if (indices.IsEmpty() || values.IsEmpty())
        LogicError("ScatterToIndices: input matrix is empty.");

    ElemType* indicesBufPtr = indices.Data();
    ElemType* valueBufPtr = values.Data();
    ElemType* buffer = Data();

    size_t num_indices = indices.GetNumElements();
    CUDA_LONG N = (CUDA_LONG)num_indices * row_elements;
    int blocksPerGrid = (int)ceil(((double)N) / GridDim::maxThreadsPerBlock);
    hipLaunchKernelGGL((_scatterToIndices<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, indicesBufPtr, valueBufPtr, buffer, row_elements, num_indices, N);
    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::InnerProductWithShiftNeg(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const size_t shift, const size_t nt)
{
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId()) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    if (a.IsEmpty() || b.IsEmpty())
        LogicError("Scale:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int k = (int) b.GetNumRows();
    const int l = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && k > 0 && l > 0); // converting from size_t to int may cause overflow
    assert(m == k && n == l);                 // converting from size_t to int may cause overflow
    if (m != k || n != l)
        InvalidArgument("Matrices a and b should have same dimension.");

    c.RequireSize(nt + 1, n);

    if (true)
    {
        c.PrepareDevice();

        dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
        dim3 block_tail((nt + 1 + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (n + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

        SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_innerProductWithShiftNeg<ElemType>), dim3(block_tail), dim3(thread_tail), 0, t_stream, c.Data(), a.Data(), b.Data(), m, n, shift, nt + 1);
#else
        ElemType* hostC = c.Data();
        const ElemType* hostA = a.Data();
        const ElemType* hostB = b.Data();
        const CUDA_LONG hostN = m; // a.GetNumRows();
        const CUDA_LONG hostM = n; // a.GetNumCols();
        const CUDA_LONG hostShift = shift;
        const CUDA_LONG hostNTPlusOne = nt + 1;
        hipLaunchKernelGGL((_innerProductWithShiftNeg<ElemType>), dim3(block_tail), dim3(thread_tail), 0, t_stream, hostC, hostA, hostB, hostN, hostM, hostShift, hostNTPlusOne);
#endif
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::GetARowByIndex(const GPUMatrix<ElemType>& a, const size_t m)
{
    if (a.IsEmpty())
        LogicError("GetARowByIndex: Matrix is empty.");

    RequireSize(1, a.GetNumCols());

    int n = a.GetNumRows();
    int P = a.GetNumCols();

    if (m >= n)
        LogicError("GetARowByIndex: m is out of range.");

    int blocksPerGrid = (int) ceil(((double) P) / GridDim::maxThreadsPerBlock);

    a.PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_getARowByIndex<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), n, P, m);
#else
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const int hostO = n;
    const int hostP = P;
    const int hostM = m;
    hipLaunchKernelGGL((_getARowByIndex<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostO, hostP, hostM);
#endif
    //      hipLaunchKernelGGL((_assignElementProductOf<ElemType>), dim3(block_tail), dim3(thread_tail), 0, t_stream, Data(), a.Data(), b.Data(), nt);
    return *this;
}

// Calculate CTC score
// prob (input): the posterior output from the network
// alpha, beta (output): alpha and beta for forward-backward calculation.
// phoneSeq (input): phone ID sequence for each utterance in this minibatch, each col is one utterance
// phoneBoundary (input): phone boundary (frame index) of each phone for each utterance in this minibatch, each col is one utterance
// totalScore (output): total CTC score
// uttToChanInd (input):  map from utterance ID to minibatch channel ID. We need this because each channel may contain more than one utterance.
// uttBeginFrame(input): the position of the first frame of each utterance in the minibatch channel. We need this because each channel may contain more than one utterance.
// uttFrameNum (input): the frame number of each utterance. The size of this vector =  the number of all utterances in this minibatch
// uttPhoneNum (input): the phone number of each utterance. The size of this vector =  the number of all utterances in this minibatch
// numParallelSequences (input): channel number in this minibatch
// maxFrameNum (input): the maximum channel frame number
// delayConstraint -- label output delay constraint introduced during training that allows to have shorter delay during inference.
//      Alpha and Beta scores outside of the delay boundary are set to zero.
//      Setting this parameter smaller will result in shorted delay between label output during decoding, yet may hurt accuracy
//      delayConstraint=-1 means no constraint
template<class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignCTCScore(const GPUMatrix<ElemType>& prob,
    GPUMatrix<ElemType>& alpha,
    GPUMatrix<ElemType>& beta,
    const GPUMatrix<ElemType> phoneSeq,
    const GPUMatrix<ElemType> phoneBoundary,
    GPUMatrix<ElemType> &totalScore,
    const std::vector<size_t>& uttToChanInd,
    const std::vector<size_t> & uttBeginFrame,
    const std::vector<size_t> & uttFrameNum,
    const std::vector<size_t> & uttPhoneNum,
    const size_t numParallelSequences,
    const size_t maxFrameNum,
    const size_t blankTokenId,
    const int delayConstraint,
    const bool isColWise)
{
    if (isColWise)
    {
        PrepareDevice();
        // Total number of phones
        long totalPhoneNum = prob.GetNumRows();
        size_t uttNum = uttFrameNum.size();

        // Max number of phones in utterances in this minibatch
        size_t maxPhoneNum = phoneSeq.GetNumRows();

        size_t *gpuFrameNum;
        CUDA_CALL(hipMalloc((void **)&gpuFrameNum, uttNum * sizeof(size_t)));
        CUDA_CALL(hipMemcpy(gpuFrameNum, uttFrameNum.data(), uttNum * sizeof(size_t), hipMemcpyHostToDevice));

        size_t *gpuPhoneNum;
        CUDA_CALL(hipMalloc((void **)&gpuPhoneNum, uttNum * sizeof(size_t)));
        CUDA_CALL(hipMemcpy(gpuPhoneNum, uttPhoneNum.data(), uttNum * sizeof(size_t), hipMemcpyHostToDevice));

        size_t *gpuBeginFrame;
        CUDA_CALL(hipMalloc((void **)&gpuBeginFrame, uttNum * sizeof(size_t)));
        CUDA_CALL(hipMemcpy(gpuBeginFrame, uttBeginFrame.data(), uttNum * sizeof(size_t), hipMemcpyHostToDevice));

        size_t *gpuUttToChanInd;
        CUDA_CALL(hipMalloc((void **)&gpuUttToChanInd, uttNum * sizeof(size_t)));
        CUDA_CALL(hipMemcpy(gpuUttToChanInd, uttToChanInd.data(), uttNum * sizeof(size_t), hipMemcpyHostToDevice));

        hipEvent_t done = nullptr;
        CUDA_CALL(hipEventCreate(&done));
        dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
        // x dimension is for utterances
        // y dimention is for phone sequence in each utterance
        // Ensure that we allocate correct number of blocks for given number of utterances and max number of phones in those utterances
        dim3 block_tail((uttNum + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (maxPhoneNum + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);
        for (long t = 0; t < maxFrameNum; t++)
        {
#ifdef __HIP_ENABLE_ORG__
            hipLaunchKernelGGL((_assignAlphaScore), dim3(block_tail), dim3(thread_tail), 0, t_stream, static_cast<const ElemType*>(prob.Data()), static_cast<ElemType*>(alpha.Data()), static_cast<ElemType*>(phoneSeq.Data()), static_cast<ElemType*>(phoneBoundary.Data()), static_cast<const size_t*>(gpuUttToChanInd),
                static_cast<const size_t*>(gpuFrameNum), static_cast<const size_t*>(gpuBeginFrame), static_cast<const size_t*>(gpuPhoneNum), static_cast<size_t>(numParallelSequences), static_cast<const size_t>(uttNum), static_cast<const size_t>(t), static_cast<const size_t>(maxPhoneNum), static_cast<const size_t>(totalPhoneNum), static_cast<const size_t>(blankTokenId), static_cast<const int>(delayConstraint));
#else
            const ElemType *hostProb = prob.Data();
            ElemType *hostAlphaScore = alpha.Data();
            ElemType *hostPhoneSeq = phoneSeq.Data();
            ElemType *hostPhoneBound = phoneBoundary.Data();
            const size_t *hostUttToChanInd = gpuUttToChanInd;
            const size_t *hostUttFrameNum = gpuFrameNum;
            const size_t *hostUttBeginFrame = gpuBeginFrame;
            const size_t *hostUttPhoneNum = gpuPhoneNum;
            size_t hostNumChannels = numParallelSequences;
            const size_t hostUttNum = uttNum;
            const size_t  hostT = t;
            const size_t hostMaxPhoneNum = maxPhoneNum; // Maximum length of utterance in this MB
            const size_t hostTotalPhoneNum = totalPhoneNum; // Total number of phones
            const size_t hostBlankTokenId = blankTokenId;
            const int hostDelayConstraint = delayConstraint;
            hipLaunchKernelGGL((_assignAlphaScore), dim3(block_tail), dim3(thread_tail), 0, t_stream, hostProb, hostAlphaScore, hostPhoneSeq, hostPhoneBound, hostUttToChanInd, hostUttFrameNum, hostUttBeginFrame, hostUttPhoneNum, hostNumChannels, hostUttNum, hostT, hostMaxPhoneNum, hostTotalPhoneNum, hostBlankTokenId, hostDelayConstraint);
#endif
        }

        for (long t = maxFrameNum - 1; t >= 0; t--)
        {
#ifdef __HIP_ENABLE_ORG__
            hipLaunchKernelGGL((_assignBetaScore), dim3(block_tail), dim3(thread_tail), 0, t_stream, static_cast<const ElemType*>(prob.Data()), static_cast<ElemType*>(beta.Data()), static_cast<ElemType*>(phoneSeq.Data()), static_cast<ElemType*>(phoneBoundary.Data()), static_cast<const size_t*>(gpuUttToChanInd),
                static_cast<const size_t*>(gpuFrameNum), static_cast<const size_t*>(gpuBeginFrame), static_cast<const size_t*>(gpuPhoneNum), static_cast<const size_t>(numParallelSequences), static_cast<const size_t>(uttNum), static_cast<const size_t>(t), static_cast<const size_t>(maxPhoneNum), static_cast<const size_t>(totalPhoneNum), static_cast<const size_t>(blankTokenId), static_cast<const int>(delayConstraint));
#else
            const ElemType *hostProb = prob.Data();
            ElemType *hostBetaScore = beta.Data();
            ElemType *hostPhoneSeq = phoneSeq.Data();
            ElemType *hostPhoneBound = phoneBoundary.Data();
            const size_t *hostUttToChanInd = gpuUttToChanInd;
            const size_t *hostUttFrameNum = gpuFrameNum;
            const size_t *hostUttBeginFrame = gpuBeginFrame;
            const size_t *hostUttPhoneNum = gpuPhoneNum;
            const size_t hostNumChannels = numParallelSequences;
            const size_t hostUttNum = uttNum;
            const size_t  hostT = t;
            const size_t hostMaxPhoneNum = maxPhoneNum;
            const size_t hostTotalPhoneNum = totalPhoneNum;
            const size_t hostBlankTokenId = blankTokenId;
            const int hostDelayConstraint = delayConstraint;
            hipLaunchKernelGGL((_assignBetaScore), dim3(block_tail), dim3(thread_tail), 0, t_stream, hostProb, hostBetaScore, hostPhoneSeq, hostPhoneBound, hostUttToChanInd,
                hostUttFrameNum, hostUttBeginFrame, hostUttPhoneNum, hostNumChannels, hostUttNum, hostT, hostMaxPhoneNum, hostTotalPhoneNum, hostBlankTokenId, hostDelayConstraint);
#endif
        }

        ElemType zerVar = 0.0;
        totalScore.SetColumn(&zerVar, 0);
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_assignTotalScore), dim3(uttNum), dim3(1), 0, t_stream, static_cast<ElemType*>(beta.Data()), static_cast<ElemType*>(totalScore.Data()), static_cast<const size_t>(uttNum), static_cast<const size_t*>(gpuUttToChanInd), static_cast<const size_t*>(gpuBeginFrame), static_cast<const size_t>(numParallelSequences), static_cast<const size_t>(maxPhoneNum));
#else
        ElemType *hostBetaScore = beta.Data();
        ElemType *hostTotalScore = totalScore.Data();
        const size_t hostUttNum = uttNum;
        const size_t *hostUttToChanInd = gpuUttToChanInd;
        const size_t *hostUttBeginFrame = gpuBeginFrame;
        const size_t hostNumChannels = numParallelSequences;
        const size_t hostMaxPhoneNum = maxPhoneNum;
        hipLaunchKernelGGL((_assignTotalScore), dim3(uttNum), dim3(1), 0, t_stream, hostBetaScore, hostTotalScore, hostUttNum, hostUttToChanInd, hostUttBeginFrame, hostNumChannels, hostMaxPhoneNum);
#endif
        dim3 block_tail_2((uttNum + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (maxFrameNum + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_assignCTCScore), dim3(block_tail_2), dim3(thread_tail), 0, t_stream, static_cast<ElemType*>(Data()), static_cast<ElemType*>(prob.Data()), static_cast<ElemType*>(alpha.Data()), static_cast<ElemType*>(beta.Data()), static_cast<ElemType*>(phoneSeq.Data()), static_cast<const size_t>(uttNum), static_cast<const size_t*>(gpuUttToChanInd),
            static_cast<const size_t*>(gpuBeginFrame), static_cast<const size_t*>(gpuPhoneNum), static_cast<const size_t*>(gpuFrameNum), static_cast<const long>(numParallelSequences), static_cast<const long>(maxPhoneNum), static_cast<const long>(totalPhoneNum));
#else
        ElemType *hostCTCscore = Data();
        ElemType *hostProb = prob.Data();
        ElemType *hostAlphaScore = alpha.Data();
        ElemType *hostBetaScore1 = beta.Data();
        ElemType *hostPhoneSeq = phoneSeq.Data();
        const size_t hostUttNum1 = uttNum;
        const size_t *hostUttToChanInd1 = gpuUttToChanInd;
        const size_t *hostUttBeginFrame1 = gpuBeginFrame;
        const size_t *hostUttPhoneNum = gpuPhoneNum;
        const size_t *hostUttFrameNum = gpuFrameNum;
        const long hostNumChannels1 = numParallelSequences;
        const long hostMaxPhoneNum1 = maxPhoneNum;
        const long hostTotalPhoneNum = totalPhoneNum;
        hipLaunchKernelGGL((_assignCTCScore), dim3(block_tail_2), dim3(thread_tail), 0, t_stream, hostCTCscore, hostProb, hostAlphaScore, hostBetaScore1, hostPhoneSeq, hostUttNum1, hostUttToChanInd1,
            hostUttBeginFrame1, hostUttPhoneNum, hostUttFrameNum, hostNumChannels1, hostMaxPhoneNum1, hostTotalPhoneNum);
#endif
        CUDA_CALL(hipFree(gpuFrameNum));
        CUDA_CALL(hipFree(gpuPhoneNum));
        CUDA_CALL(hipFree(gpuBeginFrame));
        CUDA_CALL(hipFree(gpuUttToChanInd));

        CUDA_CALL(hipEventRecord(done));
        CUDA_CALL(hipEventSynchronize(done));
        CUDA_CALL(hipEventDestroy(done));
    }
    else
    {
        NOT_IMPLEMENTED;
    }

    return *this;
}

template <class ElemType>
void GPUMatrix<ElemType>::ConductRowElementMultiplyWithShift(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, GPUMatrix<ElemType>& c, const size_t shift, const bool isafixed)
{
    if (a.GetComputeDeviceId() != b.GetComputeDeviceId() || b.GetComputeDeviceId() != c.GetComputeDeviceId()) // different GPUs
        InvalidArgument("All matrices must be on the same GPU");

    if (a.IsEmpty() || b.IsEmpty())
        LogicError("Scale:  one of the input matrices is empty.");

    const int m = (int) a.GetNumRows();
    const int n = (int) a.GetNumCols();
    const int O = (int) b.GetNumRows();
    const int P = (int) b.GetNumCols();

    assert(m > 0 && n > 0 && O > 0 && P > 0); // converting from size_t to int may cause overflow
    if (m != 1 || n != P)
        InvalidArgument("Matrices a and b should have same dimension.");

    c.RequireSize(O, P);

    if (true)
    {
        c.PrepareDevice();

        dim3 thread_tail(DEFAULT_THREAD_PER_DIM, DEFAULT_THREAD_PER_DIM);
        dim3 block_tail((O + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM, (P + DEFAULT_THREAD_PER_DIM - 1) / DEFAULT_THREAD_PER_DIM);

        SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_conductRowElementMultiplyWithShift<ElemType>), dim3(block_tail), dim3(thread_tail), 0, t_stream, c.Data(), a.Data(), b.Data(), O, P, shift, isafixed);
#else
        ElemType* hostUs = c.Data();
        const ElemType* hostA = a.Data();
        const ElemType* hostB = b.Data();
        const int hostO = O;
        const int hostP = P;
        const int hostShift = shift;
        const bool hostIsafixed = isafixed;
        hipLaunchKernelGGL((_conductRowElementMultiplyWithShift<ElemType>), dim3(block_tail), dim3(thread_tail), 0, t_stream, hostUs, hostA, hostB, hostO, hostP, hostShift, hostIsafixed);
#endif
    }
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignElementProductOfWithShift(const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const size_t shift)
{
    if (a.IsEmpty() || b.IsEmpty())
        LogicError("AssignElementProductOfWithShift: Matrix is empty.");

    assert(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols());
    if (!(a.GetNumRows() == b.GetNumRows() && a.GetNumCols() == b.GetNumCols()))
        InvalidArgument("The input matrix dimensions do not match.");

    // int O = a.GetNumRows();
    int P = a.GetNumCols();

    RequireSize(1, P);
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
    a.PrepareDevice();
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_assignElementProductOfWithShift<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, Data(), a.Data(), b.Data(), shift, N);
#else
    ElemType* hostUs = Data();
    const ElemType* hostA = a.Data();
    const ElemType* hostB = b.Data();
    const int hostShift = shift;
    const CUDA_LONG hostN = N;
    hipLaunchKernelGGL((_assignElementProductOfWithShift<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostUs, hostA, hostB, hostShift, hostN);
#endif
    return *this;
}

//sequence training
template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::DropFrame(const GPUMatrix<ElemType>& label, const GPUMatrix<ElemType>& gamma, const ElemType& threshhold)
{
    if (IsEmpty())
        LogicError("DropFrame: Matrix is empty.");

    PrepareDevice();

    long N = (long) GetNumCols(); // one kernel per column
    int blocksPerGrid = (int) ceil(N * 1.0 / GridDim::maxThreadsPerBlock);
    SyncGuard syncGuard;
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_DropFrame), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, static_cast<ElemType*>(Data()), static_cast<const ElemType*>(label.Data()), static_cast<const ElemType*>(gamma.Data()), static_cast<const ElemType>(threshhold), (long) m_numCols, (long) m_numRows);
#else
    ElemType* hostA = Data();
    const ElemType* hostLabel = label.Data();
    const ElemType* hostGamma = gamma.Data();
    const ElemType hostFramedropthreshhold = threshhold;
    const long hostM_numCols = m_numCols;
    const long hostM_numRows = m_numRows;
    hipLaunchKernelGGL((_DropFrame), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostA, hostLabel, hostGamma, hostFramedropthreshhold, hostM_numCols, hostM_numRows);
#endif
    return *this;
}

template <class ElemType>
GPUMatrix<ElemType>& GPUMatrix<ElemType>::AssignSequenceError(const ElemType hsmoothingWeight, const GPUMatrix<ElemType>& label,
                                                              const GPUMatrix<ElemType>& dnnoutput, const GPUMatrix<ElemType>& gamma, ElemType alpha)
{
    if (IsEmpty())
        LogicError("AssignSequenceError: Matrix is empty.");

    PrepareDevice();

    SyncGuard syncGuard;
    long N = (LONG64) label.GetNumElements();
    int blocksPerGrid = (int) ceil(1.0 * N / GridDim::maxThreadsPerBlock);
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_AssignSequenceError), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, static_cast<const ElemType>(hsmoothingWeight), static_cast<ElemType*>(Data()), static_cast<const ElemType*>(label.Data()), static_cast<const ElemType*>(dnnoutput.Data()), static_cast<const ElemType*>(gamma.Data()), static_cast<ElemType>(alpha), static_cast<long>(N));
#else
    const ElemType hostHsmoothingWeight = hsmoothingWeight;
    ElemType* hostError = Data();
    const ElemType* hostLabel = label.Data();
    const ElemType* hostDnnoutput = dnnoutput.Data();
    const ElemType* hostGamma = gamma.Data();
    ElemType hostAlpha = alpha;
    const long hostN = N;
    hipLaunchKernelGGL((_AssignSequenceError), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, t_stream, hostHsmoothingWeight, hostError, hostLabel, hostDnnoutput, hostGamma, hostAlpha, hostN);
#endif
    return *this;
}

#pragma endregion Static BLAS Functions

/// f = logadd(f, vec) to get the logadd sum of vector elments
template <class ElemType>
ElemType GPUMatrix<ElemType>::LogSumOfElements() const
{
    if (IsEmpty())
        LogicError("SumOfElements: Matrix is empty");

    ElemType* d_sum = TracingGPUMemoryAllocator::Allocate<ElemType>(GetComputeDeviceId(), 1);

    ElemType h_sum;
    CUDA_LONG N = (CUDA_LONG) GetNumElements();
    int blocksPerGrid = (int) ceil(((double) N) / GridDim::maxThreadsPerBlock);
#ifdef __HIP_ENABLE_ORG__
    hipLaunchKernelGGL((_reductionLogAddSum<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, Data(),
                                                                                  d_sum, 1, N);
#else
    const ElemType* hostData = Data();
    ElemType* hostSum = d_sum;
    const size_t hostSum_size = 1;
    CUDA_LONG hostN = N;
    hipLaunchKernelGGL((_reductionLogAddSum<ElemType>), dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, hostData, hostSum, hostSum_size, hostN);
#endif
    CUDA_CALL(hipMemcpy(&h_sum, d_sum, sizeof(ElemType), hipMemcpyDeviceToHost));
    TracingGPUMemoryAllocator::Free<ElemType>(GetComputeDeviceId(), d_sum);

    return h_sum;
}

template <class ElemType>
void GPUMatrix<ElemType>::RCRFBackwardCompute(
    const GPUMatrix<ElemType>& alpha, GPUMatrix<ElemType>& beta,
    const GPUMatrix<ElemType>& /*lbls*/,
    const GPUMatrix<ElemType>& pos_scores, const GPUMatrix<ElemType>& pair_scores, const int shift)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    if (alpha.IsEmpty() || pos_scores.IsEmpty() || pair_scores.IsEmpty())
        LogicError("RCRFBackwardCompute: one of the input matrices is empty.");

    if (alpha.GetNumRows() != pos_scores.GetNumRows() || alpha.GetNumCols() != pos_scores.GetNumCols())
        LogicError("RCRFBackwardCompute: matrix dimensions mismatched.");

    size_t iNumLab = alpha.GetNumRows();
    size_t iNumPos = alpha.GetNumCols();

    alpha.PrepareDevice();
    beta.RequireSize(iNumLab, iNumPos);

    ElemType* d_zeta = TracingGPUMemoryAllocator::Allocate<ElemType>(alpha.GetComputeDeviceId(), iNumLab);

    CUDA_LONG N = iNumLab;
    // TODO: change all three '512' to 'GridDim::maxThreadsPerBlock' (not doing this now since I cannot test it)
    int blocksPerGrid = (int) ceil(1.0 * N / 512);
    size_t szMemSize;
    for (int t = iNumPos - 1; t >= 0; t--)
    {
        szMemSize = sizeof(comp_t) * iNumLab;
        // This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == iNumLab.
        assert(iNumLab <= 1024);
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_rcrfBackwardComputeZetaMax1024Labels<ElemType>), dim3(blocksPerGrid), dim3(512), szMemSize, 0, t, iNumPos, alpha.Data(), d_zeta, pair_scores.Data(), iNumLab, shift);
#else
        const size_t hostT = t;
        const size_t hostINumPos = iNumPos;
        const ElemType* hostGalpha = alpha.Data();
        ElemType* hostGzeta = d_zeta;
        const ElemType* hostGpair_scores = pair_scores.Data();
        const size_t hostINumLab = iNumLab;
        const int hostShift = shift;
        hipLaunchKernelGGL((_rcrfBackwardComputeZetaMax1024Labels<ElemType>), dim3(blocksPerGrid), dim3(512), szMemSize, 0, hostT, hostINumPos, hostGalpha, hostGzeta, hostGpair_scores, hostINumLab, hostShift);
#endif
        szMemSize = iNumLab * 3;
        szMemSize *= sizeof(comp_t);
        // This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == 3 * iNumLab.
        assert(iNumLab <= 1024);
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_rcrfBackwardComputeMax1024Labels<ElemType>), dim3(blocksPerGrid), dim3(512), szMemSize, 0, t, iNumPos, alpha.Data(), beta.Data(),
                                                                                           d_zeta, pair_scores.Data(), iNumLab, shift);
#else
        const size_t hostT1= t;
        const size_t hostINumPos1= iNumPos;
        const ElemType* hostGalpha1 = alpha.Data();       // column slice at current time t
        ElemType* hostGbeta = beta.Data();             // column slices with [row, 2] at current time t for 
        const ElemType* hostGzeta1 = d_zeta;        // column slices with [row, 2] at current time t for [
        const ElemType* hostGpair_scores1 = pair_scores.Data(); // column slice at current time t
        const size_t hostINumLab1 = iNumLab;
        const int hostShift1 = shift;
        hipLaunchKernelGGL((_rcrfBackwardComputeMax1024Labels<ElemType>), dim3(blocksPerGrid), dim3(512), szMemSize, 0, hostT1, hostINumPos1, hostGalpha1, hostGbeta,
                                                                                           hostGzeta1, hostGpair_scores1, hostINumLab1, hostShift1);
#endif
    }
    /*
        error = hipGetErrorString(hipPeekAtLastError());
        printf("%s\n", error);
        error = hipGetErrorString(hipDeviceSynchronize());
        printf("%s\n", error);
        */
    TracingGPUMemoryAllocator::Free<ElemType>(alpha.GetComputeDeviceId(), d_zeta);
}

/**
    Compute the gradient for the first order Markov transition probabilities
    It uses equations derived in R. Collobert's paper "Natural language processing (almost) from scratch"
    */
template <class ElemType>
void GPUMatrix<ElemType>::RCRFTransGrdCompute(const GPUMatrix<ElemType>& lbls,
                                              const GPUMatrix<ElemType>& alpha,
                                              const GPUMatrix<ElemType>& beta,
                                              const GPUMatrix<ElemType>& pair_scores,
                                              GPUMatrix<ElemType>& grd,
                                              const int startLbl,
                                              const int shift)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    assert(shift == 1);
    int iNumPos = alpha.GetNumCols();
    int iNumLab = alpha.GetNumRows();

    ElemType* d_zeta = TracingGPUMemoryAllocator::Allocate<ElemType>(alpha.GetComputeDeviceId(), iNumLab);

    CUDA_LONG N = iNumLab;
    // TODO: change all three '512' to 'GridDim::maxThreadsPerBlock' (not doing this now since I cannot test it)
    int blocksPerGrid = (int)ceil(1.0 * N / 512);
    size_t szMemSize;
    for (int t = 0; t < iNumPos; t++)
    {
        szMemSize = sizeof(comp_t) * iNumLab;
        // This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == iNumLab.
        assert(iNumLab <= 1024);
        // BUGBUG: This is launched with 512 threads per block, but allocates shared mem as if there is only one block. Likewise for all 4 of these functions.
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_rcrfTransGrdComputeZetaMax1024Labels<ElemType>), dim3(blocksPerGrid), dim3(512), szMemSize, 0, t - 1, iNumPos, alpha.Data(), d_zeta, pair_scores.Data(), iNumLab, startLbl, shift);
#else
        const size_t hostT = t - 1;
        const size_t hostINumPos = iNumPos;
        const ElemType* hostGalpha = alpha.Data();
        ElemType* hostGzeta = d_zeta;
        const ElemType* hostGpair_scores = pair_scores.Data();
        const size_t hostINumLab = iNumLab;
        const size_t hostStart_lbl = startLbl;
        const int hostShift = shift;
        hipLaunchKernelGGL((_rcrfTransGrdComputeZetaMax1024Labels<ElemType>), dim3(blocksPerGrid), dim3(512), szMemSize, 0, hostT, hostINumPos, hostGalpha, hostGzeta, hostGpair_scores, hostINumLab, hostStart_lbl, hostShift);
#endif
        szMemSize = iNumLab * 3;
        szMemSize *= sizeof(comp_t);
        // This function assumes iNumLab <= 1024 and that shared memory == total (!) number of threads == iNumLab.
        assert(iNumLab <= 1024);
#ifdef __HIP_ENABLE_ORG__
        hipLaunchKernelGGL((_rcrfTransGrdComputeMax1024Labels<ElemType>), dim3(blocksPerGrid), dim3(512), szMemSize, 0, t, startLbl, alpha.Data(), beta.Data(),
                                                                                           d_zeta, pair_scores.Data(), lbls.Data(), grd.Data(), iNumPos, iNumLab, shift);
#else
        int hostT1 = t;
        const size_t hostStart_lbl1 = startLbl;
        const ElemType* hostGalpha1 = alpha.Data();
        const ElemType* hostGbeta1 = beta.Data();
        const ElemType* hostGzeta1 = d_zeta;
        const ElemType* hostGpair_scores1 = pair_scores.Data();
        const ElemType* hostLbls = lbls.Data();
        ElemType* hostGrd = grd.Data();
        const size_t hostINumPos1 = iNumPos;
        const size_t hostINumLab1 = iNumLab;
        const int hostShift1 = shift;
        hipLaunchKernelGGL((_rcrfTransGrdComputeMax1024Labels<ElemType>), dim3(blocksPerGrid), dim3(512), szMemSize, 0, hostT1, hostStart_lbl1, hostGalpha1, hostGbeta1,
                                                                                           hostGzeta1, hostGpair_scores1, hostLbls, hostGrd, hostINumPos1, hostINumLab1, hostShift1);
#endif
    }
    TracingGPUMemoryAllocator::Free<ElemType>(alpha.GetComputeDeviceId(), d_zeta);
};

// -----------------------------------------------------------------------
// TensorView entry points from Matrix.cpp
// -----------------------------------------------------------------------

// helper to provide a vector of ones of at least the given number of elements
// TODO: Use this to implement ComputationNode::ConstOnes? Or do we even need that anymore?
template <class ElemType>
static shared_ptr<GPUMatrix<ElemType>> GetOnesVector(size_t N, DEVICEID_TYPE deviceId)
{
    // using a dynamically allocated array so this will never get freed, avoiding free-after-DLL-unload issues.
    // and using shared_ptrs since we don't want to leak more than CacheSize elements
    // when using a plain array we would have to control lifetime of the object and destructor would be called for every element in the array at the end
    const int CacheSize = 32;
    static shared_ptr<GPUMatrix<ElemType>> * onesCache = new shared_ptr<GPUMatrix<ElemType>>[CacheSize]; // cache of objects

    if (deviceId >= CacheSize){
        LogicError("GetOnesVector: onesCache[] too small (%d entries), increase (you need %d) and recompile.", CacheSize, (int)deviceId + 1);
    }

    auto p = onesCache[deviceId];
    if (!p || p->GetNumRows() < N) // must (re-)allocate
    {
        p = make_shared<GPUMatrix<ElemType>>(GPUMatrix<ElemType>::Ones(N, 1, deviceId));
        onesCache[deviceId] = p; // this will replace the pointer thread-safely (although weird race conditions may happen where a larger entry is overwritten by a smaller one; will still run correctly)
    }
    return p;
}

// perform unary operation 'op' on a giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
// This binds the N-ariness to a template parameter N, and gets the data pointers out from the matrix objects.
template <class ElemType>
void GPUMatrix<ElemType>::TensorOp(ElemType beta, const GPUMatrix<ElemType>& a, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const array<size_t, 2>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opSum    &&
        reductionOp != ElementWiseOperator::opLogSum &&
        reductionOp != ElementWiseOperator::opMin    &&
        reductionOp != ElementWiseOperator::opMax    &&
        reductionOp != ElementWiseOperator::opElementwiseProduct)
        InvalidArgument("TensorOp: Unary reduction operations other than opMax, opMin, opSum, and opLogSum are not implemented.");

    a.PrepareDevice();
    if (a.GetComputeDeviceId() != GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");

    // special case: linear processing
    // The case statement has measurable impact for unary ops (but not for binary ops it seems, due to double mem access).
    // Linear gap-free unary ops happen so regularly that we will eliminate the case statement from the CUDA kernel, and instead expand all.
    if (regularOpDims.size() == 1 && regularStrides[0][0] == 1 && regularStrides[1][0] == 1 && reducingOpDims.size() == 0)
    {
        // special case: for copy, use hipMemcpy() instead, or hipblas_axpy()
        // TODO: We should observe if these actually make a speed difference, and if not, remove these special cases.
        if (op == ElementWiseOperator::opCopy && beta == 0 && alpha == 1)
            return CUDA_CALL(hipMemcpy(Data()+ offsets[1], a.Data()+ offsets[0], sizeof(ElemType) * regularOpDims[0], hipMemcpyDeviceToDevice));
        else if (op == ElementWiseOperator::opCopy && beta == 1)
            return HIPBLAS_CALL(hipblasaxpyHelper(GetCublasHandle(GetComputeDeviceId()), (int) regularOpDims[0], &alpha, a.Data()+ offsets[0], 1, Data()+ offsets[1], 1));
        else
            return LaunchUnaryTensorOp<ElemType>(beta, a.Data()+ offsets[0], Data()+ offsets[1], alpha, op, regularOpDims[0]);
    }

    // special case: sum-reducing a matrix onto a column vector; can be done with SGEMM
    // Note: A minor risk is that with this, our own reduction function will rarely be used.
    // That function was tested to give the same results with 'double', and nearly the same with 'float' (different summation order matters).
    else if (op == ElementWiseOperator::opCopy && // we are just adding to target without any further operation
             reductionOp == ElementWiseOperator::opSum &&
#ifdef _DEBUG
             sizeof(ElemType) == sizeof(float) && // in debug don't shortcut 'double' so we have some test of our own codepath
#endif
             regularOpDims.size() == 1 && regularStrides[0][0] == 1 && regularStrides[1][0] == 1 && // we are processing a column
             reducingOpDims.size() == 1 && reducingStrides[0][0] >= (ptrdiff_t) regularOpDims[0])   // reducing across columns and no overlap
    {
        assert(reducingStrides[1][0] == 0);
        auto ARows = regularOpDims[0];    // vertical steps
        auto ACols = reducingOpDims[0];   // horizontal steps (reduction)
        auto ALd = reducingStrides[0][0]; // horizontal step width through matrix
        hipblasHandle_t cuHandle = GetCublasHandle(a.GetComputeDeviceId());
        HIPBLAS_CALL(hipblasgemmHelper(cuHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, (int) /*CRows=*/ARows, /*CCols=*/1, (int) ACols, &alpha,
                                /*A00=*/a.Data()+ offsets[0], (int) ALd,
                                /*B00=*/GetOnesVector<ElemType>(ACols, a.GetComputeDeviceId())->Data(), (int) /*BRows=*/ACols, &beta,
                                /*C00=*/Data()+ offsets[1], (int) /*CRows=*/ARows));
        return;
    }

    // TODO: Add a special case for tensor bias reduction. cudnn is ~7% faster on Image/QuickE2E.

    // regular case
    else
        return TensorOpN<ElemType, 2>(beta, array<ElemType*, 2>{a.Data(), Data()}, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

// perform binary operation 'op' on a and b giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
template <class ElemType>
void GPUMatrix<ElemType>::TensorOp(ElemType beta, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const array<size_t, 3>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 3>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 3>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opSum)
        InvalidArgument("TensorOp: The only permitted binary reduction operation is opSum.");

    a.PrepareDevice();
    if (a.GetComputeDeviceId() != GetComputeDeviceId() || b.GetComputeDeviceId() != GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");

    return TensorOpN<ElemType, 3>(beta, array<ElemType*, 3>{a.Data(), b.Data(), Data()}, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

// perform ternary operation 'op' on a, and c giving 'this', reinterpreting the matrices as tensors as specified by the dims and strides
template <class ElemType>
void GPUMatrix<ElemType>::TensorOp(ElemType beta, const GPUMatrix<ElemType>& a, const GPUMatrix<ElemType>& b, const GPUMatrix<ElemType>& c, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
                                   const array<size_t, 4>& offsets,
                                   const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 4>& regularStrides,
                                   const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 4>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opSum)
        InvalidArgument("TensorOp: The only permitted ternary reduction operation is opSum.");

    a.PrepareDevice();
    if (a.GetComputeDeviceId() != GetComputeDeviceId() || b.GetComputeDeviceId() != GetComputeDeviceId() || c.GetComputeDeviceId() != GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");
    return TensorOpN<ElemType, 4>(beta, array<ElemType*, 4>{a.Data(), b.Data(), c.Data(), Data()}, alpha, op, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

template <class ElemType>
void GPUMatrix<ElemType>::TensorArgOp(const GPUMatrix<ElemType>& a, ElementWiseOperator reductionOp,
                                      const array<size_t, 2>& offsets,
                                      const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, 2>& regularStrides,
                                      const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, 2>& reducingStrides)
{
    if (reductionOp != ElementWiseOperator::opArgmin &&
        reductionOp != ElementWiseOperator::opArgmax)
        InvalidArgument("TensorOp: Arg reduction operations other than opArgmax, and opArgmin are not implemented.");

    a.PrepareDevice();
    if (a.GetComputeDeviceId() != GetComputeDeviceId())
        InvalidArgument("All matrices must be on the same GPU");
    return TensorOpN<ElemType, 2>((ElemType) 0, array<ElemType*, 2>{a.Data(), Data()}, (ElemType) 1, ElementWiseOperator::opCopy, reductionOp, offsets, regularOpDims, regularStrides, reducingOpDims, reducingStrides);
}

// =======================================================================
// explicit instantiations business
// =======================================================================
template class GPUMatrix<float>;
template class GPUMatrix<double>;
#ifdef __HIP_ENABLE_HALF__
template class GPUMatrix<half>;
#endif /*__HIP_ENABLE_HALF__*/
template class DeviceBoundNumber<float>;
template class DeviceBoundNumber<double>;
#ifdef __HIP_ENABLE_HALF__
template class DeviceBoundNumber<half>;
#endif /*__HIP_ENABLE_HALF__*/

// instantiation of cast methods
template void GPUMatrix<char>::CastAssignValuesOf<float>(const GPUMatrix<float>* other);
template void GPUMatrix<char>::CastAssignValuesOf<double>(const GPUMatrix<double>* other);
#ifdef __HIP_ENABLE_HALF__
template void GPUMatrix<char>::CastAssignValuesOf<half>(const GPUMatrix<half>* other);
#endif /*__HIP_ENABLE_HALF__*/
template void GPUMatrix<short>::CastAssignValuesOf<float>(const GPUMatrix<float>* other);
template void GPUMatrix<short>::CastAssignValuesOf<double>(const GPUMatrix<double>* other);
#ifdef __HIP_ENABLE_HALF__
template void GPUMatrix<short>::CastAssignValuesOf<half>(const GPUMatrix<half>* other);
#endif /*__HIP_ENABLE_HALF__*/
template void GPUMatrix<int>::CastAssignValuesOf<float>(const GPUMatrix<float>* other);
template void GPUMatrix<int>::CastAssignValuesOf<double>(const GPUMatrix<double>* other);
#ifdef __HIP_ENABLE_HALF__
template void GPUMatrix<int>::CastAssignValuesOf<half>(const GPUMatrix<half>* other);
#endif /*__HIP_ENABLE_HALF__*/
template void GPUMatrix<float>::CastAssignValuesOf<float>(const GPUMatrix<float>* other);
template void GPUMatrix<float>::CastAssignValuesOf<double>(const GPUMatrix<double>* other);
#ifdef __HIP_ENABLE_HALF__
template void GPUMatrix<float>::CastAssignValuesOf<half>(const GPUMatrix<half>* other);
#endif /*__HIP_ENABLE_HALF__*/
template void GPUMatrix<double>::CastAssignValuesOf<float>(const GPUMatrix<float>* other);
template void GPUMatrix<double>::CastAssignValuesOf<double>(const GPUMatrix<double>* other);
#ifdef __HIP_ENABLE_HALF__
template void GPUMatrix<double>::CastAssignValuesOf<half>(const GPUMatrix<half>* other);
template void GPUMatrix<half>::CastAssignValuesOf<float>(const GPUMatrix<float>* other);
template void GPUMatrix<half>::CastAssignValuesOf<double>(const GPUMatrix<double>* other);
template void GPUMatrix<half>::CastAssignValuesOf<half>(const GPUMatrix<half>* other);
#endif /*__HIP_ENABLE_HALF__*/

// instantiation of templated methods
template void GPUMatrix<float>::AdaDelta<float>(GPUMatrix<float>& gradients, GPUMatrix<float>& functionValues, float learningRate, float rho, float epsilon);
template void GPUMatrix<double>::AdaDelta<double>(GPUMatrix<double>& gradients, GPUMatrix<double>& functionValues, double learningRate, double rho, double epsilon);
#ifdef __HIP_ENABLE_HALF__
template void GPUMatrix<float>::AdaDelta<half>(GPUMatrix<half>& gradients, GPUMatrix<float>& functionValues, float learningRate, float rho, float epsilon);
#endif /*__HIP_ENABLE_HALF__*/

template void GPUMatrix<float>::BatchNormalizationForward(const GPUMatrix<float>& scale, const GPUMatrix<float>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, GPUMatrix<float>& runMean, GPUMatrix<float>& runVariance, GPUMatrix<float>& out, double epsilon, GPUMatrix<float>& saveMean, GPUMatrix<float>& saveInvStdDev) const;
template void GPUMatrix<double>::BatchNormalizationForward(const GPUMatrix<double>& scale, const GPUMatrix<double>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, GPUMatrix<double>& runMean, GPUMatrix<double>& runVariance, GPUMatrix<double>& out, double epsilon, GPUMatrix<double>& saveMean, GPUMatrix<double>& saveInvStdDev) const;
#ifdef __HIP_ENABLE_HALF__
template void GPUMatrix<half>::BatchNormalizationForward(const GPUMatrix<float>& scale, const GPUMatrix<float>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, GPUMatrix<float>& runMean, GPUMatrix<float>& runVariance, GPUMatrix<half>& out, double epsilon, GPUMatrix<float>& saveMean, GPUMatrix<float>& saveInvStdDev) const;
#endif /*__HIP_ENABLE_HALF__*/

template void GPUMatrix<float>::BatchNormalizationBackward(const GPUMatrix<float>& in, GPUMatrix<float>& grad, const GPUMatrix<float>& scale, double blendFactor, const GPUMatrix<float>& saveMean, const GPUMatrix<float>& saveInvStdDev, GPUMatrix<float>& scaleGrad, GPUMatrix<float>& biasGrad) const;
template void GPUMatrix<double>::BatchNormalizationBackward(const GPUMatrix<double>& in, GPUMatrix<double>& grad, const GPUMatrix<double>& scale, double blendFactor, const GPUMatrix<double>& saveMean, const GPUMatrix<double>& saveInvStdDev, GPUMatrix<double>& scaleGrad, GPUMatrix<double>& biasGrad) const;
#ifdef __HIP_ENABLE_HALF__
template void GPUMatrix<half>::BatchNormalizationBackward(const GPUMatrix<half>& in, GPUMatrix<half>& grad, const GPUMatrix<float>& scale, double blendFactor, const GPUMatrix<float>& saveMean, const GPUMatrix<float>& saveInvStdDev, GPUMatrix<float>& scaleGrad, GPUMatrix<float>& biasGrad) const;
#endif /*__HIP_ENABLE_HALF__*/

template <class ElemType>
hipblasHandle_t GPUMatrix<ElemType>::s_cuHandle[GPUMatrix<ElemType>::MaxGpus] = {0};

template <class ElemType>
void* GPUMatrix<ElemType>::s_hiprandGenerator = NULL;

// We use Matrix<char> as the backing store for QuantizedMatrix
// Let's explicitly instantiate the methods we need for that purpose
template GPUMatrix<char>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId);
template GPUMatrix<char>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId, char* pArray, const size_t matrixFlags);
template GPUMatrix<char>::GPUMatrix(const GPUMatrix<char>&);
template GPUMatrix<char>::GPUMatrix(GPUMatrix<char>&&);
template char* GPUMatrix<char>::CopyToArray() const;
template void GPUMatrix<char>::ChangeDeviceTo(int);
template void GPUMatrix<char>::Resize(size_t, size_t, bool);
template void GPUMatrix<char>::RequireSize(size_t, size_t, bool);

template GPUMatrix<char>::~GPUMatrix();
template GPUMatrix<char> GPUMatrix<char>::ColumnSlice(size_t startColumn, size_t numCols) const;
template GPUMatrix<char>& GPUMatrix<char>::operator=(GPUMatrix<char>&&);
template GPUMatrix<char>::GPUMatrix(int);
template void GPUMatrix<char>::SetValue(const char);
template void GPUMatrix<char>::SetValue(const size_t numRows, const size_t numCols, int deviceId, char* pArray, size_t matrixFlags, DataTransferer* transferer);
//template void GPUMatrix<char>::SetValue(CPUMatrix<char> const&);
template void GPUMatrix<char>::SetValue(GPUMatrix<char> const&);
//template void GPUMatrix<char>::SetValue(CPUSparseMatrix<char> const&);
//template void GPUMatrix<char>::SetValue(GPUSparseMatrix<char> const&);
template void GPUMatrix<char>::CopySection(size_t numRows, size_t numCols, char* dst, size_t colStride) const;
template void GPUMatrix<char>::Reshape(const size_t, const size_t);
template GPUMatrix<char>& GPUMatrix<char>::operator*=(char);
template DEVICEID_TYPE GPUMatrix<char>::PrepareDevice(DEVICEID_TYPE deviceId) const;

// Support <short>
template GPUMatrix<short>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId);
template GPUMatrix<short>::GPUMatrix(const size_t numRows, const size_t numCols, int deviceId, short* pArray, const size_t matrixFlags);
template GPUMatrix<short>::GPUMatrix(const GPUMatrix<short>&);
template GPUMatrix<short>::GPUMatrix(GPUMatrix<short>&&);
template short* GPUMatrix<short>::CopyToArray() const;
template void GPUMatrix<short>::ChangeDeviceTo(int);
template void GPUMatrix<short>::Resize(size_t, size_t, bool);
template void GPUMatrix<short>::RequireSize(size_t, size_t, bool);

template GPUMatrix<short>::~GPUMatrix();
template GPUMatrix<short> GPUMatrix<short>::ColumnSlice(size_t startColumn, size_t numCols) const;
template GPUMatrix<short>& GPUMatrix<short>::operator=(GPUMatrix<short>&&);
template GPUMatrix<short>::GPUMatrix(int);
template void GPUMatrix<short>::SetValue(const short);
template void GPUMatrix<short>::SetValue(const size_t numRows, const size_t numCols, int deviceId, short* pArray, size_t matrixFlags, DataTransferer* transferer);
//template void GPUMatrix<short>::SetValue(CPUMatrix<short> const&);
template void GPUMatrix<short>::SetValue(GPUMatrix<short> const&);
//template void GPUMatrix<short>::SetValue(CPUSparseMatrix<short> const&);
//template void GPUMatrix<short>::SetValue(GPUSparseMatrix<short> const&);
template void GPUMatrix<short>::CopySection(size_t numRows, size_t numCols, short* dst, size_t colStride) const;
template void GPUMatrix<short>::Reshape(const size_t, const size_t);
template GPUMatrix<short>& GPUMatrix<short>::operator*=(short);
template DEVICEID_TYPE GPUMatrix<short>::PrepareDevice(DEVICEID_TYPE deviceId) const;

template GPUMatrix<int>::GPUMatrix(const size_t, const size_t, int, int*, const size_t);
template GPUMatrix<int>::~GPUMatrix();

template int* TracingGPUMemoryAllocator::Allocate<int>(int, size_t);
template size_t* TracingGPUMemoryAllocator::Allocate<size_t>(int, size_t);
template long* TracingGPUMemoryAllocator::Allocate<long>(int, size_t);
template short* TracingGPUMemoryAllocator::Allocate<short>(int, size_t);
template char* TracingGPUMemoryAllocator::Allocate<char>(int, size_t);
template float* TracingGPUMemoryAllocator::Allocate<float>(int, size_t);
template double* TracingGPUMemoryAllocator::Allocate<double>(int, size_t);
#ifdef __HIP_ENABLE_HALF__
template half* TracingGPUMemoryAllocator::Allocate<half>(int, size_t);
#endif /*__HIP_ENABLE_HALF__*/

template void TracingGPUMemoryAllocator::Free<int>(int, int*, bool);
template void TracingGPUMemoryAllocator::Free<size_t>(int, size_t*, bool);
template void TracingGPUMemoryAllocator::Free<short>(int, short*, bool);
template void TracingGPUMemoryAllocator::Free<char>(int, char*, bool);
template void TracingGPUMemoryAllocator::Free<float>(int, float*, bool);
template void TracingGPUMemoryAllocator::Free<double>(int, double*, bool);
#ifdef __HIP_ENABLE_HALF__
template void TracingGPUMemoryAllocator::Free<half>(int, half*, bool);
#endif /*__HIP_ENABLE_HALF__*/

}}}

// !!!!This is from helper_cuda.h which comes with CUDA samples!!!! Consider if it is beneficial to just include all helper_cuda.h
// TODO: This is duplicated in BestGpu.cpp
// Beginning of GPU Architecture definitions
int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
        {
            {0x10, 8},   // Tesla Generation (SM 1.0) G80 class
            {0x11, 8},   // Tesla Generation (SM 1.1) G8x class
            {0x12, 8},   // Tesla Generation (SM 1.2) G9x class
            {0x13, 8},   // Tesla Generation (SM 1.3) GT200 class
            {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
            {0x21, 48},  // Fermi Generation (SM 2.1) GF10x class
            {0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
            {0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
            {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }
    return nGpuArchCoresPerSM[7].Cores;
};
// end of GPU Architecture definitions

//inline CUDA_LONG _GetFreeMemoryOnCUDADevice(int devId)
//{
//    CUdevice cudaDevice;
//    CUresult result = cuDeviceGet(&cudaDevice, devId);
//    if(result!= CUDA_SUCCESS)
//    {
//        return 0;
//    }
//
//    // create cuda context
//    CUcontext cudaContext;
//    result = cuCtxCreate(&cudaContext, CU_CTX_SCHED_AUTO, cudaDevice);
//    if(result != CUDA_SUCCESS)
//    {
//        return 0;
//    }
//
//    // get the amount of free memory on the graphics card
//    size_t free;
//    size_t total;
//    result = cuMemGetInfo(&free, &total);
//    if (result!=CUDA_SUCCESS)
//    {
//        return 0;
//    }
//    else
//        return (CUDA_LONG)free;
//}

#endif // CPUONLY
