//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "GPUMatrixCUDAKernels.cuh"
#include "CuDnnFactories.h"
#include "Convolution.cuh"
#include "GPUMatrix.h"
#include <typeinfo>
#include <typeindex>
#include "CuDnnCommon.h"
#include "half.hpp"

//temporary...
#define _HIPDBG_

// We want tensor core be enabled in order to get(v7)/find tensor core results. But if algo without tensorcore is faster, the only way to force faster algo is to turn it off. Since re-tuning can happen quite often in CNTK, it gets bad if we don't do it carefully. It also require move to get_v7 and we can't test until we can run fp16.
// For now, let's keep it simple and enable tensor core all the time for fp16.
#include <cxxabi.h>

template <>
const char* CudaErrString<hipdnnStatus_t>(hipdnnStatus_t x)
{
    return hipdnnGetErrorString(x);
}

// A note on the formats: CNTK originally used NHWC for input/output tensors and CHWN for kernels.
// Such formats have very limited support in cuDNN and not used in other frameworks.
// CNTK with cuDNN by default uses NCHW formats for both inputs/outputs and kernels.
#define TENSOR_FORMAT HIPDNN_TENSOR_NCHW
#define FILTER_FORMAT HIPDNN_TENSOR_NCHW

namespace Microsoft { namespace MSR { namespace CNTK {

class CuDnnKernel
{
public:
    CuDnnKernel(const ConvolveGeometry& geometry, hipdnnDataType_t dataType)
        : m_kernel(nullptr)
    {
        HIPDNN_CALL(hipdnnCreateFilterDescriptor(&m_kernel));
        // Set cuDNN kernel dimensions. cuDNN uses row-major format while TensorShape - column-major
        // so conversion is required.
        const auto& filt = geometry.KernelShape();
        size_t mapCount = geometry.GetMapCount(geometry.InputShape().GetRank() - 1);
        if (mapCount != geometry.MapCount().GetNumElements())
            InvalidArgument("cuDNN does not support map tensor of this configuration.");

        const size_t minDimSize = (size_t)4;    // minimum descriptor dim size is 4 for cuDNN
        const size_t filt_size = filt.GetRank();
        size_t dim_size = std::max(filt_size + 1, minDimSize);
        SmallVector<int> dims(dim_size, 1);
        for (int i = 0; i < filt_size -1; i++)
            dims[dim_size - 1 - i] = (int)filt[i];
        // Set map count(aka K) dimension.
        dims[0] = (int)mapCount;
        dims[1] = (int)filt[filt_size - 1];
        int numElems = 1;
        for(int i=0; i<(int)dim_size;i++) numElems *= dims[i];
        m_isOdd = (numElems%2==1);
        HIPDNN_CALL(hipdnnSetFilterNdDescriptor(m_kernel, dataType, FILTER_FORMAT, (int)dim_size, dims.data()));
    }

    ~CuDnnKernel()
    {
        if (m_kernel != nullptr)
        {
            hipdnnDestroyFilterDescriptor(m_kernel);
            m_kernel = nullptr;
        }
    }

    operator hipdnnFilterDescriptor_t() const
    {
        return m_kernel;
    }

    bool isOdd()
    {
        return m_isOdd;
    }

    DISABLE_COPY_AND_MOVE(CuDnnKernel);

private:
    hipdnnFilterDescriptor_t m_kernel;
    bool m_isOdd;
};

class CuDnnConv
{
public:
    CuDnnConv(const ConvolveGeometry& geometry, hipdnnDataType_t dataType)
        : m_conv(nullptr)
    {
        HIPDNN_CALL(hipdnnCreateConvolutionDescriptor(&m_conv));
        // Set cuDNN convolution parameters. cuDNN uses row-major format while TensorShape - column-major
        // so conversion is required. Also, for 2D convolutions (which have 3D tensor shapes)
        // cuDNN uses 2D descriptors while for 3D convolutions - 3D so we need to ignore
        // rightmost dimension in ConvolveGeometry tensors.
        const size_t minDimSize = (size_t)2;    // minimum stride and pad size 2 for cuDNN
        size_t stride_size = geometry.InputShape().GetRank() - 1;
        size_t dim_size = std::max(stride_size, minDimSize);
        SmallVector<int> stride(dim_size, 1);
        SmallVector<int> pad(dim_size, 0);
        SmallVector<int> dilation(dim_size, 1);
        for (int i = 0; i < stride_size; i++)
        {
            stride[dim_size - 1 - i] = (int)geometry.GetStride(i);
            pad[dim_size - 1 - i] = geometry.GetLowerPad(i);
            dilation[dim_size - 1 - i] = (int)geometry.GetDilation(i);
        }
        HIPDNN_CALL(hipdnnSetConvolutionNdDescriptor(m_conv, (int)dim_size, pad.data(),
                                                   stride.data(), dilation.data(),
                                                   HIPDNN_CROSS_CORRELATION, dataType == HIPDNN_DATA_HALF ? HIPDNN_DATA_FLOAT : dataType));
#if !defined(__HIP_PLATFORM_HCC__)
        // allow tensor core for fp16 by default
        if(dataType == HIPDNN_DATA_HALF)
            HIPDNN_CALL(hipdnnSetConvolutionMathType(m_conv, HIPDNN_TENSOR_OP_MATH));
#endif
    }

    ~CuDnnConv()
    {
        if (m_conv != nullptr)
        {
            hipdnnDestroyConvolutionDescriptor(m_conv);
            m_conv = nullptr;
        }
    }

    operator hipdnnConvolutionDescriptor_t() const
    {
        return m_conv;
    }

    DISABLE_COPY_AND_MOVE(CuDnnConv);

private:
    hipdnnConvolutionDescriptor_t m_conv;
};

class CuDnnPool
{
public:
    CuDnnPool(const ConvolveGeometry& geometry, PoolKind kind, bool forceDeterministicAlgorithms, bool poolIncludePad)
        : m_pool(nullptr)
    {
        assert(bool(kind == PoolKind::Max || kind == PoolKind::Average));

        HIPDNN_CALL(hipdnnCreatePoolingDescriptor(&m_pool));
        // Set cuDNN pooling parameters. cuDNN uses row-major format while TensorShape - column-major
        // so conversion is required. Same as in convolution descriptor, cuDNN uses 2D descriptors
        // for 3D inputs.
        const size_t minDimSize = (size_t)2;    // minimum stride and pad size 2 for cuDNN
        size_t stride_size = geometry.InputShape().GetRank() - 1;
        size_t dim_size = std::max(stride_size, minDimSize);
        SmallVector<int> dims(dim_size, 1);
        SmallVector<int> stride(dim_size, 1);
        SmallVector<int> pad(dim_size, 0);
        auto kernelShape = geometry.KernelShape();
        for (int i = 0; i < stride_size; i++)
        {
            dims[dim_size - 1 - i] = (int)kernelShape[i];
            stride[dim_size - 1 - i] = (int)geometry.GetStride(i);
            pad[dim_size - 1 - i] = geometry.GetLowerPad(i);
        }
        hipdnnPoolingMode_t poolMode = HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        if (poolIncludePad)
            poolMode = HIPDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

        if (kind == PoolKind::Max)
        {
            if (forceDeterministicAlgorithms && (hipdnnGetVersion() >= 6000))
                poolMode = HIPDNN_POOLING_MAX_DETERMINISTIC;
            else
                poolMode = HIPDNN_POOLING_MAX;
        }

        // Must use HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING to get the same results as in reference engine.
        HIPDNN_CALL(hipdnnSetPoolingNdDescriptor(m_pool,
                                                 poolMode,
                                                 HIPDNN_PROPAGATE_NAN,       
                                                 (int)dim_size, dims.data(), pad.data(), stride.data()));
    }

    ~CuDnnPool()
    {
        if (m_pool != nullptr)
        {
            hipdnnDestroyPoolingDescriptor(m_pool);
            m_pool = nullptr;
        }
    }

    operator hipdnnPoolingDescriptor_t() const
    {
        return m_pool;
    }

    DISABLE_COPY_AND_MOVE(CuDnnPool);

private:
    hipdnnPoolingDescriptor_t m_pool;
};

enum class AutotuningState : int
{
    Init = 0,          // initial state
    PendingTuning = 1, // memory of all nodes have been allocated, it's safe to do tuning now
    Running = 2        // done tuning, no long performing auto-tuning, code is running normally
};

template <class ElemType>
class CuDnnConvolutionEngine : public ConvolutionEngine<ElemType>
{
public:
    using Base = ConvolutionEngine<ElemType>;
    using typename Base::Mat;

public:
    CuDnnConvolutionEngine(ConvolveGeometryPtr geometry, DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout,
                           size_t maxTempMemSizeInSamples, PoolKind poolKind, bool forceDeterministicAlgorithms,
                           bool poolIncludePad, bool inputHasFreeDimension)
        : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad),
          m_isConvGeometryComputed(geometry->ComputeConvGeometryExplicit()),
          m_mpRowCol(geometry->MpRowCol().size(), 1, (float*)const_cast<int*>(geometry->MpRowCol().data()),deviceId,  IsGpu(deviceId) ? matrixFlagNormal : matrixFlagDontOwnBuffer), 
          m_mpRowIwht(geometry->MpRowIwht().size(), 1, (float*)const_cast<int*>(geometry->MpRowIwht().data()),deviceId,  IsGpu(deviceId) ? matrixFlagNormal : matrixFlagDontOwnBuffer), 
          m_mpRowRun(geometry->MpRowRun().size(), 1, (float*)const_cast<int*>(geometry->MpRowRun().data()),deviceId,  IsGpu(deviceId) ? matrixFlagNormal : matrixFlagDontOwnBuffer), 
          m_runs(geometry->Runs().size(), 1, (float*)const_cast<int*>(geometry->Runs().data()),deviceId,  IsGpu(deviceId) ? matrixFlagNormal : matrixFlagDontOwnBuffer), 
          m_cudnn(CuDnn::Instance()),
          m_dataType(CuDnnTensor::GetDataType<ElemType>()),
          m_forceDeterministicAlgorithms(forceDeterministicAlgorithms),
          m_inputHasFreeDimension(inputHasFreeDimension)
    {
        auto inShape = geometry->InputShape();
        auto outShape = geometry->OutputShape();

        const size_t minDimSize = (size_t)3;    // minimum input and output size are 3 for cuDNN
        size_t input_size = inShape.GetRank();
        size_t dim_size = std::max(input_size, minDimSize);
        SmallVector<size_t> inputDims(dim_size, 1);
        SmallVector<size_t> outputDims(dim_size, 1);
        for (int i = 0; i < input_size - 1; i++)
        {
            inputDims[dim_size - 1 - i] = inShape[input_size - 1 - i];
            outputDims[dim_size - 1 - i] = outShape[input_size - 1 - i];
        }
        inputDims[0] = inShape[0];
        outputDims[0] = outShape[0];
        m_inT.Set(TensorShape(inputDims), m_dataType);
        m_outT.Set(TensorShape(outputDims), m_dataType);
    }

    virtual bool ImplementsGradientOverwriteOptimization() const override { return true; }

protected:
    using Base::m_geometry;
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_maxTempMemSizeInSamples;
    using Base::m_poolKind;
    using Base::m_poolIncludePad;

    void EnsureCompatible() override
    {
        if (m_imageLayout != ImageLayoutKind::CHW)
            RuntimeError("cuDNN convolution engine supports only CHW/cudnn layout.");
        if (!IsGpu(m_deviceId))
            RuntimeError("cuDNN convolution engine supports GPU devices only.");
    }

    void EnsureConvolutionInitialized() override
    {
        if (m_kernelT == nullptr)
        {
            m_kernelT = std::make_unique<CuDnnKernel>(*m_geometry, m_dataType);
            m_conv = std::make_unique<CuDnnConv>(*m_geometry, m_dataType);
        }
    }

    void ForwardCore(const Mat& in, const Mat& kernel, Mat& out, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        
#ifdef _HIPDBG_
        std::cout<<"CNTK: ENTER ForwardCore " << std::endl;
#endif
        
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [&,this](int& calgo, hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount]) -> hipdnnStatus_t
        {
#ifdef _HIPDBG_
            std::cout<<"CNTK: ENTER finder"<<std::endl;
#endif
            return hipdnnFindConvolutionForwardAlgorithmEx(*m_cudnn, m_inT, ptr(in), *m_kernelT, ptr(kernel), *m_conv, m_outT, ptr(out), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
        };
        // Find max Memory needed while running static finder. Workaround for hipdnnFind fail. Number of algo is constant as in hipdnn 5.1
        auto staticFinder = [&,this](hipdnnConvolutionFwdAlgo_t& algo, bool noMem) -> hipdnnStatus_t
        {

            std::cout<<"CNTK: ENTER staticFinder ---- Why?"<<std::endl;

#ifdef __HIP_PLATFORM_NVCC__
            if(!noMem)
                return hipdnnGetConvolutionForwardAlgorithm(*m_cudnn, m_inT, *m_kernelT, *m_conv, m_outT, HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace.BufferSize(), &algo);
            return hipdnnGetConvolutionForwardAlgorithm(*m_cudnn, m_inT, *m_kernelT, *m_conv, m_outT, HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &algo);
#elif defined (__HIP_PLATFORM_HCC__)
            int calgo;
            hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount];
            hipdnnStatus_t retVal = finder(calgo, algoPerf);
            algo = algoPerf[0].algo;
            return retVal;
#endif

        };
        // find deterministic algorithm 
        auto deterministicFinder = [&, this](int& calgo, hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount]) -> hipdnnStatus_t
        {
#ifdef _HIPDBG_
            std::cout<<"CNTK: ENTER deterministicFinder"<<std::endl;
#endif
            
            auto result = finder(calgo, algoPerf);
#ifdef _HIPDBG_
            std::cout<<"CNTK: After Finder - selected Algo : " << (*algoPerf).algo <<std::endl;
#endif
#if defined (__HIP_PLATFORM_HCC__)
            auto found = std::find_if(algoPerf, algoPerf + calgo,
                [](const hipdnnConvolutionFwdAlgoPerf_t& a) { return a.algo == HIPDNN_CONVOLUTION_FWD_ALGO_GEMM && a.status == HIPDNN_STATUS_SUCCESS; });
#else
            auto found = std::find_if(algoPerf, algoPerf + calgo,
                [](const hipdnnConvolutionFwdAlgoPerf_t& a) { return a.algo == HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM && a.status == HIPDNN_STATUS_SUCCESS; });
#endif

            if (found == algoPerf + calgo )
                RuntimeError("cuDNN could not find a deterministic algorithm. Set 'forceDeterministicAlgorithms=false' in your configuration.");

            algoPerf[0] = *found;   // copy the deterministic algorithm to first entry
#ifdef _HIPDBG_
            std::cout<<"After Found - selected Algo : " << (*algoPerf).algo <<std::endl;
#endif
            calgo = 1;              // set count of algorithms
#ifdef _HIPDBG_
            std::cout<<"CNTK: EXIT deterministicFinder"<<std::endl;
#endif
            return result;
        };
        // find workspace size needed to auto-tune all algorithms, as well as the size needed for deterministic algorithm 
        auto workspaceSizeFinder = [&, this]() -> hipdnnStatus_t
        {
            size_t tmpSize;
            hipdnnStatus_t err = HIPDNN_STATUS_EXECUTION_FAILED;
            
            std::cout<<"CNTK: ENTER workspaceSizeFinder "<< MaxAlgoCount << std::endl;
            
            for (int i = 0; i < MaxAlgoCount; i++)
            {
#ifdef _HIPDBG_
                std::cout<<"CNTK: Invoking workspaceSizeFinder"<<std::endl;
#endif
                auto err0 = hipdnnGetConvolutionForwardWorkspaceSize(*m_cudnn, m_inT, *m_kernelT, *m_conv, m_outT, (hipdnnConvolutionFwdAlgo_t)i, &tmpSize);
                if (err0 == HIPDNN_STATUS_SUCCESS)
                {
                    if (m_fwdAlgo.MaxAlgoWorkspaceSize < tmpSize)
                        m_fwdAlgo.MaxAlgoWorkspaceSize = tmpSize;
                    if ((hipdnnConvolutionFwdAlgo_t)i == HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
                        m_fwdAlgo.DeterministicAlgoWorkspaceSize = tmpSize;
                    err = err0;
                }
            }
#ifdef _HIPDBG_
            std::cout<<"CNTK: EXIT workspaceSizeFinder " << std::endl;
#endif
            return err;
        };
        FindBestAlgo(batchSize, m_fwdAlgo, workspaceSizeFinder, deterministicFinder, finder, staticFinder, workspace);
        if(m_dataType == HIPDNN_DATA_HALF) HIPDNN_CALL(hipdnnSetConvolutionMathType(*m_conv, m_fwdAlgo.AlgoMathType));
#if !defined(__HIP_PLATFORM_HCC__)
        else HIPDNN_CALL(hipdnnSetConvolutionMathType(*m_conv, HIPDNN_DEFAULT_MATH));
#endif
        // Perform forward convolution operation.
        std::cout<<"CNTK: Invoking hipdnnConvolutionForward"<<std::endl;
        std::cout<<"CNTK: SelectedAlgo"<<m_fwdAlgo.selectedAlgo<<std::endl;
        std::cout<<"CNTK: workspace Buffer Size"<<workspace.BufferSize()<<std::endl;
        HIPDNN_CALL(hipdnnConvolutionForward(*m_cudnn, &C::One, m_inT, ptr(in), *m_kernelT, ptr(kernel), *m_conv, m_fwdAlgo.selectedAlgo, ptr(workspace), workspace.BufferSize(), &C::Zero, m_outT, ptr(out)));
        std::cout<<"CNTK: EXIT ForwardCore " << std::endl;
    }

    void BackwardDataCore(const Mat& srcGrad, const Mat& kernel, Mat& grad, bool accumulateGradient, Mat& workspace) override
    {
        size_t batchSize = srcGrad.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [&,this](int& calgo, hipdnnConvolutionBwdDataAlgoPerf_t algoPerf[MaxAlgoCount]) -> hipdnnStatus_t
        {
            hipdnnStatus_t result;
            if (accumulateGradient)
            {
                // hipdnnFindConvolutionBackwardDataAlgorithmEx will overwrite the output buffer, thus we create a temporary buffer here
                // note this memory allocation might fail, so use try...catch for safety 
                auto gradReplace = Matrix<ElemType>((grad.BufferSize() + sizeof(ElemType) - 1)/sizeof(ElemType), 1, m_deviceId);
                result = hipdnnFindConvolutionBackwardDataAlgorithmEx(*m_cudnn, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_inT, ptr(gradReplace), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
                gradReplace.ReleaseMemory();
            }
            else
                result = hipdnnFindConvolutionBackwardDataAlgorithmEx(*m_cudnn, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_inT, ptr(grad), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
            return result;
        };
        // Find max Memory needed while running static finder. Workaround for hipdnnFind fail. Number of algo is constant as in hipdnn 5.1
        auto staticFinder = [&,this](hipdnnConvolutionBwdDataAlgo_t& algo, bool noMem) -> hipdnnStatus_t
        {
#ifdef __HIP_PLATFORM_NVCC__
            if(!noMem)
                return hipdnnGetConvolutionBackwardDataAlgorithm(*m_cudnn, *m_kernelT, m_outT, *m_conv, m_inT, HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, workspace.BufferSize(), &algo);
            return hipdnnGetConvolutionBackwardDataAlgorithm(*m_cudnn, *m_kernelT, m_outT, *m_conv, m_inT, HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, &algo);
#elif defined (__HIP_PLATFORM_HCC__)
            int calgo;
            hipdnnConvolutionBwdDataAlgoPerf_t algoPerf[MaxAlgoCount];
            hipdnnStatus_t retVal = finder(calgo, algoPerf);
            algo = algoPerf[0].algo;
            return retVal;
#endif

        };
        // find deterministic algorithm 
        auto deterministicFinder = [&, this](int& calgo, hipdnnConvolutionBwdDataAlgoPerf_t algoPerf[MaxAlgoCount]) -> hipdnnStatus_t
        {
            auto result = finder(calgo, algoPerf);
#ifdef __HIP_PLATFORM_HCC__
            auto found = std::find_if(algoPerf, algoPerf + calgo,
                [](const hipdnnConvolutionBwdDataAlgoPerf_t& a) { return a.algo == HIPDNN_CONVOLUTION_BWD_DATA_ALGO_0 && a.status == HIPDNN_STATUS_SUCCESS; });
#elif defined __HIP_PLATFORM_NVCC__
            auto found = std::find_if(algoPerf, algoPerf + calgo,
                [](const hipdnnConvolutionBwdDataAlgoPerf_t& a) { return a.algo == HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1 && a.status == HIPDNN_STATUS_SUCCESS; });
#endif
            if (found == algoPerf + calgo)
                RuntimeError("cuDNN could not find a deterministic algorithm. Set 'forceDeterministicAlgorithms=false' in your configuration.");

            algoPerf[0] = *found;   // copy the deterministic algorithm to first entry
            calgo = 1;              // set count of algorithms
            return result;
        };
        // finde workspace size needed to auto-tune all algorithms, as well as the size needed for deterministic algorithm 
        auto workspaceSizeFinder = [&, this]() -> hipdnnStatus_t
        {
            size_t tmpSize;
            hipdnnStatus_t err = HIPDNN_STATUS_EXECUTION_FAILED;
            for (int i = 0; i < MaxAlgoCount; i++)
            {
                auto err0 = hipdnnGetConvolutionBackwardDataWorkspaceSize(*m_cudnn, *m_kernelT, m_outT, *m_conv, m_inT, (hipdnnConvolutionBwdDataAlgo_t)i, &tmpSize);
                if (err0 == HIPDNN_STATUS_SUCCESS)
                {
                    if (m_backDataAlgo.MaxAlgoWorkspaceSize < tmpSize)
                        m_backDataAlgo.MaxAlgoWorkspaceSize = tmpSize;
                    if ((hipdnnConvolutionBwdDataAlgo_t)i == HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1)
                        m_backDataAlgo.DeterministicAlgoWorkspaceSize = tmpSize;
                    err = err0;
                }
            }
            return err;
        };
        FindBestAlgo(batchSize, m_backDataAlgo, workspaceSizeFinder, deterministicFinder, finder, staticFinder, workspace);
        // Compute gradients with respect to the output tensor (data).
        if(m_dataType == HIPDNN_DATA_HALF) HIPDNN_CALL(hipdnnSetConvolutionMathType(*m_conv, m_backDataAlgo.AlgoMathType));
#if !defined(__HIP_PLATFORM_HCC__)
        else HIPDNN_CALL(hipdnnSetConvolutionMathType(*m_conv, HIPDNN_DEFAULT_MATH));
#endif

        std::cout<<"CNTK: SelectedAlgo"<<m_backDataAlgo.selectedAlgo<<std::endl;
        std::cout<<"accumulateGradient"<<accumulateGradient<<std::endl;
        HIPDNN_CALL(hipdnnConvolutionBackwardData(*m_cudnn, &C::One, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_backDataAlgo.selectedAlgo, ptr(workspace), workspace.BufferSize(), accumulateGradient ? &C::One : &C::Zero, m_inT, ptr(grad)));
    }

    void BackwardKernelCore(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool accumulateGradient, bool /*allowReuse*/, Mat& workspace) override
    {
#if 1
        size_t batchSize = in.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [&,this](int& calgo, hipdnnConvolutionBwdFilterAlgoPerf_t algoPerf[MaxAlgoCount]) -> hipdnnStatus_t
        {
            hipdnnStatus_t result;
            if (accumulateGradient)
            {
                // hipdnnFindConvolutionBackwardFilterAlgorithmEx will overwrite the output buffer, thus we create a temporary buffer here
                // note this memory allocation might fail, so use try...catch for safety 
                auto kernelGradReplace = Matrix<ElemType>((kernelGrad.BufferSize() + sizeof(ElemType) - 1)/sizeof(ElemType), 1, m_deviceId);
                result = hipdnnFindConvolutionBackwardFilterAlgorithmEx(*m_cudnn, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, *m_kernelT, ptr(kernelGradReplace), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
                kernelGradReplace.ReleaseMemory();
            }
            else
                result = hipdnnFindConvolutionBackwardFilterAlgorithmEx(*m_cudnn, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, *m_kernelT, ptr(kernelGrad), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
            return result;
        };
        // Find max Memory needed while running static finder. Workaround for hipdnnFind fail. Number of algo is constant as in hipdnn 5.1
        auto staticFinder = [&,this](hipdnnConvolutionBwdFilterAlgo_t& algo, bool noMem) -> hipdnnStatus_t
        {
#ifdef __HIP_PLATFORM_NVCC__
            if(!noMem)
                return hipdnnGetConvolutionBackwardFilterAlgorithm(*m_cudnn, m_inT, m_outT, *m_conv, *m_kernelT, HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, workspace.BufferSize(), &algo);
            // special case for half/odd filter
            if(m_kernelT->isOdd() && m_dataType == HIPDNN_DATA_HALF)
            {
                size_t tmpSize = 0;
                algo = (hipdnnConvolutionBwdFilterAlgo_t) 1;
                auto err = hipdnnGetConvolutionBackwardFilterWorkspaceSize(*m_cudnn, m_inT, m_outT, *m_conv, *m_kernelT, algo, &tmpSize);
                workspace.Resize((tmpSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
                return err;
            }
            return hipdnnGetConvolutionBackwardFilterAlgorithm(*m_cudnn, m_inT, m_outT, *m_conv, *m_kernelT, HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &algo);
#elif defined (__HIP_PLATFORM_HCC__)
            int calgo;
            hipdnnConvolutionBwdFilterAlgoPerf_t algoPerf[MaxAlgoCount];
            hipdnnStatus_t retVal = finder(calgo, algoPerf);
            algo = algoPerf[0].algo;
            return retVal;
#endif
        };
        // find deterministic algorithm 
        auto deterministicFinder = [&, this](int& calgo, hipdnnConvolutionBwdFilterAlgoPerf_t algoPerf[MaxAlgoCount])->hipdnnStatus_t
        {
            auto result = finder(calgo, algoPerf); 
            auto found = std::find_if(algoPerf, algoPerf + calgo,
                [](const hipdnnConvolutionBwdFilterAlgoPerf_t& a) { return a.algo == HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1 && a.status == HIPDNN_STATUS_SUCCESS; });
            if (found == algoPerf + calgo)
                RuntimeError("cuDNN could not find a deterministic algorithm. Set 'forceDeterministicAlgorithms=false' in your configuration.");

            algoPerf[0] = *found;   // copy the deterministic algorithm to first entry
            calgo = 1;              // set count of algorithms
            return result;
        };
        // finde workspace size needed to auto-tune all algorithms, as well as the size needed for deterministic algorithm 
        auto workspaceSizeFinder = [&, this]() -> hipdnnStatus_t
        {
            size_t tmpSize;
            hipdnnStatus_t err = HIPDNN_STATUS_EXECUTION_FAILED;
            for (int i = 0; i < MaxAlgoCount; i++)
            {
                auto err0 = hipdnnGetConvolutionBackwardFilterWorkspaceSize(*m_cudnn, m_inT, m_outT, *m_conv, *m_kernelT, (hipdnnConvolutionBwdFilterAlgo_t)i, &tmpSize);
                if (err0 == HIPDNN_STATUS_SUCCESS)
                {
                    if (m_backFiltAlgo.MaxAlgoWorkspaceSize < tmpSize)
                        m_backFiltAlgo.MaxAlgoWorkspaceSize = tmpSize;
                    if ((hipdnnConvolutionBwdFilterAlgo_t)i == HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
                        m_backFiltAlgo.DeterministicAlgoWorkspaceSize = tmpSize;
                    err = err0;
                }
            }
            return err;
        };
        FindBestAlgo(batchSize, m_backFiltAlgo, workspaceSizeFinder, deterministicFinder, finder, staticFinder, workspace);
        // Compute gradients with respect to the output tensor (data).
        if(m_dataType == HIPDNN_DATA_HALF) HIPDNN_CALL(hipdnnSetConvolutionMathType(*m_conv, m_backFiltAlgo.AlgoMathType));
#if !defined(__HIP_PLATFORM_HCC__)
        else HIPDNN_CALL(hipdnnSetConvolutionMathType(*m_conv, HIPDNN_DEFAULT_MATH));
#endif
        //m_backFiltAlgo.selectedAlgo =  HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        std::cout<<"Invoking hipdnnconvolution Backward filter"<<std::endl;
        HIPDNN_CALL(hipdnnConvolutionBackwardFilter(*m_cudnn, &C::One, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, m_backFiltAlgo.selectedAlgo, ptr(workspace), workspace.BufferSize(), accumulateGradient ? &C::One : &C::Zero, *m_kernelT, ptr(kernelGrad)));
#else
        // Use Native reference BackwardKernel
        const int BlockSize = 256;
        int* d_mpRowColData, *d_mpRowIwhtData, *d_mpRowRunData, *d_mRunsData;
        hipMalloc((void**)&d_mpRowColData, sizeof(int) * m_mpRowCol.GetAllocatedSize());
        hipMalloc((void**)&d_mpRowIwhtData, sizeof(int) * m_mpRowIwht.GetAllocatedSize());
        hipMalloc((void**)&d_mpRowRunData, sizeof(int) * m_mpRowRun.GetAllocatedSize());
        hipMalloc((void**)&d_mRunsData, sizeof(int) * m_runs.GetAllocatedSize());

        hipMemcpy(d_mpRowColData, m_mpRowCol.Data(), sizeof(int) * m_mpRowCol.GetAllocatedSize(), hipMemcpyHostToDevice);
        hipMemcpy(d_mpRowIwhtData, m_mpRowIwht.Data(), sizeof(int) *  m_mpRowIwht.GetAllocatedSize(), hipMemcpyHostToDevice);
        hipMemcpy(d_mpRowRunData, m_mpRowRun.Data(), sizeof(int) * m_mpRowRun.GetAllocatedSize(), hipMemcpyHostToDevice);
        hipMemcpy(d_mRunsData, m_runs.Data(), sizeof(int) * m_runs.GetAllocatedSize(), hipMemcpyHostToDevice);
        
        auto gdim = dim3((srcGrad.GetNumRows() + BlockSize - 1)/ BlockSize, std::min((int)srcGrad.GetNumCols(), 65535));
        SyncGuard syncGuard;
        hipLaunchKernelGGL((kConvolutionBackwardKernelAcc<ElemType>), dim3(gdim), dim3(BlockSize), 0, 0, (int)srcGrad.GetNumCols(), (int)in.GetNumRows(), (int)srcGrad.GetNumRows(),
                                                                   ptr(in), d_mpRowColData, d_mpRowIwhtData, d_mpRowRunData, 
                                                                   d_mRunsData, ptr(srcGrad), ptr(kernelGrad), accumulateGradient);

        hipFree(d_mpRowColData);
        hipFree(d_mpRowIwhtData);
        hipFree(d_mpRowRunData);
        hipFree(d_mRunsData);
#endif
    }

    void EnsurePoolingInitialized() override
    {
        if (m_pool == nullptr)
            m_pool = std::make_unique<CuDnnPool>(*m_geometry, m_poolKind, m_forceDeterministicAlgorithms, m_poolIncludePad);
    }

    void ForwardPoolingCore(const Mat& in, Mat& out) override
    {
        size_t batchSize = in.GetNumCols();
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);
        HIPDNN_CALL(hipdnnPoolingForward(*m_cudnn, *(m_pool), &C::One, m_inT, ptr(in), &C::Zero, m_outT, ptr(out)));
    }

    void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad, bool accumulateGradient) override
    {
        size_t batchSize = in.GetNumCols();
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);
        HIPDNN_CALL(hipdnnPoolingBackward(*m_cudnn, *(m_pool), &C::One, m_outT, ptr(out), m_outT, ptr(srcGrad),
                                          m_inT, ptr(in), accumulateGradient ? &C::One : &C::Zero, m_inT, ptr(grad)));
    }

    void MaxUnpoolingCore(const Mat& out, const Mat& poolIn, Mat& in) override
    {
        UNUSED(out);
        UNUSED(poolIn);
        UNUSED(in);
        // Not implemented but potentially can make a fallback to reference engine.
        LogicError("MaxUnpooling is not implemented for cuDNN engine.");
    }

private:
    using C = Consts<ElemType>;
#ifdef __HIP_PLATFORM_HCC__
    static const int MaxAlgoCount = 4;
#elif defined __HIP_PLATFORM_NVCC__
    static const int MaxAlgoCount = 10;
#endif

#ifdef __HIP_PLATFORM_NVCC__
    hipdnnStatus_t convertType(cudnnConvolutionFwdAlgo_t in, hipdnnConvolutionFwdAlgo_t* out)
    {
        return cudnnTohipConvolutionFwdAlgo(in, out);
    }
    hipdnnStatus_t convertType(cudnnConvolutionBwdDataAlgo_t in, hipdnnConvolutionBwdDataAlgo_t* out)
    {
        return cudnnTohipConvolutionBwdDataAlgo(in, out);
    }
    hipdnnStatus_t convertType(cudnnConvolutionBwdFilterAlgo_t in, hipdnnConvolutionBwdFilterAlgo_t* out)
    {
        return cudnnTohipConvolutionBwdFilterAlgo(in, out);
    }
    hipdnnStatus_t convertType(cudnnMathType_t in, hipdnnMathType_t *out)
    {
	    return cudnnTohipMathType(in, out);
    }
#endif


    template <typename TAlgo, typename TWorkspaceSizeFinder, typename TDeterministicFinder, typename TFinder, typename TStaticFinder>
    void FindBestAlgo(size_t batchSize, TAlgo& algo, TWorkspaceSizeFinder workspaceSizeFinder, TDeterministicFinder deterministicFinder, TFinder finder, TStaticFinder staticFinder, Mat& workspace)
    {
#ifdef _HIPDBG_
        std::cout  << "CNTK: ENTER FindBestAlgo" << std::endl;
#endif
        
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);

        // keep running if nothing changes
        if (!algo.NeedAutotuning(batchSize, workspace.BufferSize()))
            return;

#ifdef _HIPDBG_
        std::cout  << "CNTK: FindBestAlgo: 1, forceDeterministicAlgorithms =" << m_forceDeterministicAlgorithms << std::endl;
#endif
        // if batchsize changes again when just finish init, go back to init again
        if (algo.autotuningState == AutotuningState::PendingTuning && batchSize > algo.LastBatchAlgoMBSize)
            algo.autotuningState = AutotuningState::Init;

        // batchSize is bigger than the one when initialize current workspace, need free up space and go back to init
        if (algo.autotuningState == AutotuningState::Running && batchSize > algo.maxMBSizeSeen)
        {
#ifdef _HIPDBG_
            std::cout  << "CNTK: FindBestAlgo: 2" << std::endl;
#endif
            hipDeviceSynchronize(); // make sure no in-flight GPU kernels using workspace before release its memory
            workspace.Resize(0,0,0,false);
            algo.RecordAlgoBatchSizeWorkspaceSize(true, algo.selectedAlgo, 0, 0);
            algo.autotuningState = AutotuningState::Init;
        }
        else if (algo.autotuningState == AutotuningState::Running && !m_forceDeterministicAlgorithms && !m_inputHasFreeDimension)  // batchSize changes to be smaller than MaxAlgoMBSize, need to re-do tuning if non-deterministic
            algo.autotuningState = AutotuningState::PendingTuning;

#ifdef _HIPDBG_
        std::cout  << "CNTK: FindBestAlgo: 3" << std::endl;
#endif
        
        typename TAlgo::typeT algoPerf[MaxAlgoCount];
        int calgo = 0;
        // In initState, where memory allocation for nodes are not completed, we only run the algorithm with no workspace.
        // In the special case when m_forceDeterministicAlgorithms, we allocate some memory and use the deterministic algorithm.
        // In the special case when m_inputHasFreeDimension, we only run the algorithm with no workspace.
        if (algo.autotuningState == AutotuningState::Init)
        {

#ifdef _HIPDBG_
            std::cout  << "CNTK: FindBestAlgo: 4" << std::endl;
#endif
            
            // find workspace size needed for finderEx and deterministic algorithm
            HIPDNN_CALL(workspaceSizeFinder());
            
#ifdef _HIPDBG_
            std::cout  << "CNTK: FindBestAlgo: 5: " << m_forceDeterministicAlgorithms << std::endl;
            std::cout  << "CNTK: FindBestAlgo: 5_1 MaxAlgoMBSize: " << algo.MaxAlgoMBSize << std::endl;
            //std::cout  << "CNTK: FindBestAlgo: 5_2 supportsStaticFinder: " << supportsStaticFinder << std::endl;
#endif

            if (m_forceDeterministicAlgorithms)
            {
                workspace.Resize((algo.DeterministicAlgoWorkspaceSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);

#ifdef _HIPDBG_
                std::cout  << "CNTK: FindBestAlgo: 6  WS size = " << workspace.BufferSize() << std::endl;
#endif
                
                HIPDNN_CALL(deterministicFinder(calgo, algoPerf));
                assert(calgo == 1);                                 // only one deterministic algorithm will be returned
                algo.RecordAlgoBatchSizeWorkspaceSize(true, (*algoPerf).algo, batchSize, (*algoPerf).memory);
                algo.autotuningState = AutotuningState::Running;    // no further need for tuning since this is deterministic, directly enter running state
            }
            else
            {
#ifdef _HIPDBG_
                std::cout  << "CNTK: FindBestAlgo: 7" << std::endl;
#endif
                // This branch handles two cases: a) When first MB comes through, and b) When input has free dimensions.
                // If the handling of these two cases changes, we may need to create separate branches for them.
                HIPDNN_CALL(staticFinder(algo.selectedAlgo, true));
                
#ifdef _HIPDBG_
                std::cout  << "CNTK: FindBestAlgo: 8" << std::endl;
#endif
                algo.maxMBSizeSeen = batchSize;
                // Here MaxAlgoWorkspaceSize is temporarily storing 'possible' need changed by staticFinder.
                // Thus we don't set maxAlgo records and those will be tuned later.
                algo.RecordAlgoBatchSizeWorkspaceSize(false, algo.selectedAlgo, batchSize, 0);
                algo.autotuningState = m_inputHasFreeDimension ? AutotuningState::Running : AutotuningState::PendingTuning;
            }
#ifdef _HIPDBG_
            std::cout  << "CNTK: FindBestAlgo: 9" << std::endl;
#endif
            
            return;
        }

        // we allocate workspace and find algorithm if batchSize is higher than ever seen
        if (algo.MaxAlgoMBSize == 0)    // MaxAlgoMBSize is 0 only after Init. After this heavy tuning, MaxAlgoMBSize will be set to >0, thus we tune just once.
        {
#ifdef _HIPDBG_
            std::cout  << "CNTK: FindBestAlgo: 10" << std::endl;
#endif
            size_t curSize = workspace.BufferSize();

            // To control memory usage. No one seems to be using this flag
            size_t inputSampleSize = m_geometry->InputShape().GetNumElements();
            size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : inputSampleSize * m_maxTempMemSizeInSamples * sizeof(ElemType);

            try
            {   // first try allocate as much to run FindEX, this may fail when accumulate is on (in which case additional memory is allocated in finder()), thus we do try...catch...
                size_t free, total, resizeTo = 0;
                CUDA_CALL(hipMemGetInfo(&free, &total));

#ifdef _HIPDBG_
                std::cout  << "CNTK: FindBestAlgo: 11" << std::endl;
#endif
                
                free += workspace.BufferSize();
                // We reserve 2% of the total GPU memory because CuDNN seem to behave erroneously when there is no memory left
                if(free > (total/50))
                    resizeTo = free - (total/50) + sizeof(ElemType);
                // We don't need memory more than workspace we learned in workspaceSizeFinder
                resizeTo = min(resizeTo, algo.MaxAlgoWorkspaceSize);
                resizeTo = min(resizeTo, maxMem);
                if(resizeTo > 0)
                    workspace.Resize((resizeTo + sizeof(ElemType) - 1) / sizeof(ElemType), 1);     // resize the workspace so that we can run the finder

#ifdef _HIPDBG_
                std::cout  << "CNTK: FindBestAlgo: 12" << std::endl;
#endif

                // Pending State now, let's do a find and get algorithm Perfs
                calgo = 0;
                HIPDNN_CALL(finder(calgo, algoPerf));

#ifdef _HIPDBG_
                std::cout  << "CNTK: FindBestAlgo: 13" << std::endl;
#endif
                assert(calgo > 0);
                
                auto res = algoPerf;        // first returned algorithm is the fastest
                algo.RecordAlgoBatchSizeWorkspaceSize(true, (*res).algo, batchSize, (*res).memory);
#ifdef __HIP_PLATFORM_NVCC__
                algo.AlgoMathType = (*res).mathType;
#elif defined __HIP_PLATFORM_HCC__
                algo.AlgoMathType = HIPDNN_DEFAULT_MATH; //TODO: PRAS_AMD
#endif
                algo.autotuningState = AutotuningState::Running;
                if (algo.MaxAlgoWorkspaceSize < curSize)   // need to shrink the workspace
                    workspace.Resize((curSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);
                else
                    workspace.Resize((algo.MaxAlgoWorkspaceSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);
            }
            catch (...)
            {   // when it fails, it means accumulate is on, and allocation of temporary buffer failed. We resize to curSize and try again
                fprintf(stderr, "Retrying with reduced workspace memory for convolution\n");
                workspace.Resize((curSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);
                try
                {
                    calgo = 0;
                    HIPDNN_CALL(finder(calgo, algoPerf));
                    assert(calgo > 0);
                    auto res = algoPerf;    // first returned algorithm is the fastest
                    algo.RecordAlgoBatchSizeWorkspaceSize(true, (*res).algo, batchSize, (*res).memory);
#ifdef __HIP_PLATFORM_NVCC__
                    algo.AlgoMathType = (*res).mathType;
#elif defined __HIP_PLATFORM_HCC__
                    algo.AlgoMathType = HIPDNN_DEFAULT_MATH; //TODO: PRAS_AMD
#endif
                    algo.autotuningState = AutotuningState::Running;
                }
                catch (...)
                {   // fails again, let's fall back to cudnnGet
                    fprintf(stderr, "Fall back to use static finder to get the algorithm for convolution\n");
                    HIPDNN_CALL(staticFinder(algo.selectedAlgo, false));
                    algo.RecordAlgoBatchSizeWorkspaceSize(true, algo.selectedAlgo, batchSize, curSize);
                    algo.autotuningState = AutotuningState::Running;
                }
            }
        }
        else if (batchSize == algo.MaxAlgoMBSize && workspace.BufferSize() >= algo.MaxAlgoWorkspaceSize) // Use stored algo when batchsize go back to max. Likely happen when last batch in epoch lacking data
        {
            algo.RecordAlgoBatchSizeWorkspaceSize(false, algo.maxAlgo, batchSize, algo.MaxAlgoWorkspaceSize);
            algo.autotuningState = AutotuningState::Running;
        }
        else    // use fast/static method to get algorithm when batchsize get smaller. Avoid severe slowdown when batchsize change frequently
        {
            HIPDNN_CALL(staticFinder(algo.selectedAlgo, false));
            algo.RecordAlgoBatchSizeWorkspaceSize(false, algo.selectedAlgo, batchSize, workspace.BufferSize());
            algo.autotuningState = AutotuningState::Running;
        }
        
        std::cout  << "CNTK: EXIT FindBestAlgo" << std::endl;
        return;
    }

    static ElemType* ptr(Mat& src)
    {
        return src.Data();
    }
    static const ElemType* ptr(const Mat& src)
    {
        return src.Data();
    }

private:
    template <typename T>
    struct ConvAlgoInfo
    {
        typedef T typeT;
        ConvAlgoInfo()
            : LastBatchAlgoMBSize(0), MaxAlgoMBSize(0), maxMBSizeSeen(0), autotuningState(AutotuningState::Init), MaxAlgoWorkspaceSize(0), LastBatchAlgoWorkspaceSize(0), AlgoMathType(HIPDNN_TENSOR_OP_MATH)
        {
        }
        // Variables to stores states
        size_t maxMBSizeSeen; // Max minibatch size seen. If batch size exceed this number, redo tuning from scratch. maxAlgo is tuned for batchsize following this batch.

        size_t MaxAlgoMBSize;   // Batch size when current work space is allocated. If batch size returns to this size, directly pick the maxAlgo
        size_t MaxAlgoWorkspaceSize;   // First temporarily store possible workspace size for any algorithm, then store size for  maxAlgo after tunning

        size_t LastBatchAlgoWorkspaceSize;  // workspace size for selectedAlgo
        size_t LastBatchAlgoMBSize;        // minibatch size for selectedAlgo

        size_t DeterministicAlgoWorkspaceSize;  // workspace size for deterministic algorithm

        AutotuningState autotuningState;    // state of auto-tuning: Init, PendingTuning and Running
        decltype(T::algo) selectedAlgo;     // currently selected algorithm
        decltype(T::algo) maxAlgo;          // algorithm that was selected when the current workspace is allocated

        hipdnnMathType_t AlgoMathType;

        bool NeedAutotuning(size_t batchSize, size_t workspaceSize)
        {
            // NVIDIA:
            // It is not safe to assume that previously selected algorithm requires less or the same amount of workspace when minibatch size decrease
            // Need to re-run auto-tuner everytime minibatch size grow.
            // Use faster(may not be optimal) method to get algorithm when batchsize decrease
            // Should remain reasonable performance when minibatch size changes frequently (e.g. distributed reading).
            return (autotuningState != AutotuningState::Running ||
                    batchSize != LastBatchAlgoMBSize ||
                    workspaceSize < LastBatchAlgoWorkspaceSize);
        }

        // Record algorithm, batchsize and workspace right after tuning/init. Next batch will check to decide whether keep using recorded algorithm.
        // If just tuned for MaxAlgo, also record that since maxAlgo tuning is heavy.
        template <typename U>
        void RecordAlgoBatchSizeWorkspaceSize(bool justTunedForMaxAlgo, U newAlgo, size_t batchSize, size_t workspaceSize)
        {
            selectedAlgo = newAlgo;
            LastBatchAlgoMBSize = batchSize;
            LastBatchAlgoWorkspaceSize = workspaceSize;

            if (justTunedForMaxAlgo)
            {
                maxAlgo = newAlgo;
                MaxAlgoMBSize = batchSize;
                MaxAlgoWorkspaceSize = workspaceSize;
            }
        }
    };

    // IMP NOTE: Make sure that in the declaration below m_isConvGeometryComputed is declared
    // before m_mpRowCol. This ordering is required to ensure the right order of initialization
    // in the initializer list in the ctor (above) of this class.
    bool m_isConvGeometryComputed;


    Matrix<float> m_mpRowCol;   
    // Convolution-specific maps.
    Matrix<float> m_mpRowIwht;
    Matrix<float> m_mpRowRun;
    Matrix<float> m_runs;
 
    CuDnn::ptr_t m_cudnn;
    hipdnnDataType_t m_dataType;
    CuDnnTensor m_inT;
    CuDnnTensor m_outT;
    // Convolution specific.
    std::unique_ptr<CuDnnKernel> m_kernelT;
    std::unique_ptr<CuDnnConv> m_conv;
    // Pooling specific.
    std::unique_ptr<CuDnnPool> m_pool;

    ConvAlgoInfo<hipdnnConvolutionFwdAlgoPerf_t> m_fwdAlgo;
    ConvAlgoInfo<hipdnnConvolutionBwdDataAlgoPerf_t> m_backDataAlgo;
    ConvAlgoInfo<hipdnnConvolutionBwdFilterAlgoPerf_t> m_backFiltAlgo;

    // Flag indicating whether only deterministic algorithms should be used.
    bool m_forceDeterministicAlgorithms;
    bool m_inputHasFreeDimension;
};

template <class ElemType>
std::unique_ptr<ConvolutionEngine<ElemType>> CuDnnConvolutionEngineFactory<ElemType>::Create(ConvolveGeometryPtr geometry,
                                                                                             DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout,
                                                                                             size_t maxTempMemSizeInSamples, PoolKind poolKind,
                                                                                             bool forceDeterministicAlgorithms, bool poolIncludePad,
                                                                                             bool inputHasFreeDimension)
{
    return std::make_unique<CuDnnConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind,
                                                              forceDeterministicAlgorithms, poolIncludePad, inputHasFreeDimension);
}

template <class ElemType>
bool CuDnnConvolutionEngineFactory<ElemType>::IsSupported(DEVICEID_TYPE deviceId, ConvolveGeometryPtr geometry, PoolKind poolKind)
{
    // REVIEW alexeyk: IsSupported check should be performed by cuDNN itself. Is there a good way to do that?

    hipDeviceProp_t props = {0};
    // Note that hipGetDeviceProperties also sets CUDA last error so need to check/clear both.
    if (deviceId < 0 || (hipGetDeviceProperties(&props, deviceId) | hipGetLastError()) != hipSuccess || props.major < 3)
        return false;

    const auto& input = geometry->InputShape();
    const auto& kernel = geometry->KernelShape();
    const auto& sharing = geometry->Sharing();
    const auto& mapCount = geometry->MapCount();

    const auto& inputRank = input.GetRank();
    const auto& kernelRank = kernel.GetRank();
    const auto& mapRank = mapCount.GetRank();
    // cuDNN supports 2D and 3D convolutions at the moment with full sharing.
    // In case map count size > 1, then it should have all ones except last dimension.
    // If pooling is requested, then cuDNN supports only 2D/3D inputs and 2D pooling kernels.
    bool retVal = (inputRank <= 4 &&
                   std::find(begin(sharing), end(sharing), false) == sharing.end() &&
                   mapCount.GetNumElements() == mapCount[mapRank - 1] &&
                   (poolKind == PoolKind::None ||
                   (inputRank <= 3 && (kernelRank < 3 || kernel[2] == 1))));

    // cuDNN as of version 6.0 does not handle asymmetric padding for even size kernel convolution correctly. We need to detect asymmetric
    // padding due to auto-padding and choose the reference convolution implementation instead
    // a special case is when stride >= input, this means we will have a single output, and thus asymmetric padding is not an issue
    if (poolKind == PoolKind::None)     // only for convolution, pooling seems fine
    {
        for (int i = 0; i < kernelRank; i++)
        {
            auto lowerPad = geometry->GetLowerPad(i);
            auto upperPad = geometry->GetUpperPad(i);
            auto stride = geometry->GetStride(i);
            if (kernel[i] % 2 == 0 && lowerPad < upperPad && stride < input[i])
            {
                fprintf(stderr, "WARNING: Detected asymmetric padding issue with even kernel size and lowerPad (%d) < higherPad (%d) (i=%d), cuDNN will not be able to produce correct result. Switch to reference engine (VERY SLOW). \n", lowerPad, upperPad, i);
                retVal = false;
                break;
            }
        }
    }
    return retVal;
}

template class CuDnnConvolutionEngineFactory<float>;
template class CuDnnConvolutionEngineFactory<double>;
#ifdef __HIP_ENABLE_HALF__
template class CuDnnConvolutionEngineFactory<half>;
#endif //__HIP_ENABLE_HALF__
} } }
