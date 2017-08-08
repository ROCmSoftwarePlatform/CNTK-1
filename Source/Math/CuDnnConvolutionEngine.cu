//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CuDnnFactories.h"
#include "GPUMatrix.h"
#include <typeinfo>
#include <typeindex>
#include "CuDnnCommon.h"
#include <typeinfo>
#include <cxxabi.h>

template <>
const char* CudaErrString<hipdnnStatus_t>(hipdnnStatus_t x)
{
    return (const char*)0;//TODO:__add__ hipdnnGetErrorString(x);
}

// A note on the formats: CNTK originally used NHWC for input/output tensors and CHWN for kernels.
// Such formats have very limited support in cuDNN and not used in other frameworks.
// CNTK with cuDNN by default uses NCHW formats for both inputs/outputs and kernels.
#define TENSOR_FORMAT HIPDNN_TENSOR_NCHW
#define FILTER_FORMAT HIPDNN_TENSOR_NCHW

namespace Microsoft { namespace MSR { namespace CNTK {

static bool IsGpu(DEVICEID_TYPE deviceId)
{
    return deviceId >= 0;
}

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
        //TODO: __add__ HIPDNN_CALL(hipdnnSetFilterNdDescriptor(m_kernel, dataType, FILTER_FORMAT, (int)dim_size, dims.data()));
    }

    ~CuDnnKernel()
    {
        if (m_kernel != nullptr)
        {
            //TODO: __add__ hipdnnDestroyFilterDescriptor(m_kernel);
            m_kernel = nullptr;
        }
    }

    operator hipdnnFilterDescriptor_t() const
    {
        return m_kernel;
    }

    DISABLE_COPY_AND_MOVE(CuDnnKernel);

private:
    hipdnnFilterDescriptor_t m_kernel;
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
        for (int i = 0; i < stride_size; i++)
        {
            stride[dim_size - 1 - i] = (int)geometry.GetStride(i);
            pad[dim_size - 1 - i] = geometry.GetLowerPad(i);
        }
        SmallVector<int> upscale(dim_size, 1);
        /*TODO: __add__ HIPDNN_CALL(hipdnnSetConvolutionNdDescriptor(m_conv, (int)dim_size, pad.data(),
                                                   stride.data(), upscale.data(),
                                                   HIPDNN_CROSS_CORRELATION, dataType));*/
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
        assert(kind == PoolKind::Max || kind == PoolKind::Average);

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
        // deterministic maxpool is not working when kernel size > stride size in cuDNN. We ignore this flag for now. 
        if (forceDeterministicAlgorithms) {}
        // Must use HIPDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING to get the same results as in reference engine.
        /*TODO: __add__ HIPDNN_CALL(hipdnnSetPoolingNdDescriptor(m_pool,
                                               kind == PoolKind::Max ? HIPDNN_POOLING_MAX : poolMode,
                                               HIPDNN_PROPAGATE_NAN,
                                               (int)dim_size, dims.data(), pad.data(), stride.data()));*/
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
                           size_t maxTempMemSizeInSamples, PoolKind poolKind, bool forceDeterministicAlgorithms, bool poolIncludePad)
        : Base(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, poolIncludePad),
          m_hipdnn(CuDnn::Instance()),
          m_dataType(CuDnnTensor::GetDataType<ElemType>()),
          m_forceDeterministicAlgorithms(forceDeterministicAlgorithms)
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
            RuntimeError("cuDNN convolution engine supports only CHW/hipdnn layout.");
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
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [&,this](int& calgo, hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount]) -> hipdnnStatus_t
        {
            return hipdnnFindConvolutionForwardAlgorithmEx(*m_hipdnn, m_inT, ptr(in), *m_kernelT, ptr(kernel), *m_conv, m_outT, ptr(out), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
        };
        // Find max Memory needed while running static finder. Workaround for hipdnnFind fail. Number of algo is constant as in hipdnn 5.1
        auto staticFinder = [&,this](hipdnnConvolutionFwdAlgo_t& algo, bool noMem) -> hipdnnStatus_t
        {
            if(!noMem)
                return hipdnnGetConvolutionForwardAlgorithm(*m_hipdnn, m_inT, *m_kernelT, *m_conv, m_outT, HIPDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, workspace.BufferSize(), &algo);
            return hipdnnGetConvolutionForwardAlgorithm(*m_hipdnn, m_inT, *m_kernelT, *m_conv, m_outT, HIPDNN_CONVOLUTION_FWD_NO_WORKSPACE, 0, &algo);
        };
        // find deterministic algorithm 
        auto deterministicFinder = [&, this](int& calgo, hipdnnConvolutionFwdAlgoPerf_t algoPerf[MaxAlgoCount]) -> hipdnnStatus_t
        {
            auto result = finder(calgo, algoPerf); 
            auto found = std::find_if(algoPerf, algoPerf + calgo,
                [](const hipdnnConvolutionFwdAlgoPerf_t& a) { return a.algo == HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM && a.status == HIPDNN_STATUS_SUCCESS; });
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
                auto err0 = hipdnnGetConvolutionForwardWorkspaceSize(*m_hipdnn, m_inT, *m_kernelT, *m_conv, m_outT, (hipdnnConvolutionFwdAlgo_t)i, &tmpSize);
                if (err0 == HIPDNN_STATUS_SUCCESS)
                {
                    if (m_fwdAlgo.AlgoWorkspaceSize < tmpSize)
                        m_fwdAlgo.AlgoWorkspaceSize = tmpSize;
                    if ((hipdnnConvolutionFwdAlgo_t)i == HIPDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
                        m_fwdAlgo.DeterministicAlgoWorkspaceSize = tmpSize;
                    err = err0; 
                }
            }
            return err; 
        }; 
        FindBestAlgo(batchSize, m_fwdAlgo, workspaceSizeFinder, deterministicFinder, finder, staticFinder, workspace);
        // Perform forward convolution operation.
        HIPDNN_CALL(hipdnnConvolutionForward(*m_hipdnn, &C::One, m_inT, ptr(in), *m_kernelT, ptr(kernel), *m_conv, m_fwdAlgo.selectedAlgo, ptr(workspace), workspace.BufferSize(), &C::Zero, m_outT, ptr(out)));
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
                result = hipdnnFindConvolutionBackwardDataAlgorithmEx(*m_hipdnn, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_inT, ptr(gradReplace), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
                gradReplace.ReleaseMemory();
            }
            else
                result = hipdnnFindConvolutionBackwardDataAlgorithmEx(*m_hipdnn, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_inT, ptr(grad), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
            return result;
        };
        // Find max Memory needed while running static finder. Workaround for hipdnnFind fail. Number of algo is constant as in hipdnn 5.1
        auto staticFinder = [&,this](hipdnnConvolutionBwdDataAlgo_t& algo, bool noMem) -> hipdnnStatus_t
        {
            if(!noMem)
                return hipdnnGetConvolutionBackwardDataAlgorithm(*m_hipdnn, *m_kernelT, m_outT, *m_conv, m_inT, HIPDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, workspace.BufferSize(), &algo);
            return hipdnnGetConvolutionBackwardDataAlgorithm(*m_hipdnn, *m_kernelT, m_outT, *m_conv, m_inT, HIPDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, &algo);
        };
        // find deterministic algorithm 
        auto deterministicFinder = [&, this](int& calgo, hipdnnConvolutionBwdDataAlgoPerf_t algoPerf[MaxAlgoCount]) -> hipdnnStatus_t
        {
            auto result = finder(calgo, algoPerf);
            auto found = std::find_if(algoPerf, algoPerf + calgo,
                [](const hipdnnConvolutionBwdDataAlgoPerf_t& a) { return a.algo == HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1 && a.status == HIPDNN_STATUS_SUCCESS; });
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
                auto err0 = hipdnnGetConvolutionBackwardDataWorkspaceSize(*m_hipdnn, *m_kernelT, m_outT, *m_conv, m_inT, (hipdnnConvolutionBwdDataAlgo_t)i, &tmpSize);
                if (err0 == HIPDNN_STATUS_SUCCESS)
                {
                    if (m_backDataAlgo.AlgoWorkspaceSize < tmpSize)
                        m_backDataAlgo.AlgoWorkspaceSize = tmpSize;
                    if ((hipdnnConvolutionBwdDataAlgo_t)i == HIPDNN_CONVOLUTION_BWD_DATA_ALGO_1)
                        m_backDataAlgo.DeterministicAlgoWorkspaceSize = tmpSize;
                    err = err0; 
                }
            }
            return err;
        }; 
        FindBestAlgo(batchSize, m_backDataAlgo, workspaceSizeFinder, deterministicFinder, finder, staticFinder, workspace);
        // Compute gradients with respect to the output tensor (data).
        HIPDNN_CALL(hipdnnConvolutionBackwardData(*m_hipdnn, &C::One, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_backDataAlgo.selectedAlgo, ptr(workspace), workspace.BufferSize(), accumulateGradient ? &C::One : &C::Zero, m_inT, ptr(grad)));
    }

    void BackwardKernelCore(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool accumulateGradient, bool /*allowReuse*/, Mat& workspace) override
    {
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
                result = hipdnnFindConvolutionBackwardFilterAlgorithmEx(*m_hipdnn, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, *m_kernelT, ptr(kernelGradReplace), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
                kernelGradReplace.ReleaseMemory();
            }
            else
                result = hipdnnFindConvolutionBackwardFilterAlgorithmEx(*m_hipdnn, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, *m_kernelT, ptr(kernelGrad), MaxAlgoCount, &calgo, algoPerf, ptr(workspace), workspace.BufferSize());
            return result;
        };
        // Find max Memory needed while running static finder. Workaround for hipdnnFind fail. Number of algo is constant as in hipdnn 5.1
        auto staticFinder = [&,this](hipdnnConvolutionBwdFilterAlgo_t& algo, bool noMem) -> hipdnnStatus_t
        {
            if(!noMem)
                return hipdnnGetConvolutionBackwardFilterAlgorithm(*m_hipdnn, m_inT, m_outT, *m_conv, *m_kernelT, HIPDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, workspace.BufferSize(), &algo);
            return hipdnnGetConvolutionBackwardFilterAlgorithm(*m_hipdnn, m_inT, m_outT, *m_conv, *m_kernelT, HIPDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &algo);
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
                auto err0 = hipdnnGetConvolutionBackwardFilterWorkspaceSize(*m_hipdnn, m_inT, m_outT, *m_conv, *m_kernelT, (hipdnnConvolutionBwdFilterAlgo_t)i, &tmpSize);
                if (err0 == HIPDNN_STATUS_SUCCESS)
                {
                    if (m_backFiltAlgo.AlgoWorkspaceSize < tmpSize)
                        m_backFiltAlgo.AlgoWorkspaceSize = tmpSize;
                    if ((hipdnnConvolutionBwdFilterAlgo_t)i == HIPDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
                        m_backFiltAlgo.DeterministicAlgoWorkspaceSize = tmpSize;
                    err = err0; 
                }
            }
            return err;
        }; 
        FindBestAlgo(batchSize, m_backFiltAlgo, workspaceSizeFinder, deterministicFinder, finder, staticFinder, workspace);
        // Compute gradients with respect to the output tensor (data).
        HIPDNN_CALL(hipdnnConvolutionBackwardFilter(*m_hipdnn, &C::One, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, m_backFiltAlgo.selectedAlgo, ptr(workspace), workspace.BufferSize(), accumulateGradient ? &C::One : &C::Zero, *m_kernelT, ptr(kernelGrad)));
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
        HIPDNN_CALL(hipdnnPoolingForward(*m_hipdnn, *(m_pool), &C::One, m_inT, ptr(in), &C::Zero, m_outT, ptr(out)));
    }

    void BackwardPoolingCore(const Mat& out, const Mat& srcGrad, const Mat& in, Mat& grad) override
    {
        size_t batchSize = in.GetNumCols();
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);
        HIPDNN_CALL(hipdnnPoolingBackward(*m_hipdnn, *(m_pool), &C::One, m_outT, ptr(out), m_outT, ptr(srcGrad),
                                        m_inT, ptr(in), &C::One, m_inT, ptr(grad)));
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

    static const int MaxAlgoCount = 10;

    void convert_type(cudnnConvolutionFwdAlgo_t in, hipdnnConvolutionFwdAlgo_t* out)
    {
	hipdnnStatus_t status_hipdnn;
	status_hipdnn = cudnnTohipConvolutionFwdAlgo(in, out);
    }
    void convert_type(cudnnConvolutionBwdDataAlgo_t in, hipdnnConvolutionBwdDataAlgo_t* out)
    {
        hipdnnStatus_t status_hipdnn;
        status_hipdnn = cudnnTohipConvolutionBwdDataAlgo(in, out);
    }
    void convert_type(cudnnConvolutionBwdFilterAlgo_t in, hipdnnConvolutionBwdFilterAlgo_t* out)
    {
        hipdnnStatus_t status_hipdnn;
        status_hipdnn = cudnnTohipConvolutionBwdFilterAlgo(in, out);
    }

    template <typename TAlgo, typename TWorkspaceSizeFinder, typename TDeterministicFinder, typename TFinder, typename TStaticFinder>
    void FindBestAlgo(size_t batchSize, TAlgo& algo, TWorkspaceSizeFinder workspaceSizeFinder, TDeterministicFinder deterministicFinder, TFinder finder, TStaticFinder staticFinder, Mat& workspace)
    {
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);

        // keep running if nothing changes
        if ((!algo.NeedAutotuning(batchSize)) && (workspace.BufferSize() >= algo.AlgoWorkspaceSize))
            return;

        // if batchsize changes again when just finish init, go back to init again
        if (algo.autotuningState == AutotuningState::PendingTuning && batchSize > algo.MBSizeForCurrentAlgo)
            algo.autotuningState = AutotuningState::Init;

        // batchSize is bigger than the one when initialize current workspace, need free up space and go back to init
        if (algo.autotuningState == AutotuningState::Running && batchSize > algo.maxMBSizeSeen)
        {
            algo.autotuningState = AutotuningState::Init;
            hipDeviceSynchronize(); // make sure no in-flight GPU kernels using workspace before release its memory
            workspace.Resize(0,0,0,false);
            algo.AlgoWorkspaceSize = 0;
            algo.MBSizeForCurrentWorkspace = 0;
        } 
        else if (algo.autotuningState == AutotuningState::Running && !m_forceDeterministicAlgorithms)  // batchSize changes to be smaller than MBSizeForCurrentWorkspace, need to re-do tuning if non-deterministic
            algo.autotuningState = AutotuningState::PendingTuning;

        typename TAlgo::typeT algoPerf[MaxAlgoCount];
        int calgo = 0;
        // in initState, where memory allocation for nodes are not completed, we only run the algorithm with no workspace
        // or in the special case when m_forceDeterministicAlgorithms, we allocate some memory and use the deterministic algorithm 
        if (algo.autotuningState == AutotuningState::Init)
        {
            // find workspace size needed for finderEx and deterministic algorithm 
            HIPDNN_CALL(workspaceSizeFinder()); 
            if (m_forceDeterministicAlgorithms)
            {
                workspace.Resize((algo.DeterministicAlgoWorkspaceSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);
                HIPDNN_CALL(deterministicFinder(calgo, algoPerf));
                assert(calgo == 1);                                 // only one deterministic algorithm will be returned 
                algo.MBSizeForCurrentAlgo = batchSize;
		/*const char *type = typeid(algo.selectedAlgo).name();
        	int status;
        	char *res = abi::__cxa_demangle(type, NULL, NULL, &status);
		typename TAlgo::typeL sel_algo;
		hipdnnStatus_t status_hipdnn;
        	if(strcmp(res,"hipdnnConvolutionFwdAlgo_t")==0)
		{
			status_hipdnn = cudnnTohipConvolutionFwdAlgo((*algoPerf).algo, &sel_algo);
		}
		//std::cout<<endl<<res;
		else if(strcmp(res, "hipdnnConvolutionBwdDataAlgo_t")==0)
			status_hipdnn = cudnnTohipConvolutionBwdDataAlgo((*algoPerf).algo, &sel_algo);
		else if(strcmp(res, "hipdnnConvolutionBwdFilterAlgo_t")==0)
			status_hipdnn = cudnnTohipConvolutionBwdFilterAlgo((*algoPerf).algo, &sel_algo);
		else
			status_hipdnn = HIPDNN_STATUS_SUCCESS;*/
		typename TAlgo::typeL sel_algo;
		convert_type((*algoPerf).algo, &sel_algo);
                algo.selectedAlgo = sel_algo;               // deterministic algorithm is the first in the list  
                algo.maxAlgo = algo.selectedAlgo;
		algo.autotuningState = AutotuningState::Running;    // no further need for tuning since this is deterministic, directly enter running state 
                algo.AlgoWorkspaceSize = (*algoPerf).memory;
            }
            else
            {
                HIPDNN_CALL(staticFinder(algo.selectedAlgo, true));
                algo.maxMBSizeSeen = batchSize;
                algo.MBSizeForCurrentAlgo = batchSize;
                algo.autotuningState = AutotuningState::PendingTuning;
            }
            return;
        }

        // we allocate workspace and find algorithm if batchSize is higher than ever seen
        if (algo.MBSizeForCurrentWorkspace == 0)    // no workspace memory has been allocated for this node
        {
            size_t curSize = workspace.BufferSize();

            // To control memory usage. No one seems to be using this flag
            size_t inputSampleSize = m_geometry->InputShape().GetNumElements();
            size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : inputSampleSize * m_maxTempMemSizeInSamples * sizeof(ElemType);

            try
            {   // first try allocate as much to run FindEX, this may fail when accumulate is on (in which case additional memory is allocated in finder()), thus we do try...catch...
                size_t free, total, resizeTo = 0;
                CUDA_CALL(hipMemGetInfo(&free, &total));
                free += workspace.BufferSize();
                // We reserve 2% of the total GPU memory because CuDNN seem to behave erroneously when there is no memory left
                if(free > (total/50))
                    resizeTo = free - (total/50) + sizeof(ElemType);
                // We don't need memory more than workspace we learned in workspaceSizeFinder 
                resizeTo = min(resizeTo, algo.AlgoWorkspaceSize); 
                resizeTo = min(resizeTo, maxMem); 
                if(resizeTo > 0)
                    workspace.Resize((resizeTo + sizeof(ElemType) - 1) / sizeof(ElemType), 1);     // resize the workspace so that we can run the finder
                algo.MBSizeForCurrentWorkspace = batchSize;

                // Pending State now, let's do a find and get algorithm Perfs
                calgo = 0; 
                HIPDNN_CALL(finder(calgo, algoPerf));
                assert(calgo > 0); 
                auto res = algoPerf;        // first returned algorithm is the fastest 
                algo.MBSizeForCurrentAlgo = batchSize;
		typename TAlgo::typeL sel_algo;
                convert_type((*algoPerf).algo, &sel_algo);
                algo.selectedAlgo = sel_algo;
                algo.maxAlgo = algo.selectedAlgo;
                algo.autotuningState = AutotuningState::Running;
                algo.AlgoWorkspaceSize = (*res).memory;
                if (algo.AlgoWorkspaceSize < curSize)   // need to shrink the workspace
                    workspace.Resize((curSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);
                else
                    workspace.Resize((algo.AlgoWorkspaceSize + sizeof(ElemType) - 1) / sizeof(ElemType), 1, 0, false);
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
                    algo.MBSizeForCurrentAlgo = batchSize;
                    typename TAlgo::typeL sel_algo;
                    convert_type((*algoPerf).algo, &sel_algo);
		    algo.selectedAlgo = sel_algo;
                    algo.maxAlgo = algo.selectedAlgo;
                    algo.autotuningState = AutotuningState::Running;
                    algo.AlgoWorkspaceSize = (*res).memory;
                } 
                catch (...) 
                {   // fails again, let's fall back to hipdnnGet
                    fprintf(stderr, "Fall back to use static finder to get the algorithm for convolution\n");
                    HIPDNN_CALL(staticFinder(algo.selectedAlgo, false));
                    algo.MBSizeForCurrentAlgo = batchSize;
                    algo.maxAlgo = algo.selectedAlgo;
                    algo.autotuningState = AutotuningState::Running;
                    algo.AlgoWorkspaceSize = curSize;
                }
            }
        }
        else if (batchSize == algo.MBSizeForCurrentWorkspace && workspace.BufferSize() >= algo.AlgoWorkspaceSize) // Use stored algo when batchsize go back to max. Likely happen when last batch in epoch lacking data
        {
            algo.selectedAlgo = algo.maxAlgo;
            algo.MBSizeForCurrentAlgo = batchSize;
            algo.autotuningState = AutotuningState::Running;
        }
        else    // use fast/static method to get algorithm when batchsize get smaller, assuming workspace size doesn't expand. Avoid severe slowdown when batchsize change frequently
        {
            HIPDNN_CALL(staticFinder(algo.selectedAlgo, false));
            algo.MBSizeForCurrentAlgo = batchSize;
            algo.autotuningState = AutotuningState::Running;
        }
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
    template <typename T, typename L>
    struct ConvAlgoInfo
    {
        typedef T typeT;
	typedef L typeL;
        ConvAlgoInfo()
            : MBSizeForCurrentAlgo(0), MBSizeForCurrentWorkspace(0), maxMBSizeSeen(0),autotuningState(AutotuningState::Init), AlgoWorkspaceSize(0)
        {
        }
        // Current mini-batch size, needed for re-computing statistics in auto-tuner.
        size_t maxMBSizeSeen;               // maximum minibatch size that's seen for the current tuning. If batch size exceed this number, redo tuning from scratch  
        size_t MBSizeForCurrentAlgo;        // minibatch size for the currently adopted algorithm
        size_t MBSizeForCurrentWorkspace;   // minibatch size when the current work space is allocated, if bath size returns to this size, directly pick the maxAlgo 
        size_t AlgoWorkspaceSize;           // maximum workspace size for any algorithm 
        size_t DeterministicAlgoWorkspaceSize;  // workspace size for deterministic algorithm 
        AutotuningState autotuningState;    // state of auto-tuning: Init, PendingTuning and Running 
        decltype(static_cast<L>(T::algo)) selectedAlgo;     // currently selected algorithm 
        decltype(static_cast<L>(T::algo)) maxAlgo;          // algorithm that was selected when the current workspace is allocated 

        bool NeedAutotuning(size_t batchSize)
        {
            // We assume no other dimensions of tensors can change so we don't check it.
            // REVIEW alexeyk: review once we get response from NVIDIA.
            // NVIDIA response:
            // It is not safe to assume that previously selected algorithm requires less or the same amount of workspace when minibatch size decrease
            // Need to re-run auto-tuner everytime minibatch size grow.
            // Use faster(may not be optimal) method to get algorithm when batchsize decrease
            // Should remain reasonable performance when minibatch size changes frequently (e.g. distributed reading).
            return (autotuningState != AutotuningState::Running || batchSize != MBSizeForCurrentAlgo);
        }
    };

    CuDnn::ptr_t m_hipdnn;
    hipdnnDataType_t m_dataType;
    CuDnnTensor m_inT;
    CuDnnTensor m_outT;
    // Convolution specific.
    std::unique_ptr<CuDnnKernel> m_kernelT;
    std::unique_ptr<CuDnnConv> m_conv;
    // Pooling specific.
    std::unique_ptr<CuDnnPool> m_pool;

    ConvAlgoInfo<hipdnnConvolutionFwdAlgoPerf_t, hipdnnConvolutionFwdAlgo_t> m_fwdAlgo;
    ConvAlgoInfo<hipdnnConvolutionBwdDataAlgoPerf_t, hipdnnConvolutionBwdDataAlgo_t> m_backDataAlgo;
    ConvAlgoInfo<hipdnnConvolutionBwdFilterAlgoPerf_t, hipdnnConvolutionBwdFilterAlgo_t> m_backFiltAlgo;

    // Flag indicating whether only deterministic algorithms should be used.
    bool m_forceDeterministicAlgorithms;
};

template <class ElemType>
std::unique_ptr<ConvolutionEngine<ElemType>> CuDnnConvolutionEngineFactory<ElemType>::Create(ConvolveGeometryPtr geometry,
                                                                                             DEVICEID_TYPE deviceId, ImageLayoutKind imageLayout,
                                                                                             size_t maxTempMemSizeInSamples, PoolKind poolKind,
                                                                                             bool forceDeterministicAlgorithms, bool poolIncludePad)
{
    return std::make_unique<CuDnnConvolutionEngine<ElemType>>(geometry, deviceId, imageLayout, maxTempMemSizeInSamples, poolKind, forceDeterministicAlgorithms, poolIncludePad);
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
                   inputRank <= 3 && (kernelRank < 3 || kernel[2] == 1)));

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

} } }
