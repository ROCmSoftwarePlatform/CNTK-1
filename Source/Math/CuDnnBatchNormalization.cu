//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CuDnnFactories.h"
#include "BatchNormalizationEngine.h"
#include "CuDnnCommon.h"
#include "GPUMatrix.h"

#ifdef HIP_COMPILE
#define HIPDNN_BN_MIN_EPSILON 1e-5 //TODO: __add__ replace with the correct macro
#endif
namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class CuDnnBatchNormEngine : public BatchNormEngine<ElemType>
{
public:
    using Base = BatchNormEngine<ElemType>;
    using typename Base::Mat;

public:
#ifdef CUDA_COMPILE
    CuDnnBatchNormEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                        bool spatial, ImageLayoutKind imageLayout)
                        : Base(deviceId, inOutT, spatial, imageLayout),
                        m_cudnn(CuDnn::Instance()),
                        m_inOutCuDnnT(GetInOutTensor(inOutT), CuDnnTensor::GetDataType<ElemType>()),
                        m_scaleBiasCuDnnT(GetScaleBiasTensor(inOutT, spatial), CuDnnTensor::GetDataType<ElemType>()),
                        m_cudnnEpsilon(CUDNN_BN_MIN_EPSILON)
    {
    }
#elif defined HIP_COMPILE
    CuDnnBatchNormEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                        bool spatial, ImageLayoutKind imageLayout)
                        : Base(deviceId, inOutT, spatial, imageLayout),
                        m_cudnn(CuDnn::Instance()),
                        m_inOutCuDnnT(GetInOutTensor(inOutT), CuDnnTensor::GetDataType<ElemType>()),
                        m_scaleBiasCuDnnT(GetScaleBiasTensor(inOutT, spatial), CuDnnTensor::GetDataType<ElemType>()),
                        m_cudnnEpsilon(HIPDNN_BN_MIN_EPSILON)
    {
    }
#endif

protected:
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_inOutT;
    using Base::m_spatial;

    void EnsureCompatible() override
    {
#ifdef CUDA_COMPILE
	    if (m_spatial && m_imageLayout == ImageLayoutKind::HWC)
            InvalidArgument("cuDNN batch normalization supports only cudnn(CHW) layout.");
        if (m_inOutT.GetRank() > 4)
	        InvalidArgument("cuDNN batch normalization supports tensors of max 4 dimensions.");
#elif defined HIP_COMPILE
        if (m_spatial && m_imageLayout == ImageLayoutKind::HWC)
            InvalidArgument("hipDNN batch normalization supports only hipdnn(CHW) layout.");
        if (m_inOutT.GetRank() > 4)
            InvalidArgument("hipDNN batch normalization supports tensors of max 4 dimensions.");
#endif
    }

    void ForwardCore(const Mat& in, const Mat& scale, const Mat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, Mat& runMean, Mat& runVariance,
                     Mat& out, double epsilon, Mat& savedMean, Mat& savedInvStdDev) override
    {
        // TODO batchSize == 1

        // REVIEW alexeyk: there might be a way to do this in cuDNN.
        if (blendFactor != 0 && (blendFactor != 1 || expAvgFactor > 0))
            InvalidArgument("cuDNN batch normalization engine currently supports blendTimeConstant of 0 or 1 only.");

        m_inOutCuDnnT.UpdateBatchSize(in.GetNumCols());
#ifdef CUDA_COMPILE
	    cudnnBatchNormMode_t mode = m_spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
	    // cuDNN will fail with BAD_PARAM if epsilon < CUDNN_BN_MIN_EPSILON.
	    m_cudnnEpsilon = max(epsilon, CUDNN_BN_MIN_EPSILON);
#elif defined HIP_COMPILE
        hipdnnBatchNormMode_t mode = m_spatial ? HIPDNN_BATCHNORM_SPATIAL : HIPDNN_BATCHNORM_PER_ACTIVATION;
        // cuDNN will fail with BAD_PARAM if epsilon < HIPDNN_BN_MIN_EPSILON.
        m_cudnnEpsilon = max(epsilon, HIPDNN_BN_MIN_EPSILON);
#endif
        if (inferenceOnly)
        {
            assert(expAvgFactor == 0 && blendFactor == 1);
            savedMean.Resize(0, 0);      // (these are not produced in this case)
            savedInvStdDev.Resize(0, 0);
#ifdef CUDA_COMPILE
	        CUDNN_CALL2(cudnnBatchNormalizationForwardInference(*m_cudnn, mode, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(out),
                                                                  m_scaleBiasCuDnnT, ptr(scale), ptr(bias), ptr(runMean), ptr(runVariance), m_cudnnEpsilon),
			            "\nProbably hitting cuDNN limit on batch size, try reducing minibatch size");
#elif defined HIP_COMPILE
            HIPDNN_CALL2(hipdnnBatchNormalizationForwardInference(*m_cudnn, mode, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(out),
                                                                  m_scaleBiasCuDnnT, ptr(scale), ptr(bias), ptr(runMean), ptr(runVariance), m_cudnnEpsilon),
                        "\nProbably hitting cuDNN limit on batch size, try reducing minibatch size");
#endif
        }
        else
        {
            savedMean.Resize(runMean);
            savedInvStdDev.Resize(runMean);
#ifdef CUDA_COMPILE
            CUDNN_CALL(cudnnBatchNormalizationForwardTraining(*m_cudnn, mode, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in),
                                                              m_inOutCuDnnT, ptr(out), m_scaleBiasCuDnnT, ptr(scale), ptr(bias), expAvgFactor, ptr(runMean), ptr(runVariance),
							      m_cudnnEpsilon, ptr(savedMean), ptr(savedInvStdDev)));
#elif defined HIP_COMPILE
            HIPDNN_CALL(hipdnnBatchNormalizationForwardTraining(*m_cudnn, mode, const_cast<void*>(static_cast<const void*>(&C::One)),const_cast<void*>(static_cast<const void*>(&C::Zero)),
								 m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(out), m_scaleBiasCuDnnT, const_cast<void*>(static_cast<const void*>(ptr(scale))), 
								 const_cast<void*>(static_cast<const void*>(ptr(bias))), expAvgFactor, const_cast<void*>(static_cast<const void*>(ptr(runMean))), 			 						     const_cast<void*>(static_cast<const void*>(ptr(runVariance))),m_cudnnEpsilon, ptr(savedMean), ptr(savedInvStdDev)));
#endif
        }
    }

    void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, double blendFactor, const Mat& savedMean, const Mat& savedInvStdDev,
                      Mat& scaleGrad, Mat& biasGrad, bool accumulateDataGrad) override
    {
        UNUSED(blendFactor);  // BUGBUG: It should be used.
        m_inOutCuDnnT.UpdateBatchSize(srcGrad.GetNumCols());
#ifdef CUDA_COMPILE
	    cudnnBatchNormMode_t mode = m_spatial ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION;
        // REVIEW alexeyk: change betaParamDiff to 1 and update CNTK BN engine.
        CUDNN_CALL(cudnnBatchNormalizationBackward(*m_cudnn, mode, &C::One, accumulateDataGrad ? &C::One : &C::Zero, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(srcGrad), m_inOutCuDnnT, ptr(grad),
                                                                       m_scaleBiasCuDnnT, ptr(scale), ptr(scaleGrad), ptr(biasGrad), m_cudnnEpsilon, ptr(savedMean), ptr(savedInvStdDev)));
#elif defined HIP_COMPILE
        hipdnnBatchNormMode_t mode = m_spatial ? HIPDNN_BATCHNORM_SPATIAL : HIPDNN_BATCHNORM_PER_ACTIVATION;
        // REVIEW alexeyk: change betaParamDiff to 1 and update CNTK BN engine.
        HIPDNN_CALL(hipdnnBatchNormalizationBackward(*m_cudnn, mode, &C::One, accumulateDataGrad ? &C::One : &C::Zero, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(srcGrad), m_inOutCuDnnT, ptr(grad),
                                                      m_scaleBiasCuDnnT, ptr(scale), ptr(scaleGrad), ptr(biasGrad), m_cudnnEpsilon, ptr(savedMean), ptr(savedInvStdDev)));
#endif
    }

private:
    static ElemType* ptr(Mat& src)
    {
        return src.Data();
    }
    static const ElemType* ptr(const Mat& src)
    {
        return src.Data();
    }

    static TensorShape GetInOutTensor(const TensorShape& inOutT)
    {
        // cuDNN supports only 3D and 4D tensors (in cuDNN docs it's 4D and 5D dues to N dimension)
        // even for non-spatial inputs so expand the tensor if needed.
        if (inOutT.GetRank() > 2)
            return inOutT;

        const size_t outRank = 3;
        SmallVector<size_t> v(std::max(inOutT.GetRank(), outRank), 1);
        for (size_t i = outRank - inOutT.GetRank(), j = 0; i < outRank; i++, j++)
            v[i] = inOutT[j];

        return TensorShape(v);
    }

    static TensorShape GetScaleBiasTensor(const TensorShape& inOutT, bool spatial)
    {
        if (!spatial)
            return GetInOutTensor(inOutT);

        const auto& t = GetInOutTensor(inOutT);
        SmallVector<size_t> v(t.GetRank(), 1);
        v[v.size() - 1] = t[t.GetRank() - 1];
        return TensorShape(v);
    }

private:
    using C = Consts<ElemType>;

    CuDnn::ptr_t m_cudnn;
    CuDnnTensor m_inOutCuDnnT;
    CuDnnTensor m_scaleBiasCuDnnT;
    double m_cudnnEpsilon;
};

template class CuDnnBatchNormEngine<float>;
template class CuDnnBatchNormEngine<double>;

template <typename ElemType>
std::unique_ptr<BatchNormEngine<ElemType>> CuDnnBatchNormEngineFactory<ElemType>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                                                         bool spatial, ImageLayoutKind imageLayout)
{
    return std::make_unique<CuDnnBatchNormEngine<ElemType>>(deviceId, inOutT, spatial, imageLayout);
}

template class CuDnnBatchNormEngineFactory<float>;
template class CuDnnBatchNormEngineFactory<double>;

CudaTimer::~CudaTimer()
{
    // TODO: Should not throw if std::uncaught_exception()
#ifdef CUDA_COMPILE
    if (m_start != nullptr)
        CUDA_CALL(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(m_start)));
    if (m_stop != nullptr)
	CUDA_CALL(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(m_stop)));
#elif defined HIP_COMPILE
    if (m_start != nullptr)
        CUDA_CALL(hipEventDestroy(reinterpret_cast<hipEvent_t>(m_start)));
    if (m_stop != nullptr)
        CUDA_CALL(hipEventDestroy(reinterpret_cast<hipEvent_t>(m_stop)));
#endif
}
void CudaTimer::Start()
{
#ifdef CUDA_COMPILE
    cudaEvent_t start;
    cudaEvent_t stop;
    if (m_start != nullptr)
        CUDA_CALL(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(m_start)));
    if (m_stop != nullptr)
        CUDA_CALL(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(m_stop)));
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    m_start = start;
    m_stop = stop;
    CUDA_CALL(cudaEventRecord(start, GetStream()));
#elif defined HIP_COMPILE
    hipEvent_t start;
    hipEvent_t stop;
    if (m_start != nullptr)
        CUDA_CALL(hipEventDestroy(reinterpret_cast<hipEvent_t>(m_start)));
    if (m_stop != nullptr)
        CUDA_CALL(hipEventDestroy(reinterpret_cast<hipEvent_t>(m_stop)));
    CUDA_CALL(hipEventCreate(&start));
    CUDA_CALL(hipEventCreate(&stop));
    m_start = start;
    m_stop = stop;
    CUDA_CALL(hipEventRecord(start, GetStream()));
#endif
}
void CudaTimer::Stop()
{
#ifdef CUDA_COMPILE
    CUDA_CALL(cudaEventRecord(reinterpret_cast<cudaEvent_t>(m_stop), GetStream()));
    CUDA_CALL(cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(m_stop)));
#elif defined HIP_COMPILE
    CUDA_CALL(hipEventRecord(reinterpret_cast<hipEvent_t>(m_stop), GetStream()));
    CUDA_CALL(hipEventSynchronize(reinterpret_cast<hipEvent_t>(m_stop)));
#endif
}
float CudaTimer::Elapsed()
{
    float ms;
#ifdef CUDA_COMPILE
    CUDA_CALL(cudaEventElapsedTime(&ms, reinterpret_cast<cudaEvent_t>(m_start), reinterpret_cast<cudaEvent_t>(m_stop)));
#elif defined HIP_COMPILE
    CUDA_CALL(hipEventElapsedTime(&ms, reinterpret_cast<hipEvent_t>(m_start), reinterpret_cast<hipEvent_t>(m_stop)));
#endif
    return ms;
}

} } }
