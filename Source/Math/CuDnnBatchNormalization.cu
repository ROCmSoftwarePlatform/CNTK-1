//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CuDnnFactories.h"
#include "BatchNormalizationEngine.h"
#include "CuDnnCommon.h"
#include "GPUMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class InoutType, class StatType>
class CuDnnBatchNormEngine : public BatchNormEngine<InoutType, StatType>
{
public:
    using Base = BatchNormEngine<InoutType, StatType>;
    using typename Base::InoutMat;
    using typename Base::StatMat;

public:
    CuDnnBatchNormEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                        bool spatial, ImageLayoutKind imageLayout)
                        : Base(deviceId, inOutT, spatial, imageLayout),
                        m_cudnn(CuDnn::Instance()),
                        m_inOutCuDnnT(GetInOutTensor(inOutT), CuDnnTensor::GetDataType<InoutType>()),
                        m_scaleBiasCuDnnT(GetScaleBiasTensor(inOutT, spatial), CuDnnTensor::GetDataType<StatType>()),
                        m_cudnnEpsilon(HIPDNN_BN_MIN_EPSILON)
    {
    }

protected:
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_inOutT;
    using Base::m_spatial;

    void EnsureCompatible() override
    {
        if (m_spatial && m_imageLayout == ImageLayoutKind::HWC)
            InvalidArgument("hipDNN batch normalization supports only hipdnn(CHW) layout.");
        if (m_inOutT.GetRank() > 4)
            InvalidArgument("hipDNN batch normalization supports tensors of max 4 dimensions.");
    }

    void ForwardCore(const InoutMat& in, const StatMat& scale, const StatMat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, StatMat& runMean, StatMat& runVariance,
                     InoutMat& out, double epsilon, StatMat& savedMean, StatMat& savedInvStdDev) override
    {
        // TODO batchSize == 1

        // REVIEW alexeyk: there might be a way to do this in cuDNN.
        if (blendFactor != 0 && (blendFactor != 1 || expAvgFactor > 0))
            InvalidArgument("cuDNN batch normalization engine currently supports blendTimeConstant of 0 or 1 only.");

        m_inOutCuDnnT.UpdateBatchSize(in.GetNumCols());
        hipdnnBatchNormMode_t mode = m_spatial ? HIPDNN_BATCHNORM_SPATIAL_PERSISTENT : HIPDNN_BATCHNORM_PER_ACTIVATION;
        if (inferenceOnly) mode = m_spatial ? HIPDNN_BATCHNORM_SPATIAL : HIPDNN_BATCHNORM_PER_ACTIVATION;
        // cuDNN will fail with BAD_PARAM if epsilon < HIPDNN_BN_MIN_EPSILON.
        m_cudnnEpsilon = max(epsilon, HIPDNN_BN_MIN_EPSILON);
        if (inferenceOnly)
        {
            assert(expAvgFactor == 0 && blendFactor == 1);
            savedMean.Resize(0, 0);      // (these are not produced in this case)
            savedInvStdDev.Resize(0, 0);
            //TODO NEEL: Add support for BNInference
            //HIPDNN_CALL2(hipnnBatchNormalizationForwardInference(*m_cudnn, mode, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(out),
              //                                                    m_scaleBiasCuDnnT, ptr(scale), ptr(bias), ptr(runMean), ptr(runVariance), m_cudnnEpsilon),
                    //    "\nProbably hitting cuDNN limit on batch size, try reducing minibatch size");
        }
        else
        {
            savedMean.Resize(runMean);
            savedInvStdDev.Resize(runMean);
            HIPDNN_CALL(hipdnnBatchNormalizationForwardTraining(*m_cudnn, mode, (void*)&C::One, (void*)&C::Zero, m_inOutCuDnnT, ptr(in),
                                                              m_inOutCuDnnT, ptr(out), m_scaleBiasCuDnnT, (void*)ptr(scale), (void*)ptr(bias), expAvgFactor, ptr(runMean), ptr(runVariance),
                                                              m_cudnnEpsilon, ptr(savedMean), ptr(savedInvStdDev)));
        }
    }

    void BackwardCore(const InoutMat& in, const InoutMat& srcGrad, InoutMat& grad, const StatMat& scale, double blendFactor, const StatMat& savedMean, const StatMat& savedInvStdDev,
                      StatMat& scaleGrad, StatMat& biasGrad, bool accumulateDataGrad) override
    {
        UNUSED(blendFactor);  // BUGBUG: It should be used.
        m_inOutCuDnnT.UpdateBatchSize(srcGrad.GetNumCols());
        hipdnnBatchNormMode_t mode = m_spatial ? HIPDNN_BATCHNORM_SPATIAL_PERSISTENT : HIPDNN_BATCHNORM_PER_ACTIVATION;
        // REVIEW alexeyk: change betaParamDiff to 1 and update CNTK BN engine.
        HIPDNN_CALL(hipdnnBatchNormalizationBackward(*m_cudnn, mode, &C::One, accumulateDataGrad ? &C::One : &C::Zero, &C::One, &C::Zero, m_inOutCuDnnT, ptr(in), m_inOutCuDnnT, ptr(srcGrad), m_inOutCuDnnT, ptr(grad),
                                                      m_scaleBiasCuDnnT, ptr(scale), ptr(scaleGrad), ptr(biasGrad), m_cudnnEpsilon, ptr(savedMean), ptr(savedInvStdDev)));
    }

private:
    template<typename ElemType>
    static ElemType* ptr(Matrix<ElemType>& src)
    {
        return src.Data();
    }

    template<typename ElemType>
    static const ElemType* ptr(const Matrix<ElemType>& src)
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
    using C = Consts<StatType>;

    CuDnn::ptr_t m_cudnn;
    CuDnnTensor m_inOutCuDnnT;
    CuDnnTensor m_scaleBiasCuDnnT;
    double m_cudnnEpsilon;
};

template class CuDnnBatchNormEngine<float, float>;
template class CuDnnBatchNormEngine<double, double>;
template class CuDnnBatchNormEngine<half, float>;

template <typename InoutType, typename StatType>
std::unique_ptr<BatchNormEngine<InoutType, StatType>> CuDnnBatchNormEngineFactory<InoutType, StatType>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                                                         bool spatial, ImageLayoutKind imageLayout)
{
    return std::make_unique<CuDnnBatchNormEngine<InoutType, StatType>>(deviceId, inOutT, spatial, imageLayout);
}

template class CuDnnBatchNormEngineFactory<float, float>;
template class CuDnnBatchNormEngineFactory<double, double>;
template class CuDnnBatchNormEngineFactory<half, float>;

CudaTimer::~CudaTimer()
{
    // TODO: Should not throw if std::uncaught_exception()
    if (m_start != nullptr)
        CUDA_CALL(hipEventDestroy(reinterpret_cast<hipEvent_t>(m_start)));
    if (m_stop != nullptr)
        CUDA_CALL(hipEventDestroy(reinterpret_cast<hipEvent_t>(m_stop)));
}
void CudaTimer::Start()
{
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
}
void CudaTimer::Stop()
{
    CUDA_CALL(hipEventRecord(reinterpret_cast<hipEvent_t>(m_stop), GetStream()));
    CUDA_CALL(hipEventSynchronize(reinterpret_cast<hipEvent_t>(m_stop)));
}
float CudaTimer::Elapsed()
{
    float ms;
    CUDA_CALL(hipEventElapsedTime(&ms, reinterpret_cast<hipEvent_t>(m_start), reinterpret_cast<hipEvent_t>(m_stop)));
    return ms;
}

} } }
