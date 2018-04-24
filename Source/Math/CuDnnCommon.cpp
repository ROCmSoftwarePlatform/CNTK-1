//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "GPUMatrix.h"
#include "CuDnnCommon.h"
#include "half.hpp"

namespace Microsoft { namespace MSR { namespace CNTK {
#ifndef CPUONLY
MATH_API std::size_t GetCUDNNVersion()
{
    return hipdnnGetVersion();
}
#endif
template <>
const float Consts<float>::One = 1;
template <>
const double Consts<double>::One = 1;
template <>
const float Consts<float>::Zero = 0;
template <>
const double Consts<double>::Zero = 0;

#ifdef __HIP_ENABLE_HALF__
const float Consts<half>::Zero = 0;
const float Consts<half>::One = 1;
#endif //__HIP_ENABLE_HALF__

CuDnnTensor::CuDnnTensor()
    : m_tensor(nullptr)
{
}

CuDnnTensor::CuDnnTensor(const TensorShape& src, hipdnnDataType_t dataType)
    : m_tensor(nullptr)
{
    Set(src, dataType);
}

CuDnnTensor::~CuDnnTensor()
{
    if (m_tensor != nullptr)
    {
        hipdnnDestroyTensorDescriptor(m_tensor);
        m_tensor = nullptr;
    }
}

void CuDnnTensor::Set(const TensorShape& src, hipdnnDataType_t dataType)
{
    HIPDNN_CALL(hipdnnCreateTensorDescriptor(&m_tensor));
    // Set cuDNN tensor dimensions. cuDNN uses row-major format while TensorShape - column-major
    // so conversion is required. N dimension will be set to 1.
    const auto& stridesSrc = src.GetStrides();
    SmallVector<int> dims(src.GetRank() + 1);
    SmallVector<int> strides(stridesSrc.size() + 1);
    assert(dims.size() == strides.size());
    for (int i = 0; i < src.GetRank(); i++)
    {
        dims[dims.size() - 1 - i] = (int)src[i];
        strides[dims.size() - 1 - i] = (int)stridesSrc[i];
    }
    // Set "minibatch"(aka N) dimension.
    dims[0] = 1;
    strides[0] = strides[1] * dims[1];
    HIPDNN_CALL(hipdnnSetTensorNdDescriptor(m_tensor, dataType, (int)dims.size(), dims.data(), strides.data()));
}

void CuDnnTensor::UpdateBatchSize(size_t batchSize)
{
    // Currently cuDNN supports only 2D and 3D convlutions anyway (so max 5D tensors).
    const int MaxDims = 5;
    int dims[MaxDims];
    int strides[MaxDims];
    int nbDims = 0;
    hipdnnDataType_t dataType;
    // According to NVIDIA, Get/Set functions are very fast so it's safe to call them in a loop.
    HIPDNN_CALL(hipdnnGetTensorNdDescriptor(m_tensor, MaxDims, &dataType, &nbDims, dims, strides));
    assert(nbDims <= MaxDims);
    dims[0] = (int)batchSize;
    HIPDNN_CALL(hipdnnSetTensorNdDescriptor(m_tensor, dataType, nbDims, dims, strides));
}

template <typename ElemType>
hipdnnDataType_t CuDnnTensor::GetDataType()
{
    if (typeid(ElemType) == typeid(float))
        return HIPDNN_DATA_FLOAT;
    else if (typeid(ElemType) == typeid(double))
        return HIPDNN_DATA_DOUBLE;
#ifdef __HIP_ENABLE_HALF__
    else if (typeid(ElemType) == typeid(half))
        return HIPDNN_DATA_HALF;
#endif //__HIP_ENABLE_HALF__
    else
        InvalidArgument("hipDNN engine currently supports only single and double precision data types.");
}

template hipdnnDataType_t CuDnnTensor::GetDataType<float>();
template hipdnnDataType_t CuDnnTensor::GetDataType<double>();
#ifdef __HIP_ENABLE_HALF__
template hipdnnDataType_t CuDnnTensor::GetDataType<half>();
#endif //__HIP_ENABLE_HALF__
CuDnn::ptr_t CuDnn::Instance()
{
    auto createNew = []()
    {
        int deviceId;
        CUDA_CALL(hipGetDevice(&deviceId));
        hipDeviceProp_t props = {0};
        if (hipGetDeviceProperties(&props, deviceId) != hipSuccess || props.major < 3)
            RuntimeError("cuDNN requires device with compute capability 3.0 or higher.");
        hipdnnHandle_t* hipdnn = new hipdnnHandle_t;
        HIPDNN_CALL(hipdnnCreate(hipdnn));
        HIPDNN_CALL(hipdnnSetStream(*hipdnn, GetStream()));
        return hipdnn;
    };

    static std::shared_ptr<hipdnnHandle_t> m_instance = std::shared_ptr<hipdnnHandle_t>(createNew(), [](hipdnnHandle_t* src)
    {
        assert(*src != nullptr);
        auto err = hipdnnDestroy(*src);
        assert(err == HIPDNN_STATUS_SUCCESS);
#ifdef NDEBUG
        UNUSED(err);
#endif
        delete src;
    });
    return m_instance;
}

} } }
