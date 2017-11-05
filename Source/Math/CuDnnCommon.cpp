//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "GPUMatrix.h"
#include "CuDnnCommon.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <>
const float Consts<float>::One = 1;
template <>
const double Consts<double>::One = 1;
template <>
const float Consts<float>::Zero = 0;
template <>
const double Consts<double>::Zero = 0;

CuDnnTensor::CuDnnTensor()
    : m_tensor(nullptr)
{
}

#ifdef CUDA_COMPILE
CuDnnTensor::CuDnnTensor(const TensorShape& src, cudnnDataType_t dataType)
    : m_tensor(nullptr)
{
    Set(src, dataType); 
}
#elif defined HIP_COMPILE
CuDnnTensor::CuDnnTensor(const TensorShape& src, hipdnnDataType_t dataType)
    : m_tensor(nullptr)
{
    Set(src, dataType); 
}
#endif

CuDnnTensor::~CuDnnTensor()
{
    if (m_tensor != nullptr)
    {
#ifdef CUDA_COMPILE
	    cudnnDestroyTensorDescriptor(m_tensor);
#elif defined HIP_COMPILE
        hipdnnDestroyTensorDescriptor(m_tensor);
#endif
        m_tensor = nullptr;
    }
}

#ifdef CUDA_COMPILE
void CuDnnTensor::Set(const TensorShape& src, cudnnDataType_t dataType)
{
    CUDNN_CALL(cudnnCreateTensorDescriptor(&m_tensor));
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
    CUDNN_CALL(cudnnSetTensorNdDescriptor(m_tensor, dataType, (int)dims.size(), dims.data(), strides.data()));
}
#elif defined HIP_COMPILE
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
#endif

void CuDnnTensor::UpdateBatchSize(size_t batchSize)
{
    // Currently cuDNN supports only 2D and 3D convlutions anyway (so max 5D tensors).
    const int MaxDims = 5;
    int dims[MaxDims];
    int strides[MaxDims];
    int nbDims = 0;
#ifdef CUDA_COMPILE
    cudnnDataType_t dataType;
    // According to NVIDIA, Get/Set functions are very fast so it's safe to call them in a loop.
    CUDNN_CALL(cudnnGetTensorNdDescriptor(m_tensor, MaxDims, &dataType, &nbDims, dims, strides));
    assert(nbDims <= MaxDims);
    dims[0] = (int)batchSize;
    CUDNN_CALL(cudnnSetTensorNdDescriptor(m_tensor, dataType, nbDims, dims, strides));
#elif defined HIP_COMPILE
    hipdnnDataType_t dataType;
    // According to NVIDIA, Get/Set functions are very fast so it's safe to call them in a loop.
    HIPDNN_CALL(hipdnnGetTensorNdDescriptor(m_tensor, MaxDims, &dataType, &nbDims, dims, strides));
    assert(nbDims <= MaxDims);
    dims[0] = (int)batchSize;
    HIPDNN_CALL(hipdnnSetTensorNdDescriptor(m_tensor, dataType, nbDims, dims, strides));
#endif
}

#ifdef CUDA_COMPILE
template <typename ElemType>
cudnnDataType_t CuDnnTensor::GetDataType()
{
    if (typeid(ElemType) == typeid(float))
        return CUDNN_DATA_FLOAT;
    else if (typeid(ElemType) == typeid(double))
        return CUDNN_DATA_DOUBLE;
    else
        InvalidArgument("cuDNN engine currently supports only single and double precision data types.");
}
#elif defined HIP_COMPILE
template <typename ElemType>
hipdnnDataType_t CuDnnTensor::GetDataType()
{
    if (typeid(ElemType) == typeid(float))
        return HIPDNN_DATA_FLOAT;
    else if (typeid(ElemType) == typeid(double))
        return HIPDNN_DATA_DOUBLE;
    else
        InvalidArgument("hipDNN engine currently supports only single and double precision data types.");
}
#endif

#ifdef CUDA_COMPILE
template cudnnDataType_t CuDnnTensor::GetDataType<float>();
template cudnnDataType_t CuDnnTensor::GetDataType<double>();
#elif defined HIP_COMPILE
template hipdnnDataType_t CuDnnTensor::GetDataType<float>();
template hipdnnDataType_t CuDnnTensor::GetDataType<double>();
#endif

CuDnn::ptr_t CuDnn::Instance()
{
    auto createNew = []()
    {
        int deviceId;
#ifdef CUDA_COMPILE
	    CUDA_CALL(cudaGetDevice(&deviceId));
        cudaDeviceProp props = {0};
        if (cudaGetDeviceProperties(&props, deviceId) != cudaSuccess || props.major < 3)
            RuntimeError("cuDNN requires device with compute capability 3.0 or higher.");
        cudnnHandle_t* cudnn = new cudnnHandle_t;
        CUDNN_CALL(cudnnCreate(cudnn));
        CUDNN_CALL(cudnnSetStream(*cudnn, GetStream()));
	    return cudnn;
#elif defined HIP_COMPILE
        CUDA_CALL(hipGetDevice(&deviceId));
        hipDeviceProp_t props = {0};
        if (hipGetDeviceProperties(&props, deviceId) != hipSuccess || props.major < 3)
            RuntimeError("cuDNN requires device with compute capability 3.0 or higher.");
        hipdnnHandle_t* hipdnn = new hipdnnHandle_t;
        HIPDNN_CALL(hipdnnCreate(hipdnn));
        HIPDNN_CALL(hipdnnSetStream(*hipdnn, GetStream()));
        return hipdnn;
#endif
    };

#ifdef CUDA_COMPILE
    static std::shared_ptr<cudnnHandle_t> m_instance = std::shared_ptr<cudnnHandle_t>(createNew(), [](cudnnHandle_t* src)
    {
        assert(*src != nullptr);
        auto err = cudnnDestroy(*src);
        assert(err == CUDNN_STATUS_SUCCESS);
#ifdef NDEBUG
        UNUSED(err);
#endif
        delete src;
    });
#elif defined HIP_COMPILE
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
#endif
    return m_instance;
}

} } }
