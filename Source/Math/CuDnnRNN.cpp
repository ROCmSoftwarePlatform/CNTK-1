//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Matrix.h"
#include "GPUMatrix.h"
#include "TensorShape.h"
#include "TensorView.h"
#include <typeinfo>
#include <typeindex>
#include "CuDnnCommon.h"
#include "CuDnnRNN.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
class CuDnnTensorDescriptor
{
private:
#ifdef CUDA_COMPILE
    cudnnTensorDescriptor_t m_tensorDesc;
#elif defined HIP_COMPILE
    hipdnnTensorDescriptor_t m_tensorDesc;
#endif
public:
    CuDnnTensorDescriptor(size_t hiddenSize, size_t miniBatch, size_t numLayers) : m_tensorDesc(nullptr)
    {
#ifdef CUDA_COMPILE
	cudnnDataType_t m_dataType = CuDnnTensor::GetDataType<ElemType>();
        int dimA[3] = { (int)hiddenSize, (int)miniBatch, (int)numLayers };
        int strideA[3] = { 1, dimA[0], dimA[0] * dimA[1] };
        CUDNN_CALL(cudnnCreateTensorDescriptor(&m_tensorDesc));
	CUDNN_CALL(cudnnSetTensorNdDescriptor(m_tensorDesc, m_dataType, 3, dimA, strideA));
#elif defined HIP_COMPILE
        hipdnnDataType_t m_dataType = CuDnnTensor::GetDataType<ElemType>();
        int dimA[3] = { (int)hiddenSize, (int)miniBatch, (int)numLayers };
        int strideA[3] = { 1, dimA[0], dimA[0] * dimA[1] };
        HIPDNN_CALL(hipdnnCreateTensorDescriptor(&m_tensorDesc));
        HIPDNN_CALL(hipdnnSetTensorNdDescriptor(m_tensorDesc, m_dataType, 3, dimA, strideA));
#endif
    }

    ~CuDnnTensorDescriptor()
    {
#ifdef CUDA_COMPILE
	cudnnDestroyTensorDescriptor(m_tensorDesc);
#elif defined HIP_COMPILE
        hipdnnDestroyTensorDescriptor(m_tensorDesc);
#endif
    }

#ifdef CUDA_COMPILE
    operator cudnnTensorDescriptor_t() const
    {
        return m_tensorDesc;
    }
#elif defined HIP_COMPILE
    operator hipdnnTensorDescriptor_t() const
    {
        return m_tensorDesc;
    }
#endif

    DISABLE_COPY_AND_MOVE(CuDnnTensorDescriptor);
};

#ifdef CUDA_COMPILE
template <class ElemType>
void CuDnnRNNExecutor<ElemType>::SetDescriptors(size_t dim, const vector<size_t>& numSequencesForFrame, vector<cudnnTensorDescriptor_t>& descriptors)
{
    for (size_t i = 0; i < numSequencesForFrame.size(); i++)
    {
        if (descriptors.size() <= i)
        {
            descriptors.push_back(cudnnTensorDescriptor_t());
            CUDNN_CALL(cudnnCreateTensorDescriptor(&descriptors[i]));
        }
        // these dimensions are what CUDNN expects: (the minibatch dimension, the data dimension, and the number 1 (because each descriptor describes one frame of data)
        int dims[3] = { (int)numSequencesForFrame[i], (int)dim, 1 };
        int strides[3] = { dims[2] * dims[1], dims[2], 1 };
        CUDNN_CALL(cudnnSetTensorNdDescriptor(descriptors[i], m_dataType, 3, dims, strides));
    }
}
#elif defined HIP_COMPILE
template <class ElemType>
void CuDnnRNNExecutor<ElemType>::SetDescriptors(size_t dim, const vector<size_t>& numSequencesForFrame, vector<hipdnnTensorDescriptor_t>& descriptors)
{
    for (size_t i = 0; i < numSequencesForFrame.size(); i++)
    {
        if (descriptors.size() <= i)
        {
            descriptors.push_back(hipdnnTensorDescriptor_t());
            HIPDNN_CALL(hipdnnCreateTensorDescriptor(&descriptors[i]));
        }
        // these dimensions are what HIPDNN expects: (the minibatch dimension, the data dimension, and the number 1 (because each descriptor describes one frame of data)
        int dims[3] = { (int)numSequencesForFrame[i], (int)dim, 1 };
        int strides[3] = { dims[2] * dims[1], dims[2], 1 };
        HIPDNN_CALL(hipdnnSetTensorNdDescriptor(descriptors[i], m_dataType, 3, dims, strides));
    }
}
#endif

template <class ElemType>
void CuDnnRNNExecutor<ElemType>::ForwardCore(
    const GPUMatrix<ElemType>& weightsW,
    const GPUMatrix<ElemType>& inputX, GPUMatrix<ElemType>& outputY,
    const vector<size_t>& numSequencesForFrame,
    const RnnAttributes& rnnAttributes,
    GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace
    )
{
    // test that the RNN shape is correct
    if (!m_rnnT->IsCompatible(rnnAttributes))
        LogicError("RNN Layout has changed during processing");

    if (m_yDim != (m_rnnT->isBidirectional() ? 2 : 1) * m_rnnT->GetNumHidden())
        InvalidArgument("CuDnn ForwardCore: Output leading dimension must be twice hidden size for bidirectional networks");

    // set up the input and output descriptors
    SetDescriptors(m_xDim, numSequencesForFrame, xDesc);
    SetDescriptors(m_yDim, numSequencesForFrame, yDesc);

    // ensure workspace and reserve are large enough
    m_seqLength = numSequencesForFrame.size();
    size_t workSize;
    size_t reserveSize;

#ifdef CUDA_COMPILE
    // Need for every pass
    CUDNN_CALL(cudnnGetRNNWorkspaceSize(*m_cudnn, *m_rnnT, (int)m_seqLength, xDesc.data(), &workSize));
    // Only needed in training, can't be touched between passes.
    CUDNN_CALL(cudnnGetRNNTrainingReserveSize(*m_cudnn, *m_rnnT, (int)m_seqLength, xDesc.data(), &reserveSize));
#elif defined HIP_COMPILE
    // Need for every pass
    HIPDNN_CALL(hipdnnGetRNNWorkspaceSize(*m_cudnn, *m_rnnT, (int)m_seqLength, xDesc.data(), &workSize));
    // Only needed in training, can't be touched between passes.
    HIPDNN_CALL(hipdnnGetRNNTrainingReserveSize(*m_cudnn, *m_rnnT, (int)m_seqLength, xDesc.data(), &reserveSize));
#endif

    // convert from bytes to ElemType
    workSize = (workSize + sizeof(ElemType) - 1) / (sizeof(ElemType));
    reserveSize = (reserveSize + sizeof(ElemType) - 1) / sizeof(ElemType);

    reserve.Resize(reserveSize, 1);
    workspace.Resize(workSize, 1);

    wDesc = make_unique<CuDnnFilter<ElemType>>(*m_rnnT, xDesc[0]);
    if (wDesc->GetSize() != weightsW.GetNumElements())
        InvalidArgument("RNN needs %ld parameters, but %ld were allocated", wDesc->GetSize(), weightsW.GetNumElements());

#ifdef CUDA_COMPILE
    CUDNN_CALL(cudnnRNNForwardTraining(
        *m_cudnn, *m_rnnT,
        (int)m_seqLength,
        xDesc.data(), inputX.Data(),
        0, 0,
        0, 0,
        *wDesc, weightsW.Data(),
        yDesc.data(), outputY.Data(),
        0, 0,
        0, 0,
        workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
	reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
#elif defined HIP_COMPILE
    HIPDNN_CALL(hipdnnRNNForwardTraining(
        *m_cudnn, *m_rnnT,
        (int)m_seqLength,
        xDesc.data(), inputX.Data(),
        0, 0,
        0, 0,
        *wDesc, weightsW.Data(),
        yDesc.data(), outputY.Data(),
        0, 0,
        0, 0,
        workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
        reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
#endif
    m_BackwardDataCalledYet = false;
}

template <class ElemType>
void CuDnnRNNExecutor<ElemType>::BackwardDataCore(
    const GPUMatrix<ElemType>& outputY, const GPUMatrix<ElemType>& outputDY, const GPUMatrix<ElemType>& weightsW, GPUMatrix<ElemType>& dx,
    const RnnAttributes& rnnAttributes,
    GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace
    )
{
    // test that the RNN shape is correct
    if (!m_rnnT->IsCompatible(rnnAttributes))
        LogicError("RNN Layout has changed during processing");

    if (!m_BackwardDataCalledYet)
    {
#ifdef CUDA_COMPILE
	CUDNN_CALL(cudnnRNNBackwardData(
            *m_cudnn, *m_rnnT,
            (int)m_seqLength,
            yDesc.data(), outputY.Data(),
            yDesc.data(), outputDY.Data(),
            0, 0,
            0, 0,
            *wDesc, weightsW.Data(),
            0, 0,
            0, 0,
            xDesc.data(), dx.Data(),
            0, 0,
            0, 0,
            workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
	    reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
#elif defined HIP_COMPILE
        HIPDNN_CALL(hipdnnRNNBackwardData(
            *m_cudnn, *m_rnnT,
            (int)m_seqLength,
            yDesc.data(), outputY.Data(),
            yDesc.data(), outputDY.Data(),
            0, 0,
            0, 0,
            *wDesc, weightsW.Data(),
            0, 0,
            0, 0,
            xDesc.data(), dx.Data(),
            0, 0,
            0, 0,
            workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
            reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
#endif
    }
    m_BackwardDataCalledYet = true;
}

template <class ElemType>
void CuDnnRNNExecutor<ElemType>::BackwardWeightsCore(const GPUMatrix<ElemType>& inputX, const GPUMatrix<ElemType>& outputY, GPUMatrix<ElemType>& dw,
    const RnnAttributes& rnnAttributes,
    GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace
    )
{
    // test that the RNN shape is correct
    if (!m_rnnT->IsCompatible(rnnAttributes))
        LogicError("RNN Layout has changed during processing");
    if (!m_BackwardDataCalledYet)
        LogicError("out of order calling you have been very bad");
#ifdef CUDA_COMPILE
    CUDNN_CALL(cudnnRNNBackwardWeights(
        *m_cudnn, *m_rnnT,
        (int)m_seqLength,
        xDesc.data(), inputX.Data(),
        0, 0,
        yDesc.data(), outputY.Data(),
        workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
        *wDesc, dw.Data(),
	reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
#elif defined HIP_COMPILE
    HIPDNN_CALL(hipdnnRNNBackwardWeights(
        *m_cudnn, *m_rnnT,
        (int)m_seqLength,
        xDesc.data(), inputX.Data(),
        0, 0,
        yDesc.data(), outputY.Data(),
        workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
        *wDesc, dw.Data(),
        reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
#endif
}

template class CuDnnRNNExecutor<double>;
template class CuDnnRNNExecutor<float>;

} } }
