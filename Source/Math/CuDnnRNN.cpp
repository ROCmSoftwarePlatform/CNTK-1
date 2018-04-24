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
#include "half.hpp"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
class CuDnnTensorDescriptor
{
private:
    hipdnnTensorDescriptor_t m_tensorDesc;
public:
    CuDnnTensorDescriptor(size_t hiddenSize, size_t miniBatch, size_t numLayers) : m_tensorDesc(nullptr)
    {
        hipdnnDataType_t m_dataType = CuDnnTensor::GetDataType<ElemType>();
        int dimA[3] = { (int)hiddenSize, (int)miniBatch, (int)numLayers };
        int strideA[3] = { 1, dimA[0], dimA[0] * dimA[1] };
        HIPDNN_CALL(hipdnnCreateTensorDescriptor(&m_tensorDesc));
        HIPDNN_CALL(hipdnnSetTensorNdDescriptor(m_tensorDesc, m_dataType, 3, dimA, strideA));
    }

    ~CuDnnTensorDescriptor()
    {
        hipdnnDestroyTensorDescriptor(m_tensorDesc);
    }

    operator hipdnnTensorDescriptor_t() const
    {
        return m_tensorDesc;
    }

    DISABLE_COPY_AND_MOVE(CuDnnTensorDescriptor);
};

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

    // Need for every pass
    HIPDNN_CALL(hipdnnGetRNNWorkspaceSize(*m_cudnn, *m_rnnT, (int)m_seqLength, xDesc.data(), &workSize));
    // Only needed in training, can't be touched between passes.
    HIPDNN_CALL(hipdnnGetRNNTrainingReserveSize(*m_cudnn, *m_rnnT, (int)m_seqLength, xDesc.data(), &reserveSize));

    // convert from bytes to ElemType
    workSize = (workSize + sizeof(ElemType) - 1) / (sizeof(ElemType));
    reserveSize = (reserveSize + sizeof(ElemType) - 1) / sizeof(ElemType);

    reserve.Resize(reserveSize, 1);
    workspace.Resize(workSize, 1);

    wDesc = make_unique<CuDnnFilter<ElemType>>(*m_rnnT, xDesc[0]);
    if (wDesc->GetSize() != weightsW.GetNumElements())
        InvalidArgument("RNN needs %ld parameters, but %ld were allocated", wDesc->GetSize(), weightsW.GetNumElements());

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
    HIPDNN_CALL(hipdnnRNNBackwardWeights(
        *m_cudnn, *m_rnnT,
        (int)m_seqLength,
        xDesc.data(), inputX.Data(),
        0, 0,
        yDesc.data(), outputY.Data(),
        workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
        *wDesc, dw.Data(),
        reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
}

template class CuDnnRNNExecutor<double>;
template class CuDnnRNNExecutor<float>;
#ifdef __HIP_ENABLE_HALF__
template class CuDnnRNNExecutor<half>;
#endif //__HIP_ENABLE_HALF__
} } }
