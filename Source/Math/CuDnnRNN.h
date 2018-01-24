#pragma once

#include "Matrix.h"
#include "GPUMatrix.h"
#include "TensorShape.h"
#include <typeinfo>
#include <typeindex>
#include "CuDnnCommon.h"
#include "RNNCommon.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// The CUDNN RNN API requires a dropout descriptor for every RNN. It currently isn't used,
// so this wrapper creates a default descriptor and makes sure the lifetime of the object
// is managed properly.
class CuDnnDropout
{
    CuDnn::ptr_t m_cudnn;
    unsigned long long m_seed = 1;
public:
    CuDnnDropout(float dropout = 0.0f, unsigned long long seed = 1)
        : m_dropoutDesc(nullptr), m_cudnn(CuDnn::Instance())
    {
        HIPDNN_CALL(hipdnnCreateDropoutDescriptor(&m_dropoutDesc));
        size_t stateSize;
        void *states;
        HIPDNN_CALL(hipdnnDropoutGetStatesSize(*m_cudnn, &stateSize));

        // bugbug: possible leak. Does CuDnn release this for us?
        CUDA_CALL(hipMalloc(&states, stateSize));

        HIPDNN_CALL(hipdnnSetDropoutDescriptor(m_dropoutDesc,
            *m_cudnn,
            dropout,
            states,
            stateSize,
            seed));
    }

    ~CuDnnDropout()
    {
        if (m_dropoutDesc != nullptr)
        {
            hipdnnDestroyDropoutDescriptor(m_dropoutDesc);
        }
    }

    operator hipdnnDropoutDescriptor_t() const
    {
        return m_dropoutDesc;
    }

    DISABLE_COPY_AND_MOVE(CuDnnDropout);

private:
    hipdnnDropoutDescriptor_t m_dropoutDesc;
};
template <class ElemType>
class CuDnnRNN
{
private:
    hipdnnDataType_t m_dataType;
    hipdnnRNNDescriptor_t m_rnnDesc;
    CuDnnDropout m_dropout;
    RnnAttributes m_rnnAttributes;

    hipdnnRNNMode_t GetMode()
    {
        if      (m_rnnAttributes.m_recurrentOp == wstring(L"lstm"))    return hipdnnRNNMode_t::HIPDNN_LSTM;
        else if (m_rnnAttributes.m_recurrentOp == wstring(L"gru"))     return hipdnnRNNMode_t::HIPDNN_GRU;
        else if (m_rnnAttributes.m_recurrentOp == wstring(L"rnnReLU")) return hipdnnRNNMode_t::HIPDNN_RNN_RELU;
        else if (m_rnnAttributes.m_recurrentOp == wstring(L"rnnTanh")) return hipdnnRNNMode_t::HIPDNN_RNN_TANH;
        else InvalidArgument("Unknown cell type '%ls'. Supported values are 'lstm', 'gru', 'rnnReLU', 'rnnTanh'.", m_rnnAttributes.m_recurrentOp.c_str());
    }

public:
    CuDnnRNN(const RnnAttributes& rnnAttributes)
        : m_rnnDesc(nullptr), m_dropout(0.0f), m_rnnAttributes(rnnAttributes),
        m_dataType(CuDnnTensor::GetDataType<ElemType>())
    {
        HIPDNN_CALL(hipdnnCreateRNNDescriptor(&m_rnnDesc));
#if HIPDNN_VERSION >= 7000
        HIPDNN_CALL(hipdnnSetRNNDescriptor_v5(m_rnnDesc,
                  (int)m_rnnAttributes.m_hiddenSize,
                  (int)m_rnnAttributes.m_numLayers,
                  m_dropout,
                  HIPDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                  m_rnnAttributes.m_bidirectional ? HIPDNN_BIDIRECTIONAL : HIPDNN_UNIDIRECTIONAL,
                  GetMode(),
                  m_dataType));
#else
        HIPDNN_CALL(hipdnnSetRNNDescriptor(m_rnnDesc,
                  (int)m_rnnAttributes.m_hiddenSize,
                  (int)m_rnnAttributes.m_numLayers,
                  m_dropout,
                  HIPDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
                  m_rnnAttributes.m_bidirectional ? HIPDNN_BIDIRECTIONAL : HIPDNN_UNIDIRECTIONAL,
                  GetMode(),
                  m_dataType));
#endif
    }

    ~CuDnnRNN()
    {
        if (m_rnnDesc != nullptr)
        {
            hipdnnDestroyRNNDescriptor(m_rnnDesc);
            m_rnnDesc = nullptr;
        }
    }

    bool IsCompatible(const RnnAttributes& rnnAttributes) const
    {
        return this->m_rnnAttributes == rnnAttributes;
    }

    operator hipdnnRNNDescriptor_t() const
    {
        return m_rnnDesc;
    }

    bool isBidirectional() const { return m_rnnAttributes.m_bidirectional; }

    size_t GetNumLayers() { return m_rnnAttributes.m_numLayers; }
    size_t GetNumHidden() { return m_rnnAttributes.m_hiddenSize; }

    DISABLE_COPY_AND_MOVE(CuDnnRNN);
};

// The CUDNN RNN API describes the monolithic block of RNN parameters as a filter. This class
// wraps the concept of this filter, including the ability to calculate the necessary size
// based on the rnn descriptor.

template <class ElemType>
class CuDnnFilter
{
    hipdnnDataType_t m_dataType;
    CuDnn::ptr_t m_cudnn;
    size_t m_filterSize;
public:
    CuDnnFilter(const CuDnnRNN<ElemType>& rnn, const hipdnnTensorDescriptor_t& xDesc) :
        m_cudnn(CuDnn::Instance()), m_dataType(CuDnnTensor::GetDataType<ElemType>())
    {
        HIPDNN_CALL(hipdnnCreateFilterDescriptor(&m_filterDesc));
        try
        {
            size_t filterSize;
            HIPDNN_CALL(hipdnnGetRNNParamsSize(*m_cudnn, rnn, xDesc, &filterSize, m_dataType));

            size_t dataSize = 2; // HIPDNN_DATA_HALF

            if (m_dataType == hipdnnDataType_t::HIPDNN_DATA_DOUBLE)
                dataSize = 8;
            else if (m_dataType == hipdnnDataType_t::HIPDNN_DATA_FLOAT)
                dataSize = 4;

            // convert from bytes to items
            m_filterSize = (filterSize + dataSize - 1) / dataSize;
            int dimW[3] = { (int)m_filterSize, 1, 1 };
            HIPDNN_CALL(hipdnnSetFilterNdDescriptor(m_filterDesc, m_dataType, HIPDNN_TENSOR_NCHW, 3, dimW));
        }
        catch (exception e)
        {
            hipdnnDestroyFilterDescriptor(m_filterDesc);
            m_filterDesc = nullptr;
            throw e;
        }
    }
    ~CuDnnFilter()
    {
        assert(m_filterDesc != nullptr);
        hipdnnDestroyFilterDescriptor(m_filterDesc);
    }
    size_t GetSize() { return m_filterSize; }
    operator hipdnnFilterDescriptor_t() const
    {
        return m_filterDesc;
    }

    DISABLE_COPY_AND_MOVE(CuDnnFilter);

private:
    hipdnnFilterDescriptor_t m_filterDesc;
};

// CuDnnRNNExecutor holds the configuration and state for an instance of an RNN for CUDNN.
// It is generally attached to a GpuMatrix() object, and all calls to the RNN need to go through
// that object.

template <class ElemType>
class CuDnnRNNExecutor
{
    CuDnn::ptr_t m_cudnn;
    hipdnnDataType_t m_dataType;
    size_t m_xDim, m_yDim;
public:
    CuDnnRNNExecutor(size_t xDim, size_t yDim, const RnnAttributes& rnnAttributes ) :
        m_cudnn(CuDnn::Instance()),
        m_xDim(xDim), m_yDim(yDim),
        m_seqLength(0),
        m_dataType(CuDnnTensor::GetDataType<ElemType>()),
        m_BackwardDataCalledYet(false)
    {
        m_rnnT = std::make_unique<CuDnnRNN<ElemType>>(rnnAttributes);
    }

    void ForwardCore(const GPUMatrix<ElemType>& weightsW, const GPUMatrix<ElemType>& inputX, GPUMatrix<ElemType>& outputY, const vector<size_t>& numSequencesForFrame, const RnnAttributes& rnnAttributes, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace);
    void BackwardWeightsCore(const GPUMatrix<ElemType>& inputX, const GPUMatrix<ElemType>& outputY, GPUMatrix<ElemType>& dw, const RnnAttributes& rnnAttributes, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace);
    void BackwardDataCore(const GPUMatrix<ElemType>& outputY, const GPUMatrix<ElemType>& outputDY, const GPUMatrix<ElemType>& w, GPUMatrix<ElemType>& dx, const RnnAttributes& rnnAttributes, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace);

protected:
    std::unique_ptr<CuDnnFilter<ElemType>> wDesc;
    vector<hipdnnTensorDescriptor_t> xDesc;
    vector<hipdnnTensorDescriptor_t> yDesc;

private:
    static ElemType* ptr(GPUMatrix<ElemType>& src)
    {
        return src.Data();
    }
    static const ElemType* ptr(const GPUMatrix<ElemType>& src)
    {
        return src.Data();
    }

    void SetDescriptors(size_t dim, const vector<size_t>& numSequencesForFrame, vector<hipdnnTensorDescriptor_t>& descriptors);

private:
    std::unique_ptr<CuDnnRNN<ElemType>> m_rnnT;
    bool m_BackwardDataCalledYet;
    size_t m_seqLength;
};

} } }
