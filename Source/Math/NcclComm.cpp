//
// Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "NcclComm.h"

#ifdef USE_NCCL
#include "GPUMatrix.h"
#ifdef CUDA_COMPILE
#include <nccl.h>
#include <cuda_runtime.h>
#elif defined HIP_COMPILE
#include <hip/hip_runtime_api.h>
#ifdef __HIP_PLATFORM_NVCC__
#include <nccl.h>
#endif
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

#ifdef CUDA_COMPILE
// allows to write cudaFunction() || "error"   (CUDA runtime)
static void operator||(cudaError_t rc, const char *msg)
{
    if (rc != cudaSuccess)
        RuntimeError("%s: %s (cuda error %d)", msg, cudaGetErrorString(rc), (int) rc);
}
#elif defined HIP_COMPILE
// allows to write hipFunction() || "error"   (HIP runtime)
static void operator||(hipError_t rc, const char *msg)
{
    if (rc != hipSuccess)
        RuntimeError("%s: %s (hip error %d)", msg, hipGetErrorString(rc), (int) rc);
}
#endif

ncclRedOp_t ncclRedOpFromMpiOp(MPI_Op op)
{
    if (op == MPI_SUM) return ncclSum;
    else if (op == MPI_MAX) return ncclMax;
    else if (op == MPI_MIN) return ncclMin;
    else if (op == MPI_PROD) return ncclProd;
    else RuntimeError("Invalid MPI_Op");
}

ncclRedOp_t ncclRedOpFromMpiOp(MPI_Op op)
{
    if (op == MPI_SUM) return ncclSum;
    else if (op == MPI_MAX) return ncclMax;
    else if (op == MPI_MIN) return ncclMin;
    else if (op == MPI_PROD) return ncclProd;
    else RuntimeError("Invalid MPI_Op");
}

NcclComm::NcclComm(int deviceId, const MPIWrapperPtr& mpi)
    : m_ncclComm(nullptr), m_stream(nullptr)
{
#ifdef CUDA_COMPILE
    cudaDeviceSynchronize();
#elif defined HIP_COMPILE
    hipDeviceSynchronize();
#endif
    size_t numRanks = mpi->NumNodesInUse();
    std::vector<int> allDevs(numRanks);
    std::vector<std::array<char, MPI_MAX_PROCESSOR_NAME>> allHosts(numRanks);
    std::array<char, MPI_MAX_PROCESSOR_NAME> procName {};
    int nameLen;
    MPI_Get_processor_name(procName.data(), &nameLen);
    mpi->Allgather(&deviceId, 1, MPI_INT, allDevs.data(), 1, MPI_INT);
    mpi->Allgather(procName.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allHosts[0].data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR);

    for (size_t r = 0; r < numRanks; r++)
    {
        if (allDevs[r] == CPUDEVICE)
        {
            fprintf(stderr, "NcclComm: disabled, at least one rank using CPU device\n");
            return;
        }
        for (size_t s = 0; s < r; s++)
        {
            if (allHosts[r] == allHosts[s] && allDevs[r] == allDevs[s])
            {
                fprintf(stderr, "NcclComm: disabled, same device used by more than one rank\n");
                return;
            }
        }
    }

    ncclUniqueId ncclId;
    ncclResult_t res;

    if (mpi->IsMainNode())
    {
        res = ncclGetUniqueId(&ncclId);
        if (res != ncclSuccess)
            RuntimeError("NcclComm failed to obtain ncclUniqueId: %s", ncclGetErrorString(res));
    }

    mpi->Bcast(&ncclId, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0);

    PrepareDevice(deviceId);
    res = ncclCommInitRank(&m_ncclComm, numRanks, ncclId, mpi->CurrentNodeRank());
    if (res != ncclSuccess)
        RuntimeError("NcclComm failed to initialize: %s. Set the ENV \"NCCL_DEBUG=INFO\" for more information.", ncclGetErrorString(res));

#ifdef CUDA_COMPILE
    cudaStreamCreateWithFlags(&m_stream, cudaStreamDefault)
        || "cudaStreamCreateWithFlags failed";
    fprintf(stderr, "NcclComm: initialized\n");
#elif defined HIP_COMPILE
    hipStreamCreateWithFlags(&m_stream, hipStreamDefault)
        || "hipStreamCreateWithFlags failed";
    fprintf(stderr, "NcclComm: initialized\n");
#endif
}

NcclComm::~NcclComm()
{
#ifdef CUDA_COMPILE
    if (m_stream != nullptr)
        cudaStreamDestroy(m_stream);
#elif defined HIP_COMPILE
    if (m_stream != nullptr)
        hipStreamDestroy(m_stream);
#endif
    if (m_ncclComm != nullptr)
        ncclCommDestroy(m_ncclComm);
}

bool NcclComm::IsSupported()
{
    return m_ncclComm != nullptr;
}

void NcclComm::AllReduceImpl(void* inputbuffer, void *outputbuffer, size_t count, DataType dtype, MPI_Op op)
{
    ncclResult_t res;
    class NcclTypeLookup
    {
        ncclDataType_t ncclTypes[(int)DataType::COUNT];
    public:
        NcclTypeLookup()
        {
            ncclTypes[(int)DataType::FLOAT]  = ncclFloat;
            ncclTypes[(int)DataType::DOUBLE] = ncclDouble;
            ncclTypes[(int)DataType::INT]    = ncclInt;
        }
        ncclDataType_t Lookup(DataType dtype)
        {
            return ncclTypes[(int)dtype];
        }
    };

    static NcclTypeLookup s_ncclTypeLookup;

    res = ncclAllReduce(inputbuffer, outputbuffer, count, s_ncclTypeLookup.Lookup(dtype), ncclRedOpFromMpiOp(op), m_ncclComm, m_stream);

    if (res != ncclSuccess)
        RuntimeError("NcclComm ncclAllReduce failed: %s", ncclGetErrorString(res));
}

void NcclComm::BroadcastImpl(void* buffer, size_t count, MPI_Datatype dtype, int root)
{
    ncclResult_t res;
    if (dtype == MPI_CHAR)
    {
        res = ncclBcast(buffer, count, ncclChar, root, m_ncclComm, m_stream);
    }
    else
    {
        RuntimeError("NcclComm Broadcast supports Char type only");
    }
    if (res != ncclSuccess)
    {
        RuntimeError("NcclComm ncclBcast failed: %s", ncclGetErrorString(res));
    }
}

void NcclComm::Sync()
{
#ifdef CUDA_COMPILE
    cudaStreamSynchronize(m_stream) || "NcclComm: cudaStreamSynchronize failed";
#elif defined HIP_COMPILE
    hipStreamSynchronize(m_stream) || "NcclComm: hipStreamSynchronize failed";
#endif
}

}}} // end namespaces

#else // !USE_NCCL
namespace Microsoft { namespace MSR { namespace CNTK {

NcclComm::NcclComm(int /*deviceId*/, const MPIWrapperPtr& /*mpi*/) { }

NcclComm::~NcclComm() { }

bool NcclComm::IsSupported()
{
    return false;
}

void NcclComm::Sync() { }

}}} // end namespaces
#endif
