//
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// Make generic operators for floating point types
/* This file contains:
   Generalized library calls
   kernels to be called for not supported data type
*/
// NV_TODO: optimize speed -- pass things needed in, optimize kernel speed, add half2
// NV_TODO: investigate cub support for half

#pragma once


#ifndef CPUONLY

#include "hip/hip_runtime.h"
#include <hipblas.h>
#include <hipsparse.h>
#include <hipdnn.h>
#include <hiprand.h>
#include <hiprand_kernel.h>
#include <time.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100) // 'identifier': unreferenced formal parameter
#pragma warning(disable : 4127) // conditional expression is constant
#pragma warning(disable : 4201) // nonstandard extension used: nameless struct/union
#pragma warning(disable : 4458) // declaration of 'identifier' hides class member
#pragma warning(disable : 4515) // 'namespace': namespace uses itself
#pragma warning(disable : 4706) // assignment within conditional expression
#endif
#include <hipcub/hipcub.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include "half.hpp"
#define TRANS_TILE_DIM 32
#define BLOCK_ROWS 8
#define COPY_TILE_DIM 1024
#define COPY_BLOCK_DIM 256
#define WAVE_SIZE 64
#define GROUP_SIZE 256

typedef hiprandState_t hiprandState;

// kernel(s) for half functions with no library support
namespace {
__global__ void transposeNoOverlap(half *odata, const half *idata, const int m, const int n)
{
    __shared__ half tile[TRANS_TILE_DIM][TRANS_TILE_DIM+1];

    int x = blockIdx.x * TRANS_TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TRANS_TILE_DIM + threadIdx.y;

    for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*m + x];

    __syncthreads();

    x = blockIdx.y * TRANS_TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TRANS_TILE_DIM + threadIdx.y;

    if(x >= n) return;

    for (int j = 0; j < TRANS_TILE_DIM; j += BLOCK_ROWS){
        if((y+j) >= m) return;
        odata[(y+j)*n + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
// set up hiprand state, need to move up layer to remove calling for each generate call
__global__ void setup_state(hiprandState *state, unsigned long long seed)
{
    hiprand_init(seed, 0, 0, state);
}

__global__ void GenerateUniformHalf(hiprandState *state, half *result, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;

    hiprandState localState = *state;

    float x;
    skipahead((unsigned long long)id, &localState);
    x = hiprand_uniform(&localState);

    result[id] = x;
    if(id == n-1) *state = localState;
}

__global__ void GenerateNormalHalf(hiprandState *state, half *result, int n, half mean, half stddev)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= n) return;

    hiprandState localState = *state;

    float x;
    skipahead((unsigned long long)id, &localState);
    x = hiprand_normal(&localState);

    result[id] = (float)mean + (float)stddev * x;
    if(id == n-1) *state = localState;
}


__global__ void helperCopyHalf2Float(float *f, const half *h, const int n)
{
    int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (id >= n)
        return;
    f[id] = (float)h[id];
}

__global__ void helperCopyFloat2Half(half *h, const float *f, const int n)
{
    int id = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (id >= n)
        return;
    h[id] = (half)f[id];
}
// kernels can convert matrix between half and float. speed currently not optimized, may need to add half2
/*
__global__ void copyHalf2Float(float *odata, const half *idata, const int n)
{
    float tmp[COPY_TILE_DIM/COPY_BLOCK_DIM];

    int x = blockIdx.x * COPY_TILE_DIM + threadIdx.x;

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        tmp[j] = (float) idata[x + j*COPY_BLOCK_DIM];

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        if(x + j*COPY_BLOCK_DIM < n) odata[x + j*COPY_BLOCK_DIM] = tmp[j];
}

__global__ void copyFloat2Half(half *odata, const float *idata, const int n)
{
    float tmp[COPY_TILE_DIM/COPY_BLOCK_DIM];

    int x = blockIdx.x * COPY_TILE_DIM + threadIdx.x;

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        tmp[j] = idata[x + j*COPY_BLOCK_DIM];

    for (int j = 0; j < COPY_TILE_DIM/COPY_BLOCK_DIM; j++)
        if(x + j*COPY_BLOCK_DIM < n) odata[x + j*COPY_BLOCK_DIM] = tmp[j];
}
*/

}
__global__ void transform_csc_2_dense_kernel(ulong size,
                       const int *col_offsets,
                       const int *row_indices,
                       const float *values,
                       const int num_rows,
                       const int num_cols,
                       float *A, int subwave_size)
{
        const int global_id   = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
        const int local_id    = hipThreadIdx_x;
        const int thread_lane = local_id & (subwave_size - 1);
        const int vector_id   = global_id / subwave_size;
        const int num_vectors = 256 / subwave_size;
        for(int col = vector_id; col < num_cols; col += num_vectors)
        {
            const int col_start = col_offsets[col];
            const int col_end   = col_offsets[col+1];
            for(int j = col_start + thread_lane; j < col_end; j += subwave_size)
                A[row_indices[j] + num_rows * col] = values[j];
        }
}

inline void hipScsc2dense(int m, int n, const float *cscValA, const int *cscRowIndA, const int *cscColPtrA, float *A) {

    hipMemset(A, 0, sizeof(float) *m * n);

    int blocks = ((n-1)/256)+1;
    int subwave_size = WAVE_SIZE;
    ulong elements_per_col = (n * m) / n; // assumed number elements per col;

    if (elements_per_col < 64) {  subwave_size = 32;  }
    if (elements_per_col < 32) {  subwave_size = 16;  }
    if (elements_per_col < 16) {  subwave_size = 8;  }
    if (elements_per_col < 8)  {  subwave_size = 4;  }
    if (elements_per_col < 4)  {  subwave_size = 2;  }

    hipLaunchKernelGGL(transform_csc_2_dense_kernel, blocks, 256, 0, 0 ,(ulong)m * n, cscColPtrA, cscRowIndA, cscValA, m, n, A, subwave_size);
}

__global__ void transform_csc_2_dense_kernel(ulong size,
                       const int *col_offsets,
                       const int *row_indices,
                       const double *values,
                       const int num_rows,
                       const int num_cols,
                       double *A, int subwave_size)
{
        const int global_id   = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
        const int local_id    = hipThreadIdx_x;
        const int thread_lane = local_id & (subwave_size - 1);
        const int vector_id   = global_id / subwave_size;
        const int num_vectors = 256 / subwave_size;
        for(int col = vector_id; col < num_cols; col += num_vectors)
        {
            const int col_start = col_offsets[col];
            const int col_end   = col_offsets[col+1];
            for(int j = col_start + thread_lane; j < col_end; j += subwave_size)
                A[row_indices[j] + num_rows * col] = values[j];
        }
}

inline void hipScsc2dense(int m, int n, const double *cscValA, const int *cscRowIndA, const int *cscColPtrA, double *A) {

    hipMemset(A, 0, sizeof(double) *m * n);

    int blocks = ((n-1)/256)+1;
    int subwave_size = WAVE_SIZE;
    ulong elements_per_col = (n * m) / n; // assumed number elements per col;

    if (elements_per_col < 64) {  subwave_size = 32;  }
    if (elements_per_col < 32) {  subwave_size = 16;  }
    if (elements_per_col < 16) {  subwave_size = 8;  }
    if (elements_per_col < 8)  {  subwave_size = 4;  }
    if (elements_per_col < 4)  {  subwave_size = 2;  }

    hipLaunchKernelGGL(transform_csc_2_dense_kernel, blocks, 256, 0, 0 ,(ulong)m * n, cscColPtrA, cscRowIndA, cscValA, m, n, A, subwave_size);
}


// Generalize library calls to be use in template functions

// gemm
inline hipblasStatus_t hipblasgemmHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
{
    return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline hipblasStatus_t hipblasgemmHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double* alpha, const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc)
{
    return hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
inline hipblasStatus_t hipblasgemmHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const half* alpha, const half* A, int lda, const half* B, int ldb, const half* beta, half* C, int ldc)
{
    // This does true FP16 computation which is slow for non-Volta GPUs
    //return hipblasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // This does pseudo FP16 computation (input/output in fp16, computation in fp32)
    float h_a = *alpha;
    float h_b = *beta;
    //hipblasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); //TODO:PRAS_2.4
    //return hipblasGemmEx(handle, transa, transb, m, n, k, &h_a, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &h_b, C, CUDA_R_16F, ldc, CUDA_R_32F, CUBLAS_GEMM_DFALT); //TODO:PRAS_2.4
    return hipblasHgemm(handle, transa, transb, m, n, k, reinterpret_cast<const hipblasHalf*>(alpha), reinterpret_cast<hipblasHalf*>(const_cast<half*>(A)), lda, reinterpret_cast<hipblasHalf*>(const_cast<half*>(B)), ldb, reinterpret_cast<const hipblasHalf*>(beta), reinterpret_cast<hipblasHalf*>(C), ldc);
}

// batched gemm
inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const float* alpha, const float *Aarray[], int lda, const float *Barray[], int ldb, const float *beta, float *Carray[], int ldc, int batchCount)
{
    return hipblasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const double* alpha, const double *Aarray[], int lda, const double *Barray[], int ldb, const double *beta, double *Carray[], int ldc, int batchCount)
{
    return hipblasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

inline hipblasStatus_t hipblasGemmBatchedHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, int k, const half* alpha, const half *Aarray[], int lda, const half *Barray[], int ldb, const half *beta, half *Carray[], int ldc, int batchCount)
{
    return HIPBLAS_STATUS_NOT_SUPPORTED;//hipblasHgemmBatched(handle, transa, transb, m, n, k, alpha, (const __half**)Aarray, lda, (const __half**)Barray, ldb, beta, (__half**)Carray, ldc, batchCount);/TODO:PRAS_2.4
}


// axpy
inline hipblasStatus_t hipblasaxpyHelper(hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy)
{
    return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}
inline hipblasStatus_t hipblasaxpyHelper(hipblasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy)
{
    return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}
inline hipblasStatus_t hipblasaxpyHelper(hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy, int a, int b)
{
    return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}
inline hipblasStatus_t hipblasaxpyHelper(hipblasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy, int a, int b)
{
    return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}

inline hipblasStatus_t hipblasaxpyHelper(hipblasHandle_t handle, int n, const half* alpha, const half* x, int incx, half* y, int incy)
{
    float tmp_alpha = *alpha;

    float *df_x, *df_y;
    hipMalloc(&df_x, n * sizeof(float));
    hipMalloc(&df_y, n * sizeof(float));

    int blocks = ((n-1)/256)+1;
    
    hipLaunchKernelGGL((helperCopyHalf2Float), dim3(blocks), dim3(256), 0, 0, df_x, x, n);
    hipLaunchKernelGGL((helperCopyHalf2Float), dim3(blocks), dim3(256), 0, 0, df_y, y, n);

    hipblasStatus_t status;
    status = hipblasSaxpy(handle, n, (const float*)&tmp_alpha, (const float*)df_x, incx, df_y, incy);

    if(status == HIPBLAS_STATUS_SUCCESS)
        hipLaunchKernelGGL((helperCopyFloat2Half), dim3(blocks), dim3(256), 0, 0, y, df_y, n);

    return status;
}
inline hipblasStatus_t hipblasaxpyHelper(hipblasHandle_t handle, int n, const half* alpha, const half* x, int incx, half* y, int incy, int num_x, int num_y) //TODO:PRAS_2.4
{
    float tmp_alpha = *alpha;

    float *df_x, *df_y;
    hipMalloc(&df_x, num_x * sizeof(float));
    hipMalloc(&df_y, num_y * sizeof(float));

    int blocks = ((n-1)/256)+1;

    hipLaunchKernelGGL((helperCopyHalf2Float), dim3(blocks), dim3(256), 0, 0, df_x, x, num_x);
    hipLaunchKernelGGL((helperCopyHalf2Float), dim3(blocks), dim3(256), 0, 0, df_y, y, num_y);

    hipblasStatus_t status;
    status = hipblasSaxpy(handle, n, (const float*)&tmp_alpha, (const float*)df_x, incx, df_y, incy);

    if(status == HIPBLAS_STATUS_SUCCESS)
        hipLaunchKernelGGL((helperCopyFloat2Half), dim3(1024), dim3(256), 0, 0, y, df_y, num_y);

    return status;
}


// transpose using geam
inline hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, float *alpha, float *A, int lda, float *beta, float *B, int ldb, float *C, int ldc)
{
    return hipblasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb, int m, int n, double *alpha, double *A, int lda, double *beta, double *B, int ldb, double *C, int ldc)
{
    return hipblasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

inline hipblasStatus_t hipblasTransposeHelper(hipblasHandle_t, hipblasOperation_t, hipblasOperation_t, int m, int n, half *, half *A, int, half *, half *, int, half *C, int)
{
    if(C != A)
    {
        dim3 dimGrid((n+TRANS_TILE_DIM-1)/TRANS_TILE_DIM, (m+TRANS_TILE_DIM-1)/TRANS_TILE_DIM, 1);
        dim3 dimBlock(TRANS_TILE_DIM, BLOCK_ROWS, 1);

        hipLaunchKernelGGL((transposeNoOverlap), dim3(dimGrid), dim3(dimBlock), 0, 0, C, A, n, m);
    }
    else
        RuntimeError("In place transpose(half) not supported."); // hipblas do not support this either. There might be bug if this actually get called.
    return (hipblasStatus_t) 0;
}


// asum
inline hipblasStatus_t hipblasasumHelper(hipblasHandle_t handle, int n, const float *x, int incx, float *result)
{
    return hipblasSasum(handle, n, x, incx, result);
}
inline hipblasStatus_t hipblasasumHelper(hipblasHandle_t handle, int n, const double *x, int incx, double *result)
{
    return hipblasDasum(handle, n, x, incx, result);
}

inline hipblasStatus_t hipblasasumHelper(hipblasHandle_t, int n, const half *x, int incx, half *result)
{
    // pass in hipdnn handle/descriptor to remove overhead?
    hipdnnHandle_t hipdnnHandle;
    hipdnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    hipdnnReduceTensorDescriptor_t reduceTensorDesc;

    hipdnnCreate(&hipdnnHandle);
    hipdnnCreateTensorDescriptor(&srcTensorDesc);
    hipdnnCreateTensorDescriptor(&dstTensorDesc);
    hipdnnCreateReduceTensorDescriptor(&reduceTensorDesc);

    hipdnnSetTensor4dDescriptorEx(srcTensorDesc, HIPDNN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
    hipdnnSetTensor4dDescriptorEx(dstTensorDesc, HIPDNN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
    hipdnnSetReduceTensorDescriptor(reduceTensorDesc,
                                   HIPDNN_REDUCE_TENSOR_NORM1,
                                   HIPDNN_DATA_FLOAT,
                                   HIPDNN_NOT_PROPAGATE_NAN,
                                   HIPDNN_REDUCE_TENSOR_NO_INDICES,
                                   HIPDNN_32BIT_INDICES);

    void *workspace = NULL;
    size_t workspaceSizeInBytes = 0;
    hipdnnGetReductionWorkspaceSize(hipdnnHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
    if(workspaceSizeInBytes > 0) hipMalloc(&workspace, workspaceSizeInBytes);

    float alpha = 1.0f;
    float beta = 0.0f;

    void *d_res;
    hipMalloc(&d_res, sizeof(half));

    hipdnnReduceTensor(hipdnnHandle,
                      reduceTensorDesc,
                      NULL,
                      0,
                      workspace,
                      workspaceSizeInBytes,
                      &alpha,
                      srcTensorDesc,
                      (void*)x,
                      &beta,
                      dstTensorDesc,
                      d_res);

    hipMemcpy((void *)result, d_res, sizeof(half), hipMemcpyDeviceToHost);

    hipdnnDestroyReduceTensorDescriptor(reduceTensorDesc);
    hipdnnDestroyTensorDescriptor(srcTensorDesc);
    hipdnnDestroyTensorDescriptor(dstTensorDesc);
    hipdnnDestroy(hipdnnHandle);
    hipFree(d_res);
    hipFree(workspace);

    return (hipblasStatus_t) 0;
}


// amax
inline hipblasStatus_t hipblasamaxHelper(hipblasHandle_t handle, int n, const float *x, int incx, int *result)
{
    return hipblasIsamax(handle, n, x, incx, result);
}
inline hipblasStatus_t hipblasamaxHelper(hipblasHandle_t handle, int n, const double *x, int incx, int *result)
{
    return hipblasIdamax(handle, n, x, incx, result);
}

inline hipblasStatus_t hipblasamaxHelper(hipblasHandle_t, int n, const half *x, int incx, int *result)
{
    unsigned int h_result_uint = 0;
    // pass in hipdnn handle/descriptor to remove overhead?
    hipdnnHandle_t hipdnnHandle;
    hipdnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    hipdnnReduceTensorDescriptor_t reduceTensorDesc;

    hipdnnCreate(&hipdnnHandle);
    hipdnnCreateTensorDescriptor(&srcTensorDesc);
    hipdnnCreateTensorDescriptor(&dstTensorDesc);
    hipdnnCreateReduceTensorDescriptor(&reduceTensorDesc);

    hipdnnSetTensor4dDescriptorEx(srcTensorDesc, HIPDNN_DATA_HALF, 1, 1, 1, n, 1, 1, 1, incx);
    hipdnnSetTensor4dDescriptorEx(dstTensorDesc, HIPDNN_DATA_HALF, 1, 1, 1, 1, 1, 1, 1, 1);
    hipdnnSetReduceTensorDescriptor(reduceTensorDesc,
                                   HIPDNN_REDUCE_TENSOR_AMAX,
                                   HIPDNN_DATA_FLOAT,
                                   HIPDNN_NOT_PROPAGATE_NAN,
                                   HIPDNN_REDUCE_TENSOR_FLATTENED_INDICES,
                                   HIPDNN_32BIT_INDICES);

    void *workspace = NULL;
    size_t workspaceSizeInBytes = 0;
    hipdnnGetReductionWorkspaceSize(hipdnnHandle, reduceTensorDesc, srcTensorDesc, dstTensorDesc, &workspaceSizeInBytes);
    if(workspaceSizeInBytes > 0) hipMalloc(&workspace, workspaceSizeInBytes);

    float alpha = 1.0f;
    float beta = 0.0f;
    void *d_max;
    hipMalloc(&d_max, sizeof(half));
    void *d_result_uint;
    hipMalloc(&d_result_uint, sizeof(unsigned int));

    hipdnnReduceTensor(hipdnnHandle,
                      reduceTensorDesc,
                      d_result_uint,
                      sizeof(unsigned int),
                      workspace,
                      workspaceSizeInBytes,
                      &alpha,
                      srcTensorDesc,
                      (void*)x,
                      &beta,
                      dstTensorDesc,
                      d_max);

    hipMemcpy(&h_result_uint, d_result_uint, sizeof(unsigned int), hipMemcpyDeviceToHost);

    hipdnnDestroyReduceTensorDescriptor(reduceTensorDesc);
    hipdnnDestroyTensorDescriptor(srcTensorDesc);
    hipdnnDestroyTensorDescriptor(dstTensorDesc);
    hipdnnDestroy(hipdnnHandle);
    hipFree(workspace);
    hipFree(d_max);
    hipFree(d_result_uint);

    *result = (int) h_result_uint;
    return (hipblasStatus_t) 0;
}


// scal
inline hipblasStatus_t hipblasscalHelper(hipblasHandle_t handle, int n, const float *alpha, float *x, int incx)
{
    return hipblasSscal(handle, n, alpha, x, incx);
}
inline hipblasStatus_t hipblasscalHelper(hipblasHandle_t handle, int n, const double *alpha, double *x, int incx)
{
    return hipblasDscal(handle, n, alpha, x, incx);
}

inline hipblasStatus_t hipblasscalHelper(hipblasHandle_t handle, int n, const half *alpha, half *x, int incx)
{
    float tmp_alpha = *alpha;
    float *float_x;
    hipMalloc(&float_x, n * sizeof(float));

    int blocks = ((n-1)/256) + 1;
    hipLaunchKernelGGL((helperCopyHalf2Float), dim3(blocks), dim3(256), 0, 0, float_x, x, n);

    hipblasStatus_t status;
    status = hipblasSscal(handle, n, (const float*)&tmp_alpha, float_x, incx);

    if(status == HIPBLAS_STATUS_SUCCESS)
        hipLaunchKernelGGL((helperCopyFloat2Half), dim3(blocks), dim3(256), 0, 0, x, float_x, n);

    return status;

    //return hipblasScalEx(handle, n, (void*)&tmp_alpha, CUDA_R_32F, (void*)x, CUDA_R_16F, incx, CUDA_R_32F); TODO:PRAS_2.4
}


inline hipblasStatus_t hipblasscalHelper(hipblasHandle_t,int,const char *,char *, int)
{
    RuntimeError("Unsupported template argument(char) in hipblas_scal");
}
inline hipblasStatus_t hipblasscalHelper(hipblasHandle_t,int,const short *,short *, int)
{
    RuntimeError("Unsupported template argument(short) in hipblas_scal");
}

// dot
inline hipblasStatus_t hipblasdotHelper(hipblasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result)
{
    return hipblasSdot(handle, n, x, incx, y, incy, result);
}
inline hipblasStatus_t hipblasdotHelper(hipblasHandle_t handle, int n, const double *x, int incx, const double *y, int incy, double *result)
{
    return hipblasDdot(handle, n, x, incx, y, incy, result);
}

inline hipblasStatus_t hipblasdotHelper(hipblasHandle_t handle, int n, const half *x, int incx, const half *y, int incy, half *result)
{
    float *float_x, *float_y, *float_result;
    hipMalloc(&float_x, n * sizeof(float));
    hipMalloc(&float_y, n * sizeof(float));
    hipMalloc(&float_result, n * sizeof(float));

    int blocks = ((n-1)/256) + 1;

    hipLaunchKernelGGL((helperCopyHalf2Float), dim3(blocks), dim3(256), 0, 0, float_x, x, n);
    hipLaunchKernelGGL((helperCopyHalf2Float), dim3(blocks), dim3(256), 0, 0, float_y, y, n);

    hipblasStatus_t status;
    status = hipblasSdot(handle, n, float_x, incx, float_y, incy, float_result);
    if(status == HIPBLAS_STATUS_SUCCESS)
        hipLaunchKernelGGL((helperCopyFloat2Half), dim3(blocks), dim3(256), 0, 0, result, float_result, n);

    return status;

    //return hipblasDotEx(handle, n, (void*)x, CUDA_R_16F, incx, (void*)y, CUDA_R_16F, incy, (void*)result, CUDA_R_16F, CUDA_R_32F); TODO:PRAS_2.4
}

// hiprand
inline hiprandStatus_t hiprandGenerateUniformHelper(hiprandGenerator_t generator, float *outputPtr, size_t num)
{
    return hiprandGenerateUniform(generator, outputPtr, num);
}
inline hiprandStatus_t hiprandGenerateUniformHelper(hiprandGenerator_t generator, double *outputPtr, size_t num)
{
    return hiprandGenerateUniformDouble(generator, outputPtr, num);
}

inline hiprandStatus_t hiprandGenerateUniformHelper(hiprandGenerator_t, half *outputPtr, size_t num)
{
    hiprandState *devStates;
    hipMalloc((void **)&devStates, sizeof(hiprandState));
    hipLaunchKernelGGL((setup_state), dim3(1), dim3(1), 0, 0, devStates, time(NULL)); // What does hiprandGenerateUniform actually doing? should also pass in state here

    dim3 dimGrid((unsigned int)(num+COPY_BLOCK_DIM-1)/COPY_BLOCK_DIM, 1, 1);
    dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
    hipLaunchKernelGGL((GenerateUniformHalf), dim3(dimGrid), dim3(dimBlock), 0, 0, devStates, outputPtr, (int)num);

    return (hiprandStatus_t) 0;
}


inline hiprandStatus_t hiprandGenerateNormalHelper(hiprandGenerator_t generator, float *outputPtr, size_t n, float mean, float stddev)
{
    return hiprandGenerateNormal(generator, outputPtr, n, mean, stddev);
}
inline hiprandStatus_t hiprandGenerateNormalHelper(hiprandGenerator_t generator, double *outputPtr, size_t n, double mean, double stddev)
{
    return hiprandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}


inline hiprandStatus_t hiprandGenerateNormalHelper(hiprandGenerator_t, half *outputPtr, size_t n, half mean, half stddev)
{
    hiprandState *devStates;
    hipMalloc((void **)&devStates, sizeof(hiprandState));
    hipLaunchKernelGGL((setup_state), dim3(1), dim3(1), 0, 0, devStates, time(NULL)); // What does hiprandGenerateUniform actually doing? should also pass in state here

    dim3 dimGrid((unsigned int)(n+COPY_BLOCK_DIM-1)/COPY_BLOCK_DIM, 1, 1);
    dim3 dimBlock(COPY_BLOCK_DIM, 1, 1);
    hipLaunchKernelGGL((GenerateNormalHalf), dim3(dimGrid), dim3(dimBlock), 0, 0, devStates, outputPtr, (int)n, mean, stddev);

    return (hiprandStatus_t) 0;
}


// hipsparse
inline hipsparseStatus_t hipsparsecsr2denseHelper(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, float *A, int lda)
{
    return hipsparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
}
inline hipsparseStatus_t hipsparsecsr2denseHelper(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, double *A, int lda)
{
    return hipsparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, csrColIndA, A, lda);
}

inline hipsparseStatus_t hipsparsecsr2denseHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const half *, const int *, const int *, half *, int)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}

inline hipsparseStatus_t hipsparsecsr2denseHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const short *, const int *, const int *, short *, int)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline hipsparseStatus_t hipsparsecsr2denseHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const char *, const int *, const int *, char *, int)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

inline hipsparseStatus_t hipsparsecsc2denseHelper(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const float *cscValA, const int *cscRowIndA, const int *cscColPtrA, float *A, int lda)
{
    hipScsc2dense(m, n, cscValA, cscRowIndA, cscColPtrA, A);
    return HIPSPARSE_STATUS_SUCCESS;
}
inline hipsparseStatus_t hipsparsecsc2denseHelper(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const double *cscValA, const int *cscRowIndA, const int *cscColPtrA, double *A, int lda)
{
    hipScsc2dense(m, n, cscValA, cscRowIndA, cscColPtrA, A);
    return HIPSPARSE_STATUS_SUCCESS;
}

inline hipsparseStatus_t hipsparsecsc2denseHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const half *, const int *, const int *, half *, int)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}

inline hipsparseStatus_t hipsparsecsc2denseHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const short *, const int *, const int *, short *, int)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline hipsparseStatus_t hipsparsecsc2denseHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const char *, const int *, const int *, char *, int)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

inline hipsparseStatus_t hipsparsecsr2cscHelper(hipsparseHandle_t handle, int m, int n, int nnz, const float *csrVal, const int *csrRowPtr, const int *csrColInd, float *cscVal, int *cscRowInd, int *cscColPtr, hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase)
{
    return hipsparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
}
inline hipsparseStatus_t hipsparsecsr2cscHelper(hipsparseHandle_t handle, int m, int n, int nnz, const double *csrVal, const int *csrRowPtr, const int *csrColInd, double *cscVal, int *cscRowInd, int *cscColPtr, hipsparseAction_t copyValues, hipsparseIndexBase_t idxBase)
{
    return hipsparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscRowInd, cscColPtr, copyValues, idxBase);
}

inline hipsparseStatus_t hipsparsecsr2cscHelper(hipsparseHandle_t, int, int, int, const half *, const int *, const int *, half *, int *, int *, hipsparseAction_t, hipsparseIndexBase_t)
{
    RuntimeError("Unsupported template argument(half) in hipsparsecsr2cscHelper");
}


inline hipsparseStatus_t hipsparsennzHelper(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const float *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr)
{
    return hipsparseSnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
}
inline hipsparseStatus_t hipsparsennzHelper(hipsparseHandle_t handle, hipsparseDirection_t dirA, int m, int n, const hipsparseMatDescr_t descrA, const double *A, int lda, int *nnzPerRowColumn, int *nnzTotalDevHostPtr)
{
    return hipsparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn, nnzTotalDevHostPtr);
}

inline hipsparseStatus_t hipsparsennzHelper(hipsparseHandle_t,hipsparseDirection_t,int,int , const hipsparseMatDescr_t, const half *, int, int *, int *)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}

inline hipsparseStatus_t hipsparsennzHelper(hipsparseHandle_t,hipsparseDirection_t,int,int , const hipsparseMatDescr_t, const short *, int, int *, int *)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline hipsparseStatus_t hipsparsennzHelper(hipsparseHandle_t,hipsparseDirection_t,int,int , const hipsparseMatDescr_t, const char *, int, int *, int *)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

inline hipsparseStatus_t hipsparsedense2csrHelper(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const float *A, int lda, const int *nnzPerRow, float *csrValA, int *csrRowPtrA, int *csrColIndA)
{
    return hipsparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
}
inline hipsparseStatus_t hipsparsedense2csrHelper(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const double *A, int lda, const int *nnzPerRow, double *csrValA, int *csrRowPtrA, int *csrColIndA)
{
    return hipsparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrValA, csrRowPtrA, csrColIndA);
}

inline hipsparseStatus_t hipsparsedense2csrHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const half *, int, const int *, half *, int *, int *)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}
#
inline hipsparseStatus_t hipsparsedense2csrHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const short *, int, const int *, short *, int *, int *)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline hipsparseStatus_t hipsparsedense2csrHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const char *, int, const int *, char *, int *, int *)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

inline hipsparseStatus_t hipsparsedense2cscHelper(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const float *A, int lda, const int *nnzPerCol, float *cscValA, int *cscRowIndA, int *cscColPtrA)
{
    return hipsparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
}
inline hipsparseStatus_t hipsparsedense2cscHelper(hipsparseHandle_t handle, int m, int n, const hipsparseMatDescr_t descrA, const double *A, int lda, const int *nnzPerCol, double *cscValA, int *cscRowIndA, int *cscColPtrA)
{
    return hipsparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscValA, cscRowIndA, cscColPtrA);
}

inline hipsparseStatus_t hipsparsedense2cscHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const half *, int, const int *, half *, int *, int *)
{
    RuntimeError("Unsupported template argument(half) in GPUSparseMatrix");
}

inline hipsparseStatus_t hipsparsedense2cscHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const short *, int, const int *, short *, int *, int *)
{
    RuntimeError("Unsupported template argument(short) in GPUSparseMatrix");
}
inline hipsparseStatus_t hipsparsedense2cscHelper(hipsparseHandle_t,int,int,const hipsparseMatDescr_t, const char *, int, const int *, char *, int *, int *)
{
    RuntimeError("Unsupported template argument(char) in GPUSparseMatrix");
}

inline hipsparseStatus_t hipsparsecsrmmHelper(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int k, int nnz, const float *alpha, const hipsparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *B, int ldb, const float *beta, float *C, int ldc)
{
    return hipsparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}
inline hipsparseStatus_t hipsparsecsrmmHelper(hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int k, int nnz, const double *alpha, const hipsparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *B, int ldb, const double *beta, double *C, int ldc)
{
    return hipsparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc);
}

inline hipsparseStatus_t hipsparsecsrmmHelper(hipsparseHandle_t, hipsparseOperation_t, int, int, int, int, const half *, const hipsparseMatDescr_t, const half *, const int *, const int *, const half *, int, const half *, half *, int)
{
    RuntimeError("Unsupported template argument(half) in hipsparsecsrmmHelper");
}


inline hipsparseStatus_t hipsparsecsrgemmHelper(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, const hipsparseMatDescr_t descrA, const int nnzA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const hipsparseMatDescr_t descrB, const int nnzB, const float *csrValB, const int *csrRowPtrB, const int *csrColIndB, const hipsparseMatDescr_t descrC, float *csrValC, const int *csrRowPtrC, int *csrColIndC)
{
    return hipsparseScsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC);
}
inline hipsparseStatus_t hipsparsecsrgemmHelper(hipsparseHandle_t handle, hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, int k, const hipsparseMatDescr_t descrA, const int nnzA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const hipsparseMatDescr_t descrB, const int nnzB, const double *csrValB, const int *csrRowPtrB, const int *csrColIndB, const hipsparseMatDescr_t descrC, double *csrValC, const int *csrRowPtrC, int *csrColIndC)
{
    return hipsparseDcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC);
}

inline hipsparseStatus_t hipsparsecsrgemmHelper(hipsparseHandle_t, hipsparseOperation_t, hipsparseOperation_t, int, int, int, const hipsparseMatDescr_t, const int, const half *, const int *, const int *, const hipsparseMatDescr_t, const int, const half *, const int *, const int *, const hipsparseMatDescr_t, half *, const int *, int *)
{
    RuntimeError("Unsupported template argument(half) in hipsparsecsrgemmHelper");
}


inline hipsparseStatus_t hipsparsecsrgeamHelper(hipsparseHandle_t handle, int m, int n, const float *alpha, const hipsparseMatDescr_t descrA, int nnzA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *beta, const hipsparseMatDescr_t descrB, int nnzB, const float *csrValB, const int *csrRowPtrB, const int *csrColIndB, const hipsparseMatDescr_t descrC, float *csrValC, int *csrRowPtrC, int *csrColIndC)
{
    return hipsparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC);
}
inline hipsparseStatus_t hipsparsecsrgeamHelper(hipsparseHandle_t handle, int m, int n, const double *alpha, const hipsparseMatDescr_t descrA, int nnzA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *beta, const hipsparseMatDescr_t descrB, int nnzB, const double *csrValB, const int *csrRowPtrB, const int *csrColIndB, const hipsparseMatDescr_t descrC, double *csrValC, int *csrRowPtrC, int *csrColIndC)
{
    return hipsparseDcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC);
}

inline hipsparseStatus_t hipsparsecsrgeamHelper(hipsparseHandle_t, int, int, const half *, const hipsparseMatDescr_t, int, const half *, const int *, const int *, const half *, const hipsparseMatDescr_t, int, const half *, const int *, const int *, const hipsparseMatDescr_t, half *, int *, int *)
{
    RuntimeError("Unsupported template argument(half) in hipsparsecsrgeamHelper");
}


inline hipsparseStatus_t hipsparsedotiHelper(hipsparseHandle_t handle, int nnz, const float *xVal, const int *xInd, const float *y, float *resultDevHostPtr, hipsparseIndexBase_t idxBase)
{
    return hipsparseSdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase);
}
inline hipsparseStatus_t hipsparsedotiHelper(hipsparseHandle_t handle, int nnz, const double *xVal, const int *xInd, const double *y, double *resultDevHostPtr, hipsparseIndexBase_t idxBase)
{
    return hipsparseDdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase);
}

inline hipsparseStatus_t hipsparsedotiHelper(hipsparseHandle_t, int, const half *, const int *, const half *, half *, hipsparseIndexBase_t)
{
    RuntimeError("Unsupported template argument(half) in hipsparsedotiHelper");
}



// Generalize cub calls
inline hipError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_keys_in, float *d_keys_out, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, int begin_bit, int end_bit, hipStream_t stream)
{
    return hipcub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream);
}
inline hipError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_keys_in, double *d_keys_out, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, int begin_bit, int end_bit, hipStream_t stream)
{
    return hipcub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit, stream);
}

inline hipError_t SortPairsDescending(void *, size_t, const half *, half *, const uint64_t *, uint64_t *, int, int, int, hipStream_t)
{
    RuntimeError("Unsupported template argument(half) in SortPairsDescending");
}


#endif // CPUONLY
