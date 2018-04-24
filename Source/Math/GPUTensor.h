//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include "CommonMatrix.h"
#include "TensorShape.h" // only for SmallVector; I was hoping to keep this out
#include "GPUMatrixCUDAKernels.cuh"
#include <array>

namespace Microsoft { namespace MSR { namespace CNTK {


// Manual Serialization strategy to bypass the serialization attempted by compiler
#if defined(__HIP_PLATFORM_HCC__)
  #include <hip/hip_hcc.h>

  template<typename T>
    class Magic_wrapper {
        T* p_ = nullptr;
    public:
        Magic_wrapper() = default;
        explicit
        Magic_wrapper(T& x)
        {

            std::cout << "calling magic wrapper:\t" ;

 #define SIZEOF(object) (char *)(&object+1) - (char *)(&object)

            std::cout << sizeof(x) << std::endl;

            hipHostMalloc(&p_, SIZEOF(x)); new (p_) T{x};  //sizeof(T)
            std::cout<<"Magic wrapped object allocated "<<this->p_<<std::endl;
        }

        ~Magic_wrapper() {
        	size_t deallocateSize = 0;;
        	std::cout<<"Attempting to deallocate pointer\t"<<this->p_<<std::endl;
        	hipMemPtrGetInfo(p_, &deallocateSize);
        	std::cout<<"deallocate Size\t"<<deallocateSize<<std::endl;
        	if (deallocateSize > 0) {
        	   //ipHostFree(p_);
        	}
        }

        operator T&() const [[hc]] { return p_[0]; }
        operator T&() [[hc]] { return p_[0]; }
    };

  template<typename T>
  Magic_wrapper<T> make_magic_wrapper(T& x)
  {
    return Magic_wrapper<T>{x};
  }
  #define reference_to_const(...) __VA_ARGS__ &
#else
  #define make_magic_wrapper(x) x
  #define reference_to_const(...) __VA_ARGS__
#endif


// GPUMatrix::TensorOp() interfaces with actual tensor code through these two functions, which are independent of the GPUMatrix class

#define C_size_t CUDA_LONG
#define C_int CUDA_LONG
#define C_unsigned_int CUDA_LONG

template <class ElemType, C_size_t N>
void TensorOpN(ElemType beta, array<ElemType*, N> pointers, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp,
               const array<size_t, N>& offsets,
               const SmallVector<size_t>& regularOpDims, const array<SmallVector<ptrdiff_t>, N>& regularStrides,
               const SmallVector<size_t>& reducingOpDims, const array<SmallVector<ptrdiff_t>, N>& reducingStrides);

template <class ElemType>
void LaunchUnaryTensorOp(ElemType beta, const ElemType* pa, ElemType* pb, ElemType alpha, ElementWiseOperator op, size_t regularOpDim);

}}}
