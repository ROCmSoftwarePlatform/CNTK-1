//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#pragma once

#include "RNGHandle.h"

#ifndef CPUONLY
#ifdef CUDA_COMPILE
#include <curand.h>
#elif defined HIP_COMPILE
#include <hiprng.h>
#endif
#endif // !CPUONLY

namespace Microsoft { namespace MSR { namespace CNTK {

class GPURNGHandle : public RNGHandle
{
public:
    GPURNGHandle(int deviceId, uint64_t seed, uint64_t offset = 0);
    virtual ~GPURNGHandle();

#ifndef CPUONLY
#ifdef CUDA_COMPILE
    curandGenerator_t Generator()
    {
        return m_generator;
    }
#elif defined HIP_COMPILE
    hiprngGenerator_t Generator()
    {
        return m_generator;
    }
#endif

private:
#ifdef CUDA_COMPILE
    curandGenerator_t m_generator;
#elif defined HIP_COMPILE
    hiprngGenerator_t m_generator;
#endif
#endif // !CPUONLY
};

}}}
