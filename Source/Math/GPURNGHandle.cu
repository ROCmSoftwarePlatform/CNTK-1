//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CPUMatrix.cpp : full implementation of all matrix functions on the CPU side
//

#include "GPURNGHandle.h"
#include "GPUMatrix.h"

namespace Microsoft { namespace MSR { namespace CNTK {

GPURNGHandle::GPURNGHandle(int deviceId, uint64_t seed, uint64_t offset)
    : RNGHandle(deviceId)
{
    unsigned long long cudaSeed = seed;
    if (GetMathLibTraceLevel() > 0)
    {
        fprintf(stderr, "(GPU): creating hiprand object with seed %llu\n", cudaSeed);
    }

    HIPRAND_CALL(hiprandCreateGenerator(&m_generator, HIPRAND_RNG_PSEUDO_XORWOW));
    HIPRAND_CALL(hiprandSetPseudoRandomGeneratorSeed(m_generator, cudaSeed));
    // TODO: __hip__ HIPRAND_CALL(hiprandSetGeneratorOrdering(m_generator, HIPRAND_ORDERING_PSEUDO_SEEDED));
    HIPRAND_CALL(hiprandSetGeneratorOffset(m_generator, offset));
}

/*virtual*/ GPURNGHandle::~GPURNGHandle()
{
    if (std::uncaught_exception())
        hiprandDestroyGenerator(m_generator);
    else
        HIPRAND_CALL(hiprandDestroyGenerator(m_generator));
}

}}}
