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
#ifdef CUDA_COMPILE
    if (GetMathLibTraceLevel() > 0)
    {
        fprintf(stderr, "(GPU): creating curand object with seed %llu\n", cudaSeed);
    }

    CURAND_CALL(curandCreateGenerator(&m_generator, CURAND_RNG_PSEUDO_XORWOW));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(m_generator, cudaSeed));
    CURAND_CALL(curandSetGeneratorOrdering(m_generator, CURAND_ORDERING_PSEUDO_SEEDED));
    CURAND_CALL(curandSetGeneratorOffset(m_generator, offset));
#elif defined HIP_COMPILE
    if (GetMathLibTraceLevel() > 0)
    {
        fprintf(stderr, "(GPU): creating hiprng object with seed %llu\n", cudaSeed);
    }

    HIPRNG_CALL(hiprngCreateGenerator(&m_generator, HIPRNG_RNG_PSEUDO_XORWOW));
    HIPRNG_CALL(hiprngSetPseudoRandomGeneratorSeed(m_generator, cudaSeed));
    HIPRNG_CALL(hiprngSetGeneratorOrdering(m_generator, HIPRNG_ORDERING_PSEUDO_SEEDED));
    HIPRNG_CALL(hiprngSetGeneratorOffset(m_generator, offset));
#endif
}

/*virtual*/ GPURNGHandle::~GPURNGHandle()
{
#ifdef CUDA_COMPILE
    if (std::uncaught_exception())
        curandDestroyGenerator(m_generator);
    else
	CURAND_CALL(curandDestroyGenerator(m_generator));
#elif defined HIP_COMPILE
    if (std::uncaught_exception())
        hiprngDestroyGenerator(m_generator);
    else
        HIPRNG_CALL(hiprngDestroyGenerator(m_generator));
#endif
}

}}}
