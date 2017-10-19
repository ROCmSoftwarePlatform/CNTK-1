//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include <cassert>
#include <stdio.h>
#include "Profiler.h"
#include "BestGpu.h" // for CPUONLY flag only

#ifndef CPUONLY
#ifdef CUDA_COMPILE
#include <cuda_profiler_api.h>
#elif defined HIP_COMPILE
#include <hip/hip_runtime_api.h>
#endif
#else
// If compiling without CUDA, defining profiler control functions as no-op stubs
#ifdef CUDA_COMPILE
void cudaProfilerStart()
{
}
void cudaProfilerStop()
{
}
#elif defined HIP_COMPILE //though a no-op stub , still separated to maintain similar naming.
void hipProfilerStart()
{
}
void hipProfilerStop()
{
}
#endif
#endif

Profiler::Profiler(int numSamples)
    : m_numSamples(numSamples),
      m_isProfilingActive(false)
{
}

Profiler::~Profiler()
{
    if (m_isProfilingActive)
        Stop();
}

void Profiler::Start()
{
    assert(!m_isProfilingActive);
    m_isProfilingActive = true;
    fprintf(stderr, "Starting profiling\n");
#ifdef CUDA_COMPILE
    cudaProfilerStart();
#elif defined HIP_COMPILE
    hipProfilerStart();
#endif
}

void Profiler::NextSample()
{
    if (m_isProfilingActive)
    {
        if (--m_numSamples == 0)
            Stop();
    }
    else
    {
        if (m_numSamples > 0)
            Start();
    }
}

void Profiler::Stop()
{
    assert(m_isProfilingActive);
#ifdef CUDA_COMPILE
    cudaProfilerStop();
#elif defined HIP_COMPILE
    hipProfilerStop();
#endif
    fprintf(stderr, "Stopping profiling\n");
    m_isProfilingActive = false;
}
