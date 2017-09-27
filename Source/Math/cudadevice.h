// cudadevice.h - holds the buffers, events, and streams used on a per device basis
//
// F. Seide, V-hansu

#pragma once

#ifdef CUDA_COMPILE
#include <cuda_runtime_api.h>`	
#elif defined HIP_COMPILE
#include <hip/hip_runtime_api.h>
#endif

#include <assert.h>
#include <math.h>
#include <vector>
#include <unordered_set>

namespace msra { namespace cuda {

const int deviceMax = 8;

// an object that lives in a device --this class just implements setdevice() and associated storage, and is shared across matrix and vectorbaseimpl
class objectondevice
{
protected:
    size_t deviceid; // CUDA card in which this matrix lives ("virtual" index amongst cards allocated to this process); default: 0
protected:
    objectondevice(size_t d)
        : deviceid(d)
    {
    }

public:
    size_t getdevice() const
    {
        return deviceid;
    }
};

// auto-class to set device (through context) inside a function
// usage at each function that calls CUDA:
//  ondevice no (deviceid);
class ondevice
{
public:
    ondevice(size_t deviceid)
    {
	#ifdef CUDA_COMPILE
	auto rc = cudaSetDevice((int)deviceid);
        if (rc != cudaSuccess)
	    RuntimeError("Cannot set cuda device: %s (cuda error %d)", cudaGetErrorString(rc), (int)rc);
	#elif defined HIP_COMPILE
        auto rc = hipSetDevice((int)deviceid);
        if (rc != hipSuccess)
            RuntimeError("Cannot set cuda device: %s (cuda error %d)", hipGetErrorString(rc), (int)rc);
	#endif
    }
};
} }
