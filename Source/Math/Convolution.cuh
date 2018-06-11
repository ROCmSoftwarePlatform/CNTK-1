//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <hip/hip_runtime.h>
#ifdef __HIP_PLATFORM_NVCC__
#include <device_launch_parameters.h>
#include <math_constants.h>
#endif
#include "half.hpp"

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// The file contains CUDA kernels that are used in reference convolution
// engine. All these kernels look very similar as they use the same
// idea of precomputed maps described in ConvolveGeometry.h
// That is, 'mpRowCol' maps each convolution output to the start of the
// input. 'mpRowIwht', 'mpRowRun' and 'runs' provide maps that allow
// to get indices of the active weight when applying the convolution.
// See ConvolveGeometry.h (MpRowCol, MpRowIwht etc) for more details.
// -----------------------------------------------------------------------

template <typename ElemType>
__global__ void kConvolutionForward(int batchSize, const ElemType* __restrict__ kernel,
                                    const int* mpRowCol, const int* mpRowIwht,
                                    const int* mpRowRun, const int* __restrict__ runs,
                                    const ElemType* __restrict__ src, int srcVecSize,
                                    ElemType* dst, int dstVecSize)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= dstVecSize)
        return;

    src += hipBlockIdx_y * srcVecSize;
    dst += hipBlockIdx_y * dstVecSize;

    for (int sample = hipBlockIdx_y; sample < batchSize; sample += hipGridDim_y)
    {
        int colBase = mpRowCol[row];
        int ivBase = mpRowIwht[row];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(0 <= colBase && colBase < srcVecSize);
#endif

        comp_t sum = 0;
        int i0 = mpRowRun[row];
        int skip = runs[i0++];
        int size = runs[i0++];
        int imask = i0 + size;
        for (int i = 0; i < size; i++)
        {
            if (runs[imask + i] == 0)
                continue;
            int dcol = runs[i0 + i];

#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < srcVecSize);
#endif

            sum += (comp_t)kernel[ivBase + skip + i] * (comp_t)src[colBase + dcol];
        }
        dst[row] = sum;
        src += hipBlockDim_y * srcVecSize;
        dst += hipBlockDim_y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kConvolutionBackwardData(int batchSize, const ElemType* __restrict__ kernel,
                                         const int* mpRowCol, const int* mpRowIwht,
                                         const int* mpRowRun, const int* __restrict__ runs,
                                         const ElemType* __restrict__ srcGrad, int srcVecSize,
                                         ElemType* grad, int dstVecSize)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= srcVecSize)
        return;

    srcGrad += hipBlockIdx_y * srcVecSize;
    grad += hipBlockIdx_y * dstVecSize;
    for (int sample = hipBlockIdx_y; sample < batchSize; sample += hipGridDim_y)
    {
        int colBase = mpRowCol[row];
        int ivBase = mpRowIwht[row];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(0 <= colBase && colBase < dstVecSize);
#endif

        comp_t g = srcGrad[row];
        int i0 = mpRowRun[row];
        int skip = runs[i0++];
        int size = runs[i0++];
        int imask = i0 + size;
        for (int i = 0; i < size; i++)
        {
            if (runs[imask + i] == 0)
                continue;
            int dcol = runs[i0 + i];

#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
#endif

            atomicAdd(&grad[colBase + dcol], (ElemType)((comp_t)g * (comp_t)kernel[ivBase + skip + i]));
        }
        srcGrad += hipBlockDim_y * srcVecSize;
        grad += hipBlockDim_y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kConvolutionBackwardDataAcc(int batchSize, const ElemType* __restrict__ kernel,
                                         const int* mpRowCol, const int* mpRowIwht,
                                         const int* mpRowRun, const int* __restrict__ runs,
                                         const ElemType* __restrict__ srcGrad, int srcVecSize,
                                         ElemType* grad, int dstVecSize, bool accumulateGradients)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= srcVecSize)
        return;

    srcGrad += hipBlockIdx_y * srcVecSize;
    grad += hipBlockIdx_y * dstVecSize;
    for (int sample = hipBlockIdx_y; sample < batchSize; sample += hipGridDim_y)
    {
        int colBase = mpRowCol[row];
        int ivBase = mpRowIwht[row];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(0 <= colBase && colBase < dstVecSize);
#endif

        comp_t g = srcGrad[row];
        int i0 = mpRowRun[row];
        int skip = runs[i0++];
        int size = runs[i0++];
        int imask = i0 + size;
        for (int i = 0; i < size; i++)
        {
            if (runs[imask + i] == 0)
                continue;
            int dcol = runs[i0 + i];

#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
#endif
       	    if (accumulateGradients) {
                atomicAdd(&grad[colBase + dcol], (ElemType)((comp_t)g * (comp_t)kernel[ivBase + skip + i]));
            }
	    else{
                grad[colBase + dcol] = 0;
                atomicAdd(&grad[colBase + dcol], (ElemType)((comp_t)g * (comp_t)kernel[ivBase + skip + i]));
            }
        }
        srcGrad += hipBlockDim_y * srcVecSize;
        grad += hipBlockDim_y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kConvolutionBackwardKernel(int batchSize, int inVecSize, int outVecSize,
                                           const ElemType* __restrict__ in,
                                           const int* mpRowCol, const int* mpRowIwht,
                                           const int* mpRowRun, const int* __restrict__ runs,
                                           const ElemType* __restrict__ srcGrad,
                                           ElemType* kernelGrad)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= outVecSize)
        return;

    in += hipBlockIdx_y * inVecSize;
    srcGrad += hipBlockIdx_y * outVecSize;
    for (int sample = hipBlockIdx_y; sample < batchSize; sample += hipGridDim_y)
    {
        int colBase = mpRowCol[row];
        int ivBase = mpRowIwht[row];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(0 <= colBase && colBase < inVecSize);
#endif

        comp_t g = srcGrad[row];
        int i0 = mpRowRun[row];
        int skip = runs[i0++];
        int size = runs[i0++];
        int imask = i0 + size;
        for (int i = 0; i < size; i++)
        {
            if (runs[imask + i] == 0)
                continue;
            int dcol = runs[i0 + i];

#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < inVecSize);
#endif

            atomicAdd(&kernelGrad[ivBase + skip + i], (ElemType)((comp_t)g * (comp_t)in[colBase + dcol]));
        }
        in += hipBlockDim_y * inVecSize;
        srcGrad += hipBlockDim_y * outVecSize;
    }
}

template <typename ElemType>
__global__ void kMaxPoolingForward(int batchSize, const int* mpRowCol, const int* mpRowIndices, const int* indices,
                                   const ElemType* __restrict__ src, int srcVecSize,
                                   ElemType* dst, int dstVecSize)
{
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= dstVecSize)
        return;

    src += hipBlockIdx_y * srcVecSize;
    dst += hipBlockIdx_y * dstVecSize;

    for (int sample = hipBlockIdx_y; sample < batchSize; sample += hipGridDim_y)
    {
        int colBase = mpRowCol[row];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(0 <= colBase && colBase < srcVecSize);
#endif

        int i0 = mpRowIndices[row];
        int size = indices[i0++];
        ElemType res = src[colBase + indices[i0]];
        for (int i = 1; i < size; i++)
        {
            int dcol = indices[i0 + i];
#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < srcVecSize);
#endif
            res = max(res, src[colBase + dcol]);
        }
        dst[row] = res;

        src += hipBlockDim_y * srcVecSize;
        dst += hipBlockDim_y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kMaxPoolingBackward(int batchSize, const ElemType* out, const ElemType* in,
                                    const int* mpRowCol, const int* mpRowIndices, const int* indices,
                                    const ElemType* __restrict__ srcGrad, int srcVecSize,
                                    ElemType* grad, int dstVecSize)
{
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= srcVecSize)
        return;

    in += hipBlockIdx_y * dstVecSize;
    out += hipBlockIdx_y * srcVecSize;
    srcGrad += hipBlockIdx_y * srcVecSize;
    grad += hipBlockIdx_y * dstVecSize;

    for (int sample = hipBlockIdx_y; sample < batchSize; sample += hipGridDim_y)
    {
        int colBase = mpRowCol[row];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(0 <= colBase && colBase < dstVecSize);
#endif

        int i0 = mpRowIndices[row];
        int size = indices[i0++];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(size > 0);
#endif

        ElemType g = srcGrad[row];
        ElemType m = out[row];
        for (int i = 0; i < size; i++)
        {
            int dcol = indices[i0 + i];

#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
#endif

            if (in[colBase + dcol] >= m)
            {
                atomicAdd(&grad[colBase + dcol], g);
                break;
            }
        }

        in += hipBlockDim_y * dstVecSize;
        out += hipBlockDim_y * srcVecSize;
        srcGrad += hipBlockDim_y * srcVecSize;
        grad += hipBlockDim_y * dstVecSize;
    }
}

inline __device__ float round_(float a)
{
    return roundf(a);
}

inline __device__ double round_(double a)
{
    return round(a);
}

// For each image, for each ROI, this function treats that ROI as an image
// and does max pooling so that it has output size pooledHeight x pooledWidth.
// The kernel operates on one location in the output tensor, computes which ROI
// and image should populate that location, computes the subset of the image
// corresponding to the ROI and which pixels in that subset should go into the
// output location, then takes the max value over that window.
// src: Images              [W x H x C x N]
// roiData: ROIs            [4 x numROIs x N],
// dst: Pooled ROIs         [PW x PH x C x numROIs x N]
// argmax: max positions    [PW x PH x C x numROIs x N]
// spatialScale             ratio of input feature map to the original image.
// where PW = Pooled Width, PH = Pooled Height, C = Channels, N = Batch Size
template <typename ElemType>
__global__ void kMaxROIPoolingForward(const int totalIterations,
    const int numROIs, const int numImg,
    const int channels, const int width, const int height,
    const int pooledWidth, const int pooledHeight, const ElemType* src,
    const ElemType* roiData, ElemType* dst, ElemType* argmax, double spatialScale)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    // index loops over all totalRois*c*pooledHeight*pooledWidth output locations.
    for (int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        index < (totalIterations); index += hipBlockDim_x * hipGridDim_x)
    {

        // output is [W x H x C x N]
        // n is the global ROI index (the new batch index)
        int pw = index % pooledWidth;
        int ph = (index / pooledWidth) % pooledHeight;
        int c = (index / pooledWidth / pooledHeight) % channels;
        int n = index / pooledWidth / pooledHeight / channels;

        // each ROI is 4 elements: (x, y, w, h)
        roiData += n * 4;

        // roi data is relative to original image size
        int roiStartW = (int)(round_(roiData[0] * spatialScale));
        int roiStartH = (int)(round_(roiData[1] * spatialScale));
        int roiEndW = (int)(round_(roiData[2] * spatialScale));
        int roiEndH = (int)(round_(roiData[3] * spatialScale));

        int roiWidth = max(roiEndW - roiStartW + 1, (int)1);
        int roiHeight = max(roiEndH - roiStartH + 1, (int)1);

        comp_t winH = (comp_t)roiHeight / (comp_t)pooledHeight;
        comp_t winW = (comp_t)roiWidth / (comp_t)pooledWidth;

        // compute window for this output location.
        int hstart = (int)(ph * winH);
        int wstart = (int)(pw * winW);
        int hend = (int)(ceilf((ph + 1) * winH));
        int wend = (int)(ceilf((pw + 1) * winW));

        // Add ROI offsets and clip to input boundaries
        hstart = min(max(hstart + roiStartH, 0), height);
        hend = min(max(hend + roiStartH, 0), height);
        wstart = min(max(wstart + roiStartW, 0), width);
        wend = min(max(wend + roiStartW, 0), width);

        bool isempty = (hend <= hstart) || (wend <= wstart);
        // Define an empty pooling region to be zero
#ifdef __HIP_PLATFORM_NVCC__
        comp_t maxval = isempty ? (comp_t)0 : (comp_t)-CUDART_INF_F;    
#elif defined __HIP_PLATFORM_HCC__
        comp_t maxval = isempty ? (comp_t)0 : (comp_t)(-(__int_as_float(0x7f800000)));//TODO: __add__ -CUDART_INF_F;;    
#endif
        int maxidx = -1;

        int imgIdx = n / numROIs;
        src += (imgIdx * channels + c) * height * width;
        for (int h = hstart; h < hend; h++)
        {
            for (int w = wstart; w < wend; w++)
            {
                int srcIndex = w + h * width;
                if (src[srcIndex] > maxval)
                {
                    maxval = src[srcIndex];
                    maxidx = srcIndex;
                }
            }
        }
        dst[index] = maxval;
        argmax[index] = maxidx;
    }
}

// The kernel operates on one location in the input to the ROIPoolingNode (one image location).
// It loops over the ROIs corresponding to that image, seeing which ones could contain the location
// in their output. For each ROI, it checks the argmax data to see if that ROI indeed chose
// this pixel location as the maximum. If so, it increments the gradient term for the input location.
template <typename ElemType>
__global__ void kMaxROIPoolingBackward(const int totalIterations,
    const int numROIs, const int numImg,
    const int channels, const int width, const int height,
    const int pooledWidth, const int pooledHeight, const ElemType* pooledGrad,
    const ElemType* roiData, ElemType* grad, const ElemType* argmax, double spatialScale)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    // index loops over all input locations (locations in the original input tensor).
    for (int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        index < (totalIterations); index += hipBlockDim_x * hipGridDim_x)
    {
        // images are laid out [W x H x C x N]
        // (n, c, h, w) is an element in the input image
        int w = index % width;
        int h = (index / width) % height;
        int c = (index / width / height) % channels;
        int n = index / width / height / channels;

        // compute range of ROIs corresponding to this image:
        int roiMin = n * numROIs;
        int roiMax = (n + 1) * numROIs;

        comp_t gradient = 0;
        for (int roiN = roiMin; roiN < roiMax; roiN++)
        {
            // each ROI is 4 elements: (x, y, w, h)
            const ElemType* roiOffset = roiData + roiN * 4;

            // ROI data is absolute pixel value in the original image size
            int roiStartW = (int)(round_(roiOffset[0] * spatialScale));
            int roiStartH = (int)(round_(roiOffset[1] * spatialScale));
            int roiEndW = (int)(round_(roiOffset[2] * spatialScale));
            int roiEndH = (int)(round_(roiOffset[3] * spatialScale));

            int roiWidth = max(roiEndW - roiStartW + 1, (int)1);
            int roiHeight = max(roiEndH - roiStartH + 1, (int)1);

            // skip this ROI if it doesn't contain our input location.
            const bool inROI = (w >= roiStartW && w < roiStartW + roiWidth &&
                h >= roiStartH && h < roiStartH + roiHeight);
            if (!inROI)
                continue;

            comp_t winH = (comp_t)roiHeight / (comp_t)pooledHeight;
            comp_t winW = (comp_t)roiWidth / (comp_t)pooledWidth;

            // what pooled nodes in the output for this ROI could have pooled this input location?
            // we use int here since the computation can yield a negative result
            int phstart = (int)((float)(h - roiStartH) / winH);
            int pwstart = (int)((float)(w - roiStartW) / winW);
            int phend = (int)(ceilf((float)(h - roiStartH + 1) / winH));
            int pwend = (int)(ceilf((float)(w - roiStartW + 1) / winW));

            phstart = min(max(phstart, 0), pooledHeight);
            pwstart = min(max(pwstart, 0), pooledWidth);
            phend = min(max(phend, 0), pooledHeight);
            pwend = min(max(pwend, 0), pooledWidth);

            // go right up to this channel of this ROI.
            int offset = (roiN * channels + c) * pooledWidth * pooledHeight;
            const ElemType* offsetPoolGrad = pooledGrad + offset;
            const ElemType* offsetArgmax = argmax + offset;

            for (int ph = phstart; ph < phend; ph++)
            {
                for (int pw = pwstart; pw < pwend; pw++)
                {
                    if ((int)offsetArgmax[ph * pooledWidth + pw] == (h * width + w))
                    {
                        gradient += (comp_t)offsetPoolGrad[ph * pooledWidth + pw];
                    }
                }
            }
        }

        atomicAdd(&grad[index], (ElemType)gradient);
    }
}

template <typename ElemType>
__global__ void kMaxUnpooling(int batchSize, const int* mpRowCol, const int* mpRowIndices, const int* indices,
                              const ElemType* __restrict__ src, const ElemType* poolIn, int srcVecSize,
                              ElemType* dst, int dstVecSize)
{
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= srcVecSize)
        return;

    src    += hipBlockIdx_y * srcVecSize;
    poolIn += hipBlockIdx_y * dstVecSize;
    dst    += hipBlockIdx_y * dstVecSize;

    for (int sample = hipBlockIdx_y; sample < batchSize; sample += hipGridDim_y)
    {
        int colBase = mpRowCol[row];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(0 <= colBase && colBase < dstVecSize);
#endif

        int i0 = mpRowIndices[row];
        int size = indices[i0++];
        ElemType curMax = poolIn[colBase + indices[i0]];
        ElemType prevMax = curMax;
        int imax = 0;
        for (int i = 1; i < size; i++)
        {
            int dcol = indices[i0 + i];

#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
#endif

            curMax = max(curMax, poolIn[colBase + dcol]);
            if (curMax > prevMax)
            {
                prevMax = curMax;
                imax = i;
            }

        }

        int dcol = indices[i0 + imax];

#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
#endif

        dst[colBase + dcol] = src[row];

        src    += hipBlockIdx_y * srcVecSize;
        poolIn += hipBlockIdx_y * dstVecSize;
        dst    += hipBlockIdx_y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kAveragePoolingForward(int batchSize, const int* mpRowCol, const int* mpRowIndices, const int* indices,
                                       const ElemType* __restrict__ src, int srcVecSize,
                                       ElemType* dst, int dstVecSize)
{
    typedef typename TypeSelector<ElemType>::comp_t comp_t;
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= dstVecSize)
        return;

    src += hipBlockIdx_y * srcVecSize;
    dst += hipBlockIdx_y * dstVecSize;

    for (int sample = hipBlockIdx_y; sample < batchSize; sample += hipGridDim_y)
    {
        int colBase = mpRowCol[row];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(0 <= colBase && colBase < srcVecSize);
#endif

        int i0 = mpRowIndices[row];
        int size = indices[i0++];
        comp_t sum = 0;
        for (int i = 0; i < size; i++)
        {
            int dcol = indices[i0 + i];

#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < srcVecSize);
#endif

            sum += (comp_t)src[colBase + dcol];
        }
        dst[row] = sum / (comp_t)size;

        src += hipBlockDim_y * srcVecSize;
        dst += hipBlockDim_y * dstVecSize;
    }
}

template <typename ElemType>
__global__ void kAveragePoolingBackward(int batchSize, const int* mpRowCol, const int* mpRowIndices, const int* indices,
                                        const ElemType* __restrict__ srcGrad, int srcVecSize,
                                        ElemType* grad, int dstVecSize)
{
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= srcVecSize)
        return;

    srcGrad += hipBlockIdx_y * srcVecSize;
    grad += hipBlockIdx_y * dstVecSize;

    for (int sample = hipBlockIdx_y; sample < batchSize; sample += hipGridDim_y)
    {
        int colBase = mpRowCol[row];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(0 <= colBase && colBase < dstVecSize);
#endif

        int i0 = mpRowIndices[row];
        int size = indices[i0++];

#if defined( __HIP_ENABLE_ASSERT__ )
        assert(size > 0);
#endif

        ElemType g = srcGrad[row] / size;
        for (int i = 0; i < size; i++)
        {
            int dcol = indices[i0 + i];

#if defined( __HIP_ENABLE_ASSERT__ )
            assert(0 <= colBase + dcol && colBase + dcol < dstVecSize);
#endif

            atomicAdd(&grad[colBase + dcol], g);
        }

        srcGrad += hipBlockDim_y * srcVecSize;
        grad += hipBlockDim_y * dstVecSize;
    }
}

}}}
