/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include "hip/hcc_detail/host_defines.h"
#include <assert.h>
#if defined(__cplusplus)
    #include <algorithm>
    #include <type_traits>
    #include <utility>
#endif

#if defined(__clang__) && (__clang_major__ > 3)
    typedef _Float16 _Float16_2 __attribute__((ext_vector_type(2)));

    struct __half_raw {
        union {
            static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

            _Float16 data;
            unsigned short x;
        };
    };


    #if defined(__cplusplus)
        #include "hip/hcc_detail/hip_fp16_math_fwd.h"
        #include "hip/hcc_detail/hip_vector_types.h"
        #include "hip/hcc_detail/host_defines.h"

        namespace std
        {
            template<> struct is_floating_point<_Float16> : std::true_type {};
        }

        template<bool cond, typename T = void>
        using Enable_if_t = typename std::enable_if<cond, T>::type;

        // BEGIN STRUCT __HALF
        struct __half {
        protected:
            union {
                static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

                _Float16 data;
                unsigned short __x;
            };
        public:
            // CREATORS
            __host__ __device__
            __half() = default;
            __host__ __device__
            __half(const __half_raw& x) : data{x.data} {}
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                __host__ __device__
                __half(decltype(data) x) : data{x} {}
                template<
                    typename T,
                    Enable_if_t<std::is_floating_point<T>{}>* = nullptr>
                __host__ __device__
                __half(T x) : data{static_cast<_Float16>(x)} {}
            #endif
            __host__ __device__
            __half(const __half&) = default;
            __host__ __device__
            __half(__half&&) = default;
            __host__ __device__
            ~__half() = default;

            // CREATORS - DEVICE ONLY
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                template<
                    typename T, Enable_if_t<std::is_integral<T>{}>* = nullptr>
                __device__
                __half(T x) : data{static_cast<_Float16>(x)} {}
            #endif

            // MANIPULATORS
            __host__ __device__
            operator __half_raw() const { return __half_raw{data}; }
            __host__ __device__
            operator volatile __half_raw() const volatile
            {
                return __half_raw{data};
            }

        };
        // END STRUCT __HALF

        namespace
        {
            // TODO: rounding behaviour is not correct.
            // float -> half | half2
            inline
            __device__
            __half __float2half(float x)
            {
                return __half_raw{static_cast<_Float16>(x)};
            }
            // half | half2 -> float
            inline
            __device__
            float __half2float(__half x)
            {
                return static_cast<__half_raw>(x).data;
            }
        } // Anonymous namespace.

    #endif // defined(__cplusplus)
#elif defined(__GNUC__)
    #pragma once

    #if defined(__cplusplus)
        #include <cstring>
    #endif

    struct __half_raw {
        unsigned short x;
    };


    #if defined(__cplusplus)
        struct __half;

        __half __float2half(float);
        float __half2float(__half);

        // BEGIN STRUCT __HALF
        struct __half {
        protected:
            unsigned short __x;
        public:
            // CREATORS
            __half() = default;
            __half(const __half_raw& x) : __x{x.x} {}
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                __half(float x) : __x{__float2half(x).__x} {}
                __half(double x) : __x{__float2half(x).__x} {}
            #endif
            __half(const __half&) = default;
            __half(__half&&) = default;
            ~__half() = default;

            // MANIPULATORS
            __half& operator=(const __half&) = default;
            __half& operator=(__half&&) = default;
            __half& operator=(const __half_raw& x) { __x = x.x; return *this; }
            #if !defined(__HIP_NO_HALF_CONVERSIONS__)
                __half& operator=(float x)
                {
                    __x = __float2half(x).__x;
                    return *this;
                }
                __half& operator=(double x)
                {
                    return *this = static_cast<float>(x);
                }
            #endif

            // ACCESSORS
            operator float() const { return __half2float(*this); }
            operator __half_raw() const { return __half_raw{__x}; }
        };
        // END STRUCT __HALF

        namespace
        {
            inline
            unsigned short __internal_float2half(
                float flt, unsigned int& sgn, unsigned int& rem)
            {
                unsigned int x{};
                std::memcpy(&x, &flt, sizeof(flt));

                unsigned int u = (x & 0x7fffffffU);
                sgn = ((x >> 16) & 0x8000U);

                // NaN/+Inf/-Inf
                if (u >= 0x7f800000U) {
                    rem = 0;
                    return static_cast<unsigned short>(
                        (u == 0x7f800000U) ? (sgn | 0x7c00U) : 0x7fffU);
                }
                // Overflows
                if (u > 0x477fefffU) {
                    rem = 0x80000000U;
                    return static_cast<unsigned short>(sgn | 0x7bffU);
                }
                // Normal numbers
                if (u >= 0x38800000U) {
                    rem = u << 19;
                    u -= 0x38000000U;
                    return static_cast<unsigned short>(sgn | (u >> 13));
                }
                // +0/-0
                if (u < 0x33000001U) {
                    rem = u;
                    return static_cast<unsigned short>(sgn);
                }
                // Denormal numbers
                unsigned int exponent = u >> 23;
                unsigned int mantissa = (u & 0x7fffffU);
                unsigned int shift = 0x7eU - exponent;
                mantissa |= 0x800000U;
                rem = mantissa << (32 - shift);
                return static_cast<unsigned short>(sgn | (mantissa >> shift));
            }

            inline
            __half __float2half(float x)
            {
                __half_raw r;
                unsigned int sgn{};
                unsigned int rem{};
                r.x = __internal_float2half(x, sgn, rem);
                if (rem > 0x80000000U || (rem == 0x80000000U && (r.x & 0x1))) ++r.x;

                return r;
            }

            inline
            float __internal_half2float(unsigned short x)
            {
                unsigned int sign = ((x >> 15) & 1);
                unsigned int exponent = ((x >> 10) & 0x1f);
                unsigned int mantissa = ((x & 0x3ff) << 13);

                if (exponent == 0x1fU) { /* NaN or Inf */
                    mantissa = (mantissa ? (sign = 0, 0x7fffffU) : 0);
                    exponent = 0xffU;
                } else if (!exponent) { /* Denorm or Zero */
                    if (mantissa) {
                        unsigned int msb;
                        exponent = 0x71U;
                        do {
                            msb = (mantissa & 0x400000U);
                            mantissa <<= 1; /* normalize */
                            --exponent;
                        } while (!msb);
                        mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
                    }
                } else {
                    exponent += 0x70U;
                }
                unsigned int u = ((sign << 31) | (exponent << 23) | mantissa);
                float f;
                memcpy(&f, &u, sizeof(u));

                return f;
            }

            inline
            float __half2float(__half x)
            {
                return __internal_half2float(static_cast<__half_raw>(x).x);
            }

        } // Anonymous namespace.

    #endif // defined(__cplusplus)
#endif // !defined(__clang__) && defined(__GNUC__)
