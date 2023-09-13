#pragma once
#include <cmath>
#include <cstddef>

#include "ranger_def.h"

namespace yuzu
{
namespace ranger
{
inline float l2Distance(const float* v1, const float* v2, size_t len)
{
    float distance = 0.f;
    float norm1 = 0.f;
    float norm2 = 0.f;

    for (size_t i = 0; i < len; ++i)
    {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
}

#pragma region SSE指令集加速欧式距离计算
template <size_t N>
inline float l2DistanceX4(const float* v1, const float* v2)
{
    float distance = 0.f;

    size_t i;
    for (i = 0; i < N - 3;)
    {
        distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        i += 1;
        distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        i += 1;
        distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        i += 1;
        distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
        i += 1;
    }

    // Calculate the remaining elements
    if constexpr (N % 4 != 0)
    {
        for (; i < N; ++i)
        {
            float diff = v1[i] - v2[i];
            distance += diff * diff;
        }
    }

    return std::sqrt(distance);
}

#ifdef USE_RANGER_SSE
inline float l2DistanceSSE(const float* v1, const float* v2, size_t len)
{
    float* pV1 = (float*)v1;
    float* pV2 = (float*)v2;
    __m128 sum = _mm_setzero_ps();

    size_t i;
    for (i = 0; i < len - 3; i += 4)
    {
        __m128 xmm1 = _mm_loadu_ps(pV1);
        __m128 xmm2 = _mm_loadu_ps(pV2);
        pV1 += 4;
        pV2 += 4;

        __m128 xmmDiff = _mm_sub_ps(xmm1, xmm2);
        // sum = _mm_fmadd_ps(xmmDiff, xmmDiff, sum);
        __m128 xmmSquared = _mm_mul_ps(xmmDiff, xmmDiff);
        sum = _mm_add_ps(sum, xmmSquared);
    }

    float result[4];
    _mm_storeu_ps(result, sum);
    float distance = result[0] + result[1] + result[2] + result[3];

    // Calculate the remaining elements
    for (; i < len; ++i)
    {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
}

inline float l2DistanceSSEAlign(const float* v1, const float* v2, size_t len)
{
    float* pV1 = (float*)v1;
    float* pV2 = (float*)v2;
    __m128 sum = _mm_setzero_ps();

    size_t i;
    for (i = 0; i < len - 3; i += 4)
    {
        __m128 xmm1 = _mm_load_ps(pV1);
        __m128 xmm2 = _mm_load_ps(pV2);
        pV1 += 4;
        pV2 += 4;

        __m128 xmmDiff = _mm_sub_ps(xmm1, xmm2);
        __m128 xmmSquared = _mm_mul_ps(xmmDiff, xmmDiff);
        sum = _mm_add_ps(sum, xmmSquared);
    }

    float result[4];
    _mm_store_ps(result, sum);
    float distance = result[0] + result[1] + result[2] + result[3];

    // Calculate the remaining elements
    for (; i < len; ++i)
    {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
}

template <size_t N>
inline float l2DistanceSSE(const float* v1, const float* v2)
{
    float* pV1 = (float*)v1;
    float* pV2 = (float*)v2;
    __m128 sum = _mm_setzero_ps();
    size_t i;
    for (i = 0; i < N - 3; i += 4)
    {
        __m128 xmm1 = _mm_loadu_ps(pV1);
        __m128 xmm2 = _mm_loadu_ps(pV2);
        pV1 += 4;
        pV2 += 4;

        __m128 xmmDiff = _mm_sub_ps(xmm1, xmm2);
        __m128 xmmSquared = _mm_mul_ps(xmmDiff, xmmDiff);
        sum = _mm_add_ps(sum, xmmSquared);
    }

    float result[4];
    _mm_storeu_ps(result, sum);
    float distance = result[0] + result[1] + result[2] + result[3];

    // Calculate the remaining elements
    if constexpr (N % 4 != 0)
    {
        for (; i < N; ++i)
        {
            float diff = v1[i] - v2[i];
            distance += diff * diff;
        }
    }

    return std::sqrt(distance);
}
#endif
#pragma endregion

#ifdef USE_RANGER_AVX
inline float l2DistanceAVX(const float* v1, const float* v2, size_t len)
{
    float* pV1 = (float*)v1;
    float* pV2 = (float*)v2;
    __m256 sum = _mm256_setzero_ps();

    size_t i;
    for (i = 0; i < len - 7; i += 8)
    {
        __m256 ymm1 = _mm256_loadu_ps(pV1);
        __m256 ymm2 = _mm256_loadu_ps(pV2);
        pV1 += 8;
        pV2 += 8;

        __m256 ymmDiff = _mm256_sub_ps(ymm1, ymm2);
        // sum = _mm256_fmadd_ps(ymmDiff, ymmDiff, sum);
        __m256 ymmSquared = _mm256_mul_ps(ymmDiff, ymmDiff);
        sum = _mm256_add_ps(sum, ymmSquared);
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    float distance = 0.0f;

    for (int j = 0; j < 8; ++j)
    {
        distance += result[j];
    }

    // Calculate the remaining elements
    for (; i < len; ++i)
    {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
}

template <size_t N>
inline float l2DistanceAVX(const float* v1, const float* v2)
{
    float* pV1 = (float*)v1;
    float* pV2 = (float*)v2;
    __m256 sum = _mm256_setzero_ps();

    size_t i;
    for (i = 0; i < N - 7; i += 8)
    {
        __m256 ymm1 = _mm256_loadu_ps(pV1);
        __m256 ymm2 = _mm256_loadu_ps(pV2);
        pV1 += 8;
        pV2 += 8;

        __m256 ymmDiff = _mm256_sub_ps(ymm1, ymm2);
        __m256 ymmSquared = _mm256_mul_ps(ymmDiff, ymmDiff);
        sum = _mm256_add_ps(sum, ymmSquared);
    }

    float result[8];
    _mm256_storeu_ps(result, sum);
    float distance = 0.0f;

    for (int j = 0; j < 8; ++j)
    {
        distance += result[j];
    }

    // 计算剩余的元素
    if constexpr (N % 8 != 0)
    {
        for (; i < N; ++i)
        {
            float diff = v1[i] - v2[i];
            distance += diff * diff;
        }
    }

    return std::sqrt(distance);
}
#endif

#ifdef USE_RANGER_AVX512
inline float l2DistanceAVX512(const float* v1, const float* v2, size_t len)
{
    float* pV1 = (float*)v1;
    float* pV2 = (float*)v2;
    __m512 sum = _mm512_setzero_ps();

    size_t i;
    for (i = 0; i < len - 15; i += 16)
    {
        __m512 ymm1 = _mm512_loadu_ps(pV1);
        __m512 ymm2 = _mm512_loadu_ps(pV2);
        pV1 += 16;
        pV2 += 16;

        __m512 ymmDiff = _mm512_sub_ps(ymm1, ymm2);
        __m512 ymmSquared = _mm512_mul_ps(ymmDiff, ymmDiff);
        sum = _mm512_add_ps(sum, ymmSquared);
    }

    float result[16];
    _mm512_storeu_ps(result, sum);
    float distance = 0.0f;

    for (int j = 0; j < 16; ++j)
    {
        distance += result[j];
    }

    // Calculate the remaining elements
    for (; i < len; ++i)
    {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
}

template <size_t N>
inline float l2DistanceAVX512(const float* v1, const float* v2)
{
    float* pV1 = (float*)v1;
    float* pV2 = (float*)v2;
    __m512 sum = _mm512_setzero_ps();

    size_t i;
    for (i = 0; i < N - 15; i += 16)
    {
        __m512 ymm1 = _mm512_loadu_ps(pV1);
        __m512 ymm2 = _mm512_loadu_ps(pV2);
        pV1 += 16;
        pV2 += 16;

        __m512 ymmDiff = _mm512_sub_ps(ymm1, ymm2);
        __m512 ymmSquared = _mm512_mul_ps(ymmDiff, ymmDiff);
        sum = _mm512_add_ps(sum, ymmSquared);
    }

    float result[16];
    _mm512_storeu_ps(result, sum);
    float distance = 0.0f;

    for (int j = 0; j < 16; ++j)
    {
        distance += result[j];
    }

    // Calculate the remaining elements
    if constexpr (N % 16 != 0)
    {
        for (; i < N; ++i)
        {
            float diff = v1[i] - v2[i];
            distance += diff * diff;
        }
    }

    return std::sqrt(distance);
}
#endif

#ifdef USE_RANGER_NEON
inline float l2DistanceNEON(const float* v1, const float* v2, size_t len)
{
    float* pV1 = (float*)v1;
    float* pV2 = (float*)v2;
    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t i;
    for (i = 0; i < len - 3; i += 4)
    {
        float32x4_t x1 = vld1q_f32(pV1);
        float32x4_t x2 = vld1q_f32(pV2);
        pV1 += 4;
        pV2 += 4;

        float32x4_t diff = vsubq_f32(x1, x2);
        float32x4_t squared = vmulq_f32(diff, diff);
        sum = vaddq_f32(sum, squared);
    }

    float32x2_t sum_low = vget_low_f32(sum);
    float32x2_t sum_high = vget_high_f32(sum);
    sum_low = vadd_f32(sum_low, sum_high);
    float distance = vget_lane_f32(sum_low, 0);

    // Calculate the remaining elements
    for (; i < len; ++i)
    {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }

    return std::sqrt(distance);
}
#endif

} // namespace ranger
} // namespace yuzu