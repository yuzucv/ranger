#include <array>
#include <iostream>
#include <pillar/utility/timeit.h>
#include <random>

#include "ranger/dist_l2.h"

std::random_device rd;
std::mt19937 gen(rd());
constexpr int FeatureDim = 4096 + 7;

template <int N>
std::array<float, N> newTensor(std::uniform_real_distribution<float>& dist)
{
    std::array<float, N> tensor;
    for (int i = 0; i < N; i++)
    {
        tensor[i] = dist(gen);
    }
    return tensor;
};

using namespace yuzu;
using namespace yuzu::ranger;

int main()
{
    std::uniform_real_distribution<float> dist;

    auto feature1 = newTensor<FeatureDim>(dist);
    auto feature2 = newTensor<FeatureDim>(dist);
    static_assert(feature1.size() == feature2.size(), "向量维度不匹配");
    constexpr auto fsize = feature1.size();
    constexpr auto epoch = 10;

    float result[16] = {0.f};

    {
        Timer t;
        result[0] = l2Distance(feature1.data(), feature2.data(), fsize);
    }

    {
        Timer t;
        result[1] = l2DistanceX4<fsize>(feature1.data(), feature2.data());
    }
#ifdef USE_RANGER_SSE
    {
        Timer t;
        result[2] = l2DistanceSSE(feature1.data(), feature2.data(), fsize);
    }

    {
        Timer t;
        result[3] = l2DistanceSSE<fsize>(feature1.data(), feature2.data());
    }
#endif
#ifdef USE_RANGER_AVX
    {
        Timer t;
        result[4] = l2DistanceAVX(feature1.data(), feature2.data(), fsize);
    }

    {
        Timer t;
        result[5] = l2DistanceAVX<fsize>(feature1.data(), feature2.data());
    }
#endif
#ifdef USE_RANGER_NEON
    {
        Timer t;
        result[6] = l2DistanceNEON(feature1.data(), feature2.data(), fsize);
    }
#endif

    std::cout << "normal l2 distance: " << result[0] << std::endl;
    std::cout << "4X l2 distance: " << result[1] << std::endl;
#ifdef USE_RANGER_SSE
    std::cout << "SSE l2 distance: " << result[2] << std::endl;
    std::cout << "SSE l2 distance: " << result[3] << std::endl;
#endif
#ifdef USE_RANGER_AVX
    std::cout << "AVX l2 distance: " << result[4] << std::endl;
    std::cout << "AVX l2 distance: " << result[5] << std::endl;
#endif
#ifdef USE_RANGER_NEON
    std::cout << "NEON l2 distance: " << result[6] << std::endl;
#endif
    return 0;
}