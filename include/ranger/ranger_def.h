#pragma once

// For SSE
#if defined(__SSE__)
#define SUPPORTS_SSE 1
#else
#define SUPPORTS_SSE 0
#endif

// For AVX and AVX2
#if defined(__AVX__) || defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX))
#define SUPPORTS_AVX 1
#else
#define SUPPORTS_AVX 0
#endif

// Check for AVX512
#if defined(__AVX512F__) || defined(__AVX512CD__) || defined(__AVX512VL__)
#define SUPPORTS_AVX512 1
#else
#define SUPPORTS_AVX512 0
#endif

#if SUPPORTS_SSE && defined(Enable_Ranger_SSE)
#define USE_RANGER_SSE
#include <xmmintrin.h>
#endif

#if SUPPORTS_AVX && defined(Enable_Ranger_AVX)
#define USE_RANGER_AVX
#include <immintrin.h>
#endif

#if SUPPORTS_AVX512 && defined(Enable_Ranger_AVX)
#define USE_RANGER_AVX512
#include <immintrin.h>
#endif
