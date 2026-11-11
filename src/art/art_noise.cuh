#pragma once
// Stateless noise math for the art pipeline.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Every function is a pure function of integer coordinates + seed:
//   - deterministic: same seed -> same picture, on CPU and GPU alike,
//   - tileable at any size: no hidden global state (future A1 tiling),
//   - shared by the GPU kernels and the CPU reference (art_ref.cu), so the
//     verify gate compares execution paths, not two divergent copies of math.
//
// Bit-level notes: hashing is integer-only (exact on both sides); noise uses
// only +,-,* and one polynomial fade, so fast-math rounding stays ~1e-7 and
// the verify gate can use a tight relL2 budget. Transcendentals (sin/cos)
// are avoided on purpose by picking gradients from a fixed 8-direction set.

#include <cmath>
#include <cstdint>

namespace art
{

// ---- integer hashing ------------------------------------------------------
// lowbias32 finalizer: three mixes of xor-shift + multiply, integer only.
__host__ __device__ inline uint32_t hashMix(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__host__ __device__ inline uint32_t hash2i(int x, int y, unsigned seed)
{
    uint32_t h = uint32_t(x) * 0x9E3779B9u + uint32_t(y) * 0x85EBCA6Bu
               + uint32_t(seed) * 0xC2B2AE35u;
    return hashMix(h);
}

// Uniform [0, 1) from the top 24 hash bits (2^-24 step, exact scaling).
__host__ __device__ inline float hash01(int x, int y, unsigned seed)
{
    return float(hash2i(x, y, seed) >> 8) * (1.0f / 16777216.0f);
}

// ---- gradient (Perlin-style) noise ----------------------------------------

// Quintic fade: C2-continuous, no derivatives visible at lattice seams.
__host__ __device__ inline float fade(float t)
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Pick a unit gradient from 8 fixed directions (cos/sin of k*pi/4), chosen
// by hash bits. No trig, no tables in device code: CPU/GPU get identical
// values. The 8-direction set is enough once fBm sums several octaves.
__host__ __device__ inline float dotGrad(int xi, int yi, float dx, float dy, unsigned seed)
{
    const float c = 0.70710678f;
    float gx, gy;
    switch (hash2i(xi, yi, seed) & 7u) {
        case 0:  gx = 1.0f;  gy = 0.0f;  break;
        case 1:  gx = c;     gy = c;     break;
        case 2:  gx = 0.0f;  gy = 1.0f;  break;
        case 3:  gx = -c;    gy = c;     break;
        case 4:  gx = -1.0f; gy = 0.0f;  break;
        case 5:  gx = -c;    gy = -c;    break;
        case 6:  gx = 0.0f;  gy = -1.0f; break;
        default: gx = c;    gy = -c;    break;
    }
    return gx * dx + gy * dy;
}

// 2D gradient noise in roughly [-1, 1]. Smooth in both axes, zero mean.
__host__ __device__ inline float gradNoise2(float x, float y, unsigned seed)
{
    const float xf = floorf(x);  // floor, so negative coords behave
    const float yf = floorf(y);
    const int xi = int(xf);
    const int yi = int(yf);
    const float dx = x - xf;
    const float dy = y - yf;
    const float u = fade(dx);
    const float v = fade(dy);

    const float n00 = dotGrad(xi,     yi,     dx,       dy,       seed);
    const float n10 = dotGrad(xi + 1, yi,     dx - 1.0f, dy,      seed);
    const float n01 = dotGrad(xi,     yi + 1, dx,       dy - 1.0f, seed);
    const float n11 = dotGrad(xi + 1, yi + 1, dx - 1.0f, dy - 1.0f, seed);

    const float top = n00 + u * (n10 - n00);
    const float bot = n01 + u * (n11 - n01);
    // 1.41421356: unit gradients peak near 0.707, rescale to ~[-1, 1].
    return (top + v * (bot - top)) * 1.41421356f;
}

// Fractal Brownian motion: summed octaves, normalized to ~[-1, 1].
// gain 0.5 / lacunarity 2.0 are exact powers of two (no rounding drift).
__host__ __device__ inline float fbm2(float x, float y, unsigned seed,
                                      int octaves, float lacunarity, float gain)
{
    float sum = 0.0f;
    float amp = 1.0f;
    float norm = 0.0f;
    for (int o = 0; o < octaves; ++o) {
        sum += amp * gradNoise2(x, y, seed + uint32_t(o) * 1013u);
        norm += amp;
        amp *= gain;
        x *= lacunarity;
        y *= lacunarity;
    }
    return sum * (1.0f / norm);
}

// Ridged fBm: |noise| inverted into sharp creases (mountain / marble veins).
__host__ __device__ inline float fbm2Ridged(float x, float y, unsigned seed,
                                           int octaves, float lacunarity, float gain)
{
    float sum = 0.0f;
    float amp = 1.0f;
    float norm = 0.0f;
    for (int o = 0; o < octaves; ++o) {
        float n = gradNoise2(x, y, seed + uint32_t(o) * 1013u);
        n = 1.0f - (n < 0.0f ? -n : n);  // fabs without libm
        sum += amp * n * n;
        norm += amp;
        amp *= gain;
        x *= lacunarity;
        y *= lacunarity;
    }
    return sum * (1.0f / norm) * 2.0f - 1.0f;
}

}  // namespace art