#pragma once
// Gray-Scott reaction-diffusion ("Turing patterns") simulation core.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Unlike the noise fields, a Turing pattern is ITERATIVE grid state, not a
// closed-form function of (x, y). Determinism is preserved by construction:
//
//   - init is a pure function of (x, y, res, seed): hash-seeded blobs,
//   - every step is a JACOBI update: each cell reads only the previous
//     buffer, so the per-cell update order is identical on CPU and GPU
//     (a Gauss-Seidel in-place update would let tile scheduling change
//     the result - that is why it is NOT used),
//   - the update math lives in this one shared inline: +, -, * only, no
//     transcendentals, strict IEEE via --fmad=false, so the two execution
//     paths land on bit-identical values and the verify gate (maxAbs == 0)
//     still holds,
//   - edges wrap around (modulo), so the grid is TILEABLE and edge-free.
//
// Pipeline role: simulate once per image (the "brush"), then the pixel
// shader samples the grid through quintic-smoothed bilinear (sampleTuring),
// like any other field texture. Coordinates stay resolution-independent.

#include <cmath>
#include <cstdint>

#include "noise.cuh"

namespace keenly
{

// Wrap a lattice index into [0, n). C % keeps the sign of the dividend, so
// normalize explicitly; identical arithmetic on both sides.
__host__ __device__ inline int wrapi(int i, int n)
{
    int r = i % n;
    return r < 0 ? r + n : r;
}

// ---- deterministic init ----------------------------------------------------
// A = 1 everywhere; B = 0 except hash-seeded nuclei. Blob centers / radii are
// hashes of (blob index, seed) and distances wrap, so the seed field is
// tileable and reproducible.

__host__ __device__ inline float turingInitA(int x, int y, unsigned seed)
{
    (void)x; (void)y; (void)seed;
    return 1.0f;
}

__host__ __device__ inline float turingInitB(int x, int y, int res, unsigned seed)
{
    // sparse point noise: a few extra nuclei beyond the blobs
    if (hash01(x, y, seed + 991u) > 0.9993f) return 1.0f;

    const float R = float(res);
    for (int i = 0; i < 16; ++i) {
        const float cx = hash01(i, 11, seed) * R;
        const float cy = hash01(i, 29, seed) * R;
        const float r = (0.012f + 0.018f * hash01(i, 47, seed)) * R;
        float dx = float(x) + 0.5f - cx;
        float dy = float(y) + 0.5f - cy;
        if (dx > 0.5f * R) dx -= R;
        if (dx < -0.5f * R) dx += R;
        if (dy > 0.5f * R) dy -= R;
        if (dy < -0.5f * R) dy += R;
        if (dx * dx + dy * dy < r * r) return 1.0f;
    }
    return 0.0f;
}

// ---- one Jacobi step ------------------------------------------------------
// a[9] / b[9] are the 3x3 neighborhoods, row-major (center at [4]).
// Classic Gray-Scott with the Sims laplacian weights (center -1,
// edge 0.2, diagonal 0.05), Da = 1, Db = 0.5, dt = 1.
//
// The expression order below is the SINGLE source of truth for the update:
// both the CUDA kernel and the CPU reference call this exact inline, which
// is what keeps the two paths bit-identical.
__host__ __device__ inline void turingUpdate(const float* a, const float* b,
                                             float f, float k,
                                             float* outA, float* outB)
{
    const float lapA = 0.2f * (a[1] + a[3] + a[5] + a[7])
                     + 0.05f * (a[0] + a[2] + a[6] + a[8]) - a[4];
    const float lapB = 0.2f * (b[1] + b[3] + b[5] + b[7])
                     + 0.05f * (b[0] + b[2] + b[6] + b[8]) - b[4];

    const float abb = a[4] * b[4] * b[4];
    const float da = lapA - abb + f * (1.0f - a[4]);
    const float db = 0.5f * lapB + abb - (f + k) * b[4];

    float na = a[4] + da;
    float nb = b[4] + db;
    *outA = na < 0.0f ? 0.0f : (na > 1.0f ? 1.0f : na);
    *outB = nb < 0.0f ? 0.0f : (nb > 1.0f ? 1.0f : nb);
}

// ---- field sampling -------------------------------------------------------
// tex: res*res grid of the B channel (row-major). (x, y) in texel units,
// wrapping. Quintic-smoothed bilinear: C2 across texel borders, so relief
// shading over the pattern shows no bilinear creases.

__host__ __device__ inline float sampleTuring(const float* tex, int res, float x, float y)
{
    const int ix0 = wrapi(int(floorf(x)), res);
    const int iy0 = wrapi(int(floorf(y)), res);
    const int ix1 = wrapi(ix0 + 1, res);
    const int iy1 = wrapi(iy0 + 1, res);
    const float frx = x - floorf(x);
    const float fry = y - floorf(y);
    const float wx = fade(frx);
    const float wy = fade(fry);

    const float v00 = tex[size_t(iy0) * res + ix0];
    const float v10 = tex[size_t(iy0) * res + ix1];
    const float v01 = tex[size_t(iy1) * res + ix0];
    const float v11 = tex[size_t(iy1) * res + ix1];

    const float top = v00 + (v10 - v00) * wx;
    const float bot = v01 + (v11 - v01) * wx;
    const float b = top + (bot - top) * wy;

    // B is in ~[0, 0.4] on classic Gray-Scott; map to ~[-1, 1] around the
    // midtone so the palette / contrast knobs behave like the noise fields.
    const float raw = (0.20f - b) * 5.0f;
    return raw < -1.0f ? -1.0f : (raw > 1.0f ? 1.0f : raw);
}

}  // namespace keenly