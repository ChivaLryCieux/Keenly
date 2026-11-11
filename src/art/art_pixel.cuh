#pragma once
// The pixel shader: one stateless function mapping (x, y, params) -> RGB.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// This is the single source of truth for the art math. The GPU kernel
// (art_render.cu) and the CPU reference (art_ref.cu) both call fieldPixel(),
// so the verify gate compares the two *execution paths* with a tight relL2
// budget instead of drifting apart through duplicated code.
//
// Coordinates are resolution-independent: the pattern spans `view` noise
// units across the image height, so 512px previews and the full A1 render
// show the same picture at different detail densities.

#include "art_noise.cuh"

namespace art
{

struct FieldParams
{
    unsigned seed = 7u;
    float view = 3.2f;       // noise coordinate extent across the image height
    int octaves = 5;
    float lacunarity = 2.0f; // exact power of two: no rounding drift
    float gain = 0.5f;       // exact power of two
    float warp = 1.2f;       // domain warp strength
    float contrast = 1.4f;   // field shaping before display
};

// Two-level domain warp (after iq): q bends the first lookup, r bends the
// second, and the final fBm is sampled through r. This is what turns plain
// fBm smoke into flowing organic structure.
__host__ __device__ inline float warpField(float x, float y, const FieldParams& p)
{
    const unsigned s = p.seed;
    const int oc = p.octaves;
    const float lac = p.lacunarity;
    const float gn = p.gain;
    const float w4 = 4.0f * p.warp;

    const float qx = fbm2(x, y, s + 101u, oc, lac, gn);
    const float qy = fbm2(x + 5.2f, y + 1.3f, s + 211u, oc, lac, gn);
    const float rx = fbm2(x + w4 * qx + 1.7f, y + w4 * qy + 9.2f, s + 307u, oc, lac, gn);
    const float ry = fbm2(x + w4 * qx + 8.3f, y + w4 * qy + 2.8f, s + 401u, oc, lac, gn);
    return fbm2(x + w4 * rx, y + w4 * ry, s + 503u, oc, lac, gn);
}

// One pixel, linear RGB. Stage-1 look: the warped field as a luminance ramp.
// Palette / shading / grain land in later stages on top of this.
__host__ __device__ inline void fieldPixel(const FieldParams& p, int x, int y, int w, int h,
                                           float* rgb)
{
    const float aspect = float(w) / float(h);
    const float u = (float(x) + 0.5f) / float(h) - 0.5f * aspect;
    const float v = (float(y) + 0.5f) / float(h) - 0.5f;

    const float f = warpField(u * p.view, v * p.view, p);
    float t = 0.5f + 0.5f * f * p.contrast;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);

    rgb[0] = t;
    rgb[1] = t;
    rgb[2] = t;
}

}  // namespace art