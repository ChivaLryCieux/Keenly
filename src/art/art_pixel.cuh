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
//
// Pipeline stage 1 (this file): warped fBm field -> palette -> relief shading.
// Output is LINEAR RGB (possibly > 1 in highlights); tonemap / grain / dither
// are stage 2 (post) and encoding happens at PNG write time.

#include "art_noise.cuh"

namespace art
{

// Color ramp: up to 5 stops, strictly increasing positions. Stops live in
// perceptual (sRGB-like) space - easier to author - and paletteLookup()
// linearizes with gamma 2.0 (a single multiply: exact on CPU and GPU).
struct Palette
{
    int count = 2;
    float pos[5] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    float rgb[5][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };
};

// Field generators. Each is a pure function of (u, v, x, y, params), so any
// of them can render at any size, tiled or whole.
enum FieldMode
{
    kFieldWarp = 0,  // two-level domain-warped fBm: flowing organic smoke
    kFieldRidge = 1, // warped ridged fBm: marble veins / rock strata
    kFieldBands = 2, // triangle-wave contours of the warped field: flow lines
    kFieldFrost = 3  // high-frequency fBm + per-pixel speckle: frosted matte
};

struct FieldParams
{
    unsigned seed = 7u;
    float view = 3.2f;       // noise coordinate extent across the image height
    int octaves = 5;
    float lacunarity = 2.0f; // exact power of two: no rounding drift
    float gain = 0.5f;       // exact power of two
    float warp = 1.2f;       // domain warp strength
    float contrast = 1.4f;   // field shaping before the palette
    float relief = 2.5f;     // height-to-normal strength for the shading
    int mode = kFieldWarp;   // which field generator
    float bands = 12.0f;     // contour frequency for kFieldBands
    Palette pal;             // color ramp (perceptual stops)
};

__host__ __device__ inline void paletteLookup(const Palette& pal, float t, float* rgb)
{
    const int last = pal.count - 1;
    if (t <= pal.pos[0]) t = pal.pos[0];
    if (t >= pal.pos[last]) {
        rgb[0] = pal.rgb[last][0];
        rgb[1] = pal.rgb[last][1];
        rgb[2] = pal.rgb[last][2];
        return;
    }
    int i = 0;
    while (i + 1 < pal.count && t > pal.pos[i + 1]) ++i;
    const float span = pal.pos[i + 1] - pal.pos[i];
    float u = (t - pal.pos[i]) / span;
    u = u * u * (3.0f - 2.0f * u);  // smoothstep: no visible band at stops
    for (int c = 0; c < 3; ++c)
        rgb[c] = pal.rgb[i][c] + u * (pal.rgb[i + 1][c] - pal.rgb[i][c]);
}

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

// Raw field value in ~[-1, 1] for one generator, coordinates (u, v) in image
// units (height = 1). (x, y) only feeds kFieldFrost's per-pixel speckle -
// and since the speckle is the same in the gradient samples, it cancels out
// of the relief shading by construction.
__host__ __device__ inline float fieldValue(const FieldParams& p, float u, float v, int x, int y)
{
    const float px = u * p.view;
    const float py = v * p.view;
    const unsigned s = p.seed;
    const int oc = p.octaves;
    const float lac = p.lacunarity;
    const float gn = p.gain;

    switch (p.mode) {
        case kFieldRidge: {
            // ridged fBm sampled through a gentle warp: marble veins
            const float qx = fbm2(px, py, s + 211u, oc, lac, gn);
            const float qy = fbm2(px + 5.2f, py + 1.3f, s + 307u, oc, lac, gn);
            return fbm2Ridged(px + p.warp * 2.0f * qx, py + p.warp * 2.0f * qy,
                              s + 401u, oc, lac, gn);
        }
        case kFieldBands: {
            // triangle-wave contours of the warped field: flow lines / wood
            // grain / topographic bands. Polynomial only (frac + folds), so
            // CPU/GPU still agree bit-for-bit.
            const float f = warpField(px, py, p) * p.bands;
            const float fr = f - floorf(f);            // frac, exact
            const float tri = fr < 0.5f ? 2.0f * fr : 2.0f * (1.0f - fr);
            return 2.0f * sstep(0.10f, 0.90f, tri) - 1.0f;
        }
        case kFieldFrost: {
            // high-frequency fBm + per-pixel speckle: sandblasted matte
            const float f = fbm2(px * 2.5f, py * 2.5f, s + 59u, oc, lac, gn);
            const float sp = hash01(x, y, s + 3433u) * 2.0f - 1.0f;
            return f * 0.85f + sp * 0.30f;
        }
        default:
            return warpField(px, py, p);
    }
}

// One pixel, linear RGB. Field value drives the palette; finite-difference
// gradients of the same field at TWO scales drive the fake normal (light
// fixed at upper-left): a coarse slope models the big forms, a fine slope
// adds micro relief - "matte embossed" depth instead of flat noise.
__host__ __device__ inline void fieldPixel(const FieldParams& p, int x, int y, int w, int h,
                                           float* rgb)
{
    const float aspect = float(w) / float(h);
    const float u = (float(x) + 0.5f) / float(h) - 0.5f * aspect;
    const float v = (float(y) + 0.5f) / float(h) - 0.5f;
    const float step = 1.2f / float(h);   // fine relief over ~1 pixel
    const float cstep = 3.6f / float(h);  // coarse relief over ~3 pixels

    const float t   = fieldValue(p, u, v, x, y);
    const float tx  = fieldValue(p, u + step, v, x, y);
    const float ty  = fieldValue(p, u, v + step, x, y);
    const float tcx = fieldValue(p, u + cstep, v, x, y);
    const float tcy = fieldValue(p, u, v + cstep, x, y);

    float tc = 0.5f + 0.5f * t * p.contrast;
    tc = tc < 0.0f ? 0.0f : (tc > 1.0f ? 1.0f : tc);
    paletteLookup(p.pal, tc, rgb);

    // perceptual -> linear (gamma 2.0, one multiply, bit-exact on both sides)
    rgb[0] *= rgb[0];
    rgb[1] *= rgb[1];
    rgb[2] *= rgb[2];

    // Unnormalized fake normal (-gx, -gy, 1) dotted with a unit light:
    // no sqrt, no clamped normalization, cheap and stable. Fine slope at
    // full strength, coarse slope at 0.35x so big forms stay smooth.
    const float gx = (tx - t) * p.relief + (tcx - t) * p.relief * 0.35f;
    const float gy = (ty - t) * p.relief + (tcy - t) * p.relief * 0.35f;
    float diff = 0.733f + 0.40f * gx + 0.55f * gy;
    diff = diff < 0.0f ? 0.0f : (diff > 1.0f ? 1.0f : diff);
    const float shade = 0.45f + 0.85f * diff;  // ambient + diffuse, ~[0.45, 1.30]

    rgb[0] *= shade;
    rgb[1] *= shade;
    rgb[2] *= shade;
}

}  // namespace art