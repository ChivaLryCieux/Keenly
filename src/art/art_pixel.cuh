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

// Color ramp: up to 6 stops, strictly increasing positions. Stops live in
// perceptual (sRGB-like) space - easier to author - and paletteLookup()
// linearizes with gamma 2.0 (a single multiply: exact on CPU and GPU).
struct Palette
{
    int count = 2;
    float pos[6] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float rgb[6][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };
};

// Field generators. Each is a pure function of (u, v, x, y, params), so any
// of them can render at any size, tiled or whole.
enum FieldMode
{
    kFieldWarp = 0,   // two-level domain-warped fBm: flowing organic smoke
    kFieldRidge = 1,  // warped ridged fBm: marble veins / rock strata
    kFieldBands = 2,  // triangle-wave contours of the warped field: flow lines
    kFieldFrost = 3,  // fine Worley facets + sparkle: crystalline frosted glass
    kFieldFacet = 4,  // hard-edged Worley crystal: flat facets, bright seams
    kFieldPrism = 5,  // folded triangle lattice: austere chevron / diamond grid
    kFieldLiquid = 6  // glossy metal: flowing warp field + specular shading
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
    float macroAmt = 1.0f;   // layered composition strength (0 = flat v1 look)
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
            const float tri = triFrac(f);
            return 2.0f * sstep(0.10f, 0.90f, tri) - 1.0f;
        }
        case kFieldFrost: {
            // crystalline frosted glass: fine Worley facets with dark seams
            // and a per-pixel speckle bed. The sparkle highlights live in the
            // shading stage (powi 90 specular).
            const WorleyResult w = worley2(px * 5.5f, py * 5.5f, s);
            const float bevel = sstep(0.0f, 0.18f, w.f2 - w.f1);
            const float sp = hash01(x, y, s + 3433u) * 2.0f - 1.0f;
            return (w.val * 0.50f + bevel * 0.45f - 0.475f) * 1.6f + sp * 0.16f;
        }
        case kFieldFacet: {
            // austere crystal: big flat facets (one tone + one light tilt per
            // cell), bright seams where facets meet. Hard geometry, no smoke.
            const WorleyResult w = worley2(px * 3.0f, py * 3.0f, s);
            const float bevel = sstep(0.0f, 0.22f, w.f2 - w.f1);
            const float tilt = hash01(w.cx, w.cy, s + 77u) * 2.0f - 1.0f;
            const float lit = 0.5f + 0.5f * tilt;  // per-facet light angle
            return w.val * 0.55f + bevel * 0.35f + lit * 0.45f - 0.675f;
        }
        case kFieldPrism: {
            // austere chevron lattice: two triangle-wave fields folded into
            // diamonds on a 30-degree rotated frame, plus a hard light ramp
            // across the picture. Everything straight, everything sharp.
            const float rx = px * 0.8660254f + py * 0.5f;
            const float ry = -px * 0.5f + py * 0.8660254f;
            const float fq = p.bands * 0.5f;
            const float b1 = triFrac(rx * fq);
            const float b2 = triFrac(ry * fq);
            const float b = b1 < b2 ? b1 : b2;
            const float t = sstep(0.02f, 0.55f, b);
            return (2.0f * t - 1.0f) * 0.85f + 0.30f * (u + v);
        }
        case kFieldLiquid: {
            // glossy liquid metal: same flowing warp field pushed wider so
            // the palette reaches its extremes, and the shading stage turns
            // specular (powi 16) with a broad glossy highlight.
            return warpField(px, py, p) * 1.30f;
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

    // ---- layered composition -------------------------------------------------
    // Three scales, and this is what separates "a painting" from "uniform
    // noise texture": macro forms decide the light/dark layout of the frame
    // and drift the hue; the main field is the flow; fine detail adds
    // surface variation. The macro layer is what compositional depth means
    // for a procedural image - regions of light and calm, not flat texture
    // everywhere. (Only evaluated once per pixel; relief gradients sample
    // the main field alone.)
    const float macro = fbm2(u * p.view * 0.42f, v * p.view * 0.42f, p.seed + 1009u,
                             p.octaves, p.lacunarity, p.gain) * p.macroAmt;
    if (p.mode == kFieldWarp || p.mode == kFieldRidge || p.mode == kFieldLiquid) {
        const float det = fbm2(u * p.view * 2.3f, v * p.view * 2.3f, p.seed + 2003u,
                               p.octaves - 1, p.lacunarity, p.gain);
        tc += 0.11f * det * p.contrast * p.macroAmt;
    }
    tc += 0.10f * macro;  // chroma drift follows the macro forms
    tc = tc < 0.0f ? 0.0f : (tc > 1.0f ? 1.0f : tc);
    paletteLookup(p.pal, tc, rgb);

    // perceptual -> linear (gamma 2.0, one multiply, bit-exact on both sides)
    rgb[0] *= rgb[0];
    rgb[1] *= rgb[1];
    rgb[2] *= rgb[2];

    // Macro light: painterly light/dark regions PLUS a soft whole-frame
    // directional gradient (light from upper-left), so the picture has a lit
    // corner and a calm corner - composition, not uniform texture.
    const float compo = macro - 0.45f * (u * 0.55f + v) * p.macroAmt;
    float cm = 0.5f + 0.5f * compo;
    cm = cm < 0.0f ? 0.0f : (cm > 1.0f ? 1.0f : cm);
    // macroAmt 0 -> flat 1.0 (the v1 look), 1 -> full painterly light
    const float light = 1.0f + (0.74f + 0.46f * cm - 1.0f) * p.macroAmt;
    rgb[0] *= light;
    rgb[1] *= light;
    rgb[2] *= light;

    // Unnormalized fake normal (-gx, -gy, 1) dotted with a unit light:
    // no sqrt, no clamped normalization, cheap and stable. Fine slope at
    // full strength, coarse slope at 0.35x so big forms stay smooth.
    const float gx = (tx - t) * p.relief + (tcx - t) * p.relief * 0.35f;
    const float gy = (ty - t) * p.relief + (tcy - t) * p.relief * 0.35f;
    float diff = 0.733f + 0.40f * gx + 0.55f * gy;
    diff = diff < 0.0f ? 0.0f : (diff > 1.0f ? 1.0f : diff);
    float shade = 0.45f + 0.85f * diff;

    // Per-mode specular character. All integer powers (powi), so still
    // bit-exact: liquid gets a broad glossy highlight, facets get hard
    // glints on bevels, frost gets micro sparkle points.
    if (p.mode == kFieldLiquid) {
        shade += 0.45f * powi(diff, 16);   // broad glossy brightening
    } else if (p.mode == kFieldFacet) {
        shade += 0.85f * powi(diff, 48);   // hard facet glints
    } else if (p.mode == kFieldFrost) {
        shade += 1.10f * powi(diff, 90);   // micro sparkle
    }

    rgb[0] *= shade;
    rgb[1] *= shade;
    rgb[2] *= shade;

    // Liquid metal gets an ADDITIVE white glint on top of the palette: a
    // reflection is light, not paint. Slightly warm-tinted like tungsten.
    if (p.mode == kFieldLiquid) {
        const float glint = powi(diff, 16) * 0.38f;
        rgb[0] += glint;
        rgb[1] += glint * 0.94f;
        rgb[2] += glint * 0.82f;
    }
}

}  // namespace art