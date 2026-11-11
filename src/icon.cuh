#pragma once
// The icon compositor: one stateless function mapping (x, y, params) -> RGBA.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Layer order, all in LINEAR light (tonemap happens in the shared post tail):
//   1. background  = fieldPixel (the art pipeline), dimmed for contrast
//   2. glow halo   = SDF distance falloff behind the glyph (powi, no exp)
//   3. glyph       = fill gradient + bevel rim + bright stroke (sdf.cuh)
//   4. frame       = tier 1-4 border with corner ornaments / gems
//   5. post tail   = post.cuh (grain damped inside the glyph), ACES at end
//   6. alpha       = rounded-rect / rect mask with ~1px antialiasing
//
// Like fieldPixel, this is a PURE function of (x, y, w, h, params) with
// strictly IEEE math only, so the GPU kernel and the CPU reference stay
// bit-identical and the verify gate (maxAbs == 0) covers icons too.
//
// Color values in IconColors are LINEAR radiance-like intensities, not
// perceptual palette stops: ACES compresses highlights, so the stroke/glow
// values run above 1.0 on purpose (the "lit" look of game icons).

#include "pixel.cuh"
#include "post.cuh"
#include "sdf.cuh"

namespace keenly
{

// Accent colors for one icon family (blue / red / gold / purple).
struct IconColors
{
    float fill[3];    // glyph body
    float hi[3];      // bevel rim / gem highlight
    float stroke[3];  // glyph outline + frame
    float glow[3];    // halo behind glyph + frame outer glow
    float shadow[3];  // deep-side of the fill gradient
};

struct IconParams
{
    FieldParams field;      // background art field (family palette lives here)
    PostParams post;        // post tail params (grain/vignette/tonemap)
    IconColors col;
    int glyph = kGlyphSword;
    int tier = 2;           // frame tier 1..4
    int roundedMask = 1;    // 1 = rounded rect, 0 = sharp rect
    float radius = 0.16f;   // corner radius, fraction of the min side
    float glyphScale = 0.86f;
    float bgDim = 0.85f;    // background dim under the glyph (contrast rule)
};

// Frame geometry in glyph units (u, v span [-1, 1] across the min side):
// frame ring sits at half-extent fi with corner rounding rf.
constexpr float kFrameFi = 0.865f;
constexpr float kFrameRf = 0.20f;

__host__ __device__ inline void iconPixel(const IconParams& p, int x, int y, int w, int h,
                                          float* rgba)
{
    const float m = float(w < h ? w : h);
    const float u = (float(x) + 0.5f - 0.5f * float(w)) / (0.5f * m);
    const float v = (float(y) + 0.5f - 0.5f * float(h)) / (0.5f * m);
    const float aa = 1.6f / m;   // ~1px smoothing in glyph units

    // ---- 1. background ----------------------------------------------------
    float rgb[3];
    fieldPixel(p.field, x, y, w, h, rgb);
    rgb[0] *= p.bgDim;
    rgb[1] *= p.bgDim;
    rgb[2] *= p.bgDim;

    // ---- glyph distance (also drives grain damping) -----------------------
    const float gs = p.glyphScale;
    const float d = glyphSdf(p.glyph, u / gs, v / gs) * gs;
    const float cov = sstep(aa, -aa, d);          // 1 inside the glyph
    const float bev = sstep(-0.16f, -0.03f, d);   // 0 at edge, 1 deep inside

    // ---- 2. glow halo behind the glyph ------------------------------------
    {
        const float halo = powi(1.0f - sstep(0.0f, 0.30f, d), 3);
        const float k = 0.30f * halo * (1.0f - cov);
        rgb[0] += p.col.glow[0] * k;
        rgb[1] += p.col.glow[1] * k;
        rgb[2] += p.col.glow[2] * k;
    }

    // ---- 3. glyph: gradient fill + bevel rim + stroke ---------------------
    if (cov > 0.0f) {
        // top-lit vertical gradient + edge-to-center brightening
        float tone = 0.78f + 0.42f * bev;
        tone *= 1.06f - 0.18f * (0.5f + 0.5f * v);
        float body[3];
        body[0] = p.col.fill[0] * tone + p.col.shadow[0] * (1.0f - bev) * 0.35f;
        body[1] = p.col.fill[1] * tone + p.col.shadow[1] * (1.0f - bev) * 0.35f;
        body[2] = p.col.fill[2] * tone + p.col.shadow[2] * (1.0f - bev) * 0.35f;

        // bevel rim highlight along the silhouette
        const float edge = sstep(-0.030f, -0.006f, d);
        for (int c = 0; c < 3; ++c)
            body[c] = body[c] * (1.0f - 0.45f * edge) + p.col.hi[c] * (0.45f * edge);

        for (int c = 0; c < 3; ++c)
            rgb[c] += (body[c] - rgb[c]) * cov;
    }

    // stroke: bright outline band around the silhouette
    {
        const float sw = 0.032f;  // ~2px at 128
        const float str = sstep(sw + aa, sw - aa, absf(d));
        const float k = str * (0.90f + 0.18f * bev);
        for (int c = 0; c < 3; ++c)
            rgb[c] += (p.col.stroke[c] * k - rgb[c]) * str;
    }

    // ---- 4. frame by tier -------------------------------------------------
    {
        const float ring = absf(sdBox(u, v, 0.0f, 0.0f, kFrameFi, kFrameFi, kFrameRf));
        const float fw = p.tier >= 2 ? 0.024f : 0.018f;
        const float frameCov = sstep(fw + aa, fw - aa, ring);

        // outer glow (t3+): soft halo just outside the ring
        if (p.tier >= 3) {
            const float out = ring - fw;
            const float halo = powi(1.0f - sstep(0.0f, 0.14f, out), 3);
            const float k = 0.22f * halo * (1.0f - frameCov);
            rgb[0] += p.col.glow[0] * k;
            rgb[1] += p.col.glow[1] * k;
            rgb[2] += p.col.glow[2] * k;
        }

        for (int c = 0; c < 3; ++c)
            rgb[c] += (p.col.stroke[c] * 1.05f - rgb[c]) * frameCov;

        // inner thin ring (t2+)
        if (p.tier >= 2) {
            const float ring2 = absf(sdBox(u, v, 0.0f, 0.0f, 0.775f, 0.775f, 0.17f));
            const float c2 = sstep(0.011f + aa, 0.011f - aa, ring2);
            for (int c = 0; c < 3; ++c)
                rgb[c] += (p.col.stroke[c] * 0.75f - rgb[c]) * c2 * 0.8f;
        }

        // corner ornaments: studs (t2), diamonds (t3), gems (t4)
        const float cc = kFrameFi - kFrameRf;  // corner arc centers
        for (int cy = -1; cy <= 1; cy += 2) {
            for (int cx = -1; cx <= 1; cx += 2) {
                const float px = float(cx) * cc;
                const float py = float(cy) * cc;
                if (p.tier >= 4) {
                    // gem: diamond body + bright core
                    const float g = sdDiamond(u, v, px, py, 0.075f);
                    const float gc = sstep(aa, -aa, g);
                    const float core = sstep(aa, -aa, sdDiamond(u, v, px, py, 0.032f));
                    for (int c = 0; c < 3; ++c) {
                        const float gem = p.col.fill[c] * 1.25f * (1.0f - 0.5f * core)
                                        + p.col.hi[c] * 1.10f * core;
                        rgb[c] += (gem - rgb[c]) * gc;
                    }
                } else if (p.tier >= 3) {
                    const float g = sdDiamond(u, v, px, py, 0.055f);
                    const float gc = sstep(aa, -aa, g);
                    for (int c = 0; c < 3; ++c)
                        rgb[c] += (p.col.stroke[c] - rgb[c]) * gc;
                } else if (p.tier >= 2) {
                    const float g = sdCircle(u, v, px, py, 0.042f);
                    const float gc = sstep(aa, -aa, g);
                    for (int c = 0; c < 3; ++c)
                        rgb[c] += (p.col.stroke[c] * 0.92f - rgb[c]) * gc;
                }
            }
        }
    }

    // ---- 5. shared post tail, grain damped inside the glyph ---------------
    const float grainMul = 1.0f - 0.65f * cov;
    postTail(rgb, x, y, w, h, p.post, grainMul);

    // ---- 6. mask alpha ----------------------------------------------------
    const float hx = float(w) / m;
    const float hy = float(h) / m;
    float rad = 0.0f;
    if (p.roundedMask) {
        rad = 2.0f * p.radius;
        const float lim = (hx < hy ? hx : hy) - 0.02f;
        rad = rad > lim ? lim : rad;
    }
    const float maskD = sdBox(u, v, 0.0f, 0.0f, hx, hy, rad);
    const float alpha = sstep(aa, -aa, maskD);

    rgba[0] = rgb[0];
    rgba[1] = rgb[1];
    rgba[2] = rgb[2];
    rgba[3] = alpha;
}

}  // namespace keenly