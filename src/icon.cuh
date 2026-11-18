#pragma once
// The icon compositor: one stateless function mapping (x, y, params) -> RGBA.
//
// ASCII-ONLY FILE ON PURPOSE (see src/cuda_utils.h for the reason).
//
// Layer order, all in LINEAR light (tonemap happens in the shared post tail):
//   1. background  = fieldPixel (family path) or rainbowBg (generic path,
//                    BgMode styles), dimmed for contrast
//   2. glow halo   = SDF distance falloff behind the glyph (powi, no exp);
//                    skipped in flat mode
//   3. glyph       = fill gradient + bevel rim + bright stroke (sdf.cuh), or
//                    a pure fgFill silhouette when flat=1
//   4. frame       = tier 1-4 border with corner ornaments / gems (tier 0
//                    skips it entirely)
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

// Generic icon backgrounds: instead of a family field, the icon sits on an
// even seven-hue rainbow field. bgMode picks the pattern language.
enum BgMode
{
    kBgField = 0,  // family field via fieldPixel (the classic path)
    kBgFluid = 1,  // organic warp field through a rainbow palette
    kBgSwirl = 2,  // conic hue wheel advected by the warp field
    kBgBands = 3,  // flowing rainbow contour stripes
    kBgRings = 4,  // concentric rainbow rings, slightly warped
    kBgMosaic = 5, // per-cell rainbow stained glass
    kBgGlow = 6    // poster glow blobs: dark center pool, colors at edges
};

struct IconParams
{
    FieldParams field;      // background art field (family palette lives here)
    PostParams post;        // post tail params (grain/vignette/tonemap)
    IconColors col;
    int glyph = kGlyphSword;
    int tier = 2;           // frame tier 0..4 (0 = no frame)
    int roundedMask = 1;    // 1 = rounded rect, 0 = sharp rect
    float radius = 0.16f;   // corner radius, fraction of the min side
    float glyphScale = 0.86f;
    float bgDim = 0.85f;    // background dim under the glyph (contrast rule)

    // ---- generic icon pipeline (bgMode != kBgField) ------------------------
    int bgMode = kBgField;      // rainbow background style (see BgMode)
    int flat = 0;               // 1 = pure fg silhouette, 0 = shaded glyph
    float fgFill[3] = {1.0f, 1.0f, 1.0f};   // pure glyph color, linear
    float fgStroke[3] = {0.0f, 0.0f, 0.0f}; // glyph keyline + frame color
    float coreLift = 0.0f;      // kBgGlow: floor of the dark pool under glyph
};

// Frame geometry in glyph units (u, v span [-1, 1] across the min side):
// frame ring sits at half-extent fi with corner rounding rf.
constexpr float kFrameFi = 0.865f;
constexpr float kFrameRf = 0.20f;

// ---- rainbow backgrounds ----------------------------------------------------
// Seven perceptual hues, evenly spaced. A wrapping palette repeats red at the
// far end so periodic hue fields (swirl/bands/rings) close the circle.

__host__ __device__ inline void rainbowStop(int i, float* c)
{
    switch (i % 7) {
        case 0:  c[0] = 0.94f; c[1] = 0.10f; c[2] = 0.08f; break;  // red
        case 1:  c[0] = 0.98f; c[1] = 0.45f; c[2] = 0.05f; break;  // orange
        case 2:  c[0] = 1.00f; c[1] = 0.85f; c[2] = 0.05f; break;  // yellow
        case 3:  c[0] = 0.16f; c[1] = 0.75f; c[2] = 0.18f; break;  // green
        case 4:  c[0] = 0.05f; c[1] = 0.70f; c[2] = 0.78f; break;  // cyan
        case 5:  c[0] = 0.12f; c[1] = 0.30f; c[2] = 0.95f; break;  // blue
        default: c[0] = 0.58f; c[1] = 0.20f; c[2] = 0.92f; break;  // violet
    }
}

// wrap=1: 8 stops at k/7, ends on red -> periodic hue wraps cleanly.
// wrap=0: 7 stops at k/6 for a one-shot red..violet ramp.
__host__ __device__ inline void rainbowPalette(Palette& pal, int wrap)
{
    pal.count = wrap ? 8 : 7;
    const float span = wrap ? 7.0f : 6.0f;
    for (int i = 0; i < pal.count; ++i) {
        pal.pos[i] = float(i) / span;
        rainbowStop(i, pal.rgb[i]);
    }
}

// Conic hue in [0, 1): which eighth of the circle plus a within-octant
// fraction. Comparisons + one IEEE division stand in for atan2, so the
// bit-exact gate still holds. Octant order follows the pixel frame
// (y downward); the permutation below walks the circle continuously.
__host__ __device__ inline float conicHue(float u, float v)
{
    const float ax = absf(u), ay = absf(v);
    const float big = mx(ax, ay);
    const float r = big > 0.0f ? mn(ax, ay) / big : 0.0f;
    int oct = 0;
    if (u < 0.0f) oct += 4;
    if (v < 0.0f) oct += 2;
    if (ay > ax) oct += 1;
    int pos;
    switch (oct) {
        case 0: pos = 0; break;
        case 1: pos = 1; break;
        case 5: pos = 2; break;
        case 4: pos = 3; break;
        case 6: pos = 4; break;
        case 7: pos = 5; break;
        case 3: pos = 6; break;
        default: pos = 7; break;  // oct 2
    }
    const float frac = (ay > ax) ? 1.0f - r : r;
    return (float(pos) + frac) * 0.125f;
}

// Gentle one-level domain warp for icon backgrounds. The stills pipeline
// uses warpField (two-level, w4 = 4*warp displacement) which reads as rich
// turbulence under a luminance palette - but under a 7-stop hue ramp the
// same displacement tears into per-pixel color confetti. Here the lookup
// bends by ~2*warp noise units at most, so the marble swirl stays smooth.
__host__ __device__ inline float rainbowField(float px, float py,
                                              const FieldParams& f)
{
    const unsigned s = f.seed;
    const int oc = f.octaves;
    const float lac = f.lacunarity;
    const float gn = f.gain;
    const float qx = fbm2(px, py, s + 101u, oc, lac, gn);
    const float qy = fbm2(px + 5.2f, py + 1.3f, s + 211u, oc, lac, gn);
    const float wv = 2.0f * f.warp;
    return fbm2(px + wv * qx, py + wv * qy, s + 503u, oc, lac, gn);
}

// Rainbow background for the generic icon pipeline. (u, v) are the icon's
// centered, min-side-normalized coords. Output is linear RGB.
__host__ __device__ inline void rainbowBg(const IconParams& p, float u, float v,
                                          float* rgb)
{
    const FieldParams& f = p.field;
    const float px = u * f.view;
    const float py = v * f.view;
    const unsigned s = f.seed;

    // Most styles drive hue through the organic field; evaluate it once and
    // reuse it for the cheap relief gradient below.
    float w0 = 0.0f;
    WorleyResult wc;
    if (p.bgMode == kBgMosaic) wc = worley2(px * 1.1f, py * 1.1f, s);
    else                       w0 = rainbowField(px, py, f);

    float hue;
    if (p.bgMode == kBgSwirl) {
        hue = conicHue(u, v) + 0.26f * w0;             // twisted color wheel
    } else if (p.bgMode == kBgBands) {
        hue = w0 * (f.bands * 0.14f);                  // flowing hue stripes
    } else if (p.bgMode == kBgRings) {
        const float r = sqrtf(u * u + v * v);
        hue = r * (f.bands * 0.30f) + 0.12f * w0;      // warped concentric
    } else if (p.bgMode == kBgMosaic) {
        hue = wc.val;
    } else {  // kBgFluid
        const float t = 0.5f + 0.5f * w0 * f.contrast;
        hue = t;
    }

    const int wrap = (p.bgMode != kBgFluid);
    Palette pal;
    rainbowPalette(pal, wrap);
    float hh = wrap ? hue - floorf(hue)
                    : (hue < 0.0f ? 0.0f : (hue > 1.0f ? 1.0f : hue));
    paletteLookup(pal, hh, rgb);

    // perceptual -> linear, same squaring as fieldPixel
    rgb[0] *= rgb[0];
    rgb[1] *= rgb[1];
    rgb[2] *= rgb[2];

    if (p.bgMode == kBgMosaic) {
        // Stained glass: darken the seams between cells.
        const float seam = sstep(0.0f, 0.07f, wc.f2 - wc.f1);
        const float m = 0.35f + 0.65f * seam;
        rgb[0] *= m;
        rgb[1] *= m;
        rgb[2] *= m;
    } else {
        // Organic depth: warp-field gradient as a fake normal, light fixed
        // upper-left (same trick as fieldPixel, tuned softer for icons).
        const float e = 0.07f;
        const float gx = (rainbowField(px + e, py, f) - w0) * f.relief;
        const float gy = (rainbowField(px, py + e, f) - w0) * f.relief;
        float diff = 0.55f + 0.32f * gx + 0.46f * gy;
        diff = diff < 0.0f ? 0.0f : (diff > 1.0f ? 1.0f : diff);
        const float shade = 0.82f + 0.28f * diff;
        // gentle luminance drift so the hue coverage stays even but the
        // surface is not dead flat
        const float lum = 0.92f + 0.16f *
            fbm2(px * 0.35f, py * 0.35f, s + 1009u, 3, f.lacunarity, f.gain)
            * f.macroAmt;
        const float m = shade * lum;
        rgb[0] *= m;
        rgb[1] *= m;
        rgb[2] *= m;
    }
}

__host__ __device__ inline void iconPixel(const IconParams& p, int x, int y, int w, int h,
                                          float* rgba)
{
    const float m = float(w < h ? w : h);
    const float u = (float(x) + 0.5f - 0.5f * float(w)) / (0.5f * m);
    const float v = (float(y) + 0.5f - 0.5f * float(h)) / (0.5f * m);
    const float aa = 1.6f / m;   // ~1px smoothing in glyph units

    // ---- 1. background ----------------------------------------------------
    float rgb[3];
    if (p.bgMode == kBgField) {
        fieldPixel(p.field, x, y, w, h, rgb);
    } else if (p.bgMode == kBgGlow) {
        // poster glow, icon-framed: u,v in [-1,1] -> the [-0.5,0.5] poster
        // domain, so the dark pool and the top/bottom edge glows land inside
        // the icon. Blob colors come from p.field.pal (a glow scheme).
        glowField(p.field, u * 0.5f, v * 0.5f, rgb, p.coreLift);
    } else {
        rainbowBg(p, u, v, rgb);
    }
    rgb[0] *= p.bgDim;
    rgb[1] *= p.bgDim;
    rgb[2] *= p.bgDim;

    // ---- glyph distance (also drives grain damping) -----------------------
    const float gs = p.glyphScale;
    const float d = glyphSdf(p.glyph, u / gs, v / gs) * gs;
    const float cov = sstep(aa, -aa, d);          // 1 inside the glyph
    const float bev = sstep(-0.16f, -0.03f, d);   // 0 at edge, 1 deep inside

    // ---- 2. glow halo behind the glyph ------------------------------------
    if (!p.flat) {
        const float halo = powi(1.0f - sstep(0.0f, 0.30f, d), 3);
        const float k = 0.30f * halo * (1.0f - cov);
        rgb[0] += p.col.glow[0] * k;
        rgb[1] += p.col.glow[1] * k;
        rgb[2] += p.col.glow[2] * k;
    }

    // ---- 3. glyph: gradient fill + bevel rim + stroke ---------------------
    if (p.flat) {
        // Pure silhouette: one flat color, no internal shading.
        for (int c = 0; c < 3; ++c)
            rgb[c] += (p.fgFill[c] - rgb[c]) * cov;
    } else if (cov > 0.0f) {
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

    // stroke: bright outline band around the silhouette (flat glyphs get a
    // pure opposite-color keyline for readability over the rainbow)
    {
        const float sw = 0.032f;  // ~2px at 128
        const float str = sstep(sw + aa, sw - aa, absf(d));
        if (p.flat) {
            for (int c = 0; c < 3; ++c)
                rgb[c] += (p.fgStroke[c] - rgb[c]) * str;
        } else {
            const float k = str * (0.90f + 0.18f * bev);
            for (int c = 0; c < 3; ++c)
                rgb[c] += (p.col.stroke[c] * k - rgb[c]) * str;
        }
    }

    // ---- 4. frame by tier -------------------------------------------------
    if (p.tier > 0) {
        const float ring = absf(sdBox(u, v, 0.0f, 0.0f, kFrameFi, kFrameFi, kFrameRf));
        const float fw = p.tier >= 2 ? 0.024f : 0.018f;
        const float frameCov = sstep(fw + aa, fw - aa, ring);

        // outer glow (t3+): soft halo just outside the ring
        if (!p.flat && p.tier >= 3) {
            const float out = ring - fw;
            const float halo = powi(1.0f - sstep(0.0f, 0.14f, out), 3);
            const float k = 0.22f * halo * (1.0f - frameCov);
            rgb[0] += p.col.glow[0] * k;
            rgb[1] += p.col.glow[1] * k;
            rgb[2] += p.col.glow[2] * k;
        }

        if (p.flat) {
            for (int c = 0; c < 3; ++c)
                rgb[c] += (p.fgStroke[c] - rgb[c]) * frameCov;
        } else {
            for (int c = 0; c < 3; ++c)
                rgb[c] += (p.col.stroke[c] * 1.05f - rgb[c]) * frameCov;
        }

        // inner thin ring (t2+)
        if (p.tier >= 2) {
            const float ring2 = absf(sdBox(u, v, 0.0f, 0.0f, 0.775f, 0.775f, 0.17f));
            const float c2 = sstep(0.011f + aa, 0.011f - aa, ring2);
            if (p.flat) {
                for (int c = 0; c < 3; ++c)
                    rgb[c] += (p.fgStroke[c] - rgb[c]) * c2 * 0.8f;
            } else {
                for (int c = 0; c < 3; ++c)
                    rgb[c] += (p.col.stroke[c] * 0.75f - rgb[c]) * c2 * 0.8f;
            }
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
                        float gem;
                        if (p.flat) {
                            gem = p.fgStroke[c] * (1.0f - core) + p.fgFill[c] * core;
                        } else {
                            gem = p.col.fill[c] * 1.25f * (1.0f - 0.5f * core)
                                + p.col.hi[c] * 1.10f * core;
                        }
                        rgb[c] += (gem - rgb[c]) * gc;
                    }
                } else if (p.tier >= 3) {
                    const float g = sdDiamond(u, v, px, py, 0.055f);
                    const float gc = sstep(aa, -aa, g);
                    for (int c = 0; c < 3; ++c) {
                        const float oc = p.flat ? p.fgStroke[c] : p.col.stroke[c];
                        rgb[c] += (oc - rgb[c]) * gc;
                    }
                } else if (p.tier >= 2) {
                    const float g = sdCircle(u, v, px, py, 0.042f);
                    const float gc = sstep(aa, -aa, g);
                    for (int c = 0; c < 3; ++c) {
                        const float oc = p.flat ? p.fgStroke[c]
                                                : p.col.stroke[c] * 0.92f;
                        rgb[c] += (oc - rgb[c]) * gc;
                    }
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