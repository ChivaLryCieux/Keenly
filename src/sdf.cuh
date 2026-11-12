#pragma once
// 2D signed-distance primitives + the skill-icon glyph library.
//
// ASCII-ONLY FILE ON PURPOSE (see src/cuda_utils.h for the reason).
//
// Bit-level discipline: every function is a pure function of its arguments
// built from +, -, *, /, sqrtf and comparisons only. Division and sqrt are
// IEEE correctly-rounded on both host and device, so the verify gate
// (maxAbs == 0) extends over the whole icon pixel. NO transcendentals: the
// sun's rays and the snowflake arms use precomputed rotation constants as
// literal multiplies, never sin/cos.
//
// Glyph space: x, y in roughly [-1, 1], y grows DOWNWARD (screen space), so
// "up" in a glyph is negative y. Distances are in the same units.

#include <cmath>
#include <cstdint>

namespace keenly
{

// ---- scalar helpers -------------------------------------------------------

__host__ __device__ inline float mn(float a, float b) { return a < b ? a : b; }
__host__ __device__ inline float mx(float a, float b) { return a > b ? a : b; }
__host__ __device__ inline float absf(float x) { return x < 0.0f ? -x : x; }

// Polynomial smooth min (quadratic trade): blends unions so they read as one
// shape instead of glued primitives. k > 0 in all call sites.
__host__ __device__ inline float smin(float a, float b, float k)
{
    const float h = 0.5f + 0.5f * (b - a) / k;
    const float t = h < 0.0f ? 0.0f : (h > 1.0f ? 1.0f : h);
    return b + (a - b) * t - k * t * (1.0f - t);
}

// ---- primitives -----------------------------------------------------------

// Disk: signed distance to the circle boundary (negative inside).
__host__ __device__ inline float sdCircle(float x, float y, float cx, float cy, float r)
{
    const float dx = x - cx;
    const float dy = y - cy;
    return sqrtf(dx * dx + dy * dy) - r;
}

// Ring: |disk| - thickness/2.
__host__ __device__ inline float sdRing(float x, float y, float cx, float cy, float r, float th)
{
    return absf(sdCircle(x, y, cx, cy, r)) - 0.5f * th;
}

// Rounded box, centered (cx, cy), half extents (hx, hy), corner radius r.
__host__ __device__ inline float sdBox(float x, float y, float cx, float cy,
                                       float hx, float hy, float r)
{
    const float qx = absf(x - cx) - (hx - r);
    const float qy = absf(y - cy) - (hy - r);
    const float ox = mx(qx, 0.0f);
    const float oy = mx(qy, 0.0f);
    return sqrtf(ox * ox + oy * oy) + mn(mx(qx, qy), 0.0f) - r;
}

// Capsule: distance to segment (ax, ay)-(bx, by) minus radius r.
__host__ __device__ inline float sdSeg(float x, float y, float ax, float ay,
                                       float bx, float by, float r)
{
    const float pax = x - ax;
    const float pay = y - ay;
    const float bax = bx - ax;
    const float bay = by - ay;
    const float dd = bax * bax + bay * bay;
    float t = (pax * bax + pay * bay) / dd;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    const float qx = pax - t * bax;
    const float qy = pay - t * bay;
    return sqrtf(qx * qx + qy * qy) - r;
}

// Triangle from three vertices, hard corners. Edge functions normalized by
// edge length (true distance), winding fixed by the sign of the first edge
// against the third vertex - so any vertex order works.
__host__ __device__ inline float sdTri(float x, float y,
                                       float ax, float ay,
                                       float bx, float by,
                                       float cx, float cy)
{
    const float e0 = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    const float e1 = (cx - bx) * (y - by) - (cy - by) * (x - bx);
    const float e2 = (ax - cx) * (y - cy) - (ay - cy) * (x - cx);
    const float w = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const float s = w < 0.0f ? 1.0f : -1.0f;
    const float l0 = sqrtf((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
    const float l1 = sqrtf((cx - bx) * (cx - bx) + (cy - by) * (cy - by));
    const float l2 = sqrtf((ax - cx) * (ax - cx) + (ay - cy) * (ay - cy));
    return mx(mx(s * e0 / l0, s * e1 / l1), s * e2 / l2);
}

// Diamond (rotated square) of "radius" s.
__host__ __device__ inline float sdDiamond(float x, float y, float cx, float cy, float s)
{
    return (absf(x - cx) + absf(y - cy)) * 0.70710678f - s;
}

// 4-point sparkle star: L4 superellipse - r. Integer powers and two IEEE
// sqrts, so still bit-exact.
__host__ __device__ inline float sdStar4(float x, float y, float cx, float cy, float r)
{
    const float dx = absf(x - cx);
    const float dy = absf(y - cy);
    const float a = dx * dx * dx * dx;
    const float b = dy * dy * dy * dy;
    return sqrtf(sqrtf(a + b)) - r;
}

// Subtract a hole from a shape: inside = inside shape AND outside hole.
__host__ __device__ inline float sdCut(float shape, float hole)
{
    return mx(shape, -hole);
}

// ---- glyph library --------------------------------------------------------
// 18 glyphs, each a small union of the primitives above. Dispatch by int so
// the kernel stays branch-simple and the manifest stays readable.

enum GlyphId
{
    kGlyphSword = 0,
    kGlyphAxe = 1,
    kGlyphBow = 2,
    kGlyphShield = 3,
    kGlyphFlame = 4,
    kGlyphDroplet = 5,
    kGlyphSnow = 6,
    kGlyphBolt = 7,
    kGlyphSparkle = 8,
    kGlyphMoon = 9,
    kGlyphSun = 10,
    kGlyphEye = 11,
    kGlyphHeart = 12,
    kGlyphSkull = 13,
    kGlyphCross = 14,
    kGlyphRune = 15,
    kGlyphCrown = 16,
    kGlyphPotion = 17,
    kGlyphCount = 18
};

// Signed distance to one glyph silhouette (negative inside).
__host__ __device__ inline float glyphSdf(int glyph, float x, float y)
{
    switch (glyph) {
        case kGlyphSword: {
            // blade + pointed tip + guard + grip + pommel
            float d = sdSeg(x, y, 0.0f, -0.50f, 0.0f, 0.30f, 0.085f);
            d = mn(d, sdTri(x, y, 0.0f, -0.74f, -0.10f, -0.44f, 0.10f, -0.44f));
            d = mn(d, sdSeg(x, y, -0.28f, 0.32f, 0.28f, 0.32f, 0.07f));
            d = mn(d, sdSeg(x, y, 0.0f, 0.32f, 0.0f, 0.56f, 0.055f));
            d = mn(d, sdCircle(x, y, 0.0f, 0.62f, 0.09f));
            return d;
        }
        case kGlyphAxe: {
            // handle + crescent blade (disk minus offset disk)
            float d = sdSeg(x, y, 0.10f, -0.58f, 0.10f, 0.62f, 0.07f);
            const float blade = sdCut(sdCircle(x, y, -0.12f, -0.26f, 0.34f),
                                      sdCircle(x, y, -0.52f, -0.26f, 0.30f));
            return smin(d, blade, 0.06f);
        }
        case kGlyphBow: {
            // crescent bow + string + arrow shaft + head
            const float limb = sdCut(sdRing(x, y, 0.0f, 0.0f, 0.58f, 0.10f),
                                     sdBox(x, y, 0.9f, 0.0f, 0.72f, 1.2f, 0.0f));
            float d = limb;
            d = mn(d, sdSeg(x, y, 0.16f, -0.55f, 0.16f, 0.55f, 0.022f));
            d = mn(d, sdSeg(x, y, -0.34f, 0.0f, 0.38f, 0.0f, 0.032f));
            d = mn(d, sdTri(x, y, 0.38f, -0.17f, 0.38f, 0.17f, 0.70f, 0.0f));
            return d;
        }
        case kGlyphShield: {
            // rounded shield body with a pointed bottom
            float d = sdBox(x, y, 0.0f, -0.06f, 0.44f, 0.40f, 0.16f);
            d = smin(d, sdTri(x, y, -0.42f, 0.10f, 0.42f, 0.10f, 0.0f, 0.62f), 0.08f);
            return d;
        }
        case kGlyphFlame: {
            // teardrop body + pointed tip
            return smin(sdCircle(x, y, 0.0f, 0.16f, 0.30f),
                        sdTri(x, y, 0.0f, -0.70f, -0.30f, 0.12f, 0.30f, 0.12f),
                        0.10f);
        }
        case kGlyphDroplet: {
            return smin(sdCircle(x, y, 0.0f, 0.18f, 0.32f),
                        sdTri(x, y, 0.0f, -0.68f, -0.28f, 0.12f, 0.28f, 0.12f),
                        0.12f);
        }
        case kGlyphSnow: {
            // 3 crossing capsules (6 arms) + perpendicular mid crossbars.
            // Perpendicular is an exact 90-degree rotate: (-dy, dx).
            float d = sdSeg(x, y, -0.60f, 0.0f, 0.60f, 0.0f, 0.052f);
            d = mn(d, sdSeg(x, y, -0.30f, -0.5196152f, 0.30f, 0.5196152f, 0.052f));
            d = mn(d, sdSeg(x, y, -0.30f, 0.5196152f, 0.30f, -0.5196152f, 0.052f));
            // crossbars at +/- 0.42 along each of the three arm directions
            d = mn(d, sdSeg(x, y, 0.42f, -0.13f, 0.42f, 0.13f, 0.042f));
            d = mn(d, sdSeg(x, y, -0.42f, -0.13f, -0.42f, 0.13f, 0.042f));
            d = mn(d, sdSeg(x, y, 0.085f, -0.547f, 0.315f, -0.422f, 0.042f));
            d = mn(d, sdSeg(x, y, -0.315f, 0.422f, -0.085f, 0.547f, 0.042f));
            d = mn(d, sdSeg(x, y, -0.315f, -0.422f, -0.085f, -0.547f, 0.042f));
            d = mn(d, sdSeg(x, y, 0.085f, 0.547f, 0.315f, 0.422f, 0.042f));
            return d;
        }
        case kGlyphBolt: {
            // lightning: three tapered zigzag capsules
            float d = sdSeg(x, y, -0.14f, -0.58f, 0.12f, -0.10f, 0.10f);
            d = mn(d, sdSeg(x, y, 0.12f, -0.10f, -0.06f, 0.02f, 0.085f));
            d = mn(d, sdSeg(x, y, -0.06f, 0.02f, 0.16f, 0.58f, 0.10f));
            return d;
        }
        case kGlyphSparkle: {
            return sdStar4(x, y, 0.0f, 0.0f, 0.62f);
        }
        case kGlyphMoon: {
            // crescent: disk minus offset disk
            return sdCut(sdCircle(x, y, 0.0f, 0.0f, 0.58f),
                         sdCircle(x, y, 0.30f, -0.14f, 0.52f));
        }
        case kGlyphSun: {
            float d = sdCircle(x, y, 0.0f, 0.0f, 0.34f);
            // 8 rays at 45-degree multiples: exact direction constants
            d = mn(d, sdSeg(x, y, 0.46f, 0.0f, 0.66f, 0.0f, 0.05f));
            d = mn(d, sdSeg(x, y, -0.66f, 0.0f, -0.46f, 0.0f, 0.05f));
            d = mn(d, sdSeg(x, y, 0.0f, -0.66f, 0.0f, -0.46f, 0.05f));
            d = mn(d, sdSeg(x, y, 0.0f, 0.46f, 0.0f, 0.66f, 0.05f));
            d = mn(d, sdSeg(x, y, 0.325f, -0.325f, 0.467f, -0.467f, 0.05f));
            d = mn(d, sdSeg(x, y, -0.467f, 0.467f, -0.325f, 0.325f, 0.05f));
            d = mn(d, sdSeg(x, y, -0.325f, -0.325f, -0.467f, -0.467f, 0.05f));
            d = mn(d, sdSeg(x, y, 0.467f, 0.467f, 0.325f, 0.325f, 0.05f));
            return d;
        }
        case kGlyphEye: {
            // lens ring (two-disk intersection) + pupil
            const float lens = mx(sdCircle(x, y, 0.0f, 0.26f, 0.54f),
                                  sdCircle(x, y, 0.0f, -0.26f, 0.54f));
            float d = absf(lens) - 0.06f;
            d = mn(d, sdCircle(x, y, 0.0f, 0.0f, 0.15f));
            return d;
        }
        case kGlyphHeart: {
            float d = sdCircle(x, y, -0.22f, -0.16f, 0.28f);
            d = mn(d, sdCircle(x, y, 0.22f, -0.16f, 0.28f));
            d = smin(d, sdSeg(x, y, -0.44f, -0.10f, 0.0f, 0.58f, 0.24f), 0.10f);
            d = smin(d, sdSeg(x, y, 0.44f, -0.10f, 0.0f, 0.58f, 0.24f), 0.10f);
            return d;
        }
        case kGlyphSkull: {
            // cranium + jaw, eyes / nose / teeth subtracted
            float d = smin(sdCircle(x, y, 0.0f, -0.12f, 0.42f),
                           sdBox(x, y, 0.0f, 0.30f, 0.24f, 0.22f, 0.10f), 0.10f);
            d = sdCut(d, sdCircle(x, y, -0.16f, -0.14f, 0.12f));
            d = sdCut(d, sdCircle(x, y, 0.16f, -0.14f, 0.12f));
            d = sdCut(d, sdDiamond(x, y, 0.0f, 0.08f, 0.08f));
            d = sdCut(d, sdBox(x, y, -0.09f, 0.40f, 0.022f, 0.10f, 0.0f));
            d = sdCut(d, sdBox(x, y, 0.09f, 0.40f, 0.022f, 0.10f, 0.0f));
            return d;
        }
        case kGlyphCross: {
            // heal cross: vertical + horizontal bar, smooth union
            return smin(sdBox(x, y, 0.0f, 0.0f, 0.15f, 0.52f, 0.05f),
                        sdBox(x, y, 0.0f, 0.0f, 0.52f, 0.15f, 0.05f), 0.05f);
        }
        case kGlyphRune: {
            // rune ring + three orbit dots + center gem
            float d = sdRing(x, y, 0.0f, 0.0f, 0.52f, 0.09f);
            d = mn(d, sdCircle(x, y, 0.0f, -0.52f, 0.09f));
            d = mn(d, sdCircle(x, y, -0.4503333f, 0.26f, 0.09f));
            d = mn(d, sdCircle(x, y, 0.4503333f, 0.26f, 0.09f));
            d = mn(d, sdDiamond(x, y, 0.0f, 0.0f, 0.16f));
            return d;
        }
        case kGlyphCrown: {
            // base band + three spikes (capsule fan reads as a crown)
            float d = sdBox(x, y, 0.0f, 0.34f, 0.52f, 0.13f, 0.05f);
            d = smin(d, sdSeg(x, y, -0.34f, 0.30f, -0.38f, -0.42f, 0.10f), 0.07f);
            d = smin(d, sdSeg(x, y, 0.0f, 0.30f, 0.0f, -0.52f, 0.10f), 0.07f);
            d = smin(d, sdSeg(x, y, 0.34f, 0.30f, 0.38f, -0.42f, 0.10f), 0.07f);
            return d;
        }
        case kGlyphPotion: {
            // flask: cork + neck + round body
            float d = sdBox(x, y, 0.0f, -0.60f, 0.13f, 0.08f, 0.03f);
            d = smin(d, sdBox(x, y, 0.0f, -0.38f, 0.11f, 0.18f, 0.05f), 0.06f);
            d = smin(d, sdCircle(x, y, 0.0f, 0.18f, 0.38f), 0.10f);
            return d;
        }
        default:
            return sdCircle(x, y, 0.0f, 0.0f, 0.5f);
    }
}

// Glyph name for the CLI / manifest, same order as GlyphId. Plain switch:
// no static tables, so it stays valid in device code too.
__host__ __device__ inline const char* glyphName(int glyph)
{
    switch (glyph) {
        case kGlyphSword:   return "sword";
        case kGlyphAxe:     return "axe";
        case kGlyphBow:     return "bow";
        case kGlyphShield:  return "shield";
        case kGlyphFlame:   return "flame";
        case kGlyphDroplet: return "droplet";
        case kGlyphSnow:    return "snow";
        case kGlyphBolt:    return "bolt";
        case kGlyphSparkle: return "sparkle";
        case kGlyphMoon:    return "moon";
        case kGlyphSun:     return "sun";
        case kGlyphEye:     return "eye";
        case kGlyphHeart:   return "heart";
        case kGlyphSkull:   return "skull";
        case kGlyphCross:   return "cross";
        case kGlyphRune:    return "rune";
        case kGlyphCrown:   return "crown";
        case kGlyphPotion:  return "potion";
        default:            return "unknown";
    }
}

}  // namespace keenly