#pragma once
// The pixel shader: one stateless function mapping (x, y, params) -> RGB.
//
// ASCII-ONLY FILE ON PURPOSE (see src/cuda_utils.h for the reason).
//
// This is the single source of truth for the art math. The GPU kernel
// (render.cu) and the CPU reference (reference.cu) both call fieldPixel(),
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

#include "noise.cuh"
#include "turing.cuh"

namespace keenly
{

// Color ramp: up to 8 stops, strictly increasing positions. Stops live in
// perceptual (sRGB-like) space - easier to author - and paletteLookup()
// linearizes with gamma 2.0 (a single multiply: exact on CPU and GPU).
// (8 = room for a wrapping rainbow: 7 hues with red repeated at both ends.)
struct Palette
{
    int count = 2;
    float pos[8] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float rgb[8][3] = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
        {0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
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
    kFieldLiquid = 6, // glossy metal: flowing warp field + specular shading
    kFieldTuring = 7, // Gray-Scott reaction-diffusion: Turing spots / labyrinths
    kFieldAgate = 8,  // warped elliptical rings around two drops: suminagashi
    kFieldWaves = 9,  // stacked waveform silhouettes: pulsar ridge plot
    kFieldTruchet = 10, // arc-tile truchet: connected circuit / textile paths
    kFieldShards = 11,  // 16-fold mirror kaleidoscope of the warp field
    kFieldDots = 12,    // halftone screens: dot radius follows a warped tone
    kFieldAura = 13,    // drifting silk ribbons: additive quartic bumps
    kFieldMoire = 14,   // two-source ring interference: op-art ripples
    kFieldDeco = 15,    // bauhaus mosaic: hashed flat motif tiles
    kFieldGlow = 16     // soft additive light blobs on black: mesh gradient
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

    // ---- Gray-Scott (kFieldTuring) -----------------------------------------
    // The reaction-diffusion grid is simulated once per image (GPU and CPU
    // paths each run their own sim from the same pure init/update code), then
    // fieldValue samples `turingTex` - a res*res grid of the B channel.
    float rdF = 0.0545f;     // feed rate
    float rdK = 0.062f;      // kill rate
    int rdRes = 256;         // simulation grid (square, tileable)
    int rdIters = 3000;      // Jacobi steps
    const float* turingTex = nullptr;  // sim result, device or host pointer
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

// ---- helpers for the geometric generators ---------------------------------

__host__ __device__ inline float distf(float ax, float ay, float bx, float by)
{
    const float dx = ax - bx;
    const float dy = ay - by;
    return sqrtf(dx * dx + dy * dy);
}

// Height of one pulsar-wave row (kFieldWaves): a low rolling baseline plus
// sharp spikes, in row-local units (1 = one row band). Capped under 2 so a
// peak can only reach into the band above - the pixel loop scans rows r..r+1.
__host__ __device__ inline float waveProfile(float px, int row, const FieldParams& p)
{
    const float fj = float(row);
    const float wave = 0.30f + 0.12f * fbm2(px * 1.3f, fj * 0.22f,
                                            p.seed + 331u, p.octaves,
                                            p.lacunarity, p.gain);
    // cubed noise tail with hard gain: low noise flattens out, rare fBm
    // crests become tall thin peaks - the pulsar silhouette lives in the
    // tail, so it needs amplification, not a threshold
    const float n = fbm2(px * 1.5f, fj * 0.15f, p.seed + 517u,
                         p.octaves, p.lacunarity, p.gain);
    const float s0 = n > 0.0f ? n : 0.0f;
    const float prof = wave + 13.0f * s0 * s0 * s0;
    return prof > 2.6f ? 2.6f : prof;
}

// Bauhaus tones: a hash quantized to four palette positions, keeping every
// tile a flat unmodulated color (the look dies if tones slide along a ramp).
__host__ __device__ inline float decoTone(float h)
{
    const int q = int(h * 4.0f) & 3;
    return 0.12f + 0.24f * float(q);
}

// One flat motif inside a unit cell (fx, fy in [0, 1)). Returns the tile tone
// in [0, 1]. Motifs are the bauhaus starter set: quarter discs, diagonal
// splits, a center dot, ring arcs and edge half-discs.
__host__ __device__ inline float decoMotif(float fx, float fy, int id,
                                           float t0, float t1)
{
    switch (id & 7) {
        case 0: return distf(fx, fy, 0.0f, 0.0f) < 0.92f ? t1 : t0;
        case 1: return distf(fx, fy, 1.0f, 0.0f) < 0.92f ? t1 : t0;
        case 2: return distf(fx, fy, 0.0f, 1.0f) < 0.92f ? t1 : t0;
        case 3: return distf(fx, fy, 1.0f, 1.0f) < 0.92f ? t1 : t0;
        case 4: return fx + fy < 1.0f ? t1 : t0;
        case 5: return fx < fy ? t1 : t0;
        case 6: {
            const float d = distf(fx, fy, 0.5f, 0.5f);
            if (d < 0.30f) return t1;
            return d < 0.46f ? t0 : 0.5f * (t0 + t1);
        }
        default: {
            // ring arc in the lower-left corner
            const float d = distf(fx, fy, 0.0f, 1.0f);
            return (d > 0.42f && d < 0.78f) ? t1 : t0;
        }
    }
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
        case kFieldTuring: {
            // Gray-Scott pattern: the simulated B grid, sampled tileably.
            // Pattern scale is independent of output size: px already carries
            // the view factor (px = u * view), so one sim grid spans exactly
            // one image height after the rdRes / view rescale.
            const float sc = float(p.rdRes) / p.view;
            return sampleTuring(p.turingTex, p.rdRes, px * sc, py * sc);
        }
        case kFieldAgate: {
            // suminagashi rings: two ink drops; the mean distance gives nested
            // elliptical bands, then the radius is advected through an fBm
            // flow so the rings shear like combed marbling.
            const float c1x = (hash01(11, 3, s) - 0.5f) * 0.70f * p.view;
            const float c1y = (hash01(23, 5, s) - 0.5f) * 0.70f * p.view;
            const float c2x = (hash01(37, 7, s) - 0.5f) * 0.70f * p.view;
            const float c2y = (hash01(53, 9, s) - 0.5f) * 0.70f * p.view;
            float r = 0.5f * (distf(px, py, c1x, c1y) + distf(px, py, c2x, c2y));
            r += p.warp * 0.18f * fbm2(px * 0.9f, py * 0.9f, s + 911u,
                                       oc, lac, gn);
            const float ring = triFrac(r * p.bands * 0.45f);
            const float wgt = 0.55f + 0.45f * fbm2(r * 1.7f, px * 0.30f - py * 0.20f,
                                                   s + 733u, 3, lac, gn);
            return (2.0f * sstep(0.08f, 0.92f, ring) - 1.0f) * wgt;
        }
        case kFieldWaves: {
            // pulsar ridge plot: rows scan back to front; each row's curve
            // sits at l = 1 - profile, fills down to its slot bottom (so
            // nearer rows occlude farther ones) and strokes a bright line.
            const float R = p.bands * 4.2f;
            const float vr = v * R + 0.5f * R;  // v in [-0.5,0.5] -> row 0..R
            const int r = int(floorf(vr));
            const float lw = 0.045f;
            float val = -1.0f;
            for (int j = r; j <= r + 2; ++j) {
                const float l = vr - float(j);
                const float lc = 1.0f - waveProfile(px, j, p);
                if (l >= lc) val = -1.0f;                 // fill occludes
                const float stroke = sstep(lw + 0.025f, lw - 0.025f, absf(l - lc));
                if (stroke > 0.0f) {
                    const float rowHi = 0.70f + 0.30f * hash01(j, 0, s + 41u);
                    val = mx(val, 2.0f * stroke * rowHi - 1.0f);
                }
            }
            return val;
        }
        case kFieldTruchet: {
            // connected arc tiles: quarter arcs at radius 1/4 and 3/4 around
            // one diagonal corner pair, so paths continue across cells; a
            // few cells break the rhythm with a pip or a blank.
            const float cs = p.view / p.bands;
            const int cx = int(floorf(px / cs));
            const int cy = int(floorf(py / cs));
            const float fx = px / cs - float(cx);
            const float fy = py / cs - float(cy);
            const float h1 = hash01(cx, cy, s);
            const float h2 = hash01(cx, cy, s + 131u);
            const float h3 = hash01(cx, cy, s + 257u);
            const float h4 = hash01(cx, cy, s + 389u);
            const float ax0 = h1 < 0.5f ? 0.0f : 1.0f;
            const float d0 = distf(fx, fy, ax0, 0.0f);
            const float d1 = distf(fx, fy, 1.0f - ax0, 1.0f);
            float band = sstep(0.115f, 0.085f, absf(d0 - 0.25f));
            band = mx(band, sstep(0.115f, 0.085f, absf(d0 - 0.75f)));
            band = mx(band, sstep(0.115f, 0.085f, absf(d1 - 0.25f)));
            band = mx(band, sstep(0.115f, 0.085f, absf(d1 - 0.75f)));
            if (h2 < 0.14f) {
                // pip cell instead of a dead blank: a lone dot keeps rhythm
                band = sstep(0.15f, 0.11f, distf(fx, fy, 0.5f, 0.5f));
            }
            // ~1 in 5 cells invert: dark paths on a lit ground
            const float inv = h3 < 0.20f ? -1.0f : 1.0f;
            return (2.0f * band - 1.0f) * inv * (0.75f + 0.25f * h4);
        }
        case kFieldShards: {
            // 16-fold dihedral kaleidoscope: fold (u, v) into a 1/16 wedge
            // using only abs, one swap and one reflection with exact 45deg
            // constants, then run the organic warp field through the fold.
            float a = absf(u);
            float b = absf(v);
            if (b > a) { const float t = a; a = b; b = t; }
            if (b > a * 0.41421356f) {
                const float na = (a + b) * 0.70710678f;
                const float nb = (a - b) * 0.70710678f;
                a = na; b = nb;
            }
            float t = warpField(a * p.view * 1.15f, b * p.view * 1.15f, p);
            const float r = sqrtf(u * u + v * v);
            return t + 0.35f * (0.5f - r);
        }
        case kFieldDots: {
            // screenprint halftone: two rotated dot lattices overprint and
            // each dot's radius follows a low-frequency warp tone, so the
            // picture is built out of ink dots like a riso print.
            float tone = 0.50f + 0.85f * warpField(px * 0.55f, py * 0.55f, p)
                       + 0.12f * fbm2(px * 0.30f, py * 0.30f, s + 521u,
                                      4, lac, gn);
            tone = tone < 0.0f ? 0.0f : (tone > 1.0f ? 1.0f : tone);
            const float cs = p.view / (p.bands * 2.5f);
            // screen A, rotated +15 deg
            const float ax = px * 0.96592582f - py * 0.25881905f;
            const float ay = px * 0.25881905f + py * 0.96592582f;
            const float fx = ax / cs - floorf(ax / cs);
            const float fy = ay / cs - floorf(ay / cs);
            const float ra = 0.04f + 0.62f * tone;
            float ink = sstep(ra + 0.085f, ra - 0.085f, distf(fx, fy, 0.5f, 0.5f));
            // screen B, rotated -30 deg, half-cell offset: fixed radius dot
            // that fades in only in the deep zones - a modulated radius would
            // spray sub-pixel speckle at the threshold instead of printing
            const float bx = px * 0.8660254f + py * 0.5f;
            const float by = -px * 0.5f + py * 0.8660254f;
            const float gx = bx / cs + 0.5f - floorf(bx / cs + 0.5f);
            const float gy = by / cs + 0.5f - floorf(by / cs + 0.5f);
            const float wb = 0.80f * sstep(0.60f, 0.82f, tone);
            if (wb > 0.0f) {
                ink = mx(ink, wb * sstep(0.285f, 0.115f,
                                         distf(gx, gy, 0.5f, 0.5f)));
            }
            // print polarity: dark ink dots on paper, not beads on black
            return 1.0f - 2.0f * ink;
        }
        case kFieldAura: {
            // drifting silk ribbons: five fBm-displaced bands, each a quartic
            // bump cross-section with a stretched filament texture inside.
            float acc = -0.55f;
            for (int i = 0; i < 6; ++i) {
                const float fi = float(i);
                const float c0 = -0.40f + 0.16f * fi
                               + (hash01(i, 3, s + 613u) - 0.5f) * 0.08f;
                const float drift = fbm2(px * (0.55f + 0.15f * fi) + fi * 7.31f,
                                         fi * 3.7f, s + 601u + unsigned(i) * 17u,
                                         oc, lac, gn);
                const float cy = c0 + 0.28f * drift;
                const float rw = 0.012f + 0.030f * hash01(i, 7, s + 887u);
                const float z = (v - cy) / rw;
                const float b0 = 1.0f - z * z;
                const float bump = b0 > 0.0f ? b0 * b0 : 0.0f;
                const float fil = 0.45f + 0.55f * fbm2(px * 11.0f,
                                                       v * 0.9f + fi * 11.0f,
                                                       s + 907u + unsigned(i) * 13u,
                                                       4, lac, gn);
                acc += bump * fil * (0.40f + 0.5f * hash01(i, 5, s + 941u));
            }
            return acc > 1.0f ? 1.0f : acc;
        }
        case kFieldMoire: {
            // two nearby ring emitters: bipolar triangle waves multiplied,
            // giving the interference beat of two wave sources - water
            // ripple op art with a slight field wobble.
            const float c1x = (hash01(11, 1, s + 349u) - 0.5f) * 0.50f * p.view;
            const float c1y = (hash01(13, 2, s + 353u) - 0.5f) * 0.50f * p.view;
            const float c2x = (hash01(17, 3, s + 359u) - 0.5f) * 0.50f * p.view;
            const float c2y = (hash01(19, 4, s + 367u) - 0.5f) * 0.50f * p.view;
            const float w = p.warp * 0.05f * fbm2(px * 0.8f, py * 0.8f, s + 419u,
                                                  oc - 2, lac, gn);
            const float k = p.bands * 1.5f;
            const float r1 = 2.0f * sstep(0.10f, 0.90f,
                triFrac((distf(px, py, c1x, c1y) + w) * k)) - 1.0f;
            const float r2 = 2.0f * sstep(0.10f, 0.90f,
                triFrac((distf(px, py, c2x, c2y) - w) * k)) - 1.0f;
            return r1 * r2;
        }
        case kFieldDeco: {
            // bauhaus quilt: a hashed flat motif per cell on a coarse grid,
            // some cells subdivide 2x2 for scale rhythm; thin grout seams.
            const float cs = p.view / (p.bands * 0.66f);
            const int cx = int(floorf(px / cs));
            const int cy = int(floorf(py / cs));
            const float fx = px / cs - float(cx);
            const float fy = py / cs - float(cy);
            const float h1 = hash01(cx, cy, s);
            const float h2 = hash01(cx, cy, s + 179u);
            const float h3 = hash01(cx, cy, s + 311u);
            const float h4 = hash01(cx, cy, s + 449u);
            float tone;
            if (h4 < 0.22f) {
                const float sfx = fx * 2.0f;
                const float sfy = fy * 2.0f;
                const int scx = int(floorf(sfx));
                const int scy = int(floorf(sfy));
                const float sh = hash01(cx * 4 + scx, cy * 4 + scy, s + 977u);
                const float sh2 = hash01(cx * 4 + scx, cy * 4 + scy, s + 983u);
                tone = decoMotif(sfx - floorf(sfx), sfy - floorf(sfy),
                                 int(sh * 8.0f), decoTone(sh2), decoTone(h3));
            } else {
                tone = decoMotif(fx, fy, int(h1 * 8.0f),
                                 decoTone(h2), decoTone(h3));
            }
            const float dx = fx < 0.5f ? fx : 1.0f - fx;
            const float dy = fy < 0.5f ? fy : 1.0f - fy;
            if (mn(dx, dy) < 0.035f) tone = 0.04f;
            return 2.0f * tone - 1.0f;
        }
        default:
            return warpField(px, py, p);
    }
}

// Soft additive glow on black: pal.count light blobs (pal.rgb = their colors)
// laid out top/bottom around a dark center pool, each advected by a gentle
// fBm so the silhouettes look organic. Colors enter as LIGHT, sum under a
// soft-winner mix, and fade into the dark ground. coreLift floors the dark
// pool: 0 = pure black center (posters), >0 keeps a dim tint (icons need a
// colored floor under the glyph for readability).
__host__ __device__ inline void glowField(const FieldParams& p, float u, float v,
                                          float* rgb, float coreLift)
{
    rgb[0] = rgb[1] = rgb[2] = 0.0f;
    const unsigned s = p.seed;
    const int oc = p.octaves;
    const float lac = p.lacunarity;
    const float gn = p.gain;
    const int n = p.pal.count;

    // Layout contract: BLACK CENTER, colors surrounding above and below.
    // Blob slots alternate top/bottom and are spread along the frame's
    // horizontal span with jitter, so the palette wraps the dark core.

    // ambient nebula floor: the palette's mean color at low intensity, wobbled
    // by a slow fBm, so "black" areas are deep tinted space instead of a void
    {
        float amb[3] = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < n; ++i) {
            amb[0] += p.pal.rgb[i][0] * p.pal.rgb[i][0];
            amb[1] += p.pal.rgb[i][1] * p.pal.rgb[i][1];
            amb[2] += p.pal.rgb[i][2] * p.pal.rgb[i][2];
        }
        const float w = 0.025f + 0.045f * fbm2(u * 1.6f, v * 1.6f, s + 2441u,
                                             3, lac, gn) / float(n);
        rgb[0] += amb[0] * w;
        rgb[1] += amb[1] * w;
        rgb[2] += amb[2] * w;
    }

    // pass 1: per-blob light intensity (compact bright core + wide faint
    // halo), so overlap zones can settle "who owns this pixel"
    float Ii[8];
    float Itot = 0.0f;
    for (int i = 0; i < n; ++i) {
        // slot i alternates bottom(-1)/top(+1); base x spreads the band's
        // blobs evenly across the width, hash adds jitter
        const float side = ((i & 1) == 0) ? 1.0f : -1.0f;
        const float bx = (float(i) + 0.5f) / float(n) * 2.0f - 1.0f;
        const float cx = bx * 0.95f + (hash01(i, 3, s) - 0.5f) * 0.42f;
        const float cy = side * (0.58f + 0.24f * hash01(i, 7, s));
        const float R = 0.50f + 0.40f * hash01(i, 13, s);

        // organic silhouette: advect the sample point by a shared fBm
        const float wa = 0.16f * fbm2(u * 2.2f + float(i) * 5.13f,
                                      v * 2.2f, s + unsigned(i) * 101u,
                                      oc - 1, lac, gn);
        const float wb = 0.16f * fbm2(u * 2.2f, v * 2.2f + float(i) * 7.31f,
                                      s + unsigned(i) * 137u,
                                      oc - 1, lac, gn);
        // horizontal stretch: edge glows sweep wide along the frame
        const float ddx = (u + wa - cx) * 0.80f;
        const float ddy = v + wb - cy;
        const float d = sqrtf(ddx * ddx + ddy * ddy);
        const float q1 = sstep(R * 0.55f, 0.0f, d);
        const float q2 = sstep(R * 1.35f, 0.0f, d);
        Ii[i] = (0.85f * q1 * q1 + 0.30f * q2 * q2)
              * (0.55f + 0.45f * hash01(i, 17, s));
        Itot += Ii[i];
    }

    // pass 2: soft-winner mixing. A blob's effective intensity scales with
    // its local ownership share (45% floor + 55% by dominance), so each
    // color keeps a territory instead of dissolving into the additive mean
    const float inv = (Itot > 1e-5f) ? 1.0f / Itot : 0.0f;
    for (int i = 0; i < n; ++i) {
        const float J = Ii[i] * (0.45f + 0.55f * Ii[i] * inv);
        rgb[0] += p.pal.rgb[i][0] * p.pal.rgb[i][0] * J;
        rgb[1] += p.pal.rgb[i][1] * p.pal.rgb[i][1] * J;
        rgb[2] += p.pal.rgb[i][2] * p.pal.rgb[i][2] * J;
    }

    // dark-core mask: a pool at frame center with a wide feathered rim. The
    // pool's field coordinate is advected by the same shared fBm as the
    // blobs, so its edge wobbles organically instead of reading as a drawn
    // ellipse; guarantees "black stays in the middle" for every seed.
    {
        const float cu = u + 0.13f * fbm2(u * 2.0f + 13.7f, v * 2.0f,
                                          s + 3311u, oc - 1, lac, gn);
        const float cv = v + 0.13f * fbm2(u * 2.0f, v * 2.0f + 9.1f,
                                          s + 3317u, oc - 1, lac, gn);
        const float rc = sqrtf(cu * cu * 0.85f + cv * cv);
        const float cm = coreLift + (1.0f - coreLift) * sstep(0.05f, 0.50f, rc);
        rgb[0] *= cm;
        rgb[1] *= cm;
        rgb[2] *= cm;
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

    if (p.mode == kFieldGlow) {
        // Soft-blob path: no palette ramp, no relief - smooth light only,
        // with a faint macro luminance drift so large areas do not go dead
        // flat. Film grain lands later in the shared post tail.
        glowField(p, u, v, rgb, 0.0f);
        const float m = 0.90f + 0.20f *
            fbm2(u * p.view * 0.35f, v * p.view * 0.35f, p.seed + 1009u,
                 3, p.lacunarity, p.gain) * p.macroAmt;
        rgb[0] *= m;
        rgb[1] *= m;
        rgb[2] *= m;
        return;
    }
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
    if (p.mode == kFieldWarp || p.mode == kFieldRidge || p.mode == kFieldLiquid
        || p.mode == kFieldAgate || p.mode == kFieldShards || p.mode == kFieldAura) {
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
    } else if (p.mode == kFieldShards) {
        shade += 0.55f * powi(diff, 32);   // jeweled glints along the folds
    } else if (p.mode == kFieldAgate) {
        shade += 0.30f * powi(diff, 24);   // soft sheen on the ring crests
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
    } else if (p.mode == kFieldAura) {
        // additive cool-white glow on the ribbon crests: light, not paint
        const float glint = powi(diff, 8) * 0.30f;
        rgb[0] += glint * 0.88f;
        rgb[1] += glint;
        rgb[2] += glint * 1.12f;
    }
}

}  // namespace keenly