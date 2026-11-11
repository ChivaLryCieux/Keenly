#pragma once
// Per-pixel post tail, shared by the still pipeline (render.cu composeKernel)
// and the icon compositor (icon.cuh).
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Everything here is a pure function of (x, y, seed): strictly IEEE math only
// (polynomials, no transcendentals), so GPU and CPU stay bit-identical and
// the verify gate can demand maxAbs == 0 over the full icon pixel.
//
// Stage order (linear light in, display-referred out):
//   split-tone -> vividness -> film grain -> vignette -> ACES
// The still pipeline adds blur-based halation/bloom BEFORE this tail (it
// needs the half-res bright pass in render.cu); icons replace that glow with
// an SDF-distance halo in icon.cuh.

#include "noise.cuh"

namespace keenly
{

// Post-processing params: film grain, glow, vignette, tonemap. Applied in
// linear light on top of the field render.
struct PostParams
{
    unsigned seed = 7u;
    float exposure = 1.0f;    // linear exposure before tonemap
    float grain = 0.05f;      // grain amplitude in linear light
    float grainScale = 2.5f;  // grain clump size in pixels
    float halation = 0.35f;   // reddish glow strength around highlights
    float bloom = 0.12f;      // neutral glow strength
    float vignette = 0.35f;   // corner darkening amount
    float splitTone = 0.35f;  // teal shadows / warm highlights grade strength
    float saturation = 1.12f; // chroma expansion around luminance (1 = off)
};

__host__ __device__ inline float luminance(const float* c)
{
    return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
}

__host__ __device__ inline float acesFilmic(float x)
{
    // Narkowicz ACES fit: cheap, stable, the standard "cinematic" curve.
    return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
}

// Per-pixel post tail, in place on a linear RGB triple. grainMul scales the
// grain amplitude locally (icons damp grain inside the glyph so the symbol
// stays readable at 64px).
__host__ __device__ inline void postTail(float* c, int x, int y, int w, int h,
                                         const PostParams& p, float grainMul)
{
    // Split-tone grade (the real teal-orange mechanism, on top of the
    // palette): shadows drift cool, highlights drift warm, and the extreme
    // highlights desaturate toward neutral like film's shoulder does.
    {
        float L = luminance(c);
        L = L < 0.0f ? 0.0f : (L > 1.0f ? 1.0f : L);
        const float k = sstep(0.05f, 0.55f, L) * p.splitTone;
        const float kh = sstep(0.75f, 1.25f, L) * p.splitTone;
        c[0] *= 0.85f + 0.30f * k;
        c[2] *= 1.15f - 0.30f * k;
        // highlight desaturation: mix toward luminance by kh * 0.5
        const float m = kh * 0.5f;
        c[0] += (L - c[0]) * m;
        c[1] += (L - c[1]) * m;
        c[2] += (L - c[2]) * m;
    }

    // Vividness: expand chroma around luminance (film-style color
    // enhancement). No-op on achromatic content. Keeps the palette alive
    // when the macro-light layer darkens the frame.
    {
        const float Ls = luminance(c);
        const float s = p.saturation - 1.0f;
        c[0] += (c[0] - Ls) * s;
        c[1] += (c[1] - Ls) * s;
        c[2] += (c[2] - Ls) * s;
    }

    // Film grain: clumped at TWO scales (fine blobs + wide silver clumps)
    // with per-channel fine components - real film has one grain layer per
    // emulsion layer, so channels are slightly decorrelated. Amplitude is
    // modulated by a film-like midtone response curve 4*L*(1-L): strongest
    // in the midtones, invisible in blacks and whites.
    const float L2 = luminance(c);
    const float lc = L2 < 0.0f ? 0.0f : (L2 > 1.0f ? 1.0f : L2);
    const float response = 4.0f * lc * (1.0f - lc);
    const float blob = 0.6f * gradNoise2(float(x) / p.grainScale, float(y) / p.grainScale,
                                         p.seed + 7717u)
                     + 0.4f * gradNoise2(float(x) / (p.grainScale * 2.7f),
                                         float(y) / (p.grainScale * 2.7f),
                                         p.seed + 5501u);
    const float fineR = hash01(x, y, p.seed + 9931u) * 2.0f - 1.0f;
    const float fineG = hash01(x, y, p.seed + 13007u) * 2.0f - 1.0f;
    const float fineB = hash01(x, y, p.seed + 16223u) * 2.0f - 1.0f;
    const float amp = p.grain * response * grainMul;
    c[0] += amp * (0.70f * blob + 0.35f * fineR);
    c[1] += amp * (0.70f * blob + 0.30f * fineG);
    c[2] += amp * (0.70f * blob + 0.25f * fineB);

    // Vignette: elliptical, follows the frame shape.
    const float rx = (float(x) + 0.5f) / float(w) * 2.0f - 1.0f;
    const float ry = (float(y) + 0.5f) / float(h) * 2.0f - 1.0f;
    const float vig = 1.0f - p.vignette * sstep(0.30f, 1.20f, rx * rx + ry * ry);
    c[0] *= vig;
    c[1] *= vig;
    c[2] *= vig;

    // ACES tonemap per channel -> display-referred [0, 1].
    c[0] = acesFilmic(c[0] < 0.0f ? 0.0f : c[0]);
    c[1] = acesFilmic(c[1] < 0.0f ? 0.0f : c[1]);
    c[2] = acesFilmic(c[2] < 0.0f ? 0.0f : c[2]);
}

}  // namespace keenly