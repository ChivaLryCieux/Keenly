#pragma once
// Art pipeline entry points: GPU renderers and the CPU reference used by the
// verify gate. All art math lives in art_pixel.cuh / art_noise.cuh.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).

#include "art_pixel.cuh"

namespace art
{

// Post-processing chain (stage 2): film grain, halation glow, vignette,
// ACES tonemap. Applied in linear light on top of the field render.
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
};

// Render the field into an RGB float buffer on the device (row-major, 3
// floats per pixel). Launch errors are fatal (CUDA_CHECK inside).
void renderFieldGpu(float* dRgb, int w, int h, const FieldParams& p);

// Full post chain on top of a rendered field buffer: half-res bright pass ->
// separable gaussian -> compose (halation + grain + vignette + ACES).
// Output is display-referred linear RGB in [0, 1]; sRGB encoding happens at
// PNG write time. Allocations are internal (one-shot stills, not a viewer).
void renderPostGpu(float* dOut, const float* dField, int w, int h, const PostParams& p);

// CPU reference: identical pixel shader, serial row-major traversal.
void renderFieldCpu(float* hRgb, int w, int h, const FieldParams& p);

}  // namespace art