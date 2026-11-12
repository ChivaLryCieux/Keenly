#pragma once
// Art pipeline entry points: GPU renderers and the CPU reference used by the
// verify gate. All art math lives in pixel.cuh / noise.cuh.
//
// ASCII-ONLY FILE ON PURPOSE (see src/cuda_utils.h for the reason).

#include "pixel.cuh"
#include "post.cuh"
#include "icon.cuh"

namespace keenly
{

// Post-processing params (film grain, halation glow, vignette, ACES) live in
// post.cuh together with the per-pixel post tail shared by stills and icons.

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

// Icon render: RGBA float buffer (row-major, 4 floats per pixel), alpha is
// the rounded-rect / rect mask coverage. One pass - the glow is SDF-driven,
// no blur chain - so icons are a single pure-function kernel.
void renderIconGpu(float* dRgba, int w, int h, const IconParams& p);

// CPU reference for the icon compositor (same iconPixel).
void renderIconCpu(float* hRgba, int w, int h, const IconParams& p);

}  // namespace keenly