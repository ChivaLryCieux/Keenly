#pragma once
// Art pipeline entry points: GPU renderers and the CPU reference used by the
// verify gate. All art math lives in art_pixel.cuh / art_noise.cuh.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).

#include "art_pixel.cuh"

namespace art
{

// Render the field into an RGB float buffer on the device (row-major, 3
// floats per pixel). Launch errors are fatal (CUDA_CHECK inside).
void renderFieldGpu(float* dRgb, int w, int h, const FieldParams& p);

// CPU reference: identical pixel shader, serial row-major traversal.
void renderFieldCpu(float* hRgb, int w, int h, const FieldParams& p);

}  // namespace art