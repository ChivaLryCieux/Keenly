// CPU reference implementation of the art pipeline.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Same fieldPixel() as the GPU kernel, plain serial loops. Comparing the two
// buffers catches kernel plumbing bugs (indexing, swapped params, launch
// config). The math itself is shared by construction - see pixel.cuh.

#include "render.h"

namespace keenly
{

void renderFieldCpu(float* hRgb, int w, int h, const FieldParams& p)
{
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            fieldPixel(p, x, y, w, h, hRgb + (size_t(y) * w + x) * 3);
}

void renderIconCpu(float* hRgba, int w, int h, const IconParams& p)
{
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            iconPixel(p, x, y, w, h, hRgba + (size_t(y) * w + x) * 4);
}

}  // namespace keenly