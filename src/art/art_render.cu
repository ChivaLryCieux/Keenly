// GPU kernels for the art pipeline.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).

#include "art_render.h"
#include "cuda_utils.h"

namespace art
{

namespace
{

__global__ void fieldKernel(float* rgb, int w, int h, FieldParams p)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    fieldPixel(p, x, y, w, h, rgb + (size_t(y) * w + x) * 3);
}

}  // namespace

void renderFieldGpu(float* dRgb, int w, int h, const FieldParams& p)
{
    const dim3 block(16, 16);
    const dim3 grid((w + int(block.x) - 1) / int(block.x),
                    (h + int(block.y) - 1) / int(block.y));
    fieldKernel<<<grid, block>>>(dRgb, w, h, p);
    CUDA_CHECK(cudaGetLastError());  // a failed launch must not look like success
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace art