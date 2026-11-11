// GPU kernels for the art pipeline.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).

#include "render.h"
#include "cuda_utils.h"

namespace keenly
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

// ---- post chain (stage 2) --------------------------------------------------
// Half-res bright pass, separable gaussian for the glow, then a per-pixel
// compose: halation + the shared post tail (post.cuh: split-tone, grain,
// vignette, ACES). All in linear light until the tail.

// Box-average 2x2 -> half res, keep only what glows (soft knee above
// threshold). This is the halation source: film highlights bleed red.
__global__ void brightKernel(float* out, const float* field, int w, int h,
                             int hw, int hh, PostParams p)
{
    const int hx = blockIdx.x * blockDim.x + threadIdx.x;
    const int hy = blockIdx.y * blockDim.y + threadIdx.y;
    if (hx >= hw || hy >= hh) return;

    const int x0 = min(hx * 2, w - 1);
    const int y0 = min(hy * 2, h - 1);
    const int x1 = min(x0 + 1, w - 1);
    const int y1 = min(y0 + 1, h - 1);

    float c[3];
    for (int ch = 0; ch < 3; ++ch) {
        c[ch] = 0.25f * (field[(size_t(y0) * w + x0) * 3 + ch] +
                         field[(size_t(y0) * w + x1) * 3 + ch] +
                         field[(size_t(y1) * w + x0) * 3 + ch] +
                         field[(size_t(y1) * w + x1) * 3 + ch]) * p.exposure;
    }
    const float knee = sstep(0.45f, 1.1f, luminance(c));
    float* o = out + (size_t(hy) * hw + hx) * 3;
    o[0] = c[0] * knee;
    o[1] = c[1] * knee;
    o[2] = c[2] * knee;
}

// Separable 9-tap gaussian (sigma ~2), dir=1 horizontal / dir=0 vertical.
// Edge pixels clamp, so borders never darken.
__global__ void blurKernel(float* out, const float* in, int w, int h, int dir)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float kW[5] = {0.2824f, 0.2200f, 0.1039f, 0.0298f, 0.0052f};

    float acc[3] = {0.0f, 0.0f, 0.0f};
    for (int t = -4; t <= 4; ++t) {
        int sx = x, sy = y;
        if (dir == 1) sx = min(max(x + t, 0), w - 1);
        else          sy = min(max(y + t, 0), h - 1);
        const float wgt = kW[t < 0 ? -t : t];
        const float* s = in + (size_t(sy) * w + sx) * 3;
        acc[0] += wgt * s[0];
        acc[1] += wgt * s[1];
        acc[2] += wgt * s[2];
    }
    float* o = out + (size_t(y) * w + x) * 3;
    o[0] = acc[0];
    o[1] = acc[1];
    o[2] = acc[2];
}

// Per-pixel compose. Everything here is a pure function of (x, y, seed), so
// any tile of the image renders identically to the whole frame (future A1
// tiling) and every frame of an animation gets fresh grain by shifting seed.
__global__ void composeKernel(float* out, const float* field, const float* glow,
                              int w, int h, PostParams p)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float* f = field + (size_t(y) * w + x) * 3;
    float c[3];
    c[0] = f[0] * p.exposure;
    c[1] = f[1] * p.exposure;
    c[2] = f[2] * p.exposure;

    // Halation (red bleed) + neutral bloom from the blurred bright pass.
    const float* g = glow + (size_t(y / 2) * ((w + 1) / 2) + x / 2) * 3;
    c[0] += g[0] * p.halation * 1.00f + g[0] * p.bloom;
    c[1] += g[1] * p.halation * 0.45f + g[1] * p.bloom;
    c[2] += g[2] * p.halation * 0.28f + g[2] * p.bloom;

    // Shared per-pixel tail (post.cuh): split-tone -> vividness -> grain ->
    // vignette -> ACES. Identical code path for stills and icons.
    postTail(c, x, y, w, h, p, 1.0f);

    float* o = out + (size_t(y) * w + x) * 3;
    o[0] = c[0];
    o[1] = c[1];
    o[2] = c[2];
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

void renderPostGpu(float* dOut, const float* dField, int w, int h, const PostParams& p)
{
    const int hw = (w + 1) / 2;
    const int hh = (h + 1) / 2;
    float *dBright = nullptr, *dTmp = nullptr;
    CUDA_CHECK(cudaMalloc(&dBright, size_t(hw) * size_t(hh) * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dTmp, size_t(hw) * size_t(hh) * 3 * sizeof(float)));

    const dim3 block(16, 16);
    const dim3 gridH((hw + 15) / 16, (hh + 15) / 16);
    const dim3 gridF((w + 15) / 16, (h + 15) / 16);

    brightKernel<<<gridH, block>>>(dBright, dField, w, h, hw, hh, p);
    blurKernel<<<gridH, block>>>(dTmp, dBright, hw, hh, 1);      // horizontal
    blurKernel<<<gridH, block>>>(dBright, dTmp, hw, hh, 0);      // vertical
    blurKernel<<<gridH, block>>>(dTmp, dBright, hw, hh, 1);      // 2nd pass: wider,
    blurKernel<<<gridH, block>>>(dBright, dTmp, hw, hh, 0);      // softer glow
    composeKernel<<<gridF, block>>>(dOut, dField, dBright, w, h, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(dBright));
    CUDA_CHECK(cudaFree(dTmp));
}

}  // namespace keenly