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

// ---- post chain (stage 2) --------------------------------------------------
// Half-res bright pass, separable gaussian for the glow, then a per-pixel
// compose: halation + film grain + vignette + ACES. All in linear light.

__device__ float luminance(const float* c)
{
    return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
}

__device__ float acesFilmic(float x)
{
    // Narkowicz ACES fit: cheap, stable, the standard "cinematic" curve.
    return (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
}

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
    const float amp = p.grain * response;
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
    float* o = out + (size_t(y) * w + x) * 3;
    o[0] = acesFilmic(c[0] < 0.0f ? 0.0f : c[0]);
    o[1] = acesFilmic(c[1] < 0.0f ? 0.0f : c[1]);
    o[2] = acesFilmic(c[2] < 0.0f ? 0.0f : c[2]);
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

}  // namespace art