// CPU reference implementation of the art pipeline.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Same fieldPixel() as the GPU kernel, plain serial loops. Comparing the two
// buffers catches kernel plumbing bugs (indexing, swapped params, launch
// config). The math itself is shared by construction - see pixel.cuh.

#include <vector>

#include "render.h"

namespace keenly
{

void renderFieldCpu(float* hRgb, int w, int h, const FieldParams& p)
{
    FieldParams q = p;

    // Gray-Scott path: same Jacobi simulation as the GPU, serial. The shared
    // turingUpdate() inline is what keeps the two bit-identical.
    std::vector<float> a0, a1, b0, b1;
    if (p.mode == kFieldTuring) {
        const int res = p.rdRes;
        const size_t cells = size_t(res) * size_t(res);
        a0.resize(cells); a1.resize(cells);
        b0.resize(cells); b1.resize(cells);
        for (int y = 0; y < res; ++y)
            for (int x = 0; x < res; ++x) {
                a0[size_t(y) * res + x] = turingInitA(x, y, p.seed);
                b0[size_t(y) * res + x] = turingInitB(x, y, res, p.seed);
            }
        float* aIn = a0.data(); float* aOut = a1.data();
        float* bIn = b0.data(); float* bOut = b1.data();
        float a[9], b[9], na, nb;
        for (int it = 0; it < p.rdIters; ++it) {
            for (int y = 0; y < res; ++y) {
                for (int x = 0; x < res; ++x) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int sx = wrapi(x + dx, res);
                            const int sy = wrapi(y + dy, res);
                            const int i = (dy + 1) * 3 + (dx + 1);
                            a[i] = aIn[size_t(sy) * res + sx];
                            b[i] = bIn[size_t(sy) * res + sx];
                        }
                    }
                    turingUpdate(a, b, p.rdF, p.rdK, &na, &nb);
                    aOut[size_t(y) * res + x] = na;
                    bOut[size_t(y) * res + x] = nb;
                }
            }
            float* t;
            t = aIn; aIn = aOut; aOut = t;
            t = bIn; bIn = bOut; bOut = t;
        }
        q.turingTex = bIn;
    }

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            fieldPixel(q, x, y, w, h, hRgb + (size_t(y) * w + x) * 3);
}

void renderIconCpu(float* hRgba, int w, int h, const IconParams& p)
{
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            iconPixel(p, x, y, w, h, hRgba + (size_t(y) * w + x) * 4);
}

}  // namespace keenly