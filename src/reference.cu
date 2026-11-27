// CPU reference implementations of the art pipeline.
//
// ASCII-ONLY FILE ON PURPOSE (see src/cuda_utils.h for the reason).
//
// Three CPU entry points, all sharing the exact same art math as the GPU
// kernels (pixel.cuh / post.cuh / icon.cuh):
//
//   renderFieldCpu         serial reference - the ground truth the GPU gate
//                          compares against. Deliberately the most obvious
//                          code in the project: plain nested loops.
//   renderFieldCpuParallel same fieldPixel, split across row bands. Used by
//                          the --cpu backend. Must be bit-identical to the
//                          serial version: fieldPixel is a pure function of
//                          (x, y, params), so any split yields the same bytes
//                          - runCpuSplitCheck proves it on every verify run.
//   renderPostCpu          the post chain (bright / blur / compose) as serial
//                          loops. Before this existed, halation/bloom/vignette
//                          had NO reference at all: the gate only compared the
//                          field, while the delivered PNG also contains post.
//                          A blur index bug or a half-res sampling offset
//                          would have shipped silently.
//
// Bit-exactness discipline: every expression below is written in the same
// association order as the corresponding kernel. With --fmad=false and no
// fast-math there is no contraction on either side, so the float results are
// identical rather than merely close.

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#include "render.h"

namespace keenly
{

namespace
{

// Serial Jacobi Gray-Scott simulation, shared by both CPU field renderers.
// The returned vector owns the final B channel; FieldParams::turingTex is a
// raw pointer into it, so the caller must keep it alive for as long as the
// pixel loop runs.
std::vector<float> simulateTuringCpu(const FieldParams& p)
{
    const int res = p.rdRes;
    const size_t cells = size_t(res) * size_t(res);
    std::vector<float> a0(cells), a1(cells), b0(cells), b1(cells);

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

    // After rdIters swaps bIn points at the final state, which lives in the
    // local b0/b1 pair - copy it out so the pointer stays valid.
    std::vector<float> out(cells);
    std::memcpy(out.data(), bIn, cells * sizeof(float));
    return out;
}

// Params copy for one render, running the RD sim first when the mode needs
// it. `tex` must outlive the pixel loop that reads q.turingTex.
FieldParams prepareParams(const FieldParams& p, std::vector<float>& tex)
{
    FieldParams q = p;
    if (p.mode == kFieldTuring) {
        tex = simulateTuringCpu(p);
        q.turingTex = tex.data();
    }
    return q;
}

// Render rows [y0, y1) of the field. The single implementation of the CPU
// pixel loop, so the serial and threaded paths cannot drift apart.
void fieldRows(float* hRgb, int w, int h, int y0, int y1, const FieldParams& q)
{
    for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x)
            fieldPixel(q, x, y, w, h, hRgb + (size_t(y) * w + x) * 3);
}

}  // namespace

void renderFieldCpu(float* hRgb, int w, int h, const FieldParams& p)
{
    std::vector<float> tex;
    const FieldParams q = prepareParams(p, tex);
    fieldRows(hRgb, w, h, 0, h, q);
}

void renderFieldCpuParallel(float* hRgb, int w, int h, const FieldParams& p,
                            int threads)
{
    std::vector<float> tex;
    const FieldParams q = prepareParams(p, tex);

    if (threads <= 1 || h <= 1) {
        fieldRows(hRgb, w, h, 0, h, q);
        return;
    }

    std::vector<std::thread> ts;
    ts.reserve(size_t(threads));
    for (int t = 0; t < threads; ++t) {
        // Integer row bands: floor(t*h/T) .. floor((t+1)*h/T). Every row is
        // covered exactly once, and fieldPixel has no cross-pixel state, so
        // the result cannot depend on the thread count.
        const int y0 = int(size_t(h) * size_t(t) / size_t(threads));
        const int y1 = int(size_t(h) * size_t(t + 1) / size_t(threads));
        if (y0 >= y1) continue;
        ts.emplace_back(fieldRows, hRgb, w, h, y0, y1, q);
    }
    for (auto& th : ts) th.join();
}

// Split rows [0, rows) across `threads` workers. Every stage of the post
// chain is a per-row pure function over read-only inputs (bright pass reads
// the field, each blur pass reads its input buffer, compose reads the final
// glow), so a row split cannot change a single float - the thread count is
// a wall-clock knob only. renderPostCpu defaults to 1 thread so the gate
// still compares against the most obvious possible code.
namespace
{
template <class F>
void parallelRows(int rows, int threads, F fn)
{
    if (threads <= 1 || rows <= 1) {
        fn(0, rows);
        return;
    }
    std::vector<std::thread> ts;
    ts.reserve(size_t(threads));
    for (int t = 0; t < threads; ++t) {
        const int y0 = int(size_t(rows) * size_t(t) / size_t(threads));
        const int y1 = int(size_t(rows) * size_t(t + 1) / size_t(threads));
        if (y0 >= y1) continue;
        ts.emplace_back(fn, y0, y1);
    }
    for (auto& th : ts) th.join();
}
}  // namespace

void renderPostCpu(float* hOut, const float* hField, int w, int h,
                   const PostParams& p, int threads)
{
    const int hw = (w + 1) / 2;
    const int hh = (h + 1) / 2;
    std::vector<float> bright(size_t(hw) * size_t(hh) * 3);
    std::vector<float> tmp(size_t(hw) * size_t(hh) * 3);

    // ---- bright pass: box-average 2x2 -> half res, soft knee --------------
    // Mirrors brightKernel including the min() edge clamps and the
    // (y0,x0) (y0,x1) (y1,x0) (y1,x1) summation order.
    parallelRows(hh, threads, [&](int hy0, int hy1) {
        for (int hy = hy0; hy < hy1; ++hy) {
            for (int hx = 0; hx < hw; ++hx) {
                const int x0 = std::min(hx * 2, w - 1);
                const int y0 = std::min(hy * 2, h - 1);
                const int x1 = std::min(x0 + 1, w - 1);
                const int y1 = std::min(y0 + 1, h - 1);

                float c[3];
                for (int ch = 0; ch < 3; ++ch) {
                    c[ch] = 0.25f * (hField[(size_t(y0) * w + x0) * 3 + ch] +
                                     hField[(size_t(y0) * w + x1) * 3 + ch] +
                                     hField[(size_t(y1) * w + x0) * 3 + ch] +
                                     hField[(size_t(y1) * w + x1) * 3 + ch]) * p.exposure;
                }
                const float knee = sstep(0.45f, 1.1f, luminance(c));
                float* o = bright.data() + (size_t(hy) * hw + hx) * 3;
                o[0] = c[0] * knee;
                o[1] = c[1] * knee;
                o[2] = c[2] * knee;
            }
        }
    });

    // ---- separable 9-tap gaussian, two passes ----------------------------
    // Same tap weights, same accumulation order, same edge clamp as
    // blurKernel. Run h,v,h,v so the final glow lives in `bright`, matching
    // renderPostGpu's buffer ping-pong exactly.
    const float kW[5] = {0.2824f, 0.2200f, 0.1039f, 0.0298f, 0.0052f};
    auto blur = [&](float* out, const float* in, int dir) {
        parallelRows(hh, threads, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                for (int x = 0; x < hw; ++x) {
                    float acc[3] = {0.0f, 0.0f, 0.0f};
                    for (int t = -4; t <= 4; ++t) {
                        int sx = x, sy = y;
                        if (dir == 1) sx = std::min(std::max(x + t, 0), hw - 1);
                        else          sy = std::min(std::max(y + t, 0), hh - 1);
                        const float wgt = kW[t < 0 ? -t : t];
                        const float* s = in + (size_t(sy) * hw + sx) * 3;
                        acc[0] += wgt * s[0];
                        acc[1] += wgt * s[1];
                        acc[2] += wgt * s[2];
                    }
                    float* o = out + (size_t(y) * hw + x) * 3;
                    o[0] = acc[0];
                    o[1] = acc[1];
                    o[2] = acc[2];
                }
            }
        });
    };
    blur(tmp.data(), bright.data(), 1);
    blur(bright.data(), tmp.data(), 0);
    blur(tmp.data(), bright.data(), 1);
    blur(bright.data(), tmp.data(), 0);

    // ---- compose: halation + bloom + shared post tail --------------------
    parallelRows(h, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            for (int x = 0; x < w; ++x) {
                const float* f = hField + (size_t(y) * w + x) * 3;
                float c[3];
                c[0] = f[0] * p.exposure;
                c[1] = f[1] * p.exposure;
                c[2] = f[2] * p.exposure;

                const float* g = bright.data()
                               + (size_t(y / 2) * size_t(hw) + size_t(x / 2)) * 3;
                c[0] += g[0] * p.halation * 1.00f + g[0] * p.bloom;
                c[1] += g[1] * p.halation * 0.45f + g[1] * p.bloom;
                c[2] += g[2] * p.halation * 0.28f + g[2] * p.bloom;

                postTail(c, x, y, w, h, p, 1.0f);

                float* o = hOut + (size_t(y) * w + x) * 3;
                o[0] = c[0];
                o[1] = c[1];
                o[2] = c[2];
            }
        }
    });
}

namespace
{
void iconRows(float* hRgba, int w, int h, int y0, int y1, const IconParams& p)
{
    for (int y = y0; y < y1; ++y)
        for (int x = 0; x < w; ++x)
            iconPixel(p, x, y, w, h, hRgba + (size_t(y) * w + x) * 4);
}
}  // namespace

void renderIconCpu(float* hRgba, int w, int h, const IconParams& p)
{
    iconRows(hRgba, w, h, 0, h, p);
}

void renderIconCpuParallel(float* hRgba, int w, int h, const IconParams& p,
                           int threads)
{
    if (threads <= 1 || h <= 1) {
        iconRows(hRgba, w, h, 0, h, p);
        return;
    }

    std::vector<std::thread> ts;
    ts.reserve(size_t(threads));
    for (int t = 0; t < threads; ++t) {
        const int y0 = int(size_t(h) * size_t(t) / size_t(threads));
        const int y1 = int(size_t(h) * size_t(t + 1) / size_t(threads));
        if (y0 >= y1) continue;
        ts.emplace_back(iconRows, hRgba, w, h, y0, y1, p);
    }
    for (auto& th : ts) th.join();
}

}  // namespace keenly
