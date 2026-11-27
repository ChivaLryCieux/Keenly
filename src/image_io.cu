#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

#include "png_writer.h"

// Linear -> sRGB for display encoding, applied at PNG write time.
static inline float srgbEncode(float v)
{
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

// Triangular-PDF dither before quantization: kills 8-bit banding in smooth
// gradients (vignette, glow). Deterministic LCG per sample index.
static inline float tpdfDither(uint32_t idx, float step)
{
    uint32_t s = idx * 1664525u + 1013904223u;
    const float r1 = float((s >> 8) & 0xFFFFFFu) * (1.0f / 16777216.0f);
    s = s * 1664525u + 1013904223u;
    const float r2 = float((s >> 8) & 0xFFFFFFu) * (1.0f / 16777216.0f);
    return (r1 - r2) * step;
}

// Split an index range across worker threads. The encode loops are pure
// per-sample functions (srgbEncode + tpdfDither(i) + quantize), so any chunk
// split produces byte-identical output - this is a wall-clock win only.
// Small ranges run inline: thread launch would cost more than the work.
template <class F>
static void parallelFor(size_t n, F fn)
{
    unsigned tc = std::thread::hardware_concurrency();
    if (tc == 0) tc = 1;
    if (tc > 16) tc = 16;
    if (tc == 1 || n < 262144) {
        fn(size_t(0), n);
        return;
    }
    std::vector<std::thread> ts;
    const size_t chunk = (n + tc - 1) / tc;
    for (unsigned t = 0; t < tc; ++t) {
        const size_t lo = size_t(t) * chunk;
        const size_t hi = std::min(n, lo + chunk);
        if (lo >= hi) break;
        ts.emplace_back([=] { fn(lo, hi); });
    }
    for (auto& th : ts) th.join();
}

int cpuThreadCount(int requested)
{
    int threads = requested;
    if (threads <= 0) {
        threads = int(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 1;
    }
    return threads;
}

// Linear RGB buffer -> PNG (sRGB encode + TPDF dither + quantize).
// Encodes in parallel across CPU cores; output bytes are identical to the
// old serial loop (per-sample pure math, deterministic LCG dither).
bool writeImage(const std::string& path, int w, int h, int depth,
                       const float* linear)
{
    std::error_code ec;
    std::filesystem::path outPath(path);
    if (outPath.has_parent_path())
        std::filesystem::create_directories(outPath.parent_path(), ec);

    const size_t n = size_t(w) * size_t(h) * 3;
    if (depth == 16) {
        std::vector<uint16_t> img(n);
        parallelFor(n, [&](size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i) {
                float v = srgbEncode(linear[i]) + tpdfDither(uint32_t(i), 1.0f / 65535.0f);
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                img[i] = uint16_t(v * 65535.0f + 0.5f);
            }
        });
        return keenly::writePng(path, w, h, 16, img.data());
    }
    std::vector<uint8_t> img(n);
    parallelFor(n, [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
            float v = srgbEncode(linear[i]) + tpdfDither(uint32_t(i), 1.0f / 255.0f);
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            img[i] = uint8_t(v * 255.0f + 0.5f);
        }
    });
    return keenly::writePng(path, w, h, 8, img.data());
}

// RGBA float buffer -> PNG (sRGB on color, straight quantize on alpha).
// RGB channels get the same TPDF dither as writeImage (icon backgrounds are
// smooth gradients - banding shows). Alpha is a coverage mask, not a color:
// no dither there, so icon edges stay clean after engine compositing.
// Chunks are per-pixel (i += 4), so the parallel split keeps alpha with its
// pixel.
bool writeImageRgba(const std::string& path, int w, int h,
                           const float* linear)
{
    std::error_code ec;
    std::filesystem::path outPath(path);
    if (outPath.has_parent_path())
        std::filesystem::create_directories(outPath.parent_path(), ec);

    const size_t n = size_t(w) * size_t(h) * 4;
    std::vector<uint8_t> img(n);
    parallelFor(n / 4, [&](size_t lo, size_t hi) {
        for (size_t pix = lo; pix < hi; ++pix) {
            const size_t i = pix * 4;
            for (int c = 0; c < 3; ++c) {
                float v = srgbEncode(linear[i + c])
                        + tpdfDither(uint32_t(i + c), 1.0f / 255.0f);
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                img[i + c] = uint8_t(v * 255.0f + 0.5f);
            }
            float a = linear[i + 3];
            a = a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a);
            img[i + 3] = uint8_t(a * 255.0f + 0.5f);
        }
    });
    return keenly::writePng(path, w, h, 8, img.data(), 4);
}

// ---- generator table -------------------------------------------------------
// Single source of truth for --mode names. Index == FieldMode enum value
// (pixel.cuh), so parseArgs, printUsage and --check-all all derive from this
