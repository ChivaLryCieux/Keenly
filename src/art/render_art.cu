// Art pipeline driver: renders generative art backgrounds on the GPU and
// writes PNG stills.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Every render runs the verify gate first (unless --no-check): the same
// pixel shader executes on CPU and GPU and the two buffers must agree to
// relL2 < 2e-4, the same "prove it on the CPU" philosophy as bench_matmul.
// The budget is not zero: nvcc --use_fast_math plus FMA contraction leave a
// measured rounding floor of ~6.5e-5 over this pipeline (it grows with the
// length of the float-op chain: 5 field evaluations + per-octave rotation).
// Real plumbing bugs land at 1e-2 and above, so 2e-4 still has ~50x of
// detection headroom.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "art_render.h"
#include "cuda_utils.h"
#include "png_writer.h"
#include "reference.h"

struct Args
{
    int width = 2560;
    int height = 1072;  // 2.39:1 cinema frame
    int depth = 16;     // PNG bits per channel: 8 or 16
    bool check = true;
    bool post = true;
    unsigned postSet = 0;  // which params were explicitly set on the CLI
    std::string preset = "cinema";
    art::FieldParams field;
    art::PostParams postp;
    std::string out = "out/field.png";
};

// ---- presets ---------------------------------------------------------------
// Palette stops are perceptual (sRGB-like); fieldPixel linearizes them.
// Each preset also carries field + post defaults (grain character differs:
// clumped 35mm grain for cinema, fine matte grain for the poster). Explicit
// CLI flags always win over preset defaults.

enum SetBits
{
    kSetView = 1u << 0,      kSetOctaves = 1u << 1,   kSetWarp = 1u << 2,
    kSetContrast = 1u << 3,  kSetRelief = 1u << 4,    kSetSeed = 1u << 5,
    kSetExposure = 1u << 6,  kSetGrain = 1u << 7,     kSetGrainScale = 1u << 8,
    kSetHalation = 1u << 9,  kSetBloom = 1u << 10,    kSetVignette = 1u << 11,
    kSetSplitTone = 1u << 12
};

static void applyPreset(art::FieldParams& f, art::PostParams& p, const std::string& name,
                        unsigned set)
{
    if (name == "mono") {
        f.pal.count = 2;
        f.pal.pos[0] = 0.0f;  f.pal.pos[1] = 1.0f;
        f.pal.rgb[0][0] = 0.03f; f.pal.rgb[0][1] = 0.03f; f.pal.rgb[0][2] = 0.04f;
        f.pal.rgb[1][0] = 0.96f; f.pal.rgb[1][1] = 0.96f; f.pal.rgb[1][2] = 0.97f;
        if (!(set & kSetContrast)) f.contrast = 1.5f;
        if (!(set & kSetGrain))      p.grain = 0.055f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.2f;
        if (!(set & kSetHalation))   p.halation = 0.30f;
        if (!(set & kSetVignette))   p.vignette = 0.35f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.0f;
    } else if (name == "cinema") {
        // teal shadows -> amber highlights (the classic teal-orange grade)
        f.pal.count = 5;
        f.pal.pos[0] = 0.00f; f.pal.pos[1] = 0.30f; f.pal.pos[2] = 0.55f;
        f.pal.pos[3] = 0.75f; f.pal.pos[4] = 1.00f;
        f.pal.rgb[0][0] = 0.02f; f.pal.rgb[0][1] = 0.04f; f.pal.rgb[0][2] = 0.07f;
        f.pal.rgb[1][0] = 0.06f; f.pal.rgb[1][1] = 0.22f; f.pal.rgb[1][2] = 0.30f;
        f.pal.rgb[2][0] = 0.20f; f.pal.rgb[2][1] = 0.42f; f.pal.rgb[2][2] = 0.46f;
        f.pal.rgb[3][0] = 0.80f; f.pal.rgb[3][1] = 0.52f; f.pal.rgb[3][2] = 0.28f;
        f.pal.rgb[4][0] = 1.00f; f.pal.rgb[4][1] = 0.92f; f.pal.rgb[4][2] = 0.78f;
        if (!(set & kSetGrain))      p.grain = 0.05f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.8f;
        if (!(set & kSetHalation))   p.halation = 0.40f;
        if (!(set & kSetVignette))   p.vignette = 0.38f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.45f;
    } else if (name == "dusk") {
        // indigo -> magenta -> coral (poster palette)
        f.pal.count = 5;
        f.pal.pos[0] = 0.00f; f.pal.pos[1] = 0.32f; f.pal.pos[2] = 0.58f;
        f.pal.pos[3] = 0.80f; f.pal.pos[4] = 1.00f;
        f.pal.rgb[0][0] = 0.05f; f.pal.rgb[0][1] = 0.04f; f.pal.rgb[0][2] = 0.12f;
        f.pal.rgb[1][0] = 0.18f; f.pal.rgb[1][1] = 0.14f; f.pal.rgb[1][2] = 0.38f;
        f.pal.rgb[2][0] = 0.62f; f.pal.rgb[2][1] = 0.22f; f.pal.rgb[2][2] = 0.48f;
        f.pal.rgb[3][0] = 0.95f; f.pal.rgb[3][1] = 0.48f; f.pal.rgb[3][2] = 0.35f;
        f.pal.rgb[4][0] = 1.00f; f.pal.rgb[4][1] = 0.88f; f.pal.rgb[4][2] = 0.72f;
        if (!(set & kSetView))       f.view = 2.8f;
        if (!(set & kSetWarp))       f.warp = 1.35f;
        if (!(set & kSetContrast))   f.contrast = 1.35f;
        if (!(set & kSetRelief))     f.relief = 2.2f;
        if (!(set & kSetExposure))   p.exposure = 1.05f;
        if (!(set & kSetGrain))      p.grain = 0.035f;
        if (!(set & kSetGrainScale)) p.grainScale = 1.3f;
        if (!(set & kSetHalation))   p.halation = 0.22f;
        if (!(set & kSetBloom))      p.bloom = 0.10f;
        if (!(set & kSetVignette))   p.vignette = 0.28f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.25f;
    } else {
        std::fprintf(stderr, "unknown preset: %s (use mono|cinema|dusk)\n", name.c_str());
        std::exit(1);
    }
}

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

static void printUsage()
{
    std::printf(
        "render_art - GPU generative art stills -> PNG\n"
        "usage: render_art [options]\n"
        "  --width  N     image width in pixels   (default 2560)\n"
        "  --height N     image height in pixels  (default 1072)\n"
        "  --depth  D     PNG bits per channel: 8 or 16 (default 16)\n"
        "  --out    P     output PNG path         (default out/field.png)\n"
        "  --preset P     mono | cinema | dusk    (default cinema)\n"
        "  --seed   S     deterministic seed      (default 7)\n"
        "  --view   F     noise extent across height (default 3.2)\n"
        "  --octaves N    fBm octaves             (default 5)\n"
        "  --warp   F     domain warp strength    (default 1.2)\n"
        "  --contrast F   field shaping           (default 1.4)\n"
        "  --relief F     shading strength        (default 2.5)\n"
        "  --exposure F   linear exposure         (default 1.0)\n"
        "  --grain F      grain amplitude         (default 0.06)\n"
        "  --grain-scale F grain clump size, px   (default 2.5)\n"
        "  --halation F   red highlight glow      (default 0.35)\n"
        "  --bloom F      neutral glow            (default 0.12)\n"
        "  --vignette F   corner darkening        (default 0.35)\n"
        "  --split-tone F teal shadows / warm highs grade (default 0.35)\n"
        "  --no-post      skip grain/glow/vignette/tonemap (raw field)\n"
        "  --no-check     skip the CPU reference gate\n");
}

static bool parseArgs(int argc, char** argv, Args& a)
{
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (key == "--help" || key == "-h") {
            printUsage();
            std::exit(0);
        } else if (key == "--width") {
            const char* v = next("--width");
            if (!v) return false;
            a.width = std::atoi(v);
        } else if (key == "--height") {
            const char* v = next("--height");
            if (!v) return false;
            a.height = std::atoi(v);
        } else if (key == "--depth") {
            const char* v = next("--depth");
            if (!v) return false;
            a.depth = std::atoi(v);
        } else if (key == "--out") {
            const char* v = next("--out");
            if (!v) return false;
            a.out = v;
        } else if (key == "--preset") {
            const char* v = next("--preset");
            if (!v) return false;
            a.preset = v;
        } else if (key == "--seed") {
            const char* v = next("--seed");
            if (!v) return false;
            a.field.seed = unsigned(std::strtoul(v, nullptr, 10));
            a.postSet |= kSetSeed;
        } else if (key == "--view") {
            const char* v = next("--view");
            if (!v) return false;
            a.field.view = float(std::atof(v));
            a.postSet |= kSetView;
        } else if (key == "--octaves") {
            const char* v = next("--octaves");
            if (!v) return false;
            a.field.octaves = std::atoi(v);
            a.postSet |= kSetOctaves;
        } else if (key == "--warp") {
            const char* v = next("--warp");
            if (!v) return false;
            a.field.warp = float(std::atof(v));
            a.postSet |= kSetWarp;
        } else if (key == "--contrast") {
            const char* v = next("--contrast");
            if (!v) return false;
            a.field.contrast = float(std::atof(v));
            a.postSet |= kSetContrast;
        } else if (key == "--relief") {
            const char* v = next("--relief");
            if (!v) return false;
            a.field.relief = float(std::atof(v));
            a.postSet |= kSetRelief;
        } else if (key == "--exposure") {
            const char* v = next("--exposure");
            if (!v) return false;
            a.postp.exposure = float(std::atof(v));
            a.postSet |= kSetExposure;
        } else if (key == "--grain") {
            const char* v = next("--grain");
            if (!v) return false;
            a.postp.grain = float(std::atof(v));
            a.postSet |= kSetGrain;
        } else if (key == "--grain-scale") {
            const char* v = next("--grain-scale");
            if (!v) return false;
            a.postp.grainScale = float(std::atof(v));
            a.postSet |= kSetGrainScale;
        } else if (key == "--halation") {
            const char* v = next("--halation");
            if (!v) return false;
            a.postp.halation = float(std::atof(v));
            a.postSet |= kSetHalation;
        } else if (key == "--bloom") {
            const char* v = next("--bloom");
            if (!v) return false;
            a.postp.bloom = float(std::atof(v));
            a.postSet |= kSetBloom;
        } else if (key == "--vignette") {
            const char* v = next("--vignette");
            if (!v) return false;
            a.postp.vignette = float(std::atof(v));
            a.postSet |= kSetVignette;
        } else if (key == "--split-tone") {
            const char* v = next("--split-tone");
            if (!v) return false;
            a.postp.splitTone = float(std::atof(v));
            a.postSet |= kSetSplitTone;
        } else if (key == "--no-post") {
            a.post = false;
        } else if (key == "--no-check") {
            a.check = false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", key.c_str());
            return false;
        }
    }
    if (a.width <= 0 || a.height <= 0 || (a.depth != 8 && a.depth != 16)) {
        std::fprintf(stderr, "bad size or depth: %dx%d depth %d\n", a.width, a.height, a.depth);
        return false;
    }
    return true;
}

// CPU reference gate: same pixel shader, serial vs parallel execution.
// relL2 < 2e-4 is the pass budget (fast-math rounding floor, see file header).
static bool runCheck(const art::FieldParams& p)
{
    const int w = 320, h = 180;
    const size_t n = size_t(w) * size_t(h) * 3;
    std::vector<float> ref(n), gpu(n);

    art::renderFieldCpu(ref.data(), w, h, p);

    float* dBuf = nullptr;
    CUDA_CHECK(cudaMalloc(&dBuf, n * sizeof(float)));
    art::renderFieldGpu(dBuf, w, h, p);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dBuf, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dBuf));

    const DiffStats d = compareArrays(gpu.data(), ref.data(), n);
    std::printf("check  : relL2 %.3e  maxAbs %.3e  (GPU vs CPU, %dx%d)\n",
                d.relL2, d.maxAbs, w, h);
    return d.relL2 < 2e-4;
}

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage();
        return 1;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    applyPreset(args.field, args.postp, args.preset, args.postSet);
    std::printf("device : %s\n", prop.name);
    std::printf("render : %dx%d  preset %s  seed %u  view %.2f  warp %.2f  octaves %d\n",
                args.width, args.height, args.preset.c_str(), args.field.seed,
                args.field.view, args.field.warp, args.field.octaves);

    if (args.check && !runCheck(args.field)) {
        std::fprintf(stderr, "CHECK FAILED: GPU disagrees with the CPU reference\n");
        return 1;
    }

    float* dField = nullptr;
    const size_t pixels = size_t(args.width) * size_t(args.height);
    CUDA_CHECK(cudaMalloc(&dField, pixels * 3 * sizeof(float)));

    GpuTimer timer;
    timer.start();
    art::renderFieldGpu(dField, args.width, args.height, args.field);
    const float msField = timer.stop();
    std::printf("field  : %.3f ms  (%.1f Mpixel/s)\n", msField, pixels / (double(msField) * 1e3));

    std::vector<float> hRgb(pixels * 3);
    if (args.post) {
        art::PostParams post = args.postp;
        post.seed = args.field.seed;  // one seed per image: grain rerolls per seed
        float* dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dOut, pixels * 3 * sizeof(float)));
        timer.start();
        art::renderPostGpu(dOut, dField, args.width, args.height, post);
        const float msPost = timer.stop();
        std::printf("post   : %.3f ms  (grain %.3f  halation %.2f  bloom %.2f  vignette %.2f  split %.2f)\n",
                    msPost, post.grain, post.halation, post.bloom, post.vignette,
                    post.splitTone);
        CUDA_CHECK(cudaMemcpy(hRgb.data(), dOut, pixels * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dOut));
    } else {
        CUDA_CHECK(cudaMemcpy(hRgb.data(), dField, pixels * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(dField));

    std::error_code ec;
    std::filesystem::path outPath(args.out);
    if (outPath.has_parent_path())
        std::filesystem::create_directories(outPath.parent_path(), ec);

    bool ok = false;
    if (args.depth == 16) {
        std::vector<uint16_t> img(pixels * 3);
        for (size_t i = 0; i < pixels * 3; ++i) {
            float v = srgbEncode(hRgb[i]) + tpdfDither(uint32_t(i), 1.0f / 65535.0f);
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            img[i] = uint16_t(v * 65535.0f + 0.5f);
        }
        ok = art::writePng(args.out, args.width, args.height, 16, img.data());
    } else {
        std::vector<uint8_t> img(pixels * 3);
        for (size_t i = 0; i < pixels * 3; ++i) {
            float v = srgbEncode(hRgb[i]) + tpdfDither(uint32_t(i), 1.0f / 255.0f);
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            img[i] = uint8_t(v * 255.0f + 0.5f);
        }
        ok = art::writePng(args.out, args.width, args.height, 8, img.data());
    }

    if (!ok) {
        std::fprintf(stderr, "failed to write %s\n", args.out.c_str());
        return 1;
    }
    std::printf("wrote  : %s  (%d-bit)\n", args.out.c_str(), args.depth);
    return 0;
}