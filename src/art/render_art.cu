// Art pipeline driver: renders generative art backgrounds on the GPU and
// writes PNG stills.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Every render runs the verify gate first (unless --no-check): the same
// pixel shader executes on CPU and GPU and the two buffers must agree to
// relL2 < 1e-4, the same "prove it on the CPU" philosophy as bench_matmul.
// The budget is not zero: nvcc --use_fast_math plus FMA contraction leave a
// measured rounding floor of ~1.4e-5 over this pipeline (6e-6 without
// fast-math). Real plumbing bugs land at 1e-2 and above, so 1e-4 still has
// ~100x of detection headroom.

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
    std::string preset = "cinema";
    art::FieldParams field;
    std::string out = "out/field.png";
};

// ---- presets ---------------------------------------------------------------
// Palette stops are perceptual (sRGB-like); fieldPixel linearizes them.

static void applyPreset(art::FieldParams& f, const std::string& name)
{
    if (name == "mono") {
        f.pal.count = 2;
        f.pal.pos[0] = 0.0f;  f.pal.pos[1] = 1.0f;
        f.pal.rgb[0][0] = 0.03f; f.pal.rgb[0][1] = 0.03f; f.pal.rgb[0][2] = 0.04f;
        f.pal.rgb[1][0] = 0.96f; f.pal.rgb[1][1] = 0.96f; f.pal.rgb[1][2] = 0.97f;
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
        } else if (key == "--view") {
            const char* v = next("--view");
            if (!v) return false;
            a.field.view = float(std::atof(v));
        } else if (key == "--octaves") {
            const char* v = next("--octaves");
            if (!v) return false;
            a.field.octaves = std::atoi(v);
        } else if (key == "--warp") {
            const char* v = next("--warp");
            if (!v) return false;
            a.field.warp = float(std::atof(v));
        } else if (key == "--contrast") {
            const char* v = next("--contrast");
            if (!v) return false;
            a.field.contrast = float(std::atof(v));
        } else if (key == "--relief") {
            const char* v = next("--relief");
            if (!v) return false;
            a.field.relief = float(std::atof(v));
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
// relL2 < 1e-4 is the pass budget (fast-math rounding floor, see file header).
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
    return d.relL2 < 1e-4;
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
    applyPreset(args.field, args.preset);
    std::printf("device : %s\n", prop.name);
    std::printf("render : %dx%d  preset %s  seed %u  view %.2f  warp %.2f  octaves %d\n",
                args.width, args.height, args.preset.c_str(), args.field.seed,
                args.field.view, args.field.warp, args.field.octaves);

    if (args.check && !runCheck(args.field)) {
        std::fprintf(stderr, "CHECK FAILED: GPU disagrees with the CPU reference\n");
        return 1;
    }

    float* dRgb = nullptr;
    const size_t pixels = size_t(args.width) * size_t(args.height);
    CUDA_CHECK(cudaMalloc(&dRgb, pixels * 3 * sizeof(float)));

    GpuTimer timer;
    timer.start();
    art::renderFieldGpu(dRgb, args.width, args.height, args.field);
    const float ms = timer.stop();
    std::printf("kernel : %.3f ms  (%.1f Mpixel/s)\n", ms, pixels / (double(ms) * 1e3));

    std::vector<float> hRgb(pixels * 3);
    CUDA_CHECK(cudaMemcpy(hRgb.data(), dRgb, pixels * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dRgb));

    std::error_code ec;
    std::filesystem::path outPath(args.out);
    if (outPath.has_parent_path())
        std::filesystem::create_directories(outPath.parent_path(), ec);

    bool ok = false;
    if (args.depth == 16) {
        std::vector<uint16_t> img(pixels * 3);
        for (size_t i = 0; i < pixels * 3; ++i)
            img[i] = uint16_t(srgbEncode(hRgb[i]) * 65535.0f + 0.5f);
        ok = art::writePng(args.out, args.width, args.height, 16, img.data());
    } else {
        std::vector<uint8_t> img(pixels * 3);
        for (size_t i = 0; i < pixels * 3; ++i)
            img[i] = uint8_t(srgbEncode(hRgb[i]) * 255.0f + 0.5f);
        ok = art::writePng(args.out, args.width, args.height, 8, img.data());
    }

    if (!ok) {
        std::fprintf(stderr, "failed to write %s\n", args.out.c_str());
        return 1;
    }
    std::printf("wrote  : %s  (%d-bit)\n", args.out.c_str(), args.depth);
    return 0;
}