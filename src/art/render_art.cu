// Art pipeline driver: renders generative art backgrounds on the GPU and
// writes PNG stills.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Stage 1 (toolchain): CLI parsing, GPU buffer, PNG writer. The image is a
// test pattern - a fine ramp checks 16-bit gradient banding, the color chips
// check channel order, the frame checks bounds. The real art lands in the
// next commits.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "png_writer.h"

struct Args
{
    int width = 2560;
    int height = 1088;
    int depth = 16;  // PNG bits per channel: 8 or 16
    unsigned seed = 7u;
    std::string out = "out/test_pattern.png";
};

static void printUsage()
{
    std::printf(
        "render_art - GPU generative art stills -> PNG\n"
        "usage: render_art [options]\n"
        "  --width  N   image width in pixels   (default 2560)\n"
        "  --height N   image height in pixels  (default 1088)\n"
        "  --depth  D   PNG bits per channel: 8 or 16 (default 16)\n"
        "  --seed   S   deterministic seed      (default 7)\n"
        "  --out    P   output PNG path         (default out/test_pattern.png)\n");
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
        } else if (key == "--seed") {
            const char* v = next("--seed");
            if (!v) return false;
            a.seed = unsigned(std::strtoul(v, nullptr, 10));
        } else if (key == "--out") {
            const char* v = next("--out");
            if (!v) return false;
            a.out = v;
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

// Toolchain test pattern, linear RGB in [0, 1].
//   top band    : horizontal ramp  (banding / bit-depth check)
//   bottom band : 8 RGB chips      (channel order check)
//   4 px frame  : white border     (bounds check)
__global__ void testPatternKernel(float* rgb, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float* p = rgb + (size_t(y) * w + x) * 3;
    const float fx = float(x) / float(w - 1);
    const float fy = float(y) / float(h - 1);

    if (x < 4 || x >= w - 4 || y < 4 || y >= h - 4) {
        p[0] = p[1] = p[2] = 0.95f;
    } else if (y < h * 3 / 4) {
        p[0] = fx;
        p[1] = 0.25f + 0.5f * fy;
        p[2] = 1.0f - fx;
    } else {
        const int chip = min(int(fx * 8.0f), 7);
        p[0] = (chip & 1) ? 1.0f : 0.0f;
        p[1] = (chip & 2) ? 1.0f : 0.0f;
        p[2] = (chip & 4) ? 1.0f : 0.0f;
    }
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
    std::printf("device : %s\n", prop.name);
    std::printf("render : %dx%d  seed %u\n", args.width, args.height, args.seed);

    float* dRgb = nullptr;
    const size_t pixels = size_t(args.width) * size_t(args.height);
    CUDA_CHECK(cudaMalloc(&dRgb, pixels * 3 * sizeof(float)));

    const dim3 block(16, 16);
    const dim3 grid((args.width + block.x - 1) / block.x, (args.height + block.y - 1) / block.y);

    GpuTimer timer;
    timer.start();
    testPatternKernel<<<grid, block>>>(dRgb, args.width, args.height);
    CUDA_CHECK(cudaGetLastError());  // a failed launch must not look like a fast render
    const float ms = timer.stop();
    std::printf("kernel : %.3f ms  (%.1f Mpixel/s)\n", ms,
                pixels / (double(ms) * 1e3));

    std::vector<float> hRgb(pixels * 3);
    CUDA_CHECK(cudaMemcpy(hRgb.data(), dRgb, pixels * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::error_code ec;
    std::filesystem::path outPath(args.out);
    if (outPath.has_parent_path())
        std::filesystem::create_directories(outPath.parent_path(), ec);

    bool ok = false;
    if (args.depth == 16) {
        std::vector<uint16_t> img(pixels * 3);
        for (size_t i = 0; i < pixels * 3; ++i) {
            float v = hRgb[i];
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            img[i] = uint16_t(v * 65535.0f + 0.5f);
        }
        ok = art::writePng(args.out, args.width, args.height, 16, img.data());
    } else {
        std::vector<uint8_t> img(pixels * 3);
        for (size_t i = 0; i < pixels * 3; ++i) {
            float v = hRgb[i];
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

    CUDA_CHECK(cudaFree(dRgb));
    return 0;
}