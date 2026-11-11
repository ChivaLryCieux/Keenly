// Art pipeline driver: renders generative art backgrounds on the GPU and
// writes PNG stills.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// Every render runs the verify gate first (unless --no-check): the same
// pixel shader executes on CPU and GPU and the two buffers must be
// BIT-IDENTICAL (maxAbs == 0). The art target compiles strict IEEE math
// (--fmad=false, no fast-math) precisely so this holds: measured relL2 = 0
// for every field mode. If the gate ever fires, a non-IEEE operation
// (transcendental, fast-math division, FMA contraction) has slipped into
// art_noise/art_pixel and reproducibility is gone.

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
    std::string grid;      // non-empty: render the 3x3 showcase to this path
    int tile = 800;        // showcase cell size
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
    kSetSplitTone = 1u << 12, kSetBands = 1u << 13
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
    } else if (name == "ember") {
        // black-red -> orange -> gold (fire / molten metal)
        f.pal.count = 5;
        f.pal.pos[0] = 0.00f; f.pal.pos[1] = 0.30f; f.pal.pos[2] = 0.58f;
        f.pal.pos[3] = 0.80f; f.pal.pos[4] = 1.00f;
        f.pal.rgb[0][0] = 0.03f; f.pal.rgb[0][1] = 0.01f; f.pal.rgb[0][2] = 0.01f;
        f.pal.rgb[1][0] = 0.35f; f.pal.rgb[1][1] = 0.06f; f.pal.rgb[1][2] = 0.02f;
        f.pal.rgb[2][0] = 0.85f; f.pal.rgb[2][1] = 0.30f; f.pal.rgb[2][2] = 0.06f;
        f.pal.rgb[3][0] = 1.00f; f.pal.rgb[3][1] = 0.65f; f.pal.rgb[3][2] = 0.15f;
        f.pal.rgb[4][0] = 1.00f; f.pal.rgb[4][1] = 0.95f; f.pal.rgb[4][2] = 0.75f;
        if (!(set & kSetContrast))   f.contrast = 1.5f;
        if (!(set & kSetGrain))      p.grain = 0.05f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.6f;
        if (!(set & kSetHalation))   p.halation = 0.50f;
        if (!(set & kSetBloom))      p.bloom = 0.18f;
        if (!(set & kSetVignette))   p.vignette = 0.40f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.15f;
    } else if (name == "ice") {
        // deep blue-black -> steel -> pale cyan -> white (glacier / glass)
        f.pal.count = 5;
        f.pal.pos[0] = 0.00f; f.pal.pos[1] = 0.32f; f.pal.pos[2] = 0.62f;
        f.pal.pos[3] = 0.85f; f.pal.pos[4] = 1.00f;
        f.pal.rgb[0][0] = 0.02f; f.pal.rgb[0][1] = 0.05f; f.pal.rgb[0][2] = 0.10f;
        f.pal.rgb[1][0] = 0.10f; f.pal.rgb[1][1] = 0.28f; f.pal.rgb[1][2] = 0.45f;
        f.pal.rgb[2][0] = 0.45f; f.pal.rgb[2][1] = 0.75f; f.pal.rgb[2][2] = 0.85f;
        f.pal.rgb[3][0] = 0.85f; f.pal.rgb[3][1] = 0.95f; f.pal.rgb[3][2] = 0.98f;
        f.pal.rgb[4][0] = 1.00f; f.pal.rgb[4][1] = 1.00f; f.pal.rgb[4][2] = 1.00f;
        if (!(set & kSetGrain))      p.grain = 0.04f;
        if (!(set & kSetGrainScale)) p.grainScale = 1.8f;
        if (!(set & kSetHalation))   p.halation = 0.15f;
        if (!(set & kSetBloom))      p.bloom = 0.15f;
        if (!(set & kSetVignette))   p.vignette = 0.30f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.20f;
    } else if (name == "moss") {
        // dark green-brown -> moss -> sage -> warm sand (forest floor)
        f.pal.count = 5;
        f.pal.pos[0] = 0.00f; f.pal.pos[1] = 0.30f; f.pal.pos[2] = 0.60f;
        f.pal.pos[3] = 0.82f; f.pal.pos[4] = 1.00f;
        f.pal.rgb[0][0] = 0.05f; f.pal.rgb[0][1] = 0.06f; f.pal.rgb[0][2] = 0.03f;
        f.pal.rgb[1][0] = 0.15f; f.pal.rgb[1][1] = 0.25f; f.pal.rgb[1][2] = 0.10f;
        f.pal.rgb[2][0] = 0.42f; f.pal.rgb[2][1] = 0.55f; f.pal.rgb[2][2] = 0.25f;
        f.pal.rgb[3][0] = 0.75f; f.pal.rgb[3][1] = 0.72f; f.pal.rgb[3][2] = 0.45f;
        f.pal.rgb[4][0] = 0.95f; f.pal.rgb[4][1] = 0.92f; f.pal.rgb[4][2] = 0.80f;
        if (!(set & kSetGrain))      p.grain = 0.045f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.4f;
        if (!(set & kSetHalation))   p.halation = 0.25f;
        if (!(set & kSetVignette))   p.vignette = 0.33f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.30f;
    } else {
        std::fprintf(stderr, "unknown preset: %s (use mono|cinema|dusk|ember|ice|moss)\n",
                     name.c_str());
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

// Linear RGB buffer -> PNG (sRGB encode + TPDF dither + quantize).
static bool writeImage(const std::string& path, int w, int h, int depth,
                       const std::vector<float>& linear)
{
    std::error_code ec;
    std::filesystem::path outPath(path);
    if (outPath.has_parent_path())
        std::filesystem::create_directories(outPath.parent_path(), ec);

    const size_t n = size_t(w) * size_t(h) * 3;
    if (depth == 16) {
        std::vector<uint16_t> img(n);
        for (size_t i = 0; i < n; ++i) {
            float v = srgbEncode(linear[i]) + tpdfDither(uint32_t(i), 1.0f / 65535.0f);
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            img[i] = uint16_t(v * 65535.0f + 0.5f);
        }
        return art::writePng(path, w, h, 16, img.data());
    }
    std::vector<uint8_t> img(n);
    for (size_t i = 0; i < n; ++i) {
        float v = srgbEncode(linear[i]) + tpdfDither(uint32_t(i), 1.0f / 255.0f);
        v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        img[i] = uint8_t(v * 255.0f + 0.5f);
    }
    return art::writePng(path, w, h, 8, img.data());
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
        "  --preset P     mono|cinema|dusk|ember|ice|moss (default cinema)\n"
        "  --mode   M     warp|ridge|bands|frost|facet|prism|liquid (default warp)\n"
        "  --bands  N     contour frequency for bands mode (default 12)\n"
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
        "  --no-check     skip the CPU reference gate\n"
        "  --grid   P     render a 3x3 style showcase grid to PNG path P\n"
        "  --tile   N     showcase cell size in pixels (default 800)\n");
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
        } else if (key == "--mode") {
            const char* v = next("--mode");
            if (!v) return false;
            if (std::strcmp(v, "warp") == 0)        a.field.mode = 0;
            else if (std::strcmp(v, "ridge") == 0)  a.field.mode = 1;
            else if (std::strcmp(v, "bands") == 0)  a.field.mode = 2;
            else if (std::strcmp(v, "frost") == 0)  a.field.mode = 3;
            else if (std::strcmp(v, "facet") == 0)  a.field.mode = 4;
            else if (std::strcmp(v, "prism") == 0)  a.field.mode = 5;
            else if (std::strcmp(v, "liquid") == 0) a.field.mode = 6;
            else {
                std::fprintf(stderr, "unknown mode: %s "
                             "(use warp|ridge|bands|frost|facet|prism|liquid)\n", v);
                return false;
            }
        } else if (key == "--bands") {
            const char* v = next("--bands");
            if (!v) return false;
            a.field.bands = float(std::atof(v));
            a.postSet |= kSetBands;
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
        } else if (key == "--grid") {
            const char* v = next("--grid");
            if (!v) return false;
            a.grid = v;
        } else if (key == "--tile") {
            const char* v = next("--tile");
            if (!v) return false;
            a.tile = std::atoi(v);
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
    if (a.tile < 64) {
        std::fprintf(stderr, "bad tile size: %d\n", a.tile);
        return false;
    }
    return true;
}

// CPU reference gate: same pixel shader, serial vs parallel execution.
// Bit-exact pass condition (maxAbs == 0), see file header for why this is
// achievable and what breaking it means.
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
    std::printf("check  : relL2 %.3e  maxAbs %.3e  (GPU vs CPU, %dx%d, expect 0)\n",
                d.relL2, d.maxAbs, w, h);
    return d.maxAbs == 0.0;
}

// 3x3 showcase grid: nine style cells (generator x palette x seed) rendered
// tile by tile and pasted into one image with a dark gap grid. Each cell is
// a full pipeline render (field + post) and runs the bit-exact verify gate.
static int runGrid(const Args& args)
{
    struct Cell
    {
        const char* preset;
        int mode;
        unsigned seed;
    };
    static const Cell cells[9] = {
        {"ice",    art::kFieldFacet,  5u},   // austere crystal facets
        {"mono",   art::kFieldPrism,  6u},   // black-white chevron grid
        {"ember",  art::kFieldPrism,  21u},  // molten diamond lattice
        {"ice",    art::kFieldFrost,  8u},   // frosted glass
        {"mono",   art::kFieldFrost,  2u},   // matte monochrome frost
        {"dusk",   art::kFieldLiquid, 3u},   // oil-slick metal
        {"cinema", art::kFieldLiquid, 7u},   // teal-orange liquid metal
        {"ember",  art::kFieldFacet,  11u},  // cracked lava
        {"cinema", art::kFieldWarp,   7u}    // anchor: the loved organic look
    };

    const int tile = args.tile;
    const int gap = 16;
    const int gw = 3 * tile + 4 * gap;
    std::vector<float> img(size_t(gw) * size_t(gw) * 3, 0.018f);  // gap background
    std::vector<float> cellImg(size_t(tile) * size_t(tile) * 3);

    for (int i = 0; i < 9; ++i) {
        art::FieldParams f;
        art::PostParams p;
        applyPreset(f, p, cells[i].preset, 0u);
        f.mode = cells[i].mode;
        f.seed = cells[i].seed;
        p.seed = cells[i].seed;

        std::printf("cell %d : %-6s mode %d seed %u\n", i + 1, cells[i].preset,
                    f.mode, f.seed);
        if (args.check && !runCheck(f)) {
            std::fprintf(stderr, "CHECK FAILED on cell %d\n", i + 1);
            return 1;
        }

        const size_t n = size_t(tile) * size_t(tile) * 3;
        float *dField = nullptr, *dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dField, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
        art::renderFieldGpu(dField, tile, tile, f);
        art::renderPostGpu(dOut, dField, tile, tile, p);
        CUDA_CHECK(cudaMemcpy(cellImg.data(), dOut, n * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dField));
        CUDA_CHECK(cudaFree(dOut));

        const int cx = gap + (i % 3) * (tile + gap);
        const int cy = gap + (i / 3) * (tile + gap);
        for (int y = 0; y < tile; ++y)
            std::memcpy(&img[(size_t(cy + y) * gw + cx) * 3],
                        &cellImg[size_t(y) * tile * 3],
                        size_t(tile) * 3 * sizeof(float));
    }

    if (!writeImage(args.grid, gw, gw, args.depth, img)) {
        std::fprintf(stderr, "failed to write %s\n", args.grid.c_str());
        return 1;
    }
    std::printf("wrote  : %s  (%dx%d, %d-bit)\n", args.grid.c_str(), gw, gw, args.depth);
    return 0;
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

    if (!args.grid.empty())
        return runGrid(args);

    applyPreset(args.field, args.postp, args.preset, args.postSet);
    std::printf("render : %dx%d  preset %s  mode %d  seed %u  view %.2f  warp %.2f  octaves %d\n",
                args.width, args.height, args.preset.c_str(), args.field.mode,
                args.field.seed, args.field.view, args.field.warp, args.field.octaves);

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

    if (!writeImage(args.out, args.width, args.height, args.depth, hRgb)) {
        std::fprintf(stderr, "failed to write %s\n", args.out.c_str());
        return 1;
    }
    std::printf("wrote  : %s  (%d-bit)\n", args.out.c_str(), args.depth);
    return 0;
}