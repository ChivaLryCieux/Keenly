// Art pipeline driver: renders generative art backgrounds on the GPU and
// writes PNG stills.
//
// ASCII-ONLY FILE ON PURPOSE (see src/cuda_utils.h for the reason).
//
// Every render runs the verify gate first (unless --no-check): the same
// pixel shader executes on CPU and GPU and the two buffers must be
// BIT-IDENTICAL (maxAbs == 0). The art target compiles strict IEEE math
// (--fmad=false, no fast-math) precisely so this holds: measured relL2 = 0
// for every field mode. If the gate ever fires, a non-IEEE operation
// (transcendental, fast-math division, FMA contraction) has slipped into
// art math (noise.cuh / pixel.cuh) and reproducibility is gone.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "render.h"
#include "cuda_utils.h"
#include "png_writer.h"
#include "compare.h"

struct Args
{
    int width = 2560;
    int height = 1072;  // 2.39:1 cinema frame
    int depth = 16;     // PNG bits per channel: 8 or 16
    bool check = true;
    bool post = true;
    bool checkAll = false;  // run the gate on every generator and exit
    bool listModes = false; // print the generator list and exit
    bool cpu = false;       // render on the CPU instead of the GPU (--cpu)
    int  cpuThreads = 0;    // 0 = hardware_concurrency
    unsigned postSet = 0;  // which params were explicitly set on the CLI
    std::string preset = "cinema";
    std::string grid;      // non-empty: render the 3x3 showcase to this path
    int tile = 800;        // showcase cell size
    keenly::FieldParams field;
    keenly::PostParams postp;
    float film = -1.0f;      // >= 0: force the film-filter grain spec on any preset
    std::string out = "out/field.png";

    // ---- icon mode ---------------------------------------------------------
    bool iconMode = false;     // render one RGBA skill icon instead of a still
    std::string family = "blue";  // blue|red|gold|purple
    std::string glyph = "sword";  // glyph name, see sdf.cuh glyphName()
    int tier = 2;              // frame tier 1..4 (0 = none, generic only)
    bool tierSet = false;      // --tier was given on the CLI
    int maskRound = 1;         // 1 = rounded rect, 0 = sharp rect
    float radius = 0.16f;      // corner radius, fraction of the min side
    std::string iconsDemo;     // non-empty: render the icon demo batch to DIR

    // ---- generic icon pipeline (rainbow bg + flat fg silhouette) -----------
    bool giconMode = false;    // generic icon path (overrides family)
    std::string bg = "fluid";  // style: fluid|swirl|bands|rings|mosaic|glow
    std::string scheme = "aurora";   // glow scheme: aurora|sepia|flare|tide
    std::string glowDemo;      // non-empty: render the glow-icon demo batch
    std::string fg = "white";  // flat glyph color: white|black
    std::string rainbowDemo;   // non-empty: render the generic demo batch
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
    kSetSplitTone = 1u << 12, kSetBands = 1u << 13, kSetSaturation = 1u << 14,
    kSetMode = 1u << 15
};

// Glow-scheme color sets shared by the poster presets (--preset) and the
// generic icon pipeline (--bg glow --scheme). pal.rgb[i] is a blob color;
// count = number of blobs. Perceptual values - glowField squares to linear.
static bool glowScheme(keenly::Palette& pal, const std::string& name)
{
    pal.count = 4;
    if (name == "aurora") {
        pal.rgb[0][0] = 0.30f; pal.rgb[0][1] = 0.08f; pal.rgb[0][2] = 0.62f;
        pal.rgb[1][0] = 0.04f; pal.rgb[1][1] = 0.55f; pal.rgb[1][2] = 0.78f;
        pal.rgb[2][0] = 0.30f; pal.rgb[2][1] = 0.80f; pal.rgb[2][2] = 0.38f;
        pal.rgb[3][0] = 0.08f; pal.rgb[3][1] = 0.28f; pal.rgb[3][2] = 0.82f;
    } else if (name == "sepia") {
        pal.rgb[0][0] = 0.42f; pal.rgb[0][1] = 0.24f; pal.rgb[0][2] = 0.10f;
        pal.rgb[1][0] = 0.78f; pal.rgb[1][1] = 0.55f; pal.rgb[1][2] = 0.32f;
        pal.rgb[2][0] = 0.98f; pal.rgb[2][1] = 0.88f; pal.rgb[2][2] = 0.70f;
        pal.rgb[3][0] = 0.92f; pal.rgb[3][1] = 0.60f; pal.rgb[3][2] = 0.18f;
    } else if (name == "flare") {
        pal.rgb[0][0] = 0.42f; pal.rgb[0][1] = 0.10f; pal.rgb[0][2] = 0.62f;
        pal.rgb[1][0] = 0.85f; pal.rgb[1][1] = 0.25f; pal.rgb[1][2] = 0.55f;
        pal.rgb[2][0] = 0.98f; pal.rgb[2][1] = 0.45f; pal.rgb[2][2] = 0.12f;
        pal.rgb[3][0] = 0.95f; pal.rgb[3][1] = 0.68f; pal.rgb[3][2] = 0.12f;
    } else if (name == "tide") {
        pal.rgb[0][0] = 0.55f; pal.rgb[0][1] = 0.07f; pal.rgb[0][2] = 0.09f;
        pal.rgb[1][0] = 0.04f; pal.rgb[1][1] = 0.48f; pal.rgb[1][2] = 0.52f;
        pal.rgb[2][0] = 0.80f; pal.rgb[2][1] = 0.95f; pal.rgb[2][2] = 0.85f;
        pal.rgb[3][0] = 0.35f; pal.rgb[3][1] = 0.12f; pal.rgb[3][2] = 0.55f;
    } else {
        return false;
    }
    return true;
}

static void applyPreset(keenly::FieldParams& f, keenly::PostParams& p, const std::string& name,
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
        if (!(set & kSetSaturation)) p.saturation = 1.0f;
    } else if (name == "cinema") {
        // teal shadows -> amber highlights (the classic teal-orange grade),
        // 6 stops with a desaturated bridge so the teal-to-amber crossing
        // never turns muddy brown.
        f.pal.count = 6;
        f.pal.pos[0] = 0.00f; f.pal.pos[1] = 0.20f; f.pal.pos[2] = 0.42f;
        f.pal.pos[3] = 0.60f; f.pal.pos[4] = 0.80f; f.pal.pos[5] = 1.00f;
        f.pal.rgb[0][0] = 0.015f; f.pal.rgb[0][1] = 0.030f; f.pal.rgb[0][2] = 0.050f;
        f.pal.rgb[1][0] = 0.040f; f.pal.rgb[1][1] = 0.135f; f.pal.rgb[1][2] = 0.185f;
        f.pal.rgb[2][0] = 0.115f; f.pal.rgb[2][1] = 0.315f; f.pal.rgb[2][2] = 0.375f;
        f.pal.rgb[3][0] = 0.360f; f.pal.rgb[3][1] = 0.400f; f.pal.rgb[3][2] = 0.385f;
        f.pal.rgb[4][0] = 0.735f; f.pal.rgb[4][1] = 0.490f; f.pal.rgb[4][2] = 0.300f;
        f.pal.rgb[5][0] = 0.980f; f.pal.rgb[5][1] = 0.905f; f.pal.rgb[5][2] = 0.780f;
        if (!(set & kSetGrain))      p.grain = 0.05f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.8f;
        if (!(set & kSetHalation))   p.halation = 0.40f;
        if (!(set & kSetVignette))   p.vignette = 0.38f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.45f;
        if (!(set & kSetSaturation)) p.saturation = 1.22f;
    } else if (name == "dusk") {
        // indigo -> magenta -> coral (poster palette), 6 stops
        f.pal.count = 6;
        f.pal.pos[0] = 0.00f; f.pal.pos[1] = 0.22f; f.pal.pos[2] = 0.46f;
        f.pal.pos[3] = 0.66f; f.pal.pos[4] = 0.84f; f.pal.pos[5] = 1.00f;
        f.pal.rgb[0][0] = 0.040f; f.pal.rgb[0][1] = 0.035f; f.pal.rgb[0][2] = 0.110f;
        f.pal.rgb[1][0] = 0.140f; f.pal.rgb[1][1] = 0.110f; f.pal.rgb[1][2] = 0.330f;
        f.pal.rgb[2][0] = 0.420f; f.pal.rgb[2][1] = 0.185f; f.pal.rgb[2][2] = 0.460f;
        f.pal.rgb[3][0] = 0.720f; f.pal.rgb[3][1] = 0.290f; f.pal.rgb[3][2] = 0.470f;
        f.pal.rgb[4][0] = 0.950f; f.pal.rgb[4][1] = 0.500f; f.pal.rgb[4][2] = 0.360f;
        f.pal.rgb[5][0] = 1.000f; f.pal.rgb[5][1] = 0.880f; f.pal.rgb[5][2] = 0.730f;
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
    } else if (name == "gold") {
        // dark bronze -> ochre -> bright gold -> ivory (holy / light / royal)
        f.pal.count = 5;
        f.pal.pos[0] = 0.00f; f.pal.pos[1] = 0.28f; f.pal.pos[2] = 0.55f;
        f.pal.pos[3] = 0.78f; f.pal.pos[4] = 1.00f;
        f.pal.rgb[0][0] = 0.060f; f.pal.rgb[0][1] = 0.040f; f.pal.rgb[0][2] = 0.010f;
        f.pal.rgb[1][0] = 0.300f; f.pal.rgb[1][1] = 0.180f; f.pal.rgb[1][2] = 0.050f;
        f.pal.rgb[2][0] = 0.680f; f.pal.rgb[2][1] = 0.460f; f.pal.rgb[2][2] = 0.120f;
        f.pal.rgb[3][0] = 0.950f; f.pal.rgb[3][1] = 0.720f; f.pal.rgb[3][2] = 0.280f;
        f.pal.rgb[4][0] = 1.000f; f.pal.rgb[4][1] = 0.950f; f.pal.rgb[4][2] = 0.800f;
        if (!(set & kSetContrast))   f.contrast = 1.45f;
        if (!(set & kSetGrain))      p.grain = 0.04f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.2f;
        if (!(set & kSetHalation))   p.halation = 0.42f;
        if (!(set & kSetBloom))      p.bloom = 0.15f;
        if (!(set & kSetVignette))   p.vignette = 0.34f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.18f;
    } else if (name == "violet") {
        // dark violet-black -> violet -> magenta -> pale lilac (shadow / void)
        f.pal.count = 5;
        f.pal.pos[0] = 0.00f; f.pal.pos[1] = 0.25f; f.pal.pos[2] = 0.52f;
        f.pal.pos[3] = 0.76f; f.pal.pos[4] = 1.00f;
        f.pal.rgb[0][0] = 0.050f; f.pal.rgb[0][1] = 0.020f; f.pal.rgb[0][2] = 0.100f;
        f.pal.rgb[1][0] = 0.220f; f.pal.rgb[1][1] = 0.080f; f.pal.rgb[1][2] = 0.380f;
        f.pal.rgb[2][0] = 0.500f; f.pal.rgb[2][1] = 0.180f; f.pal.rgb[2][2] = 0.620f;
        f.pal.rgb[3][0] = 0.820f; f.pal.rgb[3][1] = 0.380f; f.pal.rgb[3][2] = 0.880f;
        f.pal.rgb[4][0] = 0.970f; f.pal.rgb[4][1] = 0.880f; f.pal.rgb[4][2] = 1.000f;
        if (!(set & kSetGrain))      p.grain = 0.038f;
        if (!(set & kSetGrainScale)) p.grainScale = 1.6f;
        if (!(set & kSetHalation))   p.halation = 0.26f;
        if (!(set & kSetBloom))      p.bloom = 0.12f;
        if (!(set & kSetVignette))   p.vignette = 0.30f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.16f;
    } else if (name == "aurora") {
        // glow mode: violet / cyan / spring green bleeding in over black
        // (the "daily color scheme" mesh-gradient poster look)
        glowScheme(f.pal, "aurora");
        if (!(set & kSetMode))       f.mode = keenly::kFieldGlow;
        if (!(set & kSetGrain))      p.grain = 0.10f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.2f;
        if (!(set & kSetHalation))   p.halation = 0.22f;
        if (!(set & kSetBloom))      p.bloom = 0.10f;
        if (!(set & kSetVignette))   p.vignette = 0.30f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.0f;
        if (!(set & kSetSaturation)) p.saturation = 1.05f;
    } else if (name == "sepia") {
        // glow mode: umber / tan / cream - warm vintage lantern light
        glowScheme(f.pal, "sepia");
        if (!(set & kSetMode))       f.mode = keenly::kFieldGlow;
        if (!(set & kSetGrain))      p.grain = 0.11f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.0f;
        if (!(set & kSetHalation))   p.halation = 0.30f;
        if (!(set & kSetBloom))      p.bloom = 0.12f;
        if (!(set & kSetVignette))   p.vignette = 0.32f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.0f;
        if (!(set & kSetSaturation)) p.saturation = 1.0f;
    } else if (name == "flare") {
        // glow mode: royal purple / magenta pink / ember orange
        glowScheme(f.pal, "flare");
        if (!(set & kSetMode))       f.mode = keenly::kFieldGlow;
        if (!(set & kSetGrain))      p.grain = 0.095f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.3f;
        if (!(set & kSetHalation))   p.halation = 0.28f;
        if (!(set & kSetBloom))      p.bloom = 0.11f;
        if (!(set & kSetVignette))   p.vignette = 0.30f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.0f;
        if (!(set & kSetSaturation)) p.saturation = 1.08f;
    } else if (name == "tide") {
        // glow mode: deep red / teal / pale mint - cold tide under embers
        glowScheme(f.pal, "tide");
        if (!(set & kSetMode))       f.mode = keenly::kFieldGlow;
        if (!(set & kSetGrain))      p.grain = 0.10f;
        if (!(set & kSetGrainScale)) p.grainScale = 2.2f;
        if (!(set & kSetHalation))   p.halation = 0.25f;
        if (!(set & kSetBloom))      p.bloom = 0.10f;
        if (!(set & kSetVignette))   p.vignette = 0.32f;
        if (!(set & kSetSplitTone))  p.splitTone = 0.0f;
        if (!(set & kSetSaturation)) p.saturation = 1.04f;
    } else {
        std::fprintf(stderr, "unknown preset: %s (use mono|cinema|dusk|ember|ice|moss|gold|violet|aurora|sepia|flare|tide)\n",
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

// ---- icon families ---------------------------------------------------------
// Four game-skill color families. Each family = background field preset +
// linear accent colors for the glyph/frame (see icon.cuh for how they are
// used). Accents are LINEAR radiance values: ACES compresses highlights, so
// stroke / glow intentionally exceed 1.0 for the "lit" game-icon look.

static void applyIconFamily(keenly::IconParams& ip, const std::string& name)
{
    using namespace keenly;
    PostParams post;   // preset fills this first; icon overrides land below

    if (name == "blue") {
        applyPreset(ip.field, post, "ice", 0u);
        // Fluid backgrounds only (the organic warp field): the Worley
        // frost/facet fields read as ugly Voronoi at icon size.
        ip.field.mode = kFieldWarp;
        ip.field.view = 2.8f;
        ip.field.warp = 1.1f;
        ip.col.fill[0] = 0.10f;  ip.col.fill[1] = 0.30f;  ip.col.fill[2] = 0.75f;
        ip.col.hi[0] = 0.75f;    ip.col.hi[1] = 0.92f;    ip.col.hi[2] = 1.05f;
        ip.col.stroke[0] = 1.35f; ip.col.stroke[1] = 1.50f; ip.col.stroke[2] = 1.62f;
        ip.col.glow[0] = 0.25f;  ip.col.glow[1] = 0.65f;  ip.col.glow[2] = 1.15f;
        ip.col.shadow[0] = 0.02f; ip.col.shadow[1] = 0.07f; ip.col.shadow[2] = 0.24f;
    } else if (name == "red") {
        applyPreset(ip.field, post, "ember", 0u);
        ip.field.mode = kFieldLiquid;
        ip.field.view = 2.6f;
        ip.field.warp = 1.25f;
        ip.col.fill[0] = 0.88f;  ip.col.fill[1] = 0.14f;  ip.col.fill[2] = 0.07f;
        ip.col.hi[0] = 1.15f;    ip.col.hi[1] = 0.55f;    ip.col.hi[2] = 0.32f;
        ip.col.stroke[0] = 1.62f; ip.col.stroke[1] = 1.34f; ip.col.stroke[2] = 1.02f;
        ip.col.glow[0] = 1.15f;  ip.col.glow[1] = 0.30f;  ip.col.glow[2] = 0.08f;
        ip.col.shadow[0] = 0.24f; ip.col.shadow[1] = 0.02f; ip.col.shadow[2] = 0.01f;
    } else if (name == "gold") {
        applyPreset(ip.field, post, "gold", 0u);
        // Fluid background (warp), not the Voronoi facet field.
        ip.field.mode = kFieldWarp;
        ip.field.view = 2.7f;
        ip.field.warp = 1.15f;
        ip.col.fill[0] = 0.92f;  ip.col.fill[1] = 0.62f;  ip.col.fill[2] = 0.12f;
        ip.col.hi[0] = 1.32f;    ip.col.hi[1] = 1.10f;    ip.col.hi[2] = 0.62f;
        ip.col.stroke[0] = 1.55f; ip.col.stroke[1] = 1.46f; ip.col.stroke[2] = 1.14f;
        ip.col.glow[0] = 1.20f;  ip.col.glow[1] = 0.85f;  ip.col.glow[2] = 0.24f;
        ip.col.shadow[0] = 0.26f; ip.col.shadow[1] = 0.13f; ip.col.shadow[2] = 0.02f;
    } else if (name == "purple") {
        applyPreset(ip.field, post, "violet", 0u);
        ip.field.mode = kFieldWarp;
        ip.field.view = 2.9f;
        ip.field.warp = 1.3f;
        ip.col.fill[0] = 0.52f;  ip.col.fill[1] = 0.16f;  ip.col.fill[2] = 0.82f;
        ip.col.hi[0] = 0.98f;    ip.col.hi[1] = 0.74f;    ip.col.hi[2] = 1.18f;
        ip.col.stroke[0] = 1.44f; ip.col.stroke[1] = 1.26f; ip.col.stroke[2] = 1.62f;
        ip.col.glow[0] = 0.85f;  ip.col.glow[1] = 0.28f;  ip.col.glow[2] = 1.18f;
        ip.col.shadow[0] = 0.10f; ip.col.shadow[1] = 0.02f; ip.col.shadow[2] = 0.24f;
    } else {
        std::fprintf(stderr, "unknown icon family: %s (use blue|red|gold|purple)\n",
                     name.c_str());
        std::exit(1);
    }
    // Icon post overrides the preset's values on purpose: fine grain and a
    // subtle vignette sized for small frames, and no halation - the SDF
    // halo already does the glow job. (These used to sit BEFORE the preset
    // call, which silently overwrote every one of them.)
    ip.post = post;
    ip.post.grain = 0.030f;
    ip.post.grainScale = 1.3f;
    ip.post.halation = 0.0f;
    ip.post.bloom = 0.0f;
    ip.post.vignette = 0.20f;
    ip.post.splitTone = 0.15f;
    ip.post.saturation = 1.12f;
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

// Linear RGB buffer -> PNG (sRGB encode + TPDF dither + quantize).
// Encodes in parallel across CPU cores; output bytes are identical to the
// old serial loop (per-sample pure math, deterministic LCG dither).
static bool writeImage(const std::string& path, int w, int h, int depth,
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
static bool writeImageRgba(const std::string& path, int w, int h,
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
// one array: a new generator cannot be added to one place and forgotten in
// another, and the docs can quote a count that is verifiable here.
static const char* const kFieldModeNames[] = {
    "warp", "ridge", "bands", "frost", "facet", "prism", "liquid", "turing",
    "agate", "waves", "truchet", "shards", "dots", "aura", "moire", "deco",
    "glow"
};
static const int kFieldModeCount =
    int(sizeof(kFieldModeNames) / sizeof(kFieldModeNames[0]));

// Comma/pipe joined mode list, for usage and error text.
static std::string fieldModeList()
{
    std::string s;
    for (int m = 0; m < kFieldModeCount; ++m) {
        if (m) s += '|';
        s += kFieldModeNames[m];
    }
    return s;
}

// Case-sensitive mode name -> enum value, -1 when unknown.
static int modeFromName(const char* name)
{
    for (int m = 0; m < kFieldModeCount; ++m)
        if (std::strcmp(name, kFieldModeNames[m]) == 0) return m;
    return -1;
}

static void printUsage()
{
    std::printf(
        "Keenly - CUDA generative art studio: procedural stills -> PNG\n"
        "usage: keenly [options]\n"
        "  --width  N     image width in pixels   (default 2560)\n"
        "  --height N     image height in pixels  (default 1072)\n"
        "  --depth  D     PNG bits per channel: 8 or 16 (default 16)\n"
        "  --out    P     output PNG path         (default out/field.png)\n"
        "  --preset P     mono|cinema|dusk|ember|ice|moss|gold|violet\n"
        "                 aurora|sepia|flare|tide       (default cinema)\n"
        "  --mode   M     one of the field generators below      (default warp)\n"
        "                 see `keenly --modes` for the live list\n"
        "  --film   F     film-filter grain spec, F = grain strength multiplier\n"
        "                 (overrides preset grain; works on any style)\n"
        "  --margin F     matte white poster frame, fraction of height\n"
        "  --bands  N     contour frequency for bands mode (default 12)\n"
        "  --rd-f   F     Gray-Scott feed rate   (turing mode, default 0.0545)\n"
        "  --rd-k   F     Gray-Scott kill rate   (turing mode, default 0.062)\n"
        "  --rd-res N     simulation grid size   (turing mode, default 256)\n"
        "  --rd-iters N   simulation steps       (turing mode, default 3000)\n"
        "  --seed   S     deterministic seed      (default 7)\n"
        "  --view   F     noise extent across height (default 3.2)\n"
        "  --octaves N    fBm octaves             (default 5)\n"
        "  --warp   F     domain warp strength    (default 1.2)\n"
        "  --contrast F   field shaping           (default 1.4)\n"
        "  --relief F     shading strength        (default 2.5)\n"
        "  --exposure F   linear exposure         (default 1.0)\n"
        "  --grain F      grain amplitude         (default 0.05)\n"
        "  --grain-scale F grain clump size, px   (default 2.5)\n"
        "  --halation F   red highlight glow      (default 0.35)\n"
        "  --bloom F      neutral glow            (default 0.12)\n"
        "  --vignette F   corner darkening        (default 0.35)\n"
        "  --split-tone F teal shadows / warm highs grade (default 0.35)\n"
        "  --saturation F chroma expansion around luminance (default 1.12)\n"
        "  --macro   F    layered composition strength, 0 = flat (default 1.0)\n"
        "  --no-post      skip grain/glow/vignette/tonemap (raw field)\n"
        "  --no-check     skip the CPU reference gate\n"
        "  --check-all    run the bit-exact gate on every generator, then exit\n"
        "  --modes        print the generator list (--mode values) and exit\n"
        "  --cpu          render on the CPU instead of the GPU (no CUDA device\n"
        "                 needed; bit-identical output, ~20x slower)\n"
        "  --cpu-threads N  worker threads for --cpu (default: all cores)\n"
        "  --grid   P     render a 3x3 style showcase grid to PNG path P\n"
        "  --tile   N     showcase cell size in pixels (default 800)\n"
        "icon mode (RGBA skill icons):\n"
        "  --icon         render one icon (family/glyph/tier/mask below)\n"
        "  --family F     blue|red|gold|purple        (default blue)\n"
        "  --glyph  G     sword|axe|bow|shield|flame|droplet|snow|bolt|sparkle\n"
        "                 moon|sun|eye|heart|skull|cross|rune|crown|potion\n"
        "  --tier   N     frame tier 1..4 (0 = none)   (default 2)\n"
        "  --mask   M     round|rect                   (default round)\n"
        "  --radius F     corner radius, min-side fraction (default 0.16)\n"
        "  --icons-demo P render the icon demo batch into directory P\n"
        "generic icons (rainbow/glow background + flat fg silhouette):\n"
        "  --gicon        render one generic icon (uses --bg --fg --glyph --tier)\n"
        "  --bg     B     fluid|swirl|bands|rings|mosaic|glow (default fluid)\n"
        "  --scheme S     aurora|sepia|flare|tide      (bg=glow, default aurora)\n"
        "  --fg     C     white|black                   (default white)\n"
        "  --rainbow-demo P render the generic icon batch into directory P\n"
        "  --glow-demo  P  render the glow icon batch (scheme x fg) into dir P\n");
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
            const int mode = modeFromName(v);
            if (mode < 0) {
                std::fprintf(stderr, "unknown mode: %s (use %s)\n", v,
                             fieldModeList().c_str());
                return false;
            }
            a.field.mode = mode;
            a.postSet |= kSetMode;
        } else if (key == "--bands") {
            const char* v = next("--bands");
            if (!v) return false;
            a.field.bands = float(std::atof(v));
            a.postSet |= kSetBands;
        } else if (key == "--rd-f") {
            const char* v = next("--rd-f");
            if (!v) return false;
            a.field.rdF = float(std::atof(v));
        } else if (key == "--rd-k") {
            const char* v = next("--rd-k");
            if (!v) return false;
            a.field.rdK = float(std::atof(v));
        } else if (key == "--rd-res") {
            const char* v = next("--rd-res");
            if (!v) return false;
            a.field.rdRes = std::atoi(v);
        } else if (key == "--rd-iters") {
            const char* v = next("--rd-iters");
            if (!v) return false;
            a.field.rdIters = std::atoi(v);
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
        } else if (key == "--saturation") {
            const char* v = next("--saturation");
            if (!v) return false;
            a.postp.saturation = float(std::atof(v));
            a.postSet |= kSetSaturation;
        } else if (key == "--film") {
            const char* v = next("--film");
            if (!v) return false;
            a.film = float(std::atof(v));
        } else if (key == "--margin") {
            const char* v = next("--margin");
            if (!v) return false;
            a.postp.margin = float(std::atof(v));
        } else if (key == "--macro") {
            const char* v = next("--macro");
            if (!v) return false;
            a.field.macroAmt = float(std::atof(v));
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
        } else if (key == "--check-all") {
            a.checkAll = true;
        } else if (key == "--modes") {
            a.listModes = true;
        } else if (key == "--cpu") {
            a.cpu = true;
        } else if (key == "--cpu-threads") {
            const char* v = next("--cpu-threads");
            if (!v) return false;
            a.cpuThreads = std::atoi(v);
        } else if (key == "--icon") {
            a.iconMode = true;
        } else if (key == "--family") {
            const char* v = next("--family");
            if (!v) return false;
            a.family = v;
            a.iconMode = true;
        } else if (key == "--glyph") {
            const char* v = next("--glyph");
            if (!v) return false;
            a.glyph = v;
            a.iconMode = true;
        } else if (key == "--tier") {
            const char* v = next("--tier");
            if (!v) return false;
            a.tier = std::atoi(v);
            a.tierSet = true;
            a.iconMode = true;
        } else if (key == "--mask") {
            const char* v = next("--mask");
            if (!v) return false;
            if (std::strcmp(v, "round") == 0)      a.maskRound = 1;
            else if (std::strcmp(v, "rect") == 0)  a.maskRound = 0;
            else {
                std::fprintf(stderr, "unknown mask: %s (use round|rect)\n", v);
                return false;
            }
            a.iconMode = true;
        } else if (key == "--radius") {
            const char* v = next("--radius");
            if (!v) return false;
            a.radius = float(std::atof(v));
            a.iconMode = true;
        } else if (key == "--icons-demo") {
            const char* v = next("--icons-demo");
            if (!v) return false;
            a.iconsDemo = v;
        } else if (key == "--gicon") {
            a.giconMode = true;
            a.iconMode = true;
        } else if (key == "--bg") {
            const char* v = next("--bg");
            if (!v) return false;
            if (std::strcmp(v, "fluid") == 0 || std::strcmp(v, "swirl") == 0 ||
                std::strcmp(v, "bands") == 0 || std::strcmp(v, "rings") == 0 ||
                std::strcmp(v, "mosaic") == 0 || std::strcmp(v, "glow") == 0) {
                a.bg = v;
            } else {
                std::fprintf(stderr, "unknown bg: %s (use fluid|swirl|bands|rings|mosaic|glow)\n", v);
                return false;
            }
            a.giconMode = true;
            a.iconMode = true;
        } else if (key == "--scheme") {
            const char* v = next("--scheme");
            if (!v) return false;
            if (std::strcmp(v, "aurora") == 0 || std::strcmp(v, "sepia") == 0 ||
                std::strcmp(v, "flare") == 0 || std::strcmp(v, "tide") == 0) {
                a.scheme = v;
            } else {
                std::fprintf(stderr, "unknown scheme: %s (use aurora|sepia|flare|tide)\n", v);
                return false;
            }
            a.giconMode = true;
            a.iconMode = true;
        } else if (key == "--glow-demo") {
            const char* v = next("--glow-demo");
            if (!v) return false;
            a.glowDemo = v;
        } else if (key == "--fg") {
            const char* v = next("--fg");
            if (!v) return false;
            if (std::strcmp(v, "white") == 0 || std::strcmp(v, "black") == 0) {
                a.fg = v;
            } else {
                std::fprintf(stderr, "unknown fg: %s (use white|black)\n", v);
                return false;
            }
            a.giconMode = true;
            a.iconMode = true;
        } else if (key == "--rainbow-demo") {
            const char* v = next("--rainbow-demo");
            if (!v) return false;
            a.rainbowDemo = v;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", key.c_str());
            return false;
        }
    }
    if (a.width <= 0 || a.height <= 0 || (a.depth != 8 && a.depth != 16)) {
        std::fprintf(stderr, "bad size or depth: %dx%d depth %d\n", a.width, a.height, a.depth);
        return false;
    }
    if (a.field.rdRes < 16 || a.field.rdRes > 4096 || a.field.rdIters < 0) {
        std::fprintf(stderr, "bad Gray-Scott params: res %d iters %d\n",
                     a.field.rdRes, a.field.rdIters);
        return false;
    }
    if (a.tile < 64) {
        std::fprintf(stderr, "bad tile size: %d\n", a.tile);
        return false;
    }
    return true;
}

// Synthetic post-chain check.
//
// Why this exists: comparing post on a real field render is necessary but NOT
// sufficient. If a generator happens to produce no pixels above the bright
// pass knee (warp does exactly that at gate size), the halation/bloom terms
// are all multiplied by zero and the post stage passes vacuously - a broken
// blur or a wrong half-res offset would go unnoticed. (Measured: injecting a
// 1% halation error was caught on ridge but reported 0 on warp.)
//
// So this check feeds a deliberately HDR input: a linear ramp plus bright
// blobs, which guarantees the bright pass has energy, the blur has spread,
// the vignette and grain vary, and the ACES shoulder is exercised. It also
// asserts the coverage explicitly, so the check cannot silently go vacuous
// again if the post defaults change.
static bool runPostCheck()
{
    const int w = 320, h = 180;
    const size_t n = size_t(w) * size_t(h) * 3;
    std::vector<float> src(n);

    // Ramp (0 .. 1.6 in luminance) + two bright blobs above the 1.1 knee.
    // Values > 1.0 are legal here: the field buffer is linear light and the
    // specular path really does emit > 1 in highlights.
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float u = float(x) / float(w - 1);
            const float v = float(y) / float(h - 1);
            float c[3] = {1.60f * u, 1.20f * v, 0.90f * (1.0f - 0.5f * (u + v))};
            const float dx1 = u - 0.28f, dy1 = v - 0.35f;
            const float dx2 = u - 0.72f, dy2 = v - 0.68f;
            const float g1 = 2.60f * (1.0f - 60.0f * (dx1 * dx1 + dy1 * dy1));
            const float g2 = 2.20f * (1.0f - 45.0f * (dx2 * dx2 + dy2 * dy2));
            const float g = (g1 > 0.0f ? g1 : 0.0f) + (g2 > 0.0f ? g2 : 0.0f);
            float* p = src.data() + (size_t(y) * w + x) * 3;
            p[0] = c[0] + g;
            p[1] = c[1] + 0.85f * g;
            p[2] = c[2] + 0.70f * g;
        }
    }

    keenly::PostParams post;
    post.seed = 4242u;
    post.exposure = 1.0f;
    post.grain = 0.05f;
    post.grainScale = 2.5f;
    post.halation = 0.45f;
    post.bloom = 0.20f;
    post.vignette = 0.40f;
    post.splitTone = 0.45f;
    post.saturation = 1.15f;

    std::vector<float> ref(n), gpu(n);
    keenly::renderPostCpu(ref.data(), src.data(), w, h, post);

    float* dSrc = nullptr;
    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dSrc, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dSrc, src.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    keenly::renderPostGpu(dOut, dSrc, w, h, post);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dOut, n * sizeof(float), cudaMemcpyDeviceToHost));

    const DiffStats d = compareArrays(gpu.data(), ref.data(), n);

    // Thread-count invariance for the post chain: the --cpu backend runs this
    // stage with many threads, so prove a row split reproduces the serial
    // result exactly. (Rows are pure functions over read-only inputs, so this
    // must hold; if it ever fails, someone introduced a cross-row dependency.)
    std::vector<float> threaded(n);
    keenly::renderPostCpu(threaded.data(), src.data(), w, h, post, 7);
    const DiffStats dt = compareArrays(threaded.data(), ref.data(), n);

    // Coverage assertion: the synthetic input must actually drive the chain.
    // If the output is a flat constant, the check proved nothing.
    float lo = 1e30f, hi = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        lo = std::min(lo, ref[i]);
        hi = std::max(hi, ref[i]);
    }
    const bool covered = (hi - lo) > 0.05f;

    std::printf("post   check  : relL2 %.3e  maxAbs %.3e  (synthetic HDR, expect 0)\n",
                d.relL2, d.maxAbs);
    std::printf("        split : relL2 %.3e  maxAbs %.3e  (7 threads vs serial)\n",
                dt.relL2, dt.maxAbs);
    std::printf("        coverage: output range [%.3f, %.3f] %s\n", lo, hi,
                covered ? "ok" : "VACUOUS - post stage saw a constant image");

    CUDA_CHECK(cudaFree(dSrc));
    CUDA_CHECK(cudaFree(dOut));
    return d.maxAbs == 0.0 && dt.maxAbs == 0.0 && covered;
}

// CPU reference gate: same pixel shader, serial vs parallel execution.
// Bit-exact pass condition (maxAbs == 0), see file header for why this is
// achievable and what breaking it means.
//
// Two stages are checked, because the delivered PNG contains both:
//   1. field  - the pixel shader output
//   2. post   - bright pass + gaussian + compose (halation/bloom/grain/
//               vignette/ACES). Until this stage had a reference, a blur
//               index bug or a half-res sampling offset would have shipped
//               silently: the gate only ever compared the field.
static bool runCheck(const keenly::FieldParams& p)
{
    const int w = 320, h = 180;
    const size_t n = size_t(w) * size_t(h) * 3;
    std::vector<float> refField(n), gpuField(n);

    keenly::renderFieldCpu(refField.data(), w, h, p);

    float* dBuf = nullptr;
    CUDA_CHECK(cudaMalloc(&dBuf, n * sizeof(float)));
    keenly::renderFieldGpu(dBuf, w, h, p);
    CUDA_CHECK(cudaMemcpy(gpuField.data(), dBuf, n * sizeof(float), cudaMemcpyDeviceToHost));

    const DiffStats dField = compareArrays(gpuField.data(), refField.data(), n);
    std::printf("check  : field relL2 %.3e  maxAbs %.3e", dField.relL2, dField.maxAbs);

    // ---- post chain on this generator's own field -------------------------
    // Catches field->post plumbing (exposure, seed, buffer handoff). Note
    // this can be near-vacuous on generators with no highlights; the
    // synthetic runPostCheck below is what guarantees real coverage.
    keenly::PostParams post;
    post.seed = p.seed;
    post.exposure = 1.0f;
    post.grain = 0.05f;
    post.grainScale = 2.5f;
    post.halation = 0.35f;
    post.bloom = 0.12f;
    post.vignette = 0.35f;
    post.splitTone = 0.35f;
    post.saturation = 1.12f;

    std::vector<float> refPost(n), gpuPost(n);
    keenly::renderPostCpu(refPost.data(), refField.data(), w, h, post);

    float* dOut = nullptr;
    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
    keenly::renderPostGpu(dOut, dBuf, w, h, post);
    CUDA_CHECK(cudaMemcpy(gpuPost.data(), dOut, n * sizeof(float), cudaMemcpyDeviceToHost));

    const DiffStats dPost = compareArrays(gpuPost.data(), refPost.data(), n);
    std::printf("  post relL2 %.3e  maxAbs %.3e  (%dx%d, expect 0)\n",
                dPost.relL2, dPost.maxAbs, w, h);

    CUDA_CHECK(cudaFree(dBuf));
    CUDA_CHECK(cudaFree(dOut));
    return dField.maxAbs == 0.0 && dPost.maxAbs == 0.0;
}

// CPU split determinism: the --cpu backend renders with row bands across
// threads, so prove that any split reproduces the serial reference exactly.
// If this ever fires, someone introduced cross-pixel state into the shader
// (a thread-local RNG, an accumulator) and the thread count became visible
// in the output.
static bool runCpuSplitCheck(const keenly::FieldParams& p)
{
    const int w = 320, h = 180;
    const size_t n = size_t(w) * size_t(h) * 3;
    std::vector<float> serial(n), threaded(n);

    keenly::renderFieldCpu(serial.data(), w, h, p);
    keenly::renderFieldCpuParallel(threaded.data(), w, h, p, 7);  // odd count on purpose

    const DiffStats d = compareArrays(threaded.data(), serial.data(), n);
    std::printf("check  : cpu-split relL2 %.3e  maxAbs %.3e  (7 threads vs serial, expect 0)\n",
                d.relL2, d.maxAbs);
    return d.maxAbs == 0.0;
}

// Regression entry point (--check-all): the bit-exact gate on every field
// generator. This is what scripts\verify.ps1 runs after every change.
// defined below: the icon gate reuses the same compositor as batch renders
static bool runIconCheck(const keenly::IconParams& p);
static void applyRainbowIcon(keenly::IconParams& ip, int bgMode, bool whiteFg,
                             const std::string& scheme);

static int runCheckAll()
{
    bool allOk = true;
    for (int m = 0; m < kFieldModeCount; ++m) {
        keenly::FieldParams f;
        keenly::PostParams p;
        applyPreset(f, p, "cinema", 0u);
        f.mode = m;
        f.seed = 31u + unsigned(m) * 17u;
        // turing: a small sim keeps the CPU reference fast; the gate proves
        // the execution paths agree, not the artistic quality of 3000 steps.
        if (m == keenly::kFieldTuring) {
            f.rdRes = 96;
            f.rdIters = 200;
        }
        std::printf("mode %-6s ", kFieldModeNames[m]);
        if (!runCheck(f)) allOk = false;
        // The --cpu backend renders with row bands; prove the split is
        // invisible on every generator, not just the default one.
        if (!runCpuSplitCheck(f)) allOk = false;
    }

    // Synthetic HDR post check: guarantees the post stage has real energy on
    // input, so a broken blur or half-res offset cannot hide behind a
    // highlight-free generator.
    if (!runPostCheck()) allOk = false;

    // Icon shading path: the compositor (SDF glyph, halo, frame, mask, post
    // tail) is a different kernel than fieldPixel - one family icon and one
    // generic glow icon cover its two backgrounds.
    {
        keenly::IconParams ip;
        applyIconFamily(ip, "blue");
        ip.glyph = keenly::kGlyphSword;
        ip.tier = 2;
        std::printf("icon blue/sword/t2  ");
        if (!runIconCheck(ip)) allOk = false;

        keenly::IconParams gp;
        applyRainbowIcon(gp, keenly::kBgGlow, true, "aurora");
        gp.glyph = keenly::kGlyphFlame;
        std::printf("icon glow/aurora    ");
        if (!runIconCheck(gp)) allOk = false;
    }

    std::printf(allOk ? "ALL MODES BIT-EXACT\n" : "GATE FAILED\n");
    return allOk ? 0 : 1;
}

// 3x3 showcase grid: nine style cells rendered tile by tile and pasted into
// one image with a dark gap grid. Each cell is a full pipeline render
// (field + post) and runs the bit-exact verify gate.
//
// Current lineup: the warp anchor plus the eight modern-art generators -
// one cell per shape language so the grid reads as a style catalog. A zero
// override keeps the preset default.
static int runGrid(const Args& args)
{
    struct Cell
    {
        const char* preset;
        int mode;
        unsigned seed;
        float view;      // 0 = preset default
        float warp;
        float contrast;
        float relief;
    };
    static const Cell cells[9] = {
        {"cinema", keenly::kFieldWarp,    7u,  0.0f, 0.0f, 0.0f, 0.0f}, // hero anchor
        {"gold",   keenly::kFieldAgate,   7u,  0.0f, 0.0f, 0.0f, 0.0f}, // suminagashi
        {"mono",   keenly::kFieldWaves,   7u,  0.0f, 0.0f, 0.0f, 0.0f}, // pulsar plot
        {"dusk",   keenly::kFieldTruchet, 7u,  0.0f, 0.0f, 0.0f, 0.0f}, // arc circuit
        {"violet", keenly::kFieldShards,  7u,  0.0f, 0.0f, 0.0f, 0.0f}, // kaleidoscope
        {"ember",  keenly::kFieldDots,    7u,  0.0f, 0.0f, 0.0f, 0.0f}, // riso halftone
        {"ice",    keenly::kFieldAura,    7u,  0.0f, 0.0f, 0.0f, 0.0f}, // silk ribbons
        {"cinema", keenly::kFieldMoire,   7u,  0.0f, 0.0f, 0.0f, 0.0f}, // interference
        {"dusk",   keenly::kFieldDeco,    7u,  0.0f, 0.0f, 0.0f, 0.0f}  // bauhaus quilt
    };

    const int tile = args.tile;
    const int gap = 16;
    const int gw = 3 * tile + 4 * gap;
    std::vector<float> img(size_t(gw) * size_t(gw) * 3, 0.018f);  // gap background
    std::vector<float> cellImg(size_t(tile) * size_t(tile) * 3);
    GpuBuffer fieldBuf, outBuf;  // reused across all nine cells

    for (int i = 0; i < 9; ++i) {
        keenly::FieldParams f;
        keenly::PostParams p;
        applyPreset(f, p, cells[i].preset, 0u);
        f.mode = cells[i].mode;
        f.seed = cells[i].seed;
        p.seed = cells[i].seed;
        if (cells[i].view > 0.0f)     f.view = cells[i].view;
        if (cells[i].warp > 0.0f)     f.warp = cells[i].warp;
        if (cells[i].contrast > 0.0f) f.contrast = cells[i].contrast;
        if (cells[i].relief > 0.0f)   f.relief = cells[i].relief;

        std::printf("cell %d : %-6s mode %d seed %u\n", i + 1, cells[i].preset,
                    f.mode, f.seed);
        if (args.check && !runCheck(f)) {
            std::fprintf(stderr, "CHECK FAILED on cell %d\n", i + 1);
            return 1;
        }

        const size_t n = size_t(tile) * size_t(tile) * 3;
        float* dField = fieldBuf.ensure(n);
        float* dOut = outBuf.ensure(n);
        keenly::renderFieldGpu(dField, tile, tile, f);
        keenly::renderPostGpu(dOut, dField, tile, tile, p);
        CUDA_CHECK(cudaMemcpy(cellImg.data(), dOut, n * sizeof(float),
                              cudaMemcpyDeviceToHost));

        const int cx = gap + (i % 3) * (tile + gap);
        const int cy = gap + (i / 3) * (tile + gap);
        for (int y = 0; y < tile; ++y)
            std::memcpy(&img[(size_t(cy + y) * gw + cx) * 3],
                        &cellImg[size_t(y) * tile * 3],
                        size_t(tile) * 3 * sizeof(float));
    }

    if (!writeImage(args.grid, gw, gw, args.depth, img.data())) {
        std::fprintf(stderr, "failed to write %s\n", args.grid.c_str());
        return 1;
    }
    std::printf("wrote  : %s  (%dx%d, %d-bit)\n", args.grid.c_str(), gw, gw, args.depth);
    return 0;
}

// ---- icon mode -------------------------------------------------------------
// One RGBA skill icon = background field + SDF glyph + tier frame + rounded
// rect mask, all as one pure-function kernel (icon.cuh). The batch demo
// renders the whole family x glyph lineup plus contact sheets.

static int glyphFromName(const std::string& name)
{
    for (int g = 0; g < keenly::kGlyphCount; ++g)
        if (name == keenly::glyphName(g)) return g;
    std::fprintf(stderr, "unknown glyph: %s\n", name.c_str());
    return -1;
}

// ---- generic icon pipeline --------------------------------------------------
// Rainbow background + a pure white/black silhouette glyph. No family
// palette: the background IS the color story, the glyph is a neutral shape.

static int bgModeFromName(const std::string& name)
{
    if (name == "fluid")  return keenly::kBgFluid;
    if (name == "swirl")  return keenly::kBgSwirl;
    if (name == "bands")  return keenly::kBgBands;
    if (name == "rings")  return keenly::kBgRings;
    if (name == "mosaic") return keenly::kBgMosaic;
    if (name == "glow")   return keenly::kBgGlow;
    return -1;
}

static void applyRainbowIcon(keenly::IconParams& ip, int bgMode, bool whiteFg,
                             const std::string& scheme)
{
    using namespace keenly;
    // Field params feed rainbowBg's warp field / glowField's blobs. For the
    // rainbow styles the palette is built inside icon.cuh; for glow the
    // scheme's blob colors land in ip.field.pal here.
    ip.field.seed = 7u;
    ip.field.view = 3.0f;
    ip.field.warp = 0.40f;   // gentle warp: w4 = 4*warp displaces the lookup
                             // by ~1.6 noise units -> bends the field instead
                             // of scrambling it (still-pipeline warp 1.2 is
                             // per-pixel hue confetti on a 7-stop palette)
    ip.field.octaves = 3;      // hue wants a SMOOTH field: 5-octave fBm at
                             // icon size is per-pixel hue jitter, not swirls
    ip.field.contrast = 1.55f;
    ip.field.relief = 2.2f;
    ip.field.bands = 12.0f;
    ip.field.macroAmt = 0.6f;

    ip.post = PostParams();
    ip.post.grain = 0.018f;
    ip.post.grainScale = 2.0f;
    ip.post.halation = 0.0f;
    ip.post.bloom = 0.0f;
    ip.post.vignette = 0.18f;
    ip.post.splitTone = 0.0f;   // a teal/orange grade would skew the rainbow
    ip.post.saturation = 1.05f;

    ip.bgMode = bgMode;
    ip.flat = 1;
    ip.tier = 0;
    // dimmer rainbow under white ink, fuller brightness under black ink
    ip.bgDim = whiteFg ? 0.80f : 0.95f;

    // Linear values: pure black is exact; "white" runs hot on purpose so it
    // still reads white after the ACES shoulder (~0.92 sRGB), same trick as
    // the family strokes.
    const float W = 2.6f;
    if (whiteFg) {
        ip.fgFill[0] = W; ip.fgFill[1] = W; ip.fgFill[2] = W;
        ip.fgStroke[0] = 0.0f; ip.fgStroke[1] = 0.0f; ip.fgStroke[2] = 0.0f;
    } else {
        ip.fgFill[0] = 0.0f; ip.fgFill[1] = 0.0f; ip.fgFill[2] = 0.0f;
        ip.fgStroke[0] = W; ip.fgStroke[1] = W; ip.fgStroke[2] = W;
    }

    if (bgMode == kBgGlow) {
        // Poster-glow background: the scheme's 4 blob colors go into the
        // field palette; the glyph sits on the dark pool, so the pool gets
        // a floor (white ink likes it near-black, black ink needs tint).
        if (!glowScheme(ip.field.pal, scheme)) {
            std::fprintf(stderr, "unknown scheme: %s (use aurora|sepia|flare|tide)\n",
                         scheme.c_str());
            std::exit(1);
        }
        ip.coreLift = whiteFg ? 0.10f : 0.42f;
        ip.field.macroAmt = 0.5f;
        ip.post.grain = 0.055f;      // glow icons keep the film-grain character
        ip.post.grainScale = 2.2f;
        ip.post.vignette = 0.22f;
    }
}

// Icon gate: same iconPixel on CPU and GPU, bit-exact like the field gate.
static bool runIconCheck(const keenly::IconParams& p)
{
    const int w = 128, h = 128;
    const size_t n = size_t(w) * size_t(h) * 4;
    std::vector<float> ref(n), gpu(n);

    keenly::renderIconCpu(ref.data(), w, h, p);

    float* dBuf = nullptr;
    CUDA_CHECK(cudaMalloc(&dBuf, n * sizeof(float)));
    keenly::renderIconGpu(dBuf, w, h, p);
    CUDA_CHECK(cudaMemcpy(gpu.data(), dBuf, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dBuf));

    const DiffStats d = compareArrays(gpu.data(), ref.data(), n);
    std::printf("check  : relL2 %.3e  maxAbs %.3e  (icon GPU vs CPU, expect 0)\n",
                d.relL2, d.maxAbs);
    return d.maxAbs == 0.0;
}

// Render one RGBA icon into host memory through a reusable device scratch
// buffer. Batch runs call this repeatedly with the same GpuBuffer, so the
// device allocation happens once per batch instead of once per icon.
// Returns the GPU render time in ms.
static float renderIconToHost(const keenly::IconParams& p, int w, int h,
                              std::vector<float>& host, GpuBuffer& scratch)
{
    const size_t n = size_t(w) * size_t(h) * 4;
    host.resize(n);
    // Size-adaptive grain (icons_plan 1): clump size scales with the icon,
    // and at 32px the amplitude halves too - full-strength grain on a tiny
    // tile reads as noise, not texture.
    keenly::IconParams ps = p;
    const int minDim = (w < h ? w : h);
    ps.post.grainScale *= float(minDim) / 128.0f;
    if (minDim <= 32) ps.post.grain *= 0.5f;
    float* dBuf = scratch.ensure(n);
    GpuTimer timer;
    timer.start();
    keenly::renderIconGpu(dBuf, w, h, ps);
    const float ms = timer.stop();
    CUDA_CHECK(cudaMemcpy(host.data(), dBuf, n * sizeof(float), cudaMemcpyDeviceToHost));
    return ms;
}

static bool renderOneIcon(const Args& args, const keenly::IconParams& p,
                          int w, int h, const std::string& path)
{
    if (args.check && !runIconCheck(p)) {
        std::fprintf(stderr, "ICON CHECK FAILED: GPU disagrees with the CPU reference\n");
        return false;
    }
    GpuBuffer scratch;
    std::vector<float> img;
    const float ms = renderIconToHost(p, w, h, img, scratch);

    if (!writeImageRgba(path, w, h, img.data())) {
        std::fprintf(stderr, "failed to write %s\n", path.c_str());
        return false;
    }
    std::printf("wrote  : %s  (%dx%d RGBA, %.2f ms)\n", path.c_str(), w, h, ms);
    return true;
}

static int runIcon(const Args& args)
{
    const int glyph = glyphFromName(args.glyph);
    if (glyph < 0) return 1;
    if (args.tier < 0 || args.tier > 4 || (!args.giconMode && args.tier < 1)) {
        std::fprintf(stderr, "bad tier: %d (family: 1..4, generic: 0..4)\n",
                     args.tier);
        return 1;
    }

    keenly::IconParams ip;
    if (args.giconMode) {
        const int bm = bgModeFromName(args.bg);
        if (bm < 0) {
            std::fprintf(stderr, "unknown bg: %s\n", args.bg.c_str());
            return 1;
        }
        applyRainbowIcon(ip, bm, args.fg != "black", args.scheme);
        // generic icons take the seed knob straight: seed IS the variant id.
        ip.field.seed = args.field.seed;
        ip.post.seed = args.field.seed;
        if (args.film >= 0.0f) {
            ip.post.grain = 0.055f * args.film;
            ip.post.grainScale = 2.4f;
        }
        // a few field knobs pass through when explicitly set
        if (args.postSet & kSetView)     ip.field.view = args.field.view;
        if (args.postSet & kSetWarp)     ip.field.warp = args.field.warp;
        if (args.postSet & kSetContrast) ip.field.contrast = args.field.contrast;
        if (args.postSet & kSetRelief)   ip.field.relief = args.field.relief;
        if (args.postSet & kSetOctaves)  ip.field.octaves = args.field.octaves;
        if (args.postSet & kSetBands)    ip.field.bands = args.field.bands;
        if (!args.tierSet) ip.tier = 0;
        else               ip.tier = args.tier;
    } else {
        applyIconFamily(ip, args.family);
        ip.tier = args.tier;
        // explicit --seed picks the icon variant; default keeps the
        // established seed-7 look for backwards-compatible batches
        const unsigned sd = (args.postSet & kSetSeed) ? args.field.seed : 7u;
        ip.field.seed = sd;
        ip.post.seed = sd;
    }
    ip.glyph = glyph;
    ip.roundedMask = args.maskRound;
    ip.radius = args.radius;

    if (args.giconMode) {
        std::printf("gicon  : %s bg / %s fg / %s / t%d / %s mask / seed %u%s%s\n",
                    args.bg.c_str(), args.fg.c_str(), keenly::glyphName(glyph),
                    ip.tier, args.maskRound ? "round" : "rect", ip.field.seed,
                    args.bg == "glow" ? "  scheme " : "",
                    args.bg == "glow" ? args.scheme.c_str() : "");
    } else {
        std::printf("icon   : %s / %s / t%d / %s mask\n", args.family.c_str(),
                    keenly::glyphName(glyph), args.tier,
                    args.maskRound ? "round" : "rect");
    }
    return renderOneIcon(args, ip, args.width, args.height, args.out) ? 0 : 1;
}

// Demo batch: every family x every glyph at 128px + 64px, tier 2 rounded,
// plus one contact sheet per family and a tier-variant sheet. This is the
// "show me what this looks like" output; the manifest-driven production run
// comes later (docs/icons_plan.md Phase 2).
static int runIconsDemo(const Args& args)
{
    static const char* kFamilies[4] = {"blue", "red", "gold", "purple"};
    static const int kTile = 128;
    static const int kThumb = 64;
    const std::string root = args.iconsDemo;
    bool allOk = true;

    for (int f = 0; f < 4; ++f) {
        // Contact sheet: 6 x 3 glyphs at 128px on a dark backdrop.
        const int cols = 6;
        const int rows = (keenly::kGlyphCount + cols - 1) / cols;
        const int gap = 16;
        const int sw = cols * kTile + (cols + 1) * gap;
        const int sh = rows * kTile + (rows + 1) * gap;
        std::vector<float> sheet(size_t(sw) * size_t(sh) * 3, 0.030f);

        // Per family: the bit-exact gate runs on the first glyph (params vary
        // per glyph only by seed and glyph id, one gate per family covers the
        // code paths), then every glyph renders through ONE reused scratch
        // buffer. Each 128px icon is rendered once and feeds both its PNG and
        // the contact sheet - the old code rendered it twice.
        GpuBuffer scratch;
        std::vector<float> icon, thumb;
        for (int g = 0; g < keenly::kGlyphCount; ++g) {
            keenly::IconParams ip;
            applyIconFamily(ip, kFamilies[f]);
            ip.glyph = g;
            ip.tier = 2;
            ip.roundedMask = 1;
            ip.radius = 0.16f;
            ip.field.seed = 7u + unsigned(g) * 13u;
            ip.post.seed = ip.field.seed;

            if (g == 0 && args.check && !runIconCheck(ip)) {
                std::fprintf(stderr, "ICON CHECK FAILED on family %s\n", kFamilies[f]);
                allOk = false;
            }

            char name[128];
            std::snprintf(name, sizeof(name), "%s/%s_%s.png", kFamilies[f],
                          kFamilies[f], keenly::glyphName(g));
            const float ms128 = renderIconToHost(ip, kTile, kTile, icon, scratch);
            if (!writeImageRgba(root + "/" + name, kTile, kTile, icon.data())) {
                std::fprintf(stderr, "failed to write %s\n", name);
                allOk = false;
            }

            std::snprintf(name, sizeof(name), "%s/thumbs/%s_%s_64.png", kFamilies[f],
                          kFamilies[f], keenly::glyphName(g));
            const float ms64 = renderIconToHost(ip, kThumb, kThumb, thumb, scratch);
            if (!writeImageRgba(root + "/" + name, kThumb, kThumb, thumb.data())) {
                std::fprintf(stderr, "failed to write %s\n", name);
                allOk = false;
            }
            std::printf("glyph  : %-6s %-8s 128px %5.2f ms  64px %4.2f ms\n",
                        kFamilies[f], keenly::glyphName(g), ms128, ms64);

            // paste the 128px icon onto the contact sheet (alpha over dark)
            const int cx = gap + (g % cols) * (kTile + gap);
            const int cy = gap + (g / cols) * (kTile + gap);
            for (int y = 0; y < kTile; ++y) {
                for (int x = 0; x < kTile; ++x) {
                    const size_t ci = (size_t(y) * kTile + x) * 4;
                    const float a = icon[ci + 3];
                    float* s = &sheet[(size_t(cy + y) * sw + cx + x) * 3];
                    for (int c = 0; c < 3; ++c)
                        s[c] = icon[ci + c] * a + 0.030f * (1.0f - a);
                }
            }
        }

        char sheetName[128];
        std::snprintf(sheetName, sizeof(sheetName), "%s/sheet_%s.png",
                      kFamilies[f], kFamilies[f]);
        if (!writeImage(root + "/" + sheetName, sw, sh, 8, sheet.data())) {
            std::fprintf(stderr, "failed to write %s\n", sheetName);
            allOk = false;
        } else {
            std::printf("wrote  : %s/sheet_%s.png  (%dx%d)\n",
                        kFamilies[f], kFamilies[f], sw, sh);
        }
    }

    // Tier variant sheet: sword on gold, tiers 1..4 at 256px.
    {
        const int t = 256, gap = 16;
        const int sw = 4 * t + 5 * gap;
        std::vector<float> sheet(size_t(sw) * size_t(sw) * 3, 0.030f);
        GpuBuffer scratch;
        std::vector<float> cell;
        for (int tier = 1; tier <= 4; ++tier) {
            keenly::IconParams ip;
            applyIconFamily(ip, "gold");
            ip.glyph = keenly::kGlyphSword;
            ip.tier = tier;
            ip.roundedMask = 1;
            ip.radius = 0.16f;
            renderIconToHost(ip, t, t, cell, scratch);
            const int cx = gap + (tier - 1) * (t + gap);
            for (int y = 0; y < t; ++y) {
                for (int x = 0; x < t; ++x) {
                    const size_t ci = (size_t(y) * t + x) * 4;
                    const float a = cell[ci + 3];
                    float* s = &sheet[(size_t(gap + y) * sw + cx + x) * 3];
                    for (int c = 0; c < 3; ++c)
                        s[c] = cell[ci + c] * a + 0.030f * (1.0f - a);
                }
            }
        }
        if (!writeImage(root + "/sheet_tiers.png", sw, sw, 8, sheet.data())) {
            std::fprintf(stderr, "failed to write sheet_tiers.png\n");
            allOk = false;
        } else {
            std::printf("wrote  : %s/sheet_tiers.png  (%dx%d)\n",
                        root.c_str(), sw, sw);
        }
    }

    return allOk ? 0 : 1;
}

// Generic icon demo: every bg style x every fg color x every glyph at 128px,
// one contact sheet per (style, fg) plus a compact style-comparison sheet.
// The bit-exact gate runs once per style (same coverage argument as the
// family demo: params differ only by seed and glyph id).
static int runRainbowDemo(const Args& args)
{
    static const char* kStyles[5] = {"fluid", "swirl", "bands", "rings", "mosaic"};
    static const int kStyleIds[5] = {keenly::kBgFluid, keenly::kBgSwirl,
                                     keenly::kBgBands, keenly::kBgRings,
                                     keenly::kBgMosaic};
    static const char* kFgs[2] = {"white", "black"};
    const int kT = 128;
    const int gap = 16;
    const std::string root = args.rainbowDemo;
    bool allOk = true;

    // Overview sheet: 5 style rows x 2 fg columns, sword glyph.
    const int ovw = 2 * kT + 3 * gap;
    const int ovh = 5 * kT + 6 * gap;
    std::vector<float> overview(size_t(ovw) * size_t(ovh) * 3, 0.030f);

    for (int st = 0; st < 5; ++st) {
        for (int fi = 0; fi < 2; ++fi) {
            const bool white = (fi == 0);

            // Per (style, fg) sheet: 6 cols x 3 rows of 128px icons.
            const int cols = 6;
            const int rows = (keenly::kGlyphCount + cols - 1) / cols;
            const int sw = cols * kT + (cols + 1) * gap;
            const int sh = rows * kT + (rows + 1) * gap;
            std::vector<float> sheet(size_t(sw) * size_t(sh) * 3, 0.030f);

            GpuBuffer scratch;
            std::vector<float> icon;
            for (int g = 0; g < keenly::kGlyphCount; ++g) {
                keenly::IconParams ip;
                applyRainbowIcon(ip, kStyleIds[st], white, "aurora");
                ip.glyph = g;
                ip.field.seed = 7u + unsigned(g) * 13u;
                ip.post.seed = ip.field.seed;

                if (g == 0 && args.check && !runIconCheck(ip)) {
                    std::fprintf(stderr, "ICON CHECK FAILED on %s/%s\n",
                                 kStyles[st], kFgs[fi]);
                    allOk = false;
                }

                char name[160];
                std::snprintf(name, sizeof(name), "%s/%s_%s.png", kStyles[st],
                              kFgs[fi], keenly::glyphName(g));
                renderIconToHost(ip, kT, kT, icon, scratch);
                if (!writeImageRgba(root + "/" + name, kT, kT, icon.data())) {
                    std::fprintf(stderr, "failed to write %s\n", name);
                    allOk = false;
                }

                // paste onto the contact sheet (alpha over dark)
                const int cx = gap + (g % cols) * (kT + gap);
                const int cy = gap + (g / cols) * (kT + gap);
                for (int y = 0; y < kT; ++y) {
                    for (int x = 0; x < kT; ++x) {
                        const size_t ci = (size_t(y) * kT + x) * 4;
                        const float a = icon[ci + 3];
                        float* s = &sheet[(size_t(cy + y) * sw + cx + x) * 3];
                        for (int c = 0; c < 3; ++c)
                            s[c] = icon[ci + c] * a + 0.030f * (1.0f - a);
                    }
                }

                // overview cell = sword of this style/fg
                if (g == keenly::kGlyphSword) {
                    const int ox = gap + fi * (kT + gap);
                    const int oy = gap + st * (kT + gap);
                    for (int y = 0; y < kT; ++y) {
                        for (int x = 0; x < kT; ++x) {
                            const size_t ci = (size_t(y) * kT + x) * 4;
                            const float a = icon[ci + 3];
                            float* s = &overview[(size_t(oy + y) * ovw + ox + x) * 3];
                            for (int c = 0; c < 3; ++c)
                                s[c] = icon[ci + c] * a + 0.030f * (1.0f - a);
                        }
                    }
                }
            }

            char sheetName[160];
            std::snprintf(sheetName, sizeof(sheetName), "sheets/sheet_%s_%s.png",
                          kStyles[st], kFgs[fi]);
            if (!writeImage(root + "/" + sheetName, sw, sh, 8, sheet.data())) {
                std::fprintf(stderr, "failed to write %s\n", sheetName);
                allOk = false;
            } else {
                std::printf("wrote  : %s/%s  (%dx%d)\n", root.c_str(), sheetName,
                            sw, sh);
            }
            std::printf("style  : %-6s %-5s done\n", kStyles[st], kFgs[fi]);
        }
    }

    if (!writeImage(root + "/sheet_styles.png", ovw, ovh, 8, overview.data())) {
        std::fprintf(stderr, "failed to write sheet_styles.png\n");
        allOk = false;
    } else {
        std::printf("wrote  : %s/sheet_styles.png  (%dx%d)\n", root.c_str(),
                    ovw, ovh);
    }
    return allOk ? 0 : 1;
}

// Glow icon demo: every scheme x every fg color x every glyph at 128px.
// Same structure as the rainbow demo - one contact sheet per (scheme, fg),
// plus an overview sheet with scheme rows and fg columns.
static int runGlowDemo(const Args& args)
{
    static const char* kSchemes[4] = {"aurora", "sepia", "flare", "tide"};
    static const char* kFgs[2] = {"white", "black"};
    const int kT = 128;
    const int gap = 16;
    const std::string root = args.glowDemo;
    bool allOk = true;

    const int ovw = 2 * kT + 3 * gap;
    const int ovh = 4 * kT + 5 * gap;
    std::vector<float> overview(size_t(ovw) * size_t(ovh) * 3, 0.030f);

    for (int st = 0; st < 4; ++st) {
        for (int fi = 0; fi < 2; ++fi) {
            const bool white = (fi == 0);

            const int cols = 6;
            const int rows = (keenly::kGlyphCount + cols - 1) / cols;
            const int sw = cols * kT + (cols + 1) * gap;
            const int sh = rows * kT + (rows + 1) * gap;
            std::vector<float> sheet(size_t(sw) * size_t(sh) * 3, 0.030f);

            GpuBuffer scratch;
            std::vector<float> icon;
            for (int g = 0; g < keenly::kGlyphCount; ++g) {
                keenly::IconParams ip;
                applyRainbowIcon(ip, keenly::kBgGlow, white, kSchemes[st]);
                ip.glyph = g;
                ip.field.seed = 7u + unsigned(g) * 13u;
                ip.post.seed = ip.field.seed;

                if (g == 0 && args.check && !runIconCheck(ip)) {
                    std::fprintf(stderr, "ICON CHECK FAILED on %s/%s\n",
                                 kSchemes[st], kFgs[fi]);
                    allOk = false;
                }

                char name[160];
                std::snprintf(name, sizeof(name), "%s/%s_%s.png", kSchemes[st],
                              kFgs[fi], keenly::glyphName(g));
                renderIconToHost(ip, kT, kT, icon, scratch);
                if (!writeImageRgba(root + "/" + name, kT, kT, icon.data())) {
                    std::fprintf(stderr, "failed to write %s\n", name);
                    allOk = false;
                }

                const int cx = gap + (g % cols) * (kT + gap);
                const int cy = gap + (g / cols) * (kT + gap);
                for (int y = 0; y < kT; ++y) {
                    for (int x = 0; x < kT; ++x) {
                        const size_t ci = (size_t(y) * kT + x) * 4;
                        const float a = icon[ci + 3];
                        float* s = &sheet[(size_t(cy + y) * sw + cx + x) * 3];
                        for (int c = 0; c < 3; ++c)
                            s[c] = icon[ci + c] * a + 0.030f * (1.0f - a);
                    }
                }

                if (g == keenly::kGlyphSword) {
                    const int ox = gap + fi * (kT + gap);
                    const int oy = gap + st * (kT + gap);
                    for (int y = 0; y < kT; ++y) {
                        for (int x = 0; x < kT; ++x) {
                            const size_t ci = (size_t(y) * kT + x) * 4;
                            const float a = icon[ci + 3];
                            float* s = &overview[(size_t(oy + y) * ovw + ox + x) * 3];
                            for (int c = 0; c < 3; ++c)
                                s[c] = icon[ci + c] * a + 0.030f * (1.0f - a);
                        }
                    }
                }
            }

            char sheetName[160];
            std::snprintf(sheetName, sizeof(sheetName), "sheets/sheet_%s_%s.png",
                          kSchemes[st], kFgs[fi]);
            if (!writeImage(root + "/" + sheetName, sw, sh, 8, sheet.data())) {
                std::fprintf(stderr, "failed to write %s\n", sheetName);
                allOk = false;
            } else {
                std::printf("wrote  : %s/%s  (%dx%d)\n", root.c_str(), sheetName,
                            sw, sh);
            }
            std::printf("scheme : %-6s %-5s done\n", kSchemes[st], kFgs[fi]);
        }
    }

    if (!writeImage(root + "/sheet_schemes.png", ovw, ovh, 8, overview.data())) {
        std::fprintf(stderr, "failed to write sheet_schemes.png\n");
        allOk = false;
    } else {
        std::printf("wrote  : %s/sheet_schemes.png  (%dx%d)\n", root.c_str(),
                    ovw, ovh);
    }
    return allOk ? 0 : 1;
}

// Render one still on the CPU (--cpu). Same shader, same post chain, same
// PNG encoder as the GPU path - the only difference is where the pixels are
// computed. Output is bit-identical to the GPU render (that is what the gate
// asserts), so this is the backend for machines without an NVIDIA device and
// the way to prove the "same seed, same image, anywhere" claim.
static int runStillCpu(const Args& args, const keenly::FieldParams& field,
                       const keenly::PostParams& postIn)
{
    const int w = args.width, h = args.height;
    const size_t pixels = size_t(w) * size_t(h);
    const size_t n = pixels * 3;

    int threads = args.cpuThreads;
    if (threads <= 0) {
        threads = int(std::thread::hardware_concurrency());
        if (threads <= 0) threads = 1;
    }
    std::printf("cpu    : %d thread(s)\n", threads);

    std::vector<float> hField(n);
    auto t0 = std::chrono::steady_clock::now();
    keenly::renderFieldCpuParallel(hField.data(), w, h, field, threads);
    const float msField =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("field  : %.3f ms  (%.1f Mpixel/s)\n", msField,
                pixels / (double(msField) * 1e3));

    std::vector<float> hOut(n);
    const float* src = hField.data();
    if (args.post) {
        keenly::PostParams post = postIn;
        post.seed = field.seed;
        t0 = std::chrono::steady_clock::now();
        keenly::renderPostCpu(hOut.data(), hField.data(), w, h, post, threads);
        const float msPost =
            std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("post   : %.3f ms  (grain %.3f  halation %.2f  bloom %.2f  vignette %.2f  split %.2f)\n",
                    msPost, post.grain, post.halation, post.bloom, post.vignette,
                    post.splitTone);
        src = hOut.data();
    }

    const auto tWrite = std::chrono::steady_clock::now();
    const bool wrote = writeImage(args.out, w, h, args.depth, src);
    const float msWrite =
        std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - tWrite).count();
    if (!wrote) {
        std::fprintf(stderr, "failed to write %s\n", args.out.c_str());
        return 1;
    }
    std::printf("write  : %.3f ms  (sRGB encode + dither + stream to disk)\n", msWrite);
    std::printf("wrote  : %s  (%d-bit)\n", args.out.c_str(), args.depth);
    return 0;
}

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage();
        return 1;
    }

    if (args.listModes) {
        for (int m = 0; m < kFieldModeCount; ++m)
            std::printf("%d  %s\n", m, kFieldModeNames[m]);
        return 0;
    }

    // --cpu renders the still pipeline without touching CUDA at all, so the
    // device query has to stay behind this branch: on a machine with no
    // NVIDIA driver cudaGetDeviceProperties would abort before we get here.
    if (args.cpu) {
        applyPreset(args.field, args.postp, args.preset, args.postSet);
        if (args.film >= 0.0f) {
            args.postp.grain = 0.055f * args.film;
            args.postp.grainScale = 2.4f;
        }
        std::printf("render : %dx%d  preset %s  mode %s  seed %u  view %.2f  warp %.2f  octaves %d\n",
                    args.width, args.height, args.preset.c_str(),
                    kFieldModeNames[args.field.mode], args.field.seed,
                    args.field.view, args.field.warp, args.field.octaves);
        return runStillCpu(args, args.field, args.postp);
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device : %s\n", prop.name);

    if (args.checkAll)
        return runCheckAll();

    if (!args.iconsDemo.empty())
        return runIconsDemo(args);

    if (!args.rainbowDemo.empty())
        return runRainbowDemo(args);

    if (!args.glowDemo.empty())
        return runGlowDemo(args);

    if (args.iconMode)
        return runIcon(args);

    if (!args.grid.empty())
        return runGrid(args);

    applyPreset(args.field, args.postp, args.preset, args.postSet);
    if (args.film >= 0.0f) {
        // Dedicated film filter: a fixed 35mm-style clumped grain spec
        // (clumps ~2.4px) forced on top of whatever the preset picked.
        args.postp.grain = 0.055f * args.film;
        args.postp.grainScale = 2.4f;
    }
    std::printf("render : %dx%d  preset %s  mode %s  seed %u  view %.2f  warp %.2f  octaves %d\n",
                args.width, args.height, args.preset.c_str(),
                kFieldModeNames[args.field.mode], args.field.seed,
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
    keenly::renderFieldGpu(dField, args.width, args.height, args.field);
    const float msField = timer.stop();
    std::printf("field  : %.3f ms  (%.1f Mpixel/s)\n", msField, pixels / (double(msField) * 1e3));

    // Pinned host buffer: the multi-hundred-MB D2H at poster sizes runs at
    // full PCIe bandwidth instead of pageable memory's ~2-4GB/s.
    float* hRgb = nullptr;
    CUDA_CHECK(cudaMallocHost(&hRgb, pixels * 3 * sizeof(float)));
    if (args.post) {
        keenly::PostParams post = args.postp;
        post.seed = args.field.seed;  // one seed per image: grain rerolls per seed
        float* dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dOut, pixels * 3 * sizeof(float)));
        timer.start();
        keenly::renderPostGpu(dOut, dField, args.width, args.height, post);
        const float msPost = timer.stop();
        std::printf("post   : %.3f ms  (grain %.3f  halation %.2f  bloom %.2f  vignette %.2f  split %.2f)\n",
                    msPost, post.grain, post.halation, post.bloom, post.vignette,
                    post.splitTone);
        CUDA_CHECK(cudaMemcpy(hRgb, dOut, pixels * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dOut));
    } else {
        CUDA_CHECK(cudaMemcpy(hRgb, dField, pixels * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(dField));

    // Encode + PNG write is a CPU stage; time it with the wall clock.
    const auto tWrite = std::chrono::steady_clock::now();
    const bool wrote = writeImage(args.out, args.width, args.height, args.depth, hRgb);
    const auto tDone = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaFreeHost(hRgb));
    if (!wrote) {
        std::fprintf(stderr, "failed to write %s\n", args.out.c_str());
        return 1;
    }
    const float msWrite = std::chrono::duration<float, std::milli>(tDone - tWrite).count();
    std::printf("write  : %.3f ms  (sRGB encode + dither + stream to disk)\n", msWrite);
    std::printf("wrote  : %s  (%d-bit)\n", args.out.c_str(), args.depth);
    return 0;
}