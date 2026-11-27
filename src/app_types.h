#pragma once
#include <string>
#include "render.h"

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
    std::string iconsManifest; // non-empty: schema-driven icon manifest CSV
    std::string outDir = "out/icons"; // output root for manifest batches

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
