#include "jobs.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "catalog.h"
#include "checks.h"
#include "cuda_utils.h"
#include "image_io.h"
#include "artifact.h"
#include "manifest.h"
#include "version.h"

// 3x3 showcase grid: nine style cells rendered tile by tile and pasted into
// one image with a dark gap grid. Each cell is a full pipeline render
// (field + post) and runs the bit-exact verify gate.
//
// Current lineup: the warp anchor plus the eight modern-art generators -
// one cell per shape language so the grid reads as a style catalog. A zero
// override keeps the preset default.
int runGrid(const Args& args)
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
    std::vector<float> cpuField(size_t(tile) * size_t(tile) * 3);
    GpuBuffer fieldBuf, outBuf;  // reused across all nine cells on the GPU path
    const int threads = cpuThreadCount(args.cpuThreads);

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
        if (args.check) {
            const bool checkOk = args.cpu ? runCpuSplitCheck(f) : runCheck(f);
            if (!checkOk) {
                std::fprintf(stderr, "CHECK FAILED on cell %d\n", i + 1);
                return 1;
            }
        }

        const size_t n = size_t(tile) * size_t(tile) * 3;
        if (args.cpu) {
            keenly::renderFieldCpuParallel(cpuField.data(), tile, tile, f, threads);
            keenly::renderPostCpu(cellImg.data(), cpuField.data(), tile, tile, p, threads);
        } else {
            float* dField = fieldBuf.ensure(n);
            float* dOut = outBuf.ensure(n);
            keenly::renderFieldGpu(dField, tile, tile, f);
            keenly::renderPostGpu(dOut, dField, tile, tile, p);
            CUDA_CHECK(cudaMemcpy(cellImg.data(), dOut, n * sizeof(float),
                                  cudaMemcpyDeviceToHost));
        }

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

// Render one RGBA icon into host memory. GPU batch runs reuse the scratch
// buffer; CPU runs split rows across worker threads. Returns render time in ms.
static float renderIconToHost(const keenly::IconParams& p, int w, int h,
                              std::vector<float>& host, GpuBuffer& scratch,
                              bool useCpu, int threads)
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

    if (useCpu) {
        const auto t0 = std::chrono::steady_clock::now();
        keenly::renderIconCpuParallel(host.data(), w, h, ps, threads);
        return std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }

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
    if (args.check && !runIconBackendCheck(args.cpu, p)) {
        std::fprintf(stderr, "ICON CHECK FAILED: backend output disagrees with the reference\n");
        return false;
    }
    GpuBuffer scratch;
    std::vector<float> img;
    const float ms = renderIconToHost(p, w, h, img, scratch,
                                      args.cpu, cpuThreadCount(args.cpuThreads));

    if (!writeImageRgba(path, w, h, img.data())) {
        std::fprintf(stderr, "failed to write %s\n", path.c_str());
        return false;
    }
    std::printf("wrote  : %s  (%dx%d RGBA, %.2f ms)\n", path.c_str(), w, h, ms);
    return true;
}

int runIcon(const Args& args)
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
int runIconsDemo(const Args& args)
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

            if (g == 0 && args.check && !runIconBackendCheck(args.cpu, ip)) {
                std::fprintf(stderr, "ICON CHECK FAILED on family %s\n", kFamilies[f]);
                allOk = false;
            }

            char name[128];
            std::snprintf(name, sizeof(name), "%s/%s_%s.png", kFamilies[f],
                          kFamilies[f], keenly::glyphName(g));
            const float ms128 = renderIconToHost(ip, kTile, kTile, icon, scratch,
                                             args.cpu, cpuThreadCount(args.cpuThreads));
            if (!writeImageRgba(root + "/" + name, kTile, kTile, icon.data())) {
                std::fprintf(stderr, "failed to write %s\n", name);
                allOk = false;
            }

            std::snprintf(name, sizeof(name), "%s/thumbs/%s_%s_64.png", kFamilies[f],
                          kFamilies[f], keenly::glyphName(g));
            const float ms64 = renderIconToHost(ip, kThumb, kThumb, thumb, scratch,
                                            args.cpu, cpuThreadCount(args.cpuThreads));
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
            renderIconToHost(ip, t, t, cell, scratch,
                             args.cpu, cpuThreadCount(args.cpuThreads));
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
int runRainbowDemo(const Args& args)
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

                if (g == 0 && args.check && !runIconBackendCheck(args.cpu, ip)) {
                    std::fprintf(stderr, "ICON CHECK FAILED on %s/%s\n",
                                 kStyles[st], kFgs[fi]);
                    allOk = false;
                }

                char name[160];
                std::snprintf(name, sizeof(name), "%s/%s_%s.png", kStyles[st],
                              kFgs[fi], keenly::glyphName(g));
                renderIconToHost(ip, kT, kT, icon, scratch,
                                 args.cpu, cpuThreadCount(args.cpuThreads));
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
int runGlowDemo(const Args& args)
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

                if (g == 0 && args.check && !runIconBackendCheck(args.cpu, ip)) {
                    std::fprintf(stderr, "ICON CHECK FAILED on %s/%s\n",
                                 kSchemes[st], kFgs[fi]);
                    allOk = false;
                }

                char name[160];
                std::snprintf(name, sizeof(name), "%s/%s_%s.png", kSchemes[st],
                              kFgs[fi], keenly::glyphName(g));
                renderIconToHost(ip, kT, kT, icon, scratch,
                                 args.cpu, cpuThreadCount(args.cpuThreads));
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
int runStillCpu(const Args& args, const keenly::FieldParams& field,
                       const keenly::PostParams& postIn)
{
    const int w = args.width, h = args.height;
    const size_t pixels = size_t(w) * size_t(h);
    const size_t n = pixels * 3;

    const int threads = cpuThreadCount(args.cpuThreads);
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

int runStillGpu(const Args& args, const keenly::FieldParams& field,
                const keenly::PostParams& postIn)
{
    float* dField = nullptr;
    const size_t pixels = size_t(args.width) * size_t(args.height);
    CUDA_CHECK(cudaMalloc(&dField, pixels * 3 * sizeof(float)));

    GpuTimer timer;
    timer.start();
    keenly::renderFieldGpu(dField, args.width, args.height, field);
    const float msField = timer.stop();
    std::printf("field  : %.3f ms  (%.1f Mpixel/s)\n", msField,
                pixels / (double(msField) * 1e3));

    float* hRgb = nullptr;
    CUDA_CHECK(cudaMallocHost(&hRgb, pixels * 3 * sizeof(float)));
    if (args.post) {
        keenly::PostParams post = postIn;
        post.seed = field.seed;
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

int runIconsManifest(const Args& args)
{
    std::vector<IconAssetSpec> specs;
    std::string error;
    if (!loadIconManifest(args.iconsManifest, specs, error)) {
        std::fprintf(stderr, "manifest error: %s\n", error.c_str());
        return 1;
    }

    std::string manifestSha;
    if (!sha256FileHex(args.iconsManifest, manifestSha)) manifestSha = "unknown";

    GpuBuffer scratch;
    std::vector<float> icon;
    std::set<std::string> checkedFamilies;
    int written = 0;
    bool allOk = true;
    const char* backend = args.cpu ? "cpu" : "cuda";
    const int threads = cpuThreadCount(args.cpuThreads);

    std::printf("manifest: %s  (%zu assets, %s backend)\n",
                args.iconsManifest.c_str(), specs.size(), backend);

    for (const IconAssetSpec& spec : specs) {
        const int glyph = glyphFromName(spec.glyph);
        if (glyph < 0) {
            std::fprintf(stderr, "manifest error: unknown glyph %s\n", spec.glyph.c_str());
            return 1;
        }

        keenly::IconParams ip;
        applyIconFamily(ip, spec.family);
        ip.glyph = glyph;
        ip.tier = spec.tier;
        ip.field.seed = spec.seed;
        ip.post.seed = spec.seed;
        ip.roundedMask = spec.mask == "round" ? 1 : 0;
        ip.radius = spec.radius;

        if (args.check && checkedFamilies.insert(spec.family).second &&
            !runIconBackendCheck(args.cpu, ip)) {
            std::fprintf(stderr, "ICON CHECK FAILED for family %s\n", spec.family.c_str());
            allOk = false;
            break;
        }

        const float ms = renderIconToHost(ip, spec.size, spec.size, icon, scratch,
                                          args.cpu, threads);
        char name[192];
        std::snprintf(name, sizeof(name), "%s/%s_t%d_v%d_%d.png",
                      spec.family.c_str(), spec.glyph.c_str(), spec.tier,
                      spec.variant, spec.size);
        const std::string path = args.outDir + "/" + name;
        if (!writeImageRgba(path, spec.size, spec.size, icon.data())) {
            std::fprintf(stderr, "failed to write %s\n", path.c_str());
            allOk = false;
            continue;
        }

        std::string outputSha;
        if (!sha256FileHex(path, outputSha)) outputSha = "unknown";
        const std::string canonical = canonicalIconSpec(spec);
        const std::string specSha = sha256Hex(
            reinterpret_cast<const uint8_t*>(canonical.data()), canonical.size());

        std::ostringstream js;
        js << "{\n"
           << "  \"schema\": \"keenly.icon-artifact/v1\",\n"
           << "  \"generator\": {\"name\": \"keenly\", \"version\": \"" << KEENLY_VERSION
           << "\", \"sourceRevision\": \"" << KEENLY_SOURCE_REVISION << "\"},\n"
           << "  \"backend\": \"" << backend << "\",\n"
           << "  \"manifest\": {\"path\": \"" << jsonEscape(args.iconsManifest)
           << "\", \"schema\": \"" << kIconManifestSchema
           << "\", \"sha256\": \"" << manifestSha << "\"},\n"
           << "  \"asset\": {\n"
           << "    \"family\": \"" << jsonEscape(spec.family) << "\",\n"
           << "    \"glyph\": \"" << jsonEscape(spec.glyph) << "\",\n"
           << "    \"tier\": " << spec.tier << ",\n"
           << "    \"variant\": " << spec.variant << ",\n"
           << "    \"seed\": " << spec.seed << ",\n"
           << "    \"size\": " << spec.size << ",\n"
           << "    \"mask\": \"" << jsonEscape(spec.mask) << "\",\n"
           << "    \"radius\": " << spec.radius << "\n"
           << "  },\n"
           << "  \"canonicalSpec\": \"" << jsonEscape(canonical) << "\",\n"
           << "  \"specSha256\": \"" << specSha << "\",\n"
           << "  \"output\": {\"path\": \"" << jsonEscape(path)
           << "\", \"width\": " << spec.size << ", \"height\": " << spec.size
           << ", \"sha256\": \"" << outputSha << "\"}\n"
           << "}\n";

        const std::string sidecar = path + ".json";
        if (!writeTextFile(sidecar, js.str())) {
            std::fprintf(stderr, "failed to write %s\n", sidecar.c_str());
            allOk = false;
            continue;
        }
        ++written;
        std::printf("wrote  : %s  (%dx%d RGBA, %.2f ms)\n",
                    path.c_str(), spec.size, spec.size, ms);
    }

    std::printf("manifest: %d/%zu asset(s) written\n", written, specs.size());
    return allOk && written == int(specs.size()) ? 0 : 1;
}
