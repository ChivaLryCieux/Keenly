#include "cli.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "catalog.h"

void printUsage()
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
        "  --icons P       render schema-versioned icon manifest CSV P\n"
        "  --outdir P      output root for manifest batches (default out/icons)\n"
        "generic icons (rainbow/glow background + flat fg silhouette):\n"
        "  --gicon        render one generic icon (uses --bg --fg --glyph --tier)\n"
        "  --bg     B     fluid|swirl|bands|rings|mosaic|glow (default fluid)\n"
        "  --scheme S     aurora|sepia|flare|tide      (bg=glow, default aurora)\n"
        "  --fg     C     white|black                   (default white)\n"
        "  --rainbow-demo P render the generic icon batch into directory P\n"
        "  --glow-demo  P  render the glow icon batch (scheme x fg) into dir P\n");
}

bool parseArgs(int argc, char** argv, Args& a)
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
        } else if (key == "--icons") {
            const char* v = next("--icons");
            if (!v) return false;
            a.iconsManifest = v;
        } else if (key == "--outdir") {
            const char* v = next("--outdir");
            if (!v) return false;
            a.outDir = v;
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
