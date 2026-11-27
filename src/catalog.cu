#include "catalog.h"
#include "app_types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---- presets ---------------------------------------------------------------
// Palette stops are perceptual (sRGB-like); fieldPixel linearizes them.
// Each preset also carries field + post defaults (grain character differs:
// clumped 35mm grain for cinema, fine matte grain for the poster). Explicit
// CLI flags always win over preset defaults.


// Glow-scheme color sets shared by the poster presets (--preset) and the
// generic icon pipeline (--bg glow --scheme). pal.rgb[i] is a blob color;
// count = number of blobs. Perceptual values - glowField squares to linear.
bool glowScheme(keenly::Palette& pal, const std::string& name)
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

void applyPreset(keenly::FieldParams& f, keenly::PostParams& p, const std::string& name,
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

// ---- icon families ---------------------------------------------------------
// Four game-skill color families. Each family = background field preset +
// linear accent colors for the glyph/frame (see icon.cuh for how they are
// used). Accents are LINEAR radiance values: ACES compresses highlights, so
// stroke / glow intentionally exceed 1.0 for the "lit" game-icon look.

void applyIconFamily(keenly::IconParams& ip, const std::string& name)
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
std::string fieldModeList()
{
    std::string s;
    for (int m = 0; m < kFieldModeCount; ++m) {
        if (m) s += '|';
        s += kFieldModeNames[m];
    }
    return s;
}

// Case-sensitive mode name -> enum value, -1 when unknown.
int modeFromName(const char* name)
{
    for (int m = 0; m < kFieldModeCount; ++m)
        if (std::strcmp(name, kFieldModeNames[m]) == 0) return m;
    return -1;
}

// ---- icon mode -------------------------------------------------------------
// One RGBA skill icon = background field + SDF glyph + tier frame + rounded
// rect mask, all as one pure-function kernel (icon.cuh). The batch demo
// renders the whole family x glyph lineup plus contact sheets.

int glyphFromName(const std::string& name)
{
    for (int g = 0; g < keenly::kGlyphCount; ++g)
        if (name == keenly::glyphName(g)) return g;
    std::fprintf(stderr, "unknown glyph: %s\n", name.c_str());
    return -1;
}

// ---- generic icon pipeline --------------------------------------------------
// Rainbow background + a pure white/black silhouette glyph. No family
// palette: the background IS the color story, the glyph is a neutral shape.

int bgModeFromName(const std::string& name)
{
    if (name == "fluid")  return keenly::kBgFluid;
    if (name == "swirl")  return keenly::kBgSwirl;
    if (name == "bands")  return keenly::kBgBands;
    if (name == "rings")  return keenly::kBgRings;
    if (name == "mosaic") return keenly::kBgMosaic;
    if (name == "glow")   return keenly::kBgGlow;
    return -1;
}

void applyRainbowIcon(keenly::IconParams& ip, int bgMode, bool whiteFg,
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

int fieldModeCount()
{
    return kFieldModeCount;
}

const char* fieldModeName(int mode)
{
    return kFieldModeNames[mode];
}
