#include "checks.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "catalog.h"
#include "compare.h"
#include "cuda_utils.h"

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
bool runPostCheck()
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
bool runCheck(const keenly::FieldParams& p)
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
bool runCpuSplitCheck(const keenly::FieldParams& p)
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
int runCheckAll()
{
    bool allOk = true;
    for (int m = 0; m < fieldModeCount(); ++m) {
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
        std::printf("mode %-6s ", fieldModeName(m));
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

// Icon gate: same iconPixel on CPU and GPU, bit-exact like the field gate.
bool runIconCheck(const keenly::IconParams& p)
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

bool runIconCpuSplitCheck(const keenly::IconParams& p)
{
    const int w = 128, h = 128;
    const size_t n = size_t(w) * size_t(h) * 4;
    std::vector<float> serial(n), threaded(n);

    keenly::renderIconCpu(serial.data(), w, h, p);
    keenly::renderIconCpuParallel(threaded.data(), w, h, p, 7);

    const DiffStats d = compareArrays(threaded.data(), serial.data(), n);
    std::printf("check  : relL2 %.3e  maxAbs %.3e  (icon CPU split, expect 0)\n",
                d.relL2, d.maxAbs);
    return d.maxAbs == 0.0;
}

bool runIconBackendCheck(bool useCpu, const keenly::IconParams& p)
{
    return useCpu ? runIconCpuSplitCheck(p) : runIconCheck(p);
}
