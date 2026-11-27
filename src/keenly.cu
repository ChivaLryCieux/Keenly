#include <cstdio>

#include <cuda_runtime.h>

#include "app_types.h"
#include "catalog.h"
#include "checks.h"
#include "cli.h"
#include "cuda_utils.h"
#include "jobs.h"

static void prepareStillArgs(Args& args)
{
    applyPreset(args.field, args.postp, args.preset, args.postSet);
    if (args.film >= 0.0f) {
        args.postp.grain = 0.055f * args.film;
        args.postp.grainScale = 2.4f;
    }
    std::printf("render : %dx%d  preset %s  mode %s  seed %u  view %.2f  warp %.2f  octaves %d\n",
                args.width, args.height, args.preset.c_str(),
                fieldModeName(args.field.mode), args.field.seed,
                args.field.view, args.field.warp, args.field.octaves);
}

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage();
        return 1;
    }

    if (args.listModes) {
        for (int m = 0; m < fieldModeCount(); ++m)
            std::printf("%d  %s\n", m, fieldModeName(m));
        return 0;
    }

    if (args.cpu) {
        if (args.checkAll) {
            std::fprintf(stderr, "--check-all requires the CUDA backend; run it without --cpu\n");
            return 1;
        }
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

        prepareStillArgs(args);
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

    prepareStillArgs(args);

    if (args.check && !runCheck(args.field)) {
        std::fprintf(stderr, "CHECK FAILED: GPU disagrees with the CPU reference\n");
        return 1;
    }

    return runStillGpu(args, args.field, args.postp);
}
