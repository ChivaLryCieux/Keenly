#pragma once
#include "app_types.h"
#include "render.h"

int runGrid(const Args& args);
int runIcon(const Args& args);
int runIconsDemo(const Args& args);
int runRainbowDemo(const Args& args);
int runGlowDemo(const Args& args);
int runIconsManifest(const Args& args);
int runStillCpu(const Args& args, const keenly::FieldParams& field,
                const keenly::PostParams& postIn);
int runStillGpu(const Args& args, const keenly::FieldParams& field,
                const keenly::PostParams& postIn);
