#pragma once
#include <string>

int cpuThreadCount(int requested);
bool writeImage(const std::string& path, int w, int h, int depth,
                const float* linear);
bool writeImageRgba(const std::string& path, int w, int h,
                    const float* linear);
