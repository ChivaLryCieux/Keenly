#pragma once
#include <string>
#include "render.h"

bool glowScheme(keenly::Palette& pal, const std::string& name);
void applyPreset(keenly::FieldParams& f, keenly::PostParams& p,
                 const std::string& name, unsigned set);
void applyIconFamily(keenly::IconParams& ip, const std::string& name);
int glyphFromName(const std::string& name);
int bgModeFromName(const std::string& name);
void applyRainbowIcon(keenly::IconParams& ip, int bgMode, bool whiteFg,
                      const std::string& scheme);

int fieldModeCount();
const char* fieldModeName(int mode);
std::string fieldModeList();
int modeFromName(const char* name);
