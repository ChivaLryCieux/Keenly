#pragma once
#include "render.h"

bool runPostCheck();
bool runCheck(const keenly::FieldParams& p);
bool runCpuSplitCheck(const keenly::FieldParams& p);
int runCheckAll();
bool runIconCheck(const keenly::IconParams& p);
bool runIconCpuSplitCheck(const keenly::IconParams& p);
bool runIconBackendCheck(bool useCpu, const keenly::IconParams& p);
