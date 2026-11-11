#pragma once
// Comparison helper for the verify gate: GPU output vs CPU reference.
//
// ASCII-ONLY FILE ON PURPOSE (see cuda_utils.h for the reason).
// relL2 = ||a - b|| / ||a||, maxAbs = largest absolute difference.

#include <cmath>
#include <cstddef>

struct DiffStats
{
    double relL2;
    double maxAbs;
};

inline DiffStats compareArrays(const float* a, const float* b, size_t count)
{
    double num = 0.0, den = 0.0, maxAbs = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(a[i]) * double(a[i]);
        const double ad = d < 0.0 ? -d : d;
        if (ad > maxAbs) maxAbs = ad;
    }
    DiffStats s;
    s.relL2 = std::sqrt(num) / (std::sqrt(den) + 1e-30);
    s.maxAbs = maxAbs;
    return s;
}