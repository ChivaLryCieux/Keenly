#pragma once
#include <string>
#include <vector>

constexpr const char* kIconManifestSchema = "keenly.icon-manifest/v1";

struct IconAssetSpec
{
    std::string family;
    std::string glyph;
    int tier = 2;
    int variant = 0;
    unsigned seed = 7u;
    int size = 128;
    std::string mask = "round";
    float radius = 0.16f;
};

bool loadIconManifest(const std::string& path,
                      std::vector<IconAssetSpec>& rows,
                      std::string& error);
std::string canonicalIconSpec(const IconAssetSpec& spec);
