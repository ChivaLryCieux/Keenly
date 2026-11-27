#include "manifest.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace
{
std::string trim(const std::string& s)
{
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

std::vector<std::string> splitCsv(const std::string& line)
{
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(line);
    while (std::getline(ss, cur, ',')) out.push_back(trim(cur));
    return out;
}

bool parseInt(const std::string& s, int& out)
{
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (errno || end == s.c_str() || *end != '\0') return false;
    out = int(v);
    return true;
}

bool parseUInt(const std::string& s, unsigned& out)
{
    if (s.empty() || s[0] == '-') return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long v = std::strtoul(s.c_str(), &end, 10);
    if (errno || end == s.c_str() || *end != '\0') return false;
    out = unsigned(v);
    return true;
}

bool parseFloat(const std::string& s, float& out)
{
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const float v = std::strtof(s.c_str(), &end);
    if (errno || end == s.c_str() || *end != '\0') return false;
    out = v;
    return true;
}

bool validFamily(const std::string& s)
{
    return s == "blue" || s == "red" || s == "gold" || s == "purple";
}
}  // namespace

std::string canonicalIconSpec(const IconAssetSpec& spec)
{
    uint32_t radiusBits = 0;
    std::memcpy(&radiusBits, &spec.radius, sizeof(radiusBits));
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "keenly.icon/v1|family=%s|glyph=%s|tier=%d|variant=%d|seed=%u|size=%d|mask=%s|radius=%08x",
                  spec.family.c_str(), spec.glyph.c_str(), spec.tier, spec.variant,
                  spec.seed, spec.size, spec.mask.c_str(), radiusBits);
    return std::string(buf);
}

bool loadIconManifest(const std::string& path,
                      std::vector<IconAssetSpec>& rows,
                      std::string& error)
{
    std::ifstream in(path);
    if (!in) {
        error = "cannot open manifest: " + path;
        return false;
    }

    std::string line;
    int lineNo = 0;
    bool sawSchema = false;
    bool sawHeader = false;
    std::set<std::string> keys;
    rows.clear();

    while (std::getline(in, line)) {
        ++lineNo;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::string t = trim(line);
        if (t.empty()) continue;
        if (!sawSchema) {
            if (t != std::string("#schema: ") + kIconManifestSchema) {
                error = "line " + std::to_string(lineNo) +
                        ": first non-empty line must be #schema: " + kIconManifestSchema;
                return false;
            }
            sawSchema = true;
            continue;
        }
        if (!t.empty() && t[0] == '#') continue;
        if (!sawHeader) {
            if (t != "family,glyph,tier,variant,seed,size,mask,radius") {
                error = "line " + std::to_string(lineNo) +
                        ": expected header family,glyph,tier,variant,seed,size,mask,radius";
                return false;
            }
            sawHeader = true;
            continue;
        }

        const std::vector<std::string> f = splitCsv(t);
        if (f.size() != 8) {
            error = "line " + std::to_string(lineNo) + ": expected 8 columns";
            return false;
        }
        IconAssetSpec r;
        r.family = f[0];
        r.glyph = f[1];
        r.mask = f[6];
        if (!parseInt(f[2], r.tier) || !parseInt(f[3], r.variant) ||
            !parseUInt(f[4], r.seed) || !parseInt(f[5], r.size) ||
            !parseFloat(f[7], r.radius)) {
            error = "line " + std::to_string(lineNo) + ": numeric parse failure";
            return false;
        }
        if (!validFamily(r.family) || r.glyph.empty() ||
            r.tier < 1 || r.tier > 4 || r.variant < 0 ||
            r.size < 16 || r.size > 1024 ||
            (r.mask != "round" && r.mask != "rect") ||
            r.radius < 0.0f || r.radius > 0.5f) {
            error = "line " + std::to_string(lineNo) + ": value outside schema bounds";
            return false;
        }
        const std::string key = canonicalIconSpec(r);
        if (!keys.insert(key).second) {
            error = "line " + std::to_string(lineNo) + ": duplicate asset spec";
            return false;
        }
        rows.push_back(r);
    }

    if (!sawSchema) error = "missing schema directive";
    else if (!sawHeader) error = "missing manifest header";
    else if (rows.empty()) error = "manifest contains no assets";
    return error.empty();
}
