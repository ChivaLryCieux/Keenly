#pragma once
// Minimal dependency-free PNG writer for the art pipeline.
//
// ASCII-ONLY FILE ON PURPOSE (see src/common/cuda_utils.h for the reason).
//
// RGB only, 8-bit or 16-bit per channel. The IDAT zlib stream uses
// uncompressed ("stored") DEFLATE blocks: valid zlib, decodes everywhere,
// zero external dependencies, zero compression. So a 16-bit A1 poster is a
// big file on disk. Real compression is listed as future work in docs/art.md.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace art
{
namespace png_detail
{

struct Crc32
{
    uint32_t state = 0xFFFFFFFFu;

    // Standard reflected CRC32 table (poly 0xEDB88320), built once on first
    // use. Table-driven keeps full-size poster exports (hundreds of MB of
    // scanline data) from spending seconds in a bitwise loop.
    static const uint32_t* table()
    {
        static uint32_t t[256];
        static bool ready = false;
        if (!ready) {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int k = 0; k < 8; ++k)
                    c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
                t[i] = c;
            }
            ready = true;
        }
        return t;
    }

    void update(const uint8_t* data, size_t n)
    {
        const uint32_t* t = table();
        uint32_t c = state;
        for (size_t i = 0; i < n; ++i)
            c = t[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
        state = c;
    }

    uint32_t value() const { return state ^ 0xFFFFFFFFu; }
};

inline uint32_t adler32(const uint8_t* data, size_t n)
{
    uint32_t a = 1u, b = 0u;
    size_t off = 0;
    while (off < n) {
        // Fold modulo 65521 every 5552 bytes: the longest run that cannot
        // overflow a uint32 accumulator.
        size_t take = n - off;
        if (take > 5552) take = 5552;
        for (size_t i = 0; i < take; ++i) {
            a += data[off + i];
            b += a;
        }
        a %= 65521u;
        b %= 65521u;
        off += take;
    }
    return (b << 16) | a;
}

inline void putBe32(std::vector<uint8_t>& out, uint32_t v)
{
    out.push_back(uint8_t(v >> 24));
    out.push_back(uint8_t(v >> 16));
    out.push_back(uint8_t(v >> 8));
    out.push_back(uint8_t(v));
}

// Append `data` as a zlib stream built from stored (uncompressed) DEFLATE
// blocks. Maximum stored block payload is 65535 bytes.
inline void appendZlibStored(std::vector<uint8_t>& out, const uint8_t* data, size_t n)
{
    out.push_back(0x78);  // CMF: deflate, 32K window
    out.push_back(0x01);  // FLG: ((0x78 << 8) | 0x01) % 31 == 0, no preset dict

    size_t off = 0;
    do {
        size_t take = n - off;
        if (take > 65535) take = 65535;
        const bool last = (off + take == n);
        out.push_back(last ? uint8_t(1) : uint8_t(0));  // BFINAL, BTYPE=00 (stored)
        const uint16_t len = uint16_t(take);
        const uint16_t nlen = uint16_t(~len);
        out.push_back(uint8_t(len & 0xFF));   // LEN, little-endian
        out.push_back(uint8_t(len >> 8));
        out.push_back(uint8_t(nlen & 0xFF));  // NLEN = ~LEN
        out.push_back(uint8_t(nlen >> 8));
        out.insert(out.end(), data + off, data + off + take);
        off += take;
    } while (off < n);

    putBe32(out, adler32(data, n));  // adler32 is big-endian in zlib
}

inline void writeChunk(std::FILE* f, const char type[4], const std::vector<uint8_t>& data)
{
    std::vector<uint8_t> hdr;
    putBe32(hdr, uint32_t(data.size()));
    std::fwrite(hdr.data(), 1, 4, f);
    std::fwrite(type, 1, 4, f);
    std::fwrite(data.data(), 1, data.size(), f);

    Crc32 crc;
    crc.update(reinterpret_cast<const uint8_t*>(type), 4);
    crc.update(data.data(), data.size());
    std::vector<uint8_t> tail;
    putBe32(tail, crc.value());
    std::fwrite(tail.data(), 1, 4, f);
}

}  // namespace png_detail

// pixels: RGB interleaved, row-major, top row first.
//   bitDepth 8  -> uint8_t samples
//   bitDepth 16 -> uint16_t samples in host order (PNG stores big-endian,
//                  so byteswapped while writing)
// Returns false on I/O failure.
inline bool writePng(const std::string& path, int width, int height, int bitDepth,
                     const void* pixels)
{
    using namespace png_detail;
    if (width <= 0 || height <= 0 || pixels == nullptr) return false;
    if (bitDepth != 8 && bitDepth != 16) return false;

    const size_t samples = size_t(width) * size_t(height) * 3;
    const size_t rowBytes = size_t(width) * 3 * (bitDepth / 8);

    // Raw scanline stream: one filter byte (0 = None) before every row.
    std::vector<uint8_t> raw(size_t(height) * (rowBytes + 1));
    if (bitDepth == 8) {
        const uint8_t* src = static_cast<const uint8_t*>(pixels);
        for (int y = 0; y < height; ++y) {
            uint8_t* dst = raw.data() + size_t(y) * (rowBytes + 1);
            dst[0] = 0;
            std::memcpy(dst + 1, src + size_t(y) * rowBytes, rowBytes);
        }
    } else {
        const uint16_t* src = static_cast<const uint16_t*>(pixels);
        for (int y = 0; y < height; ++y) {
            uint8_t* dst = raw.data() + size_t(y) * (rowBytes + 1);
            dst[0] = 0;
            const uint16_t* row = src + size_t(y) * size_t(width) * 3;
            for (size_t i = 0, n = size_t(width) * 3; i < n; ++i) {
                dst[1 + 2 * i] = uint8_t(row[i] >> 8);
                dst[2 + 2 * i] = uint8_t(row[i] & 0xFF);
            }
        }
    }
    (void)samples;

    std::vector<uint8_t> z;
    z.reserve(raw.size() + (raw.size() / 65535 + 1) * 5 + 16);
    appendZlibStored(z, raw.data(), raw.size());

    std::FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "wb") != 0 || f == nullptr) return false;

    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::fwrite(sig, 1, 8, f);

    std::vector<uint8_t> ihdr;
    putBe32(ihdr, uint32_t(width));
    putBe32(ihdr, uint32_t(height));
    ihdr.push_back(uint8_t(bitDepth));  // bit depth per channel
    ihdr.push_back(2);                  // color type 2 = truecolor RGB
    ihdr.push_back(0);                  // compression method 0 = deflate
    ihdr.push_back(0);                  // filter method 0
    ihdr.push_back(0);                  // interlace 0 = none
    writeChunk(f, "IHDR", ihdr);
    writeChunk(f, "IDAT", z);
    writeChunk(f, "IEND", {});

    return std::fclose(f) == 0;
}

}  // namespace art