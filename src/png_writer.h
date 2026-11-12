#pragma once
// Minimal dependency-free PNG writer for the art pipeline.
//
// ASCII-ONLY FILE ON PURPOSE (see src/cuda_utils.h for the reason).
//
// RGB or RGBA, 8-bit or 16-bit per channel. The IDAT zlib stream uses
// uncompressed ("stored") DEFLATE blocks: valid zlib, decodes everywhere,
// zero external dependencies, zero compression. So a 16-bit A1 poster is a
// big file on disk. Real compression is listed as future work in
// docs/pipeline.md.
//
// STREAMING: scanlines are converted row by row into a 64KB stored-block
// staging buffer and flushed straight to the file. The writer never holds
// the raw scanline stream or a second full-frame IDAT copy - on the 70MP A1
// export that used to be ~800MB of transient RAM on top of the pixel buffer.
// Output bytes are IDENTICAL to the old build-the-whole-zlib-vector writer:
// same 65535-byte block boundaries (computed from the known raw stream
// length up front), same CRC32 and adler32 values.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace keenly
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

// Incremental adler32 with the same 5552-byte fold schedule as a one-shot
// pass over the whole stream (folding a/b mod 65521 never changes the
// result), so streamed output carries the exact bytes a buffer-built zlib
// stream would. Fold every 5552 bytes: the longest run that cannot overflow
// a uint32 accumulator.
struct Adler32
{
    uint32_t a = 1u, b = 0u;
    int pending = 0;  // bytes since last fold (0..5551)

    void update(const uint8_t* data, size_t n)
    {
        size_t off = 0;
        while (off < n) {
            size_t take = n - off;
            const size_t room = size_t(5552 - pending);
            if (take > room) take = room;
            for (size_t i = 0; i < take; ++i) {
                a += data[off + i];
                b += a;
            }
            pending += int(take);
            off += take;
            if (pending == 5552) {
                a %= 65521u;
                b %= 65521u;
                pending = 0;
            }
        }
    }

    uint32_t value() const
    {
        return ((b % 65521u) << 16) | (a % 65521u);
    }
};

inline void putBe32(std::vector<uint8_t>& out, uint32_t v)
{
    out.push_back(uint8_t(v >> 24));
    out.push_back(uint8_t(v >> 16));
    out.push_back(uint8_t(v >> 8));
    out.push_back(uint8_t(v));
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

// pixels: RGB(A) interleaved, row-major, top row first.
//   bitDepth 8  -> uint8_t samples
//   bitDepth 16 -> uint16_t samples in host order (PNG stores big-endian,
//                  so byteswapped while writing)
//   channels 3  -> RGB  (color type 2)
//   channels 4  -> RGBA (color type 6, straight alpha)
// Returns false on I/O failure.
inline bool writePng(const std::string& path, int width, int height, int bitDepth,
                     const void* pixels, int channels = 3)
{
    using namespace png_detail;
    if (width <= 0 || height <= 0 || pixels == nullptr) return false;
    if (bitDepth != 8 && bitDepth != 16) return false;
    if (channels != 3 && channels != 4) return false;

    const size_t rowBytes = size_t(width) * channels * (bitDepth / 8);
    const size_t rowSpan = rowBytes + 1;  // + filter byte
    const size_t totalN = size_t(height) * rowSpan;

    // Stored DEFLATE: max 65535 payload bytes per block, 5 header bytes each.
    // Block boundaries follow the raw stream exactly, so computing the IDAT
    // length up front lets the chunk header go out before the pixels do.
    const size_t nBlocks = (totalN + 65534) / 65535;
    const uint32_t idatLen = uint32_t(2 + nBlocks * 5 + totalN + 4);

    std::FILE* f = nullptr;
    if (fopen_s(&f, path.c_str(), "wb") != 0 || f == nullptr) return false;

    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::fwrite(sig, 1, 8, f);

    std::vector<uint8_t> ihdr;
    putBe32(ihdr, uint32_t(width));
    putBe32(ihdr, uint32_t(height));
    ihdr.push_back(uint8_t(bitDepth));  // bit depth per channel
    ihdr.push_back(channels == 4 ? uint8_t(6) : uint8_t(2));  // 6 = RGBA, 2 = RGB
    ihdr.push_back(0);                  // compression method 0 = deflate
    ihdr.push_back(0);                  // filter method 0
    ihdr.push_back(0);                  // interlace 0 = none
    writeChunk(f, "IHDR", ihdr);

    // ---- IDAT, streamed ----------------------------------------------------
    std::vector<uint8_t> v;
    putBe32(v, idatLen);
    std::fwrite(v.data(), 1, 4, f);
    std::fwrite("IDAT", 1, 4, f);

    Crc32 crc;
    crc.update(reinterpret_cast<const uint8_t*>("IDAT"), 4);
    const uint8_t zhdr[2] = {0x78, 0x01};  // CMF: deflate, 32K window;
                                           // FLG: ((0x78 << 8) | 0x01) % 31 == 0
    std::fwrite(zhdr, 1, 2, f);
    crc.update(zhdr, 2);

    Adler32 ad;
    std::vector<uint8_t> row(rowSpan);   // filter byte + one scanline
    std::vector<uint8_t> block(65535);   // stored-block staging
    size_t filled = 0;                   // bytes staged
    size_t off = 0;                      // raw-stream bytes emitted

    auto flushBlock = [&](bool last) {
        const size_t take = filled;
        uint8_t head[5] = {
            last ? uint8_t(1) : uint8_t(0),          // BFINAL, BTYPE=00 (stored)
            uint8_t(take & 0xFF), uint8_t(take >> 8),       // LEN, little-endian
            uint8_t((~take) & 0xFF), uint8_t((~take) >> 8)  // NLEN = ~LEN
        };
        std::fwrite(head, 1, 5, f);
        crc.update(head, 5);
        std::fwrite(block.data(), 1, take, f);
        crc.update(block.data(), take);
        off += take;
        filled = 0;
    };

    for (int y = 0; y < height; ++y) {
        row[0] = 0;  // filter type 0 (None)
        if (bitDepth == 8) {
            const uint8_t* src = static_cast<const uint8_t*>(pixels) + size_t(y) * rowBytes;
            std::memcpy(row.data() + 1, src, rowBytes);
        } else {
            const uint16_t* src = static_cast<const uint16_t*>(pixels)
                                + size_t(y) * size_t(width) * channels;
            uint8_t* dst = row.data() + 1;
            for (size_t i = 0, n = size_t(width) * channels; i < n; ++i) {
                dst[2 * i]     = uint8_t(src[i] >> 8);    // PNG is big-endian
                dst[2 * i + 1] = uint8_t(src[i] & 0xFF);
            }
        }
        ad.update(row.data(), rowSpan);

        size_t rOff = 0;
        while (rOff < rowSpan) {
            const size_t take = std::min(size_t(65535) - filled, rowSpan - rOff);
            std::memcpy(block.data() + filled, row.data() + rOff, take);
            filled += take;
            rOff += take;
            if (filled == 65535)
                flushBlock(off + 65535 == totalN);  // last only if stream ends here
        }
    }
    if (filled > 0) flushBlock(true);

    std::vector<uint8_t> tail;
    putBe32(tail, ad.value());  // adler32 is big-endian in zlib
    std::fwrite(tail.data(), 1, 4, f);
    crc.update(tail.data(), 4);
    tail.clear();
    putBe32(tail, crc.value());
    std::fwrite(tail.data(), 1, 4, f);

    writeChunk(f, "IEND", {});
    return std::fclose(f) == 0;
}

}  // namespace keenly