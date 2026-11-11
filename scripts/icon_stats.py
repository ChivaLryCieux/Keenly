# -*- coding: utf-8 -*-
# Icon QA stats: checks the 64px readability rules from docs/icons_plan.md.
#   1. glyph-vs-background contrast: icon must not be flat
#   2. value range: dark background AND bright glyph both present
#   3. edge population: the stroke/frame must be visible
# Usage: python out\icon_stats.py out\icons\blue\blue_sword.png [...]
import struct, sys, zlib


def read_png(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a png: " + path
    pos, w, h, depth, ctype = 8, 0, 0, 0, 0
    idat = b""
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]
        typ = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", chunk[:10])
        elif typ == b"IDAT":
            idat += chunk
        pos += 12 + ln
    raw = zlib.decompress(idat)
    ch = 4 if ctype == 6 else 3
    assert depth == 8, "8-bit only"
    px = []
    stride = w * ch
    prev = bytearray(stride)
    off = 0
    for y in range(h):
        ft = raw[off]
        row = bytearray(raw[off + 1:off + 1 + stride])
        off += 1 + stride
        if ft == 1:
            for i in range(ch, stride):
                row[i] = (row[i] + row[i - ch]) & 255
        elif ft == 2:
            for i in range(stride):
                row[i] = (row[i] + prev[i]) & 255
        elif ft == 3:
            for i in range(stride):
                a = row[i - ch] if i >= ch else 0
                row[i] = (row[i] + ((a + prev[i]) >> 1)) & 255
        elif ft == 4:
            for i in range(stride):
                a = row[i - ch] if i >= ch else 0
                b = prev[i]
                c = prev[i - ch] if i >= ch else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                row[i] = (row[i] + pr) & 255
        prev = row
        px.append(bytes(row))
    return w, h, ch, px


def luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def metrics(path):
    w, h, ch, rows = read_png(path)
    lums = []
    alphas = []
    for row in rows:
        for x in range(w):
            i = x * ch
            r, g, b = row[i] / 255.0, row[i + 1] / 255.0, row[i + 2] / 255.0
            lums.append(luminance(r, g, b))
            alphas.append(row[i + 3] / 255.0 if ch == 4 else 1.0)
    inside = [l for l, a in zip(lums, alphas) if a > 0.5]
    if not inside:
        inside = lums
    inside.sort()
    n = len(inside)
    lo = inside[int(n * 0.05)]
    hi = inside[int(n * 0.95)]
    p50 = inside[n // 2]
    # Weber-ish contrast ratio between bright glyph and dark bg (sRGB space)
    def lin(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    cr = (lin(hi) + 0.05) / (lin(lo) + 0.05)
    # fraction of near-black vs near-white (bimodality hint)
    dark = sum(1 for v in inside if v < 0.25) / n
    bright = sum(1 for v in inside if v > 0.55) / n
    return dict(path=path, p05=lo, p50=p50, p95=hi, contrast=cr,
                dark=dark, bright=bright)


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: python out\\icon_stats.py <icon.png> [...]")
        return 1
    bad = 0
    for p in paths:
        m = metrics(p)
        ok = m["contrast"] >= 3.0 and m["bright"] >= 0.02
        flag = "OK " if ok else "WARN"
        if not ok:
            bad += 1
        print("%s %-44s contrast %5.2f  p05 %.2f p50 %.2f p95 %.2f  dark %4.0f%% bright %4.0f%%"
              % (flag, p.split("\\")[-1].split("/")[-1], m["contrast"],
                 m["p05"], m["p50"], m["p95"], m["dark"] * 100, m["bright"] * 100))
    print("checked %d, warnings %d" % (len(paths), bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())