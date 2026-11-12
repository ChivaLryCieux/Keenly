# -*- coding: utf-8 -*-
# One-shot rename/replacement for the Keenly docs restructure.
# Reads docs/art.md (UTF-8), writes docs/pipeline.md (UTF-8).
import io

src = r'C:\Users\LRY\Desktop\BASE\On-CUDA\docs\art.md'
dst = r'C:\Users\LRY\Desktop\BASE\On-CUDA\docs\pipeline.md'

with io.open(src, 'r', encoding='utf-8') as f:
    t = f.read()

t = t.replace('# src/art —— CUDA 生成艺术模块', '# Keenly 渲染管线')
t = t.replace('experiments.md E9/E10/E11', 'experiments.md 实验 1/2/3')
t = t.replace('E9/E10/E11', '实验 1/2/3')
t = t.replace('E12', '实验 4')
t = t.replace('render_art.exe', 'keenly.exe')
t = t.replace('render_art.cu', 'keenly.cu')
t = t.replace('render_art --check', 'keenly --check')
t = t.replace('art_noise/art_pixel', 'src/noise.cuh / src/pixel.cuh')
t = t.replace('art.md', 'pipeline.md')

# Replace the file-structure listing line by line (robust to whitespace).
lines = t.split('\n')
start = next(i for i, s in enumerate(lines) if s.strip() == 'src/art/')
end = next(i for i, s in enumerate(lines) if 'keenly.cu' in s and 'CLI' in s)
new_block = [
    'src/',
    '  keenly.cu      CLI + 位级门禁 + 九宫格 showcase + PNG 写出',
    '  noise.cuh      整数哈希、8 方向梯度噪声、fBm、ridged fBm、Worley、powi',
    '  pixel.cuh      调色板、分层构图、7 个场生成器（GPU/CPU 共享的像素着色）',
    '  render.h/.cu   field kernel + post 链（bright/blur/compose）',
    '  reference.cu   CPU 参考（串行遍历，同 pixel shader）',
    '  png_writer.h   零依赖 PNG 写出（RGB 8/16-bit，stored DEFLATE）',
    '  compare.h      DiffStats / compareArrays（门禁比较）',
    '  cuda_utils.h   CUDA_CHECK + GpuTimer',
]
lines[start:end + 1] = new_block
t = '\n'.join(lines)

with io.open(dst, 'w', encoding='utf-8') as f:
    f.write(t)

print('wrote', dst, len(t), 'chars')