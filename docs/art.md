# src/art —— CUDA 生成艺术模块

用同一套 GPU 纪律（CPU 参考校验、确定性、门禁）做视觉艺术：胶片颗粒、
磨砂质感、有机流体感的生成背景，出电影画幅（2.39:1）和 A1 海报。

---

## 管线总览

```
fBm 梯度噪声 (5 octave)
  → 二级域扭曲 (two-level domain warp, after iq)     ← 有机流体感的来源
  → 调色板 (5 stop 感知空间色带) + 浮雕光照            ← 高度场梯度 → 伪法线 → 点光
  → 后期链 (全部线性光):
      半分辨率 bright pass → 9-tap 高斯 → halation 红晕 + bloom
      → 胶片颗粒 (聚团 + 4L(1-L) 中间调响应)
      → 椭圆暗角 → ACES tonemap
  → sRGB 编码 + TPDF dither → 8/16-bit PNG
```

一切像素值都是 `(x, y, seed)` 的纯函数：同 seed 复现、CPU/GPU 一致、
未来分块渲染无缝（为 600 DPI A1 留的口子）。

## 文件结构

```
src/art/
  art_noise.cuh      整数哈希、8 方向梯度噪声、fBm、ridged fBm
  art_pixel.cuh      调色板、域扭曲、浮雕光照（GPU/CPU 共享的像素着色）
  art_render.h/.cu   field kernel + post 链（bright/blur/compose）
  art_ref.cu         CPU 参考（串行遍历，同 pixel shader）
  png_writer.h       零依赖 PNG 写出（RGB 8/16-bit，stored DEFLATE）
  render_art.cu      CLI + 门禁 + preset
```

## 构建与运行

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_direct.ps1   # 全量
# 或 CMake / VS2026（见 README）

# 电影画幅（2.39:1，16-bit）
.\bin\render_art.exe --width 2560 --height 1072 --preset cinema --out out\film_frame.png

# A1 比例海报（16-bit 预览）
.\bin\render_art.exe --width 2000 --height 2832 --preset dusk --out out\a1_poster.png

# A1 300 DPI 印刷母版（594x841mm = 7014x9933 px，约 200MB）
.\bin\render_art.exe --width 7014 --height 9933 --depth 8 --preset dusk --out out\a1_poster_300dpi.png

.\bin\render_art.exe --help        # 全部旋钮
```

## 三个 preset

| preset | 色带 | 颗粒 | 用途 |
|---|---|---|---|
| `cinema` | 青黑阴影 → 琥珀高光（teal-orange） | 聚团 35mm 颗粒（scale 2.8） | 电影画幅背景 |
| `dusk` | 靛蓝 → 品红 → 珊瑚 | 细磨砂颗粒（scale 1.3） | 海报 |
| `mono` | 灰阶 | 中等 | 质感基底 / 印刷 |

CLI 显式参数优先于 preset 默认值（`render_art.cu` 用 set 位图跟踪）。

## 质感旋钮

| 旋钮 | 作用 | 调大 = |
|---|---|---|
| `--grain` | 颗粒振幅（线性光） | 更重的颗粒感 |
| `--grain-scale` | 颗粒聚团尺寸（像素） | 更粗的银盐团（2.5~3 = 35mm；1.2 = 磨砂） |
| `--halation` | 高光红晕 | 胶片高光溢出（电影感的关键） |
| `--bloom` | 中性泛光 | 更柔的高光 |
| `--vignette` | 暗角 | 更聚焦的画面 |
| `--warp` / `--view` | 域扭曲强度 / 图案尺度 | 更流体 / 更宏观的结构 |
| `--relief` | 浮雕光照强度 | 更强的立体质感（磨砂蚀刻感） |

## 验证门禁（和 bench_matmul 同一哲学）

每次渲染前自动跑 `check`：同一个 pixel shader 在 CPU（串行）和 GPU（并行）
各跑一遍，relL2 必须 < **1e-4**，否则 exit 1。

- 实测舍入底噪：**1.4e-5**（fast-math + FMA 收缩）；关掉 fast-math 是 6e-6，
  但项目统一用 fast-math，且真实 bug 量级在 1e-2 以上，1e-4 预算有 ~100x 检出余量。
- post 链（颗粒/模糊/暗角）不在数值门禁内——它们的效果用统计验证：
  高频残差（颗粒）、暗角 A/B（边角 ×0.85、中心 ×1.0）、行内唯一值数（色带）。

## 实测（RTX 2080 SUPER）

| 输出 | 尺寸 | field | post | 备注 |
|---|---|---|---|---|
| 电影画幅 | 2560×1072 | 16.0 ms | 1.3 ms | 16-bit PNG 16MB |
| A1 海报预览 | 2000×2832 | 24.4 ms | 2.2 ms | 16-bit PNG 32MB |
| A1 300 DPI | 7014×9933 | 196 ms | 23 ms | **70MP 共 219ms**，8-bit PNG 200MB |

渲染不是瓶颈，PNG 写盘是。色彩统计：cinema 均值 RGB (0.25, 0.41, 0.43)
（青调主导）、dusk (0.52, 0.23, 0.48)（品红/靛蓝），颗粒高频残差 0.04–0.09。

## 显存现实 & 分块渲染（todo）

| 输出 | 像素 | RGB fp32 单 buffer |
|---|---|---|
| 2.39:1 电影画幅 | 2.7M | 32 MB |
| A1 @300 DPI | 70M | 836 MB |
| A1 @600 DPI | 279M | **3.35 GB**（ping-pong ×3 就超 8GB） |

600 DPI 需要 tile + halo 分块渲染（conv v1 的 shared-memory halo 的图像版）。
因为像素是 `(x, y, seed)` 纯函数，field/颗粒/暗角天然可分块；只有 halation
的高斯模糊需要 halo 交换——留作下一个实验。

## 已知取舍 & 未来工作

- [ ] PNG 用 stored DEFLATE（零依赖、零压缩），文件偏大；换 fixed-Huffman 或 zlib 可缩 2-5x
- [ ] tile + halo 分块渲染，打通 A1 600 DPI
- [ ] 蓝噪声 void-and-cluster 生成 + 抖动/半调印刷质感
- [ ] curl-noise 粒子平流（水墨/风场线条）
- [ ] Gray-Scott 反应扩散（珊瑚/斑纹）、stable fluids（烟雾）
- [ ] 动画：逐帧换 seed 让颗粒「沸腾」，帧序列 → ffmpeg
- [ ] halation 目前是半分辨率最近邻回采样，可换双线性