# Keenly

**CUDA 生成艺术工作室** —— 程序化生成电影画幅与 A1 海报尺寸的背景：
胶片颗粒、磨砂质感、有机流动、液态金属；并为战棋游戏量产**技能图标**
（4 色系 × 18 符号 × 4 级边框，RGBA，128/64px）。一台可复现的「审美机器」。

核心设计一句话：**每个像素都是 `(x, y, seed)` 的纯函数**——
GPU 与 CPU 参考**逐位一致**（maxAbs = 0），同一个 seed 无限复现，
**换 seed 就是换构图，风格不变**。

![showcase](docs/showcase.png)

---

## 快速开始

```powershell
# 构建（任选其一）
powershell -ExecutionPolicy Bypass -File scripts\build_direct.ps1   # 直接 nvcc
powershell -ExecutionPolicy Bypass -File scripts\build.ps1          # CMake + Ninja

# 电影画幅（2.39:1，16-bit PNG）
.\bin\keenly.exe --preset cinema --width 2560 --height 1072 --out out\film_frame.png

# A1 海报（594x841mm 比例）
.\bin\keenly.exe --preset dusk --width 2000 --height 2832 --out out\a1_poster.png

# 九宫格：一次看九种构图/风格组合
.\bin\keenly.exe --grid out\showcase_grid.png --tile 800

# 技能图标（RGBA；--family blue|red|gold|purple、--glyph、--tier 1..4）
.\bin\keenly.exe --icon --family gold --glyph crown --tier 3 --width 128 --height 128 --out out\icon.png

# 图标整批：4 色系 x 18 符号 + 64px 缩略 + contact sheet
.\bin\keenly.exe --icons-demo out\icons

# 回归门禁（改动后跑这个）
powershell -ExecutionPolicy Bypass -File scripts\verify.ps1
```

A1 300 DPI 印刷母版：`--width 7014 --height 9933`（7000 万像素，全链约
1.3 秒：GPU 渲染 0.4s + 编码/写盘 0.9s）。

---

## 架构

```
src/
  keenly.cu       CLI + 位级门禁 + 九宫格/图标 demo + PNG 写出
  noise.cuh       整数哈希、8 方向梯度噪声、fBm、ridged fBm、Worley、powi
  turing.cuh      Gray-Scott 反应扩散：Jacobi 模拟 + 网格采样（可平铺）
  pixel.cuh       调色板、分层构图、8 个场生成器（GPU/CPU 共享的像素着色）
  sdf.cuh         2D SDF 图元 + 18 个技能符号库（纯多项式 + IEEE sqrt）
  icon.cuh        图标合成：流体背景 + 符号 + 分级边框 + 圆角遮罩 -> RGBA
  post.cuh        每像素后期尾链（split-tone/颗粒/暗角/ACES，静帧与图标共用）
  render.h/.cu    field/icon kernel + post 链（bright/blur/compose）+ RD 模拟
  reference.cu    CPU 参考（串行遍历，同 pixel/icon shader）
  png_writer.h    零依赖 PNG 写出（RGB/RGBA，8/16-bit，stored DEFLATE）
  compare.h       DiffStats / compareArrays（门禁比较）
  cuda_utils.h    CUDA_CHECK + GpuTimer
docs/
  pipeline.md     渲染管线技术文档（生成器/调色/后期/门禁原理）
  icons_plan.md   技能图标量产方案（尺寸/色系/配额/分阶段）
  experiments.md  实验记录：改了什么 → 指标怎么变 → 为什么
scripts/
  build.ps1 / build_direct.ps1   两种构建路径
  verify.ps1                      回归门禁（构建 + 位级门禁 + 冒烟渲染）
  icon_stats.py                   图标质检（对比度/值域/64px 可读性）
tools/
  grid_stats.py / comp_stats.py / validate_rd.py   分析脚本（见 tools/README.md）
```

**数据流**（详见 [docs/pipeline.md](docs/pipeline.md)）：

```
噪声 (5 octave fBm，每 octave 旋转 14°)
  → 二级域扭曲 (two-level domain warp)          ← 有机流动感
  → 分层构图（宏观明暗布局 + 色相漂移 + 细节层）   ← 「画」与「噪声贴图」的区别
  → 调色板 (6 stop) + 双尺度浮雕光照
  → 后期链 (线性光):
      halation 红晕 + bloom → split-tone 青影橙光 → vividness
      → 胶片颗粒 (双尺度聚团 + 三层色差) → 暗角 → ACES
  → sRGB + TPDF dither → 8/16-bit PNG
```

**分层构图**是 v2 的灵魂：宏观形体决定画面明暗布局、全画幅左上方向光、
色相随大形体漂移——`--macro 0` 关掉（平铺感）、`--macro 1` 全开（绘画感）。

---

## 生成器 × 调色板

**8 个场生成器**（`--mode`）：

| mode | 质感 | 数学 |
|---|---|---|
| `warp` | 流动烟雾 / 有机流体（默认，验证过的审美基线） | 二级域扭曲 fBm |
| `ridge` | 大理石纹 / 岩层 | 扭曲 ridged fBm |
| `bands` | 流线条纹 / 木纹 / 等高线 | warpField 三角波等值线 |
| `frost` | 磨砂玻璃 | 细 Worley 晶面 + 微闪点 |
| `facet` | 凛冽几何：晶体切面 / 碎裂 | Worley + 每面受光 + 硬闪 |
| `prism` | 凛冽几何：雪佛龙 / 菱格 | 30° 旋转坐标三角波折叠 |
| `liquid` | 光照质感：液态金属 / 油膜 | 宽场 + 宽高光 + 加性 glint |
| `turing` | 有机斑纹：珊瑚 / 迷宫 / 斑点 | Gray-Scott 反应扩散（图灵斑） |

**8 个调色板**（`--preset`）：`cinema`（青橙）、`dusk`（靛蓝-品红-珊瑚）、
`mono`（灰阶）、`ember`（火）、`ice`（冰川）、`moss`（森林）、
`gold`（神圣金）、`violet`（暗紫）。
preset 附带各自的颗粒/光晕默认值，CLI 显式参数优先。

**质感旋钮**：`--grain` / `--grain-scale`（35mm 聚团 ↔ 磨砂细颗粒）、
`--halation`（胶片高光红晕）、`--split-tone`（青影橙光）、`--saturation`、
`--macro`（构图强度）、`--relief`（浮雕）、`--warp` / `--view`（结构）。
全部清单见 `keenly --help`。

---

## 技能图标（战棋游戏）

一张图标 = **色系 × 符号 × 边框等级 × seed**，全部一个纯函数 kernel 出图
（`icon.cuh`，含背景/符号/边框/遮罩/后期，单张 ~0.1 ms）：

| 部件 | 内容 |
|---|---|
| 色系 | `blue`（冰霜）、`red`（火焰）、`gold`（神圣）、`purple`（暗影），各带一套点缀色（描边/高光/辉光/阴影） |
| 背景 | 统一**流体**（warp 有机流动；red 为 liquid 流动+高光）——Worley 系在图标尺寸下是难看的 Voronoi |
| 符号 | `src/sdf.cuh` 18 个：sword/axe/bow/shield/flame/droplet/snow/bolt/sparkle/moon/sun/eye/heart/skull/cross/rune/crown/potion |
| 边框 | `--tier 1..4`：素描边 → 双环角钉 → 菱形角饰+外发光 → 宝石嵌角（顺带做稀有度） |
| 尺寸 | 128×128 主力 + 64×64 直渲（纯函数分辨率无关，直渲比缩放锐）；`--mask round\|rect`、`--radius` 圆角三档 |

图标输出 **8-bit RGBA**（alpha = 遮罩覆盖率，1px AA）。可读性纪律：符号笔画
≥5px@128、亮描边、背景压暗、符号区颗粒减弱。量产方案（配额 160 张、manifest
驱动）见 [docs/icons_plan.md](docs/icons_plan.md)，质检脚本
`scripts\icon_stats.py`。

---

## 工程纪律

**位级门禁**。Keenly 用严格 IEEE 数学编译（`--fmad=false`，不加 fast-math），
所以 GPU 渲染与 CPU 参考逐位一致——`keenly --check-all` 对全部 8 个生成器
（含 Gray-Scott 迭代模拟）要求 `maxAbs == 0`，图标合成路径同样过门禁
（单张逐张；demo 批量按色系抽样），`scripts\verify.ps1` 每次改动后跑。这不是过度工程：
它同时保证了 seed 系列的可复现性、未来分块渲染的无缝性、印刷母版的可重渲性。
历史教训（实验 1/2/3）都写在 [docs/experiments.md](docs/experiments.md) 里——
fast-math 的误差预算模型被 bands 的 fold 增益（126 倍）击穿后，才换成零误差的编译约定。

**纯函数纪律**。噪声只用整数哈希 + 多项式 + IEEE sqrt；高光用整数幂连乘
（不用 libm `pow`）；没有三角函数、没有全局状态。反应扩散用 **Jacobi
ping-pong**（每格只读上一帧）保持同样的确定性。这是「换 seed 出系列」
和「任意尺寸/分块渲染」的根基。

**实验记录**。每个被数据推翻的假设都记在 [docs/experiments.md](docs/experiments.md)：
舍入底噪随链长增长、fold 的结构性增益、直方图形状判据（值域塌缩 vs 双峰）、
迭代模拟的种子尺度语义（实验 5/6）。

## 性能（RTX 2080 SUPER，健康时钟实测）

| 输出 | 尺寸 | field | post | write |
|---|---|---|---|---|
| 电影画幅 | 2560×1072 | ~40 ms | ~1.3 ms | ~66 ms |
| A1 300 DPI | 7014×9933（70MP） | ~395 ms | ~25 ms | ~0.9 s |
| 图灵斑海报 | 2000×2832 | ~95 ms（含 512²×6000 步模拟） | ~2.5 ms | ~136 ms |
| 技能图标 | 128×128 RGBA | **~0.1 ms**（单 kernel 含后期） | 同左 | ~1 ms |

write = sRGB 编码 + dither + 流式落盘（多线程）。渲染不是瓶颈，写出链
次之；文件体积仍是 stored DEFLATE（零依赖零压缩）。注意 GPU 降频能把
所有数字放大 10 倍以上，A/B 要热态交替采样（见实验 9）。

---

## 路线图

- [x] Gray-Scott 反应扩散（图灵斑，`--mode turing`，Jacobi 保位级门禁）
- [ ] 技能图标 Phase 2：manifest 批量产（配额表 160 张）+ 32px 剪影档
- [ ] tile + halo 分块渲染，打通 A1 600 DPI（单 buffer 3.35GB 超 8GB 显存）
- [ ] 逐帧换 seed 的动画路径（颗粒「沸腾」= 胶片感）→ ffmpeg 帧序列
- [ ] PNG fixed-Huffman 压缩，文件缩 2-5x
- [ ] 蓝噪声 void-and-cluster + 半调印刷质感
- [ ] curl-noise 粒子平流（水墨/风场线条）
- [ ] halation 双线性回采样

---

*Keenly —— 锐利地看，温柔地渲染。*