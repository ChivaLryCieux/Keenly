# Keenly 渲染管线

用同一套 GPU 纪律（CPU 参考校验、确定性、门禁）做视觉艺术：胶片颗粒、
磨砂质感、有机流体感的生成背景，出电影画幅（2.39:1）和 A1 海报。

---

## 管线总览

```
fBm 梯度噪声 (5 octave，每 octave 旋转 14°，消网格对齐伪影)
  → 二级域扭曲 (two-level domain warp, after iq)     ← 有机流体感的来源
  → 分层构图（三个尺度）:
      宏观形体 (0.42x)  → 画面明暗布局 + 色相漂移      ← 「画」和「噪声贴图」的区别
      主流场            → 流动结构
      细节层 (2.3x)     → 表面变化
  → 调色板 (6 stop 感知空间色带 + 中性过渡桥) + 双尺度浮雕光照
  → 后期链 (全部线性光):
      半分辨率 bright pass → 双程 9-tap 高斯 → halation 红晕 + bloom
      → split-tone 分离调色 (青影橙光 + 高光去饱和，真·调色机制)
      → vividness 色度扩张 (绕亮度展开，灰色内容无影响)
      → 胶片颗粒 (双尺度聚团 + 三层色差 + 4L(1-L) 中间调响应)
      → 椭圆暗角 → ACES tonemap
  → sRGB 编码 + TPDF dither → 8/16-bit PNG
```

分层构图是 v2 的核心：`--macro 0` 关掉（回到 v1 的平铺感），`--macro 1`
全开（宏观明暗布局 + 全画幅左上方向光 + 色相漂移）。

一切像素值都是 `(x, y, seed)` 的纯函数：同 seed 复现、CPU/GPU 一致、
未来分块渲染无缝（为 600 DPI A1 留的口子）。

## 文件结构

```
src/
  keenly.cu      CLI + 位级门禁 + 九宫格 showcase + PNG 写出
  noise.cuh      整数哈希、8 方向梯度噪声、fBm、ridged fBm、Worley、powi
  pixel.cuh      调色板、分层构图、7 个场生成器（GPU/CPU 共享的像素着色）
  render.h/.cu   field kernel + post 链（bright/blur/compose）
  reference.cu   CPU 参考（串行遍历，同 pixel shader）
  png_writer.h   零依赖 PNG 写出（RGB 8/16-bit，stored DEFLATE）
  compare.h      DiffStats / compareArrays（门禁比较）
  cuda_utils.h   CUDA_CHECK + GpuTimer
```

## 构建与运行

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_direct.ps1   # 全量
# 或 CMake / VS2026（见 README）

# 电影画幅（2.39:1，16-bit）
.\bin\keenly.exe --width 2560 --height 1072 --preset cinema --out out\film_frame.png

# A1 比例海报（16-bit 预览）
.\bin\keenly.exe --width 2000 --height 2832 --preset dusk --out out\a1_poster.png

# A1 300 DPI 印刷母版（594x841mm = 7014x9933 px，约 200MB）
.\bin\keenly.exe --width 7014 --height 9933 --depth 8 --preset dusk --out out\a1_poster_300dpi.png

.\bin\keenly.exe --help        # 全部旋钮
```

## 三个 preset

| preset | 色带 | 颗粒 | 用途 |
|---|---|---|---|
| `cinema` | 青黑阴影 → 琥珀高光（teal-orange） | 聚团 35mm 颗粒（scale 2.8） | 电影画幅背景 |
| `dusk` | 靛蓝 → 品红 → 珊瑚 | 细磨砂颗粒（scale 1.3） | 海报 |
| `mono` | 灰阶 | 中等 | 质感基底 / 印刷 |
| `ember` | 黑红 → 橙 → 金 | 中粗颗粒 + 强 halation | 火 / 熔岩 |
| `ice` | 深蓝黑 → 钢蓝 → 冰白 | 细颗粒 + 柔 bloom | 冰川 / 玻璃 |
| `moss` | 暗绿棕 → 苔绿 → 暖沙 | 中颗粒 | 森林 / 有机 |

CLI 显式参数优先于 preset 默认值（`keenly.cu` 用 set 位图跟踪）。

## 七个场生成器（--mode）

| mode | 数学 | 质感 |
|---|---|---|
| `warp` | 二级域扭曲 fBm | 流动的烟雾 / 有机流体（默认） |
| `ridge` | 扭曲的 ridged fBm | 大理石纹 / 岩层 |
| `bands` | warpField 的三角波等值线 | 流线条纹 / 木纹 / 等高线 |
| `frost` | 细 Worley 晶面 + 斑点 + powi-90 微闪 | **磨砂玻璃**（晶面磨砂，不是软噪声） |
| `facet` | Worley 细胞 + 每面受光 + 亮缝 | **凛冽几何**：晶体切面 / 碎裂 |
| `prism` | 30° 旋转坐标上的三角波折叠 | **凛冽几何**：雪佛龙 / 菱格栅（直方图双峰） |
| `liquid` | warpField 宽场 + powi-16 高光 + 加性 glint | **光照质感**：液态金属 / 油膜 |
| `turing` | Gray-Scott 反应扩散（图灵斑） | **有机斑纹**：珊瑚 / 迷宫 / 斑点 |

全部是 `(x, y, seed)` 纯函数、纯多项式 + IEEE sqrt 运算（Worley 距离、
整数幂高光）——CPU/GPU 逐位一致由此成立。高光的性格按模式分：
liquid 宽高光 + 白 glint、facet 硬闪、frost 微闪点。

### turing 模式：迭代模拟如何保住位级门禁

反应扩散不是 `(x, y)` 闭式函数，是**网格迭代状态**。确定性靠构造保证：

- **Jacobi 更新**（ping-pong 双缓冲）：每格只读上一帧缓冲，更新顺序不影响
  结果——如果用原地 Gauss-Seidel，线程调度会改变结果，GPU/CPU 就不可能全等；
- **同一份更新数学**（`turing.cuh` 的 `turingUpdate` 内联）：GPU kernel 和
  CPU 参考都调它，`--fmad=false` 严格 IEEE，实测 maxAbs = 0；
- **环绕边界**（wrapi）：网格可平铺，无边缘伪影；
- 初始化是 `(x, y, res, seed)` 的纯函数（hash 种子斑点），换 seed 换构图。

模拟每图跑一次（GPU ~1.5s @512²×6000 步），之后像素着色器用五次方平滑
双线性（quintic bilinear）采样 B 网格——C2 连续，浮雕着色下无双线性折痕。

Gray-Scott 参数 `--rd-f`（feed）/ `--rd-k`（kill）决定形态（Pearson 形态图）：
(0.0545, 0.062) 珊瑚生长、(0.0367, 0.0649) 分裂斑点、(0.022, 0.051) 蠕虫、
(0.038, 0.061) 空洞迷宫、(0.026, 0.051) 条纹、(0.034, 0.056) 混沌。

## 九宫格阵容（runGrid cells 表）

当前阵容：**有机电影家族的 9 个变体**（调色板 × 尺度/扭曲/对比度微调）——
第一轮风格实验的结论是「好看」来自构图与调色，不是形状语言的种类，
所以阵容回到验证过的家族里拉开参数。改 `keenly.cu` 里 `runGrid()`
的 cells 表即可换阵容（支持每格 view/warp/contrast/relief 覆盖）。

## 质感旋钮

| 旋钮 | 作用 | 调大 = |
|---|---|---|
| `--grain` | 颗粒振幅（线性光） | 更重的颗粒感 |
| `--grain-scale` | 颗粒聚团尺寸（像素） | 更粗的银盐团（2.5~3 = 35mm；1.2 = 磨砂） |
| `--halation` | 高光红晕 | 胶片高光溢出（电影感的关键） |
| `--bloom` | 中性泛光 | 更柔的高光 |
| `--vignette` | 暗角 | 更聚焦的画面 |
| `--split-tone` | 青影橙光分离调色 + 高光去饱和 | 更强的 teal-orange 调色（0 关闭） |
| `--warp` / `--view` | 域扭曲强度 / 图案尺度 | 更流体 / 更宏观的结构 |
| `--relief` | 浮雕光照强度 | 更强的立体质感（磨砂蚀刻感） |

## 系列出图与九宫格

像素是 `(x, y, seed)` 纯函数，**换 seed 就是换构图，风格不变**——同一 preset
可以出一整个系列：

```powershell
.\bin\keenly.exe --preset cinema --seed 21 --width 2560 --height 1072 --out out\cinema_seed21.png
.\bin\keenly.exe --preset cinema --seed 42 --width 2560 --height 1072 --out out\cinema_seed42.png

# 九宫格：9 种风格组合一次出图（4 生成器 x 6 调色板的展示集）
.\bin\keenly.exe --grid out\showcase_grid.png --tile 800 --depth 16
```

九宫格每格都是完整管线渲染（含各自的位级门禁），格子阵容在
`keenly.cu` 的 `runGrid()` 里。

## 验证门禁（和 bench_matmul 同一哲学，但更严）

每次渲染前自动跑 `check`：同一个 pixel shader 在 CPU（串行）和 GPU（并行）
各跑一遍，要求 **位级全等（maxAbs == 0）**，否则 exit 1。

- art 目标编译**严格 IEEE 数学**（`--fmad=false`，不加 fast-math），所以
  GPU 输出与 CPU 参考逐位一致（四个生成器实测 relL2 = 0.000e+00）。
  matmul/conv 仍用 fast-math 追速度——各取所需，flags 是 per-target 的。
- 历史教训（见 experiments.md 实验 1/2/3）：fast-math 时代门禁只能给预算
  （1e-4 → 2e-4），而 bands 的折叠结构把舍入差放大 126 倍直接击穿预算——
  换 IEEE 之后预算模型整个不需要了，位级全等是最强的正确性声明。
- post 链（颗粒/模糊/暗角/调色）不在数值门禁内——它们的效果用统计验证：
  高频残差（颗粒）、暗角 A/B（边角 ×0.85、中心 ×1.0）、行内唯一值数（色带）、
  饱和度变化（split-tone 生效验证）。

## 实测（RTX 2080 SUPER）

| 输出 | 尺寸 | field | post | 备注 |
|---|---|---|---|---|
| 电影画幅 | 2560×1072 | 36 ms | 1.5 ms | 16-bit PNG 16MB |
| A1 海报预览 | 2000×2832 | 41 ms | 2.2 ms | 16-bit PNG 32MB |
| A1 300 DPI | 7014×9933 | ~1.2 s | ~0.15 s | **70MP**，8-bit PNG 200MB |

（quality 轮后 field 从 3 次求值变 5 次 + 每 octave 旋转，field 时间约 ×2；
渲染仍不是瓶颈，PNG 写盘是。）色彩统计：cinema 均值 RGB (0.18, 0.42, 0.48)
（split-tone 后青调更深、暖高光更活，饱和度 0.35→0.38），dusk (0.51, 0.19, 0.58)。

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