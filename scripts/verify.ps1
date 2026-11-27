<#
.SYNOPSIS
    Keenly regression gate: build, then prove every generator is still
    bit-identical to the CPU reference.

.DESCRIPTION
    This is the regression gate. It answers two questions:

        Is every generator still BIT-EXACT against the CPU reference?
        Does the --cpu backend produce the same PNG bytes as the GPU?

    Keenly compiles strict IEEE math (--fmad=false, no fast-math) precisely so
    GPU renders match the serial CPU reference sample-for-sample (maxAbs 0).
    The gate is a hard error (exit 1) on any deviation - if it fires, a
    non-IEEE operation slipped into noise.cuh / pixel.cuh / post.cuh, or a
    post-chain bug (blur taps, half-res offset) was introduced, and
    reproducibility (tiling, seed series, print masters) is gone.

    The bit-exact comparison covers THREE things, not just the field:
      1. field shader        (per generator)
      2. post chain          (per generator + a synthetic HDR pass that
                             guarantees the bright/blur/compose stages see
                             real energy instead of a black image)
      3. cpu row-split       (thread count must not change a single float)

    A CPU/GPU PNG parity check and a tiny render smoke test follow: they
    catch launch, encode and write failures that a math-only check misses.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\verify.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\verify.ps1 -SkipBuild
#>
[CmdletBinding()]
param(
    # Skip the nvcc build step (use when you already built).
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$bin  = Join-Path $root "bin"
$exe  = Join-Path $bin "keenly.exe"

function Write-Step([string]$text) {
    Write-Host ""
    Write-Host "=== $text ===" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 0. Flag-sync guard: the bit-exact gate is only meaningful if BOTH build
#    paths compile the same strict-IEEE math. CMakeLists.txt and
#    build_direct.ps1 carry the flags in hand-edited form - catch drift here
#    before it silently turns into "the gate passes on one build path only".
# ---------------------------------------------------------------------------
Write-Step "Flag sync: CMakeLists.txt vs build_direct.ps1"
$cmakeTxt   = Get-Content -LiteralPath (Join-Path $root "CMakeLists.txt") -Raw
$directTxt  = Get-Content -LiteralPath (Join-Path $root "scripts\build_direct.ps1") -Raw

# Check CODE lines only: both files explain '--fmad=false, no fast-math' in
# comments, and matching those mentions would fire on every run.
$cmakeCode  = (($cmakeTxt  -split "`n") | Where-Object { $_ -notmatch '^\s*#' }) -join "`n"
$directCode = (($directTxt -split "`n") | Where-Object { $_ -notmatch '^\s*#' }) -join "`n"

$problems = @()
if ($cmakeCode  -notmatch '--fmad=false') { $problems += "CMakeLists.txt is missing --fmad=false" }
if ($directCode -notmatch '--fmad=false') { $problems += "build_direct.ps1 is missing --fmad=false" }
if ($cmakeCode  -match 'use_fast_math')   { $problems += "CMakeLists.txt enables --use_fast_math (breaks bit-exactness)" }
if ($directCode -match 'use_fast_math')   { $problems += "build_direct.ps1 enables --use_fast_math (breaks bit-exactness)" }
if ($cmakeCode  -notmatch 'CMAKE_CUDA_ARCHITECTURES\s+75') { $problems += "CMakeLists.txt CMAKE_CUDA_ARCHITECTURES is not 75" }
if ($directCode -notmatch 'sm_75')        { $problems += "build_direct.ps1 does not target sm_75" }
if ($cmakeCode  -notmatch 'CXX_STANDARD 17') { $problems += "CMakeLists.txt is not C++17" }
if ($directCode -notmatch 'std=c\+\+17')  { $problems += "build_direct.ps1 is not C++17" }

if ($problems.Count -gt 0) {
    $problems | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host "FLAG DRIFT: the two build paths no longer compile the same math." -ForegroundColor Red
    exit 1
}
Write-Host "flags agree: strict IEEE (--fmad=false), sm_75, C++17"

# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------
if ($SkipBuild) {
    Write-Step "Skipping build (-SkipBuild)"
} else {
    Write-Step "Building keenly"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $root "scripts\build_direct.ps1")
    if ($LASTEXITCODE -ne 0) { Write-Error "build failed with code $LASTEXITCODE"; exit 1 }
}

if (-not (Test-Path -LiteralPath $exe)) {
    Write-Error "keenly.exe not found at $exe"
    exit 1
}

# ---------------------------------------------------------------------------
# 2. Bit-exact gate on every generator
# ---------------------------------------------------------------------------
Write-Step "Verify gate: GPU vs CPU, expect maxAbs = 0 on every generator"
& $exe --check-all
if ($LASTEXITCODE -ne 0) {
    Write-Host "GATE FAILED: GPU output is not bit-identical to the CPU reference." -ForegroundColor Red
    Write-Host "A non-IEEE operation (transcendental, fast-math div, FMA fusion) has" -ForegroundColor Red
    Write-Host "slipped into src\noise.cuh / src\pixel.cuh / src\post.cuh, OR a post" -ForegroundColor Red
    Write-Host "chain bug (blur taps, half-res offset, edge clamp) was introduced." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# 3. CPU backend parity: --cpu must produce byte-identical PNGs to the GPU.
#    This is the "same seed, same image, anywhere" claim made executable: it
#    covers field + post + encode end to end, not just the float buffers the
#    gate compares.
# ---------------------------------------------------------------------------
Write-Step "CPU backend parity (GPU PNG vs --cpu PNG, byte-identical)"
$parityDir = Join-Path $root "out\_parity"
New-Item -ItemType Directory -Path $parityDir -Force | Out-Null
$gpuPng = Join-Path $parityDir "gpu.png"
$cpuPng = Join-Path $parityDir "cpu.png"

& $exe --no-check --width 512 --height 320 --depth 16 --preset dusk --seed 42 `
       --out $gpuPng | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "GPU parity render failed"; exit 1 }

& $exe --cpu --width 512 --height 320 --depth 16 --preset dusk --seed 42 `
       --out $cpuPng | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "CPU parity render failed"; exit 1 }

$gpuHash = (Get-FileHash -LiteralPath $gpuPng -Algorithm SHA256).Hash
$cpuHash = (Get-FileHash -LiteralPath $cpuPng -Algorithm SHA256).Hash
if ($gpuHash -ne $cpuHash) {
    Write-Host "PARITY FAILED: --cpu output differs from the GPU render." -ForegroundColor Red
    Write-Host "  gpu: $gpuHash" -ForegroundColor Red
    Write-Host "  cpu: $cpuHash" -ForegroundColor Red
    Write-Host "The CPU backend is a rendering path, not a reference: it must match." -ForegroundColor Red
    exit 1
}
Remove-Item -LiteralPath $parityDir -Recurse -Force
Write-Host "parity ok: $($gpuHash.Substring(0,16))... (GPU == --cpu, 16-bit PNG)"

# ---------------------------------------------------------------------------
# 4. Render smoke test (launch + PNG write path + CPU backend)
# ---------------------------------------------------------------------------
Write-Step "Render smoke test (512px still + 2x2 grid + icons)"
$smokeDir = Join-Path $root "out"
New-Item -ItemType Directory -Path $smokeDir -Force | Out-Null

& $exe --width 512 --height 512 --depth 8 --preset cinema --seed 7 `
       --out (Join-Path $smokeDir "verify_smoke.png")
if ($LASTEXITCODE -ne 0) { Write-Error "smoke render failed"; exit 1 }

& $exe --grid (Join-Path $smokeDir "verify_smoke_grid.png") --tile 256 --depth 8
if ($LASTEXITCODE -ne 0) { Write-Error "smoke grid failed"; exit 1 }

& $exe --icon --family gold --glyph crown --tier 2 --width 128 --height 128 `
       --out (Join-Path $smokeDir "verify_smoke_icon.png")
if ($LASTEXITCODE -ne 0) { Write-Error "icon smoke failed"; exit 1 }

& $exe --gicon --bg glow --scheme tide --fg white --glyph eye `
       --width 128 --height 128 --out (Join-Path $smokeDir "verify_smoke_gicon.png")
if ($LASTEXITCODE -ne 0) { Write-Error "gicon smoke failed"; exit 1 }

# Manifest production smoke: schema parse, deterministic render, PNG hash and
# provenance sidecar all have to agree end to end.
$miniManifest = Join-Path $smokeDir "_verify_manifest.csv"
$manifestOut = Join-Path $smokeDir "_verify_manifest_icons"
Set-Content -LiteralPath $miniManifest -Value @(
    '#schema: keenly.icon-manifest/v1',
    'family,glyph,tier,variant,seed,size,mask,radius',
    'blue,sword,2,0,7,64,round,0.16',
    'red,flame,3,1,123,32,rect,0.10'
)
& $exe --icons $miniManifest --outdir $manifestOut
if ($LASTEXITCODE -ne 0) { Write-Error "manifest smoke failed"; exit 1 }
$manifestPng = Join-Path $manifestOut "blue\sword_t2_v0_64.png"
$manifestSidecar = "$manifestPng.json"
if (-not (Test-Path -LiteralPath $manifestPng) -or -not (Test-Path -LiteralPath $manifestSidecar)) {
    Write-Error "manifest smoke did not produce the expected PNG and sidecar"
    exit 1
}
$sidecar = Get-Content -LiteralPath $manifestSidecar -Raw | ConvertFrom-Json
$actualHash = (Get-FileHash -LiteralPath $manifestPng -Algorithm SHA256).Hash.ToLowerInvariant()
if ($sidecar.output.sha256 -ne $actualHash) {
    Write-Error "manifest provenance hash does not match the PNG"
    exit 1
}
Remove-Item -LiteralPath $manifestOut -Recurse -Force
Remove-Item -LiteralPath $miniManifest -Force

Write-Host ""
Write-Host "PASS: field + post bit-exact, CPU/GPU PNG parity, renders write clean." -ForegroundColor Green
exit 0