<#
.SYNOPSIS
    Keenly regression gate: build, then prove every generator is still
    bit-identical to the CPU reference.

.DESCRIPTION
    This is the regression gate. It answers one question:

        Is every field generator still BIT-EXACT against the CPU reference?

    Keenly compiles strict IEEE math (--fmad=false, no fast-math) precisely so
    GPU renders match the serial CPU reference sample-for-sample (maxAbs 0).
    The gate is a hard error (exit 1) on any deviation - if it fires, a
    non-IEEE operation slipped into noise.cuh / pixel.cuh and reproducibility
    (tiling, seed series, print masters) is gone.

    A tiny render smoke test follows the gate: it catches launch and PNG
    write failures that a math-only check would miss.

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
    Write-Host "slipped into src\noise.cuh or src\pixel.cuh." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Render smoke test (launch + PNG write path)
# ---------------------------------------------------------------------------
Write-Step "Render smoke test (512px still + 2x2 grid)"
$smokeDir = Join-Path $root "out"
New-Item -ItemType Directory -Path $smokeDir -Force | Out-Null

& $exe --width 512 --height 512 --depth 8 --preset cinema --seed 7 `
       --out (Join-Path $smokeDir "verify_smoke.png")
if ($LASTEXITCODE -ne 0) { Write-Error "smoke render failed"; exit 1 }

& $exe --grid (Join-Path $smokeDir "verify_smoke_grid.png") --tile 256 --depth 8
if ($LASTEXITCODE -ne 0) { Write-Error "smoke grid failed"; exit 1 }

Write-Host ""
Write-Host "PASS: all generators bit-exact, renders write clean." -ForegroundColor Green
exit 0