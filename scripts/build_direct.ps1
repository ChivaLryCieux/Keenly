# Build Keenly by invoking nvcc directly.
#
# Use this when the Ninja/CMake path is unavailable - for example inside a
# confined agent sandbox where Ninja cannot capture child-process output and
# deadlocks. In a normal terminal prefer scripts\build.ps1 or VS2026.

$ErrorActionPreference = "Stop"

$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
$root = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $root "bin"
New-Item -ItemType Directory -Path $bin -Force | Out-Null

# -Xcompiler=/utf-8 uses the = form on purpose: embedding a quoted argument
# inside a PowerShell string here just fights the quote parser.
# -std=c++17 matches CMAKE_CXX_STANDARD / CMAKE_CUDA_STANDARD in CMakeLists.txt.
#
# --fmad=false + NO --use_fast_math: Keenly renders strict IEEE math so GPU
# output is bit-identical to the CPU reference and the verify gate can demand
# maxAbs == 0. Speed is not the bottleneck; reproducibility is the product.
$flags = "-arch=sm_75 -O3 -lineinfo --expt-relaxed-constexpr -std=c++17 --fmad=false -Xcompiler=/utf-8"

$srcArgs = (@("src\keenly.cu", "src\render.cu", "src\reference.cu") |
            ForEach-Object { "`"$root\$_`"" }) -join " "
$out = "`"$bin\keenly.exe`""

Write-Host "Building keenly ..."
$cmd = "`"$vcvars`" >nul 2>&1 && nvcc $flags -I`"$root\src`" $srcArgs -o $out"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to build keenly"; exit $LASTEXITCODE }

Write-Host ""
Write-Host "Build OK:"
Get-ChildItem "$bin\*.exe" | ForEach-Object { Write-Host "  $($_.FullName)" }