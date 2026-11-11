cmake_minimum_required(VERSION 3.20)

# Keenly - a CUDA generative art studio: procedural film-grain, frosted and
# organic backgrounds, rendered as cinema frames and A1 posters.
project(Keenly LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 2080 SUPER = Turing = compute capability 7.5.
# Pin it explicitly so nvcc does not spend time JIT-ing from a virtual arch.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# -lineinfo lets Nsight tools correlate SASS back to source lines.
# NOTE: no --use_fast_math here on purpose. Keenly builds strict IEEE math
# (--fmad=false below) so GPU renders are BIT-IDENTICAL to the CPU reference
# and the verify gate can demand maxAbs == 0. Speed is not the bottleneck;
# reproducibility is the product.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo --expt-relaxed-constexpr")

add_executable(keenly
    src/keenly.cu
    src/render.cu
    src/reference.cu
)

target_include_directories(keenly PRIVATE src)
target_compile_options(keenly PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)