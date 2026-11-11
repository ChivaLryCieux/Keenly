add_executable(render_art
    render_art.cu
    art_render.cu
    art_ref.cu
    ${PROJECT_SOURCE_DIR}/src/common/reference.cu
)

target_include_directories(render_art PRIVATE ${PROJECT_SOURCE_DIR}/src/common)

# Art renders chase reproducibility, not speed: strict IEEE math with FMA
# contraction disabled makes every kernel bit-identical to the CPU reference
# (the verify gate demands maxAbs == 0). matmul/conv keep --use_fast_math.
target_compile_options(render_art PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)