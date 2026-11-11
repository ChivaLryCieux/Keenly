add_executable(render_art
    render_art.cu
    art_render.cu
    art_ref.cu
    ${PROJECT_SOURCE_DIR}/src/common/reference.cu
)

target_include_directories(render_art PRIVATE ${PROJECT_SOURCE_DIR}/src/common)