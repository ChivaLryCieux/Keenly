add_executable(render_art
    render_art.cu
)

target_include_directories(render_art PRIVATE ${PROJECT_SOURCE_DIR}/src/common)