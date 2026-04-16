add_executable(bench_vector_add bench_vector_add.cu)

target_link_libraries(bench_vector_add PRIVATE cuda_oplib::cuda_oplib CUDA::cudart)

set_target_properties(
    bench_vector_add
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_layernorm.cu")
    add_executable(bench_layernorm bench_layernorm.cu)

    target_link_libraries(bench_layernorm PRIVATE cuda_oplib::cuda_oplib CUDA::cudart)

    set_target_properties(
        bench_layernorm
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
endif()
