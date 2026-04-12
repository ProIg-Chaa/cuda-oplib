add_executable(bench_vector_add bench_vector_add.cu)

target_link_libraries(bench_vector_add PRIVATE cuda_oplib::cuda_oplib CUDA::cudart)

set_target_properties(
    bench_vector_add
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

