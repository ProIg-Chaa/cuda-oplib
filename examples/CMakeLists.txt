add_executable(vector_add_example cpp/vector_add_example.cu)

target_link_libraries(vector_add_example PRIVATE cuda_oplib::cuda_oplib CUDA::cudart)

set_target_properties(
    vector_add_example
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

