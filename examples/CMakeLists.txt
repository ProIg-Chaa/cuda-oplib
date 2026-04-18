add_executable(vector_add_example cpp/vector_add_example.cu)

target_link_libraries(vector_add_example PRIVATE cuda_oplib::cuda_oplib CUDA::cudart)

set_target_properties(
    vector_add_example
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/layernorm_example.cu")
    add_executable(layernorm_example cpp/layernorm_example.cu)

    target_link_libraries(layernorm_example PRIVATE cuda_oplib::cuda_oplib CUDA::cudart)

    set_target_properties(
        layernorm_example
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/rmsnorm_example.cu")
    add_executable(rmsnorm_example cpp/rmsnorm_example.cu)

    target_link_libraries(rmsnorm_example PRIVATE cuda_oplib::cuda_oplib CUDA::cudart)

    set_target_properties(
        rmsnorm_example
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )
endif()
