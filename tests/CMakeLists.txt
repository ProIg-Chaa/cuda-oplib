add_executable(test_vector_add cpp/test_vector_add.cu)

target_link_libraries(test_vector_add PRIVATE cuda_oplib::cuda_oplib CUDA::cudart)

set_target_properties(
    test_vector_add
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME vector_add_smoke COMMAND test_vector_add)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/test_layernorm.cu")
    add_executable(test_layernorm cpp/test_layernorm.cu)

    target_link_libraries(test_layernorm PRIVATE cuda_oplib::cuda_oplib CUDA::cudart)

    set_target_properties(
        test_layernorm
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
    )

    add_test(NAME layernorm_smoke COMMAND test_layernorm)
endif()
