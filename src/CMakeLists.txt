set(CUDA_OPLIB_SOURCES
    kernel/vector_add.cu
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kernel/layernorm_half2.cu")
    list(APPEND CUDA_OPLIB_SOURCES
        kernel/layernorm_half2.cu
    )
endif()

add_library(cuda_oplib SHARED ${CUDA_OPLIB_SOURCES})
add_library(cuda_oplib::cuda_oplib ALIAS cuda_oplib)

target_include_directories(
    cuda_oplib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(cuda_oplib PUBLIC CUDA::cudart)
target_compile_features(cuda_oplib PUBLIC cxx_std_17)

set_target_properties(
    cuda_oplib
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME cuda_oplib
)

install(
    TARGETS cuda_oplib
    EXPORT cuda_oplibTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

