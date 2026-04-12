add_library(cuda_oplib SHARED operators/vector_add.cu)
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

