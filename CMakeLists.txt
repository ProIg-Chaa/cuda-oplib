cmake_minimum_required(VERSION 3.24)

project(cuda_oplib VERSION 0.1.0 LANGUAGES CXX CUDA)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(CUDA_OPLIB_BUILD_TESTS "Build correctness tests." ON)
option(CUDA_OPLIB_BUILD_BENCHMARKS "Build benchmark binaries." ON)
option(CUDA_OPLIB_BUILD_EXAMPLES "Build example binaries." ON)
option(CUDA_OPLIB_BUILD_PYTORCH_BINDINGS "Build PyTorch bindings." OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type." FORCE)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(STATUS
        "CMAKE_CUDA_ARCHITECTURES is not set. "
        "Pass an explicit value like 80;86;89 for reproducible releases."
    )
endif()

enable_testing()

add_subdirectory(src)

if(CUDA_OPLIB_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(CUDA_OPLIB_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(CUDA_OPLIB_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/cuda_oplib-config.cmake.in
    ${PROJECT_BINARY_DIR}/cuda_oplib-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_oplib
)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/cuda_oplib-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/cuda_oplib-config.cmake
        ${PROJECT_BINARY_DIR}/cuda_oplib-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_oplib
)

install(
    EXPORT cuda_oplibTargets
    FILE cuda_oplibTargets.cmake
    NAMESPACE cuda_oplib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_oplib
)

