#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cuda_oplib::detail {

inline void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string("CUDA call failed: ") + expr + " at " + file + ":" +
        std::to_string(line) + " (" + cudaGetErrorString(status) + ")");
}

}  // namespace cuda_oplib::detail

#define CUDA_OPLIB_CHECK_CUDA(expr) \
    ::cuda_oplib::detail::check_cuda((expr), #expr, __FILE__, __LINE__)

