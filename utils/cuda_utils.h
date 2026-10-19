#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void CheckCuda(cudaError_t status, const char* expression,
                      const char* file, int line) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": " + expression + ": " + cudaGetErrorString(status));
}

#define CUDA_CALL(call) CheckCuda((call), #call, __FILE__, __LINE__)
