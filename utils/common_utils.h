#ifndef POWERSPARSE_COMMON_UTILS_H
#define POWERSPARSE_COMMON_UTILS_H

#include <cstdint>
#include <cuda_runtime.h>

namespace PowerSparse {

__device__ constexpr uint32_t StaticPow(uint32_t base, uint32_t exponent) {
  return exponent == 0 ? 1 : base * StaticPow(base, exponent - 1);
}

// barrier for sync in kernel
struct PowerSparseBarrier {
  static constexpr int kThreadsPerBlock = 4 * 8;

  static_assert(kThreadsPerBlock % 32 == 0,
                "The thread-block size must be divisible by the warp size.");
  static_assert(kThreadsPerBlock > 0, "The thread-block size must be nonzero.");

  static constexpr int kThreadsPerOutputTile = 8;

  static_assert((kThreadsPerOutputTile % 2) == 0 || (kThreadsPerOutputTile == 1),
                "The number of threads collaborating on a tile must be "
                "a multiple of two or all threads must be independent.");

  static_assert((kThreadsPerBlock == 32) || (kThreadsPerOutputTile >= 32) ||
                (kThreadsPerOutputTile == 1),
                "Independent warps must be in separate thread-blocks "
                "when using a subwarp tiling.");

  #if __CUDA_ARCH__ >= 700
    uint32_t thread_mask = 0xffffffff;
  #endif

  __device__ __forceinline__ PowerSparseBarrier(int thread_idx_y) {
    #if __CUDA_ARCH__ >= 700
      if ((kThreadsPerOutputTile < 32) && (kThreadsPerOutputTile > 1)) {
        constexpr uint32_t kBaseSubwarpMask =
        StaticPow(2, kThreadsPerOutputTile) - 1;
        thread_mask = kBaseSubwarpMask << (thread_idx_y * kThreadsPerOutputTile);
      }
    #endif
  }

  __device__ __forceinline__ void Sync() {
    #if __CUDA_ARCH__ >= 700
      if (kThreadsPerOutputTile > 32) {
        __syncthreads();
      } else if (kThreadsPerOutputTile > 1) {
        __syncwarp(thread_mask);
      }
    #else
      if (kThreadsPerOutputTile > 32) {
        __syncthreads();
      }
    #endif
  }
  __device__ __forceinline__ uint32_t ThreadMask() const {
    #if __CUDA_ARCH__ >= 700
      return thread_mask;
    #else
      return 0xffffffff;
    #endif
  }
};

__device__ __forceinline__ void float_times_float4_then_accumulate(
  float sp_data, float4 ds_data, float *accumulator) {
  float4 *accu_ptr = reinterpret_cast<float4*>(accumulator);
  accu_ptr[0].x += sp_data * ds_data.x;
  accu_ptr[0].y += sp_data * ds_data.y;
  accu_ptr[0].z += sp_data * ds_data.z;
  accu_ptr[0].w += sp_data * ds_data.w;
}

} // namespace PowerSparse

#endif // POWERSPARSE_COMMON_UTILS_H
