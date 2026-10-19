#include "PowerSparse_SDDMM.h"

namespace PowerSparse {

namespace {

template <int kBlockItemsY, int kBlockWidth>
struct Barrier {
  static constexpr int kThreadsPerBlock = kBlockItemsY * kBlockWidth;

  static_assert(kThreadsPerBlock % 32 == 0,
                "The thread-block size must be divisible by the warp size.");
  static_assert(kThreadsPerBlock > 0, "The thread-block size must be nonzero.");

  static constexpr int kThreadsPerOutputTile = kBlockWidth;

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

  __device__ __forceinline__ Barrier(int thread_idx_y) {
    #if __CUDA_ARCH__ >= 700
      if ((kThreadsPerOutputTile < 32) && (kThreadsPerOutputTile > 1)) {
        constexpr uint32_t kBaseSubwarpMask = StaticPow(2, kThreadsPerOutputTile) - 1;
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

template <typename OutType, typename InType>
__device__ __forceinline__ OutType* OffsetCast(InType* ptr, int offset) {
  return reinterpret_cast<OutType*>(
    const_cast<char*>(reinterpret_cast<const char*>(ptr)) + offset);
}

} // anonymous namespace

__global__ void sddmm_kernel_residue_reduce32(
  int group_num, int n, int k,
  const int* __restrict__ row_indices,
  const int* __restrict__ row_offsets,
  const int* __restrict__ column_indices,
  const float* __restrict__ values,
  const float* __restrict__ lhs_matrix,
  const float* __restrict__ rhs_matrix,
  float *out) {

  static constexpr int K = 32; // 32 in 32residue, 128 in 128residue
  static constexpr int kBlockItemsK = K; // 32 in 32residue, 128 in 128residue
  static constexpr int vector_len = 4;
  static constexpr int kBlockWidth = K / vector_len; // 8 in 32residue, 32 in 128residue
  static constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth / 4;
  static constexpr int kScalarThreadItemsK = kBlockItemsK / kBlockWidth;
  static constexpr int kScalarThreadItemsX = 32 / kBlockWidth;
  static constexpr int inner_pipeline_stages = 4;

  int group_id = blockIdx.x * 4 + threadIdx.y;

  if(group_id >= group_num) return;

  int row_beg = row_offsets[group_id];
  int row_end = row_offsets[group_id + 1];
  int row_nnz = row_end - row_beg;

  group_id = row_indices[group_id];

  int Nonzeros = row_nnz;

  __shared__ int column_indices_tile_array[32 * vector_len];
  float values_tile[kScalarThreadItemsX] = {};

  int* column_indices_tile = column_indices_tile_array + 32 * threadIdx.y;

  __align__(16) float lhs_fragment[kBlockItemsK / kBlockWidth];
  __align__(16) float rhs_fragment[kBlockItemsK / kBlockWidth * 4];

  float output_fragment[32 / kBlockWidth] = {};

  Barrier<4,kBlockWidth> barrier(threadIdx.y);

  const float* sparse_values_ = values        + row_beg + threadIdx.x;
  const int* column_indices_ = column_indices + row_beg + threadIdx.x;

  int* column_idxs_tile = column_indices_tile + threadIdx.x;
  int nonzeros_ = row_nnz;

  #pragma unroll
  for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++x_item_idx) {
    if(nonzeros_ <= static_cast<int>(threadIdx.x)) break;
    *column_idxs_tile = __ldg(column_indices_);
    nonzeros_ -= kBlockWidth;
    column_idxs_tile += kBlockWidth;
    column_indices_  += kBlockWidth;
  }
  barrier.Sync();

  nonzeros_ = row_nnz;

  #pragma unroll
  for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++x_item_idx) {
    if(nonzeros_ <= static_cast<int>(threadIdx.x)) break;
    *(values_tile + x_item_idx) = __ldg(sparse_values_);
    nonzeros_ -= kBlockWidth;
    sparse_values_ += kBlockWidth;
  }

  const float4 *lhs_matrix_ = reinterpret_cast<const float4*>(lhs_matrix + group_id * k) + threadIdx.x;
  float4 *lhs_fragment_ = reinterpret_cast<float4*>(lhs_fragment);

  const float4 *rhs_matrix_base = reinterpret_cast<const float4*>(rhs_matrix) + threadIdx.x;
  float4 *rhs_fragment_ = reinterpret_cast<float4*>(rhs_fragment);

  const uint32_t kShflMask = barrier.ThreadMask();

  int x_idx = 0, t_idx = 0;
  #pragma unroll
  for(int kk = 0; kk < K / kBlockItemsK; ++kk) {
    #pragma unroll
    for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
      *(lhs_fragment_ + k_item_idx) = __ldg(lhs_matrix_);
      lhs_matrix_ += kBlockWidth;
    }

    int *column_idxs_tile = reinterpret_cast<int*>(column_indices_tile);
    nonzeros_ = Nonzeros;

    for(int i = 0; i < Nonzeros / 4; ++i) {
      float accumulator_fragment[4] = {};

      #pragma unroll
      for(int j = 0; j < 4; ++j) {
        const float4 *rhs_matrix_ = reinterpret_cast<const float4*>(rhs_matrix_base) + K / 4 * (*column_idxs_tile);

        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
          int fragment_offset = j * kThreadItemsK + k_item_idx;
          *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
          rhs_matrix_ += kBlockWidth;
        }
        ++column_idxs_tile;
      }
      #pragma unroll
      for(int j=0; j < 4; ++j) {
        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
          accumulator_fragment[j] += lhs_fragment[k_item_idx] * rhs_fragment[j * kScalarThreadItemsK + k_item_idx];
        }
      }

      int tag = threadIdx.x & 3;

      if(tag == 1) {
        float temp = accumulator_fragment[1];
        accumulator_fragment[1] = accumulator_fragment[0];
        accumulator_fragment[0] = temp;
        temp = accumulator_fragment[2];
        accumulator_fragment[2] = accumulator_fragment[3];
        accumulator_fragment[3] = temp;
      }else if(tag == 2){
        float temp = accumulator_fragment[2];
        accumulator_fragment[2] = accumulator_fragment[0];
        accumulator_fragment[0] = temp;
        temp = accumulator_fragment[3];
        accumulator_fragment[3] = accumulator_fragment[1];
        accumulator_fragment[1] = temp;
      }else if(tag == 3) {
        float temp = accumulator_fragment[3];
        accumulator_fragment[3] = accumulator_fragment[0];
        accumulator_fragment[0] = temp;
        temp = accumulator_fragment[2];
        accumulator_fragment[2] = accumulator_fragment[1];
        accumulator_fragment[1] = temp;
      }

      #pragma unroll
      for(int j = 1; j < 4; ++j) {
        accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[j],j);
      }

      #pragma unroll
      for(int j = kBlockWidth/2; j >= 4; j /= 2){
        accumulator_fragment[0] += __shfl_xor_sync(kShflMask,accumulator_fragment[0],j);
      }

      if(threadIdx.x >= t_idx && threadIdx.x < t_idx + 4){
        output_fragment[x_idx] = accumulator_fragment[0] * values_tile[x_idx];
        ++x_idx;
      }

      t_idx = (t_idx + 4) % kBlockWidth;
    }

    if((Nonzeros & 3) >= 2) {
      float accumulator_fragment[2] = {};

      #pragma unroll
      for(int j = 0; j < 2; ++j) {
        const float4 *rhs_matrix_ = reinterpret_cast<const float4*>(rhs_matrix_base) + K / 4 * (*column_idxs_tile);

        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
          int fragment_offset = j * kThreadItemsK + k_item_idx;
          *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
          rhs_matrix_ += kBlockWidth;
        }
        ++column_idxs_tile;
      }
      #pragma unroll
      for(int j=0; j < 2; ++j) {
        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
          accumulator_fragment[j] += lhs_fragment[k_item_idx] * rhs_fragment[j * kScalarThreadItemsK + k_item_idx];
        }
      }

      int tag = threadIdx.x & 1;
      if(tag == 1) {
        float temp = accumulator_fragment[1];
        accumulator_fragment[1] = accumulator_fragment[0];
        accumulator_fragment[0] = temp;
      }

      accumulator_fragment[0] += __shfl_xor_sync(kShflMask, accumulator_fragment[1],1);

      #pragma unroll
      for(int j = kBlockWidth / 2; j >= 2; j /= 2) {
        accumulator_fragment[0] += __shfl_xor_sync(kShflMask, accumulator_fragment[0],j);
      }

      if( threadIdx.x >= t_idx && threadIdx.x < t_idx + 2) {
        output_fragment[x_idx] = accumulator_fragment[0] * values_tile[x_idx];
        ++x_idx;
      }
      t_idx += 2;
    }

    if(Nonzeros & 1) {
      float accumulator_fragment[1] = {};

      const float4 *rhs_matrix_ = reinterpret_cast<const float4*>(rhs_matrix_base) + K / 4 * (*column_idxs_tile);

      #pragma unroll
      for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        *(rhs_fragment_ + k_item_idx) = __ldg(rhs_matrix_);
        rhs_matrix_ += kBlockWidth;
      }

      #pragma unroll
      for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
        accumulator_fragment[0] += lhs_fragment[k_item_idx] * rhs_fragment[k_item_idx];
      }

      #pragma unroll
      for(int j = kBlockWidth / 2 ; j >= 1; j /= 2) {
        accumulator_fragment[0] += __shfl_xor_sync(kShflMask, accumulator_fragment[0], j);
      }

      if(threadIdx.x == t_idx) {
        output_fragment[x_idx] = accumulator_fragment[0] * values_tile[x_idx];
        ++x_idx;
      }
    }
    rhs_matrix_base = OffsetCast<const float4>(rhs_matrix_base, kBlockItemsK * sizeof(float));
  }

  float* output_values_ = out + row_beg + threadIdx.x;
  #pragma unroll
  for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++ x_item_idx) {
    if(row_nnz > 0 && row_beg + x_item_idx * kBlockWidth + threadIdx.x < row_end) {
      atomicAdd(output_values_, output_fragment[x_item_idx]);
    }
    row_nnz        -= kBlockWidth;
    output_values_ += kBlockWidth;
  }
}

__global__ void sddmm_kernel_residue_reduce128(
  int group_num, int n, int k,
  const int* __restrict__ row_indices,
  const int* __restrict__ row_offsets,
  const int* __restrict__ column_indices,
  const float* __restrict__ values,
  const float* __restrict__ lhs_matrix,
  const float* __restrict__ rhs_matrix,
  float *out) {

  static constexpr int K = 128; // 32 in 32residue, 128 in 128residue
  static constexpr int kBlockItemsK = K; // 32 in 32residue, 128 in 128residue
  static constexpr int vector_len = 4;
  static constexpr int kBlockWidth = K / vector_len; // 8 in 32residue, 32 in 128residue
  static constexpr int kThreadItemsK = kBlockItemsK / kBlockWidth / 4;
  static constexpr int kScalarThreadItemsK = kBlockItemsK / kBlockWidth;
  static constexpr int kScalarThreadItemsX = 32 / kBlockWidth;
  static constexpr int inner_pipeline_stages = 4;

  int group_id = blockIdx.x * 4 + threadIdx.y;

  if(group_id >= group_num) return;

  int row_beg = row_offsets[group_id];
  int row_end = row_offsets[group_id + 1];
  int row_nnz = row_end - row_beg;

  group_id = row_indices[group_id];

  int Nonzeros = row_nnz;

  __shared__ int column_indices_tile_array[32 * vector_len];
  float values_tile[kScalarThreadItemsX] = {};

  int* column_indices_tile = column_indices_tile_array + 32 * threadIdx.y;

  __align__(16) float lhs_fragment[kBlockItemsK / kBlockWidth];
  __align__(16) float rhs_fragment[kBlockItemsK / kBlockWidth * 4];

  float output_fragment[32 / kBlockWidth] = {};

  Barrier<4,kBlockWidth> barrier(threadIdx.y);

  const float* sparse_values_ = values        + row_beg + threadIdx.x;
  const int* column_indices_ = column_indices + row_beg + threadIdx.x;

  int* column_idxs_tile = column_indices_tile + threadIdx.x;
  int nonzeros_ = row_nnz;

  #pragma unroll
  for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++x_item_idx) {
    if(nonzeros_ <= static_cast<int>(threadIdx.x)) break;
    *column_idxs_tile = __ldg(column_indices_);
    nonzeros_ -= kBlockWidth;
    column_idxs_tile += kBlockWidth;
    column_indices_  += kBlockWidth;
  }
  barrier.Sync();

  nonzeros_ = row_nnz;

  #pragma unroll
  for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++x_item_idx) {
    if(nonzeros_ <= static_cast<int>(threadIdx.x)) break;
    *(values_tile + x_item_idx) = __ldg(sparse_values_);
    nonzeros_ -= kBlockWidth;
    sparse_values_ += kBlockWidth;
  }

  const float4 *lhs_matrix_ = reinterpret_cast<const float4*>(lhs_matrix + group_id * k) + threadIdx.x;
  float4 *lhs_fragment_ = reinterpret_cast<float4*>(lhs_fragment);

  const float4 *rhs_matrix_base = reinterpret_cast<const float4*>(rhs_matrix) + threadIdx.x;
  float4 *rhs_fragment_ = reinterpret_cast<float4*>(rhs_fragment);

  const uint32_t kShflMask = barrier.ThreadMask();

  int x_idx = 0, t_idx = 0;
  #pragma unroll
  for(int kk = 0; kk < K / kBlockItemsK; ++kk) {
    #pragma unroll
    for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++ k_item_idx) {
      *(lhs_fragment_ + k_item_idx) = __ldg(lhs_matrix_);
      lhs_matrix_ += kBlockWidth;
    }

    int *column_idxs_tile = reinterpret_cast<int*>(column_indices_tile);
    nonzeros_ = Nonzeros;

    for(int i = 0; i < Nonzeros / 4; ++i) {
      float accumulator_fragment[4] = {};

      #pragma unroll
      for(int j = 0; j < 4; ++j) {
        const float4 *rhs_matrix_ = reinterpret_cast<const float4*>(rhs_matrix_base) + K / 4 * (*column_idxs_tile);

        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
          int fragment_offset = j * kThreadItemsK + k_item_idx;
          *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
          rhs_matrix_ += kBlockWidth;
        }
        ++column_idxs_tile;
      }
      #pragma unroll
      for(int j=0; j < 4; ++j) {
        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
          accumulator_fragment[j] += lhs_fragment[k_item_idx] * rhs_fragment[j * kScalarThreadItemsK + k_item_idx];
        }
      }

      int tag = threadIdx.x & 3;

      if(tag == 1) {
        float temp = accumulator_fragment[1];
        accumulator_fragment[1] = accumulator_fragment[0];
        accumulator_fragment[0] = temp;
        temp = accumulator_fragment[2];
        accumulator_fragment[2] = accumulator_fragment[3];
        accumulator_fragment[3] = temp;
      }else if(tag == 2){
        float temp = accumulator_fragment[2];
        accumulator_fragment[2] = accumulator_fragment[0];
        accumulator_fragment[0] = temp;
        temp = accumulator_fragment[3];
        accumulator_fragment[3] = accumulator_fragment[1];
        accumulator_fragment[1] = temp;
      }else if(tag == 3) {
        float temp = accumulator_fragment[3];
        accumulator_fragment[3] = accumulator_fragment[0];
        accumulator_fragment[0] = temp;
        temp = accumulator_fragment[2];
        accumulator_fragment[2] = accumulator_fragment[1];
        accumulator_fragment[1] = temp;
      }

      #pragma unroll
      for(int j = 1; j < 4; ++j) {
        accumulator_fragment[0] += __shfl_xor_sync(kShflMask, accumulator_fragment[j],j);
      }

      #pragma unroll
      for(int j = kBlockWidth/2; j >= 4; j /= 2){
        accumulator_fragment[0] += __shfl_xor_sync(kShflMask, accumulator_fragment[0],j);
      }

      if(threadIdx.x >= t_idx && threadIdx.x < t_idx + 4){
        output_fragment[x_idx] = accumulator_fragment[0] * values_tile[x_idx];
        ++x_idx;
      }

      t_idx = (t_idx + 4) % kBlockWidth;
    }

    if((Nonzeros & 3) >= 2) {
      float accumulator_fragment[2] = {};

      #pragma unroll
      for(int j = 0; j < 2; ++j) {
        const float4 *rhs_matrix_ = reinterpret_cast<const float4*>(rhs_matrix_base) + K / 4 * (*column_idxs_tile);

        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
          int fragment_offset = j * kThreadItemsK + k_item_idx;
          *(rhs_fragment_ + fragment_offset) = __ldg(rhs_matrix_);
          rhs_matrix_ += kBlockWidth;
        }
        ++column_idxs_tile;
      }
      #pragma unroll
      for(int j=0; j < 2; ++j) {
        #pragma unroll
        for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
          accumulator_fragment[j] += lhs_fragment[k_item_idx] * rhs_fragment[j * kScalarThreadItemsK + k_item_idx];
        }
      }

      int tag = threadIdx.x & 1;
      if(tag == 1) {
        float temp = accumulator_fragment[1];
        accumulator_fragment[1] = accumulator_fragment[0];
        accumulator_fragment[0] = temp;
      }

      accumulator_fragment[0] += __shfl_xor_sync(kShflMask, accumulator_fragment[1], 1);

      #pragma unroll
      for(int j = kBlockWidth/2 ; j >= 2; j /= 2) {
        accumulator_fragment[0] += __shfl_xor_sync(kShflMask, accumulator_fragment[0], j);
      }

      if( threadIdx.x >= t_idx && threadIdx.x < t_idx + 2) {
        output_fragment[x_idx] = accumulator_fragment[0] * values_tile[x_idx];
        ++x_idx;
      }
      t_idx += 2;
    }

    if(Nonzeros & 1) {
      float accumulator_fragment[1] = {};

      const float4 *rhs_matrix_ = reinterpret_cast<const float4*>(rhs_matrix_base) + K / 4 * (*column_idxs_tile);

      #pragma unroll
      for(int k_item_idx = 0; k_item_idx < kThreadItemsK; ++k_item_idx) {
        *(rhs_fragment_ + k_item_idx) = __ldg(rhs_matrix_);
        rhs_matrix_ += kBlockWidth;
      }

      #pragma unroll
      for(int k_item_idx = 0; k_item_idx < kScalarThreadItemsK; ++k_item_idx) {
        accumulator_fragment[0] += lhs_fragment[k_item_idx] * rhs_fragment[k_item_idx];
      }

      #pragma unroll
      for(int j = kBlockWidth / 2; j >= 1; j /= 2) {
        accumulator_fragment[0] += __shfl_xor_sync(kShflMask, accumulator_fragment[0], j);
      }

      if(threadIdx.x == t_idx) {
        output_fragment[x_idx] = accumulator_fragment[0] * values_tile[x_idx];
        ++x_idx;
      }
    }
    rhs_matrix_base = OffsetCast<const float4>(rhs_matrix_base, kBlockItemsK * sizeof(float));
  }

  float* output_values_ = out + row_beg + threadIdx.x;
  #pragma unroll
  for(int x_item_idx = 0; x_item_idx < kScalarThreadItemsX; ++x_item_idx) {
    if(row_nnz > 0 && row_beg + x_item_idx * kBlockWidth + threadIdx.x < row_end) {
      atomicAdd(output_values_, output_fragment[x_item_idx]);
    }
    row_nnz        -= kBlockWidth;
    output_values_ += kBlockWidth;
  }
}

} // namespace PowerSparse
