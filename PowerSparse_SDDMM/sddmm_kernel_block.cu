#include "PowerSparse_SDDMM.h"
#include "cooperative_groups.h"
#include <cooperative_groups/memcpy_async.h>

namespace PowerSparse {

namespace {

__host__ __device__ __forceinline__ int Log2(int x) {
  if (x >>= 1) return Log2(x) + 1;
  return 0;
}

template <typename OutType, typename InType>
__device__ __forceinline__ OutType* OffsetCast(InType* ptr, int offset) {
  return reinterpret_cast<OutType*>(
    const_cast<char*>(reinterpret_cast<const char*>(ptr)) + offset);
}

__device__ __forceinline__ void inner_pipeline(
  float4 *reg_rhdata_trd_vec_ptr,
  const float4 *glb_rhdata_trd_vec_ptr,
  int reduce_dim, int vector_len,
  int *shm_colind_oct_scl_ptr,
  PowerSparseBarrier barrier,
  float *reg_lhdata_trd_scl_imm,
  float *reg_rhdata_trd_scl_imm,
  float *reg_accumu_trd_scl_imm) {

  reg_rhdata_trd_vec_ptr[0] = __ldg(glb_rhdata_trd_vec_ptr + reduce_dim / vector_len * shm_colind_oct_scl_ptr[0]);
  reg_rhdata_trd_vec_ptr[1] = __ldg(glb_rhdata_trd_vec_ptr + reduce_dim / vector_len * shm_colind_oct_scl_ptr[1]);
  reg_rhdata_trd_vec_ptr[2] = __ldg(glb_rhdata_trd_vec_ptr + reduce_dim / vector_len * shm_colind_oct_scl_ptr[2]);
  reg_rhdata_trd_vec_ptr[3] = __ldg(glb_rhdata_trd_vec_ptr + reduce_dim / vector_len * shm_colind_oct_scl_ptr[3]);

  int db_flag = 1; // for double buffer
  #pragma unroll
  for(int i = 0; i < 8 - 1; ++ i) {
    reg_rhdata_trd_vec_ptr[db_flag * 4 + 0] = __ldg(glb_rhdata_trd_vec_ptr + reduce_dim / vector_len * shm_colind_oct_scl_ptr[i * 4 + 4]);
    reg_rhdata_trd_vec_ptr[db_flag * 4 + 1] = __ldg(glb_rhdata_trd_vec_ptr + reduce_dim / vector_len * shm_colind_oct_scl_ptr[i * 4 + 5]);
    reg_rhdata_trd_vec_ptr[db_flag * 4 + 2] = __ldg(glb_rhdata_trd_vec_ptr + reduce_dim / vector_len * shm_colind_oct_scl_ptr[i * 4 + 6]);
    reg_rhdata_trd_vec_ptr[db_flag * 4 + 3] = __ldg(glb_rhdata_trd_vec_ptr + reduce_dim / vector_len * shm_colind_oct_scl_ptr[i * 4 + 7]);

    barrier.Sync();
    #pragma unroll
    for(int reduce_pos = 0; reduce_pos < 4; ++ reduce_pos) {
      const float lhs_value = reg_lhdata_trd_scl_imm[reduce_pos]; // load left value
      reg_accumu_trd_scl_imm[i * 4 + 0] += lhs_value * reg_rhdata_trd_scl_imm[reduce_pos + 4 * (0 + (db_flag^1)*4)]; // accumulate
      reg_accumu_trd_scl_imm[i * 4 + 1] += lhs_value * reg_rhdata_trd_scl_imm[reduce_pos + 4 * (1 + (db_flag^1)*4)]; // accumulate
      reg_accumu_trd_scl_imm[i * 4 + 2] += lhs_value * reg_rhdata_trd_scl_imm[reduce_pos + 4 * (2 + (db_flag^1)*4)]; // accumulate
      reg_accumu_trd_scl_imm[i * 4 + 3] += lhs_value * reg_rhdata_trd_scl_imm[reduce_pos + 4 * (3 + (db_flag^1)*4)]; // accumulate
    }
    db_flag ^= 1; // double buffer翻转
  }

  #pragma unroll
  for(int reduce_pos = 0; reduce_pos < 4; ++ reduce_pos) {
    const float lhs_value = reg_lhdata_trd_scl_imm[reduce_pos]; // load left value
    reg_accumu_trd_scl_imm[28] += lhs_value * reg_rhdata_trd_scl_imm[reduce_pos + 16]; // accumulate
    reg_accumu_trd_scl_imm[29] += lhs_value * reg_rhdata_trd_scl_imm[reduce_pos + 20]; // accumulate
    reg_accumu_trd_scl_imm[30] += lhs_value * reg_rhdata_trd_scl_imm[reduce_pos + 24]; // accumulate
    reg_accumu_trd_scl_imm[31] += lhs_value * reg_rhdata_trd_scl_imm[reduce_pos + 28]; // accumulate
  }

}

__device__ __forceinline__ void shuffle_result(
  const uint32_t threadMask, float *reg) {

  #pragma unroll
  for(int base_idx = 0; base_idx < 4; ++ base_idx) { // 这个循环执行四次
    #pragma unroll
    for(int k_item_idx = 1; k_item_idx < 8; k_item_idx *= 2) { // 这个循环执行三次 (8=8)
      const int kBoundX = 8 / (k_item_idx * 2); // 第一轮4 第二轮2 第三轮1
      #pragma unroll
      for(int x_item_idx = 0; x_item_idx < kBoundX; ++ x_item_idx) { // 这个循环依次执行 4次 2次 1次
        const int idx_a = x_item_idx * 2 * 4 * k_item_idx;
        const int idx_b = (x_item_idx *2  + 1) * 4 * k_item_idx;
        const int kStep = Log2(k_item_idx);
        if((threadIdx.x >> kStep)&1){
          float t = reg[base_idx + idx_a];
          reg[base_idx + idx_a] = reg[base_idx + idx_b];
          reg[base_idx + idx_b] = t;
        }
        reg[base_idx + idx_a] += __shfl_xor_sync(threadMask,reg[base_idx + idx_b],k_item_idx,8);
      }
    }
  }
}

} // anonymous namespace

 __global__ void sddmm_kernel_block_reduce32(
  int group_num, int N, int reduce_dim,
  const int* __restrict__ lhs_row_ids,
  const int* __restrict__ col_ind,
  const float* __restrict__ sp_data,
  const float* __restrict__ lhs,
  const float* __restrict__ rhs,
  float *out, int group_size) {

  static constexpr int ReduceDim = 32; // 32 in 32residue, 128 in 128residue
  static constexpr int vector_len = 4;
  static constexpr int inner_pipeline_stages = 4;

  int group_id = blockIdx.x * 4 + threadIdx.y;
  int sp_N_beg = blockIdx.y * 32;

  auto this_block = cooperative_groups::this_thread_block();
  auto octet = cooperative_groups::tiled_partition<8>(this_block);
  if(group_id >= group_num) return;

  int lhs_row_id = lhs_row_ids[group_id];
  int sp_row_beg = group_id * group_size;

  __shared__ __align__(16) int shm_colind_tbk_scl_ptr[32 * vector_len];
  float4 reg_spdata_trd_vec_imm = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  int *shm_colind_oct_scl_ptr = shm_colind_tbk_scl_ptr + 32 * threadIdx.y;

  __align__(16) float reg_lhdata_trd_scl_imm[vector_len];
  __align__(16) float reg_rhdata_trd_scl_imm[2 * vector_len * inner_pipeline_stages];
  float4 *reg_lhdata_trd_vec_ptr = reinterpret_cast<float4*>(reg_lhdata_trd_scl_imm);
  float4 *reg_rhdata_trd_vec_ptr = reinterpret_cast<float4*>(reg_rhdata_trd_scl_imm);

  float reg_accumu_trd_scl_imm[32] = {};
  float reg_otdata_trd_scl_imm[vector_len];

  PowerSparseBarrier barrier(threadIdx.y);

  int4* shm_colind_oct_vec_ptr = reinterpret_cast<int4*>(shm_colind_oct_scl_ptr);
  const int4* glb_colind_oct_vec_ptr = reinterpret_cast<const int4*>(col_ind + sp_row_beg + sp_N_beg);
  cooperative_groups::memcpy_async(octet, shm_colind_oct_vec_ptr, 8, glb_colind_oct_vec_ptr, 8);

  const float4* glb_spdata_trd_vec_ptr = reinterpret_cast<const float4*>(sp_data + sp_row_beg + sp_N_beg) + threadIdx.x;
  reg_spdata_trd_vec_imm = __ldg(glb_spdata_trd_vec_ptr);

  const float4 *glb_lhdata_trd_vec_ptr = reinterpret_cast<const float4*>(lhs + lhs_row_id * reduce_dim) + threadIdx.x;
  const float4 *glb_rhdata_trd_vec_ptr = reinterpret_cast<const float4*>(rhs) + threadIdx.x;

  cooperative_groups::wait(octet);

  #pragma unroll
  for(int i = 0; i < ReduceDim / 32; ++i){
    *(reg_lhdata_trd_vec_ptr) = __ldg(glb_lhdata_trd_vec_ptr); // load left matrix value
    glb_lhdata_trd_vec_ptr += 8;

    inner_pipeline(reg_rhdata_trd_vec_ptr, glb_rhdata_trd_vec_ptr, ReduceDim, vector_len, shm_colind_oct_scl_ptr, barrier,
      reg_lhdata_trd_scl_imm, reg_rhdata_trd_scl_imm, reg_accumu_trd_scl_imm);

    glb_rhdata_trd_vec_ptr = OffsetCast<const float4>(glb_rhdata_trd_vec_ptr, 32 * sizeof(float));
  }

  shuffle_result(barrier.ThreadMask(), reg_accumu_trd_scl_imm);

  reg_otdata_trd_scl_imm[0] = reg_accumu_trd_scl_imm[0] * reg_spdata_trd_vec_imm.x;
  reg_otdata_trd_scl_imm[1] = reg_accumu_trd_scl_imm[1] * reg_spdata_trd_vec_imm.y;
  reg_otdata_trd_scl_imm[2] = reg_accumu_trd_scl_imm[2] * reg_spdata_trd_vec_imm.z;
  reg_otdata_trd_scl_imm[3] = reg_accumu_trd_scl_imm[3] * reg_spdata_trd_vec_imm.w;

  float *glb_otdata_trd_scl_ptr = reinterpret_cast<float*>(out + sp_row_beg + sp_N_beg) + threadIdx.x * vector_len;
  glb_otdata_trd_scl_ptr[0] = reg_otdata_trd_scl_imm[0];
  glb_otdata_trd_scl_ptr[1] = reg_otdata_trd_scl_imm[1];
  glb_otdata_trd_scl_ptr[2] = reg_otdata_trd_scl_imm[2];
  glb_otdata_trd_scl_ptr[3] = reg_otdata_trd_scl_imm[3];
}

__global__ void sddmm_kernel_block_reduce128(
  int group_num, int N, int reduce_dim,
  const int* __restrict__ lhs_row_ids,
  const int* __restrict__ col_ind,
  const float* __restrict__ sp_data,
  const float* __restrict__ lhs,
  const float* __restrict__ rhs,
  float *out, int group_size) {

  static constexpr int ReduceDim = 128; // 32 in 32residue, 128 in 128residue
  static constexpr int vector_len = 4;
  static constexpr int inner_pipeline_stages = 4;

  int group_id = blockIdx.x * 4 + threadIdx.y;
  int sp_N_beg = blockIdx.y * 32;

  auto this_block = cooperative_groups::this_thread_block();
  auto octet = cooperative_groups::tiled_partition<8>(this_block);
  if(group_id >= group_num) return;

  int lhs_row_id = lhs_row_ids[group_id];
  int sp_row_beg = group_id * group_size;

  __shared__ __align__(16) int shm_colind_tbk_scl_ptr[32 * vector_len];
  float4 reg_spdata_trd_vec_imm = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  int *shm_colind_oct_scl_ptr = shm_colind_tbk_scl_ptr + 32 * threadIdx.y;

  __align__(16) float reg_lhdata_trd_scl_imm[vector_len];
  __align__(16) float reg_rhdata_trd_scl_imm[2 * vector_len * inner_pipeline_stages];
  float4 *reg_lhdata_trd_vec_ptr = reinterpret_cast<float4*>(reg_lhdata_trd_scl_imm);
  float4 *reg_rhdata_trd_vec_ptr = reinterpret_cast<float4*>(reg_rhdata_trd_scl_imm);

  float reg_accumu_trd_scl_imm[32] = {};
  float reg_otdata_trd_scl_imm[vector_len];

  PowerSparseBarrier barrier(threadIdx.y);

  int4* shm_colind_oct_vec_ptr = reinterpret_cast<int4*>(shm_colind_oct_scl_ptr);
  const int4* glb_colind_oct_vec_ptr = reinterpret_cast<const int4*>(col_ind + sp_row_beg + sp_N_beg);
  cooperative_groups::memcpy_async(octet, shm_colind_oct_vec_ptr, 8, glb_colind_oct_vec_ptr, 8);

  const float4* glb_spdata_trd_vec_ptr = reinterpret_cast<const float4*>(sp_data + sp_row_beg + sp_N_beg) + threadIdx.x;
  reg_spdata_trd_vec_imm = __ldg(glb_spdata_trd_vec_ptr);

  const float4 *glb_lhdata_trd_vec_ptr = reinterpret_cast<const float4*>(lhs + lhs_row_id * reduce_dim) + threadIdx.x;
  const float4 *glb_rhdata_trd_vec_ptr = reinterpret_cast<const float4*>(rhs) + threadIdx.x;

  cooperative_groups::wait(octet);

  #pragma unroll
  for(int i = 0; i < ReduceDim / 32; ++i){
    *(reg_lhdata_trd_vec_ptr) = __ldg(glb_lhdata_trd_vec_ptr);
    glb_lhdata_trd_vec_ptr += 8;

    inner_pipeline(reg_rhdata_trd_vec_ptr, glb_rhdata_trd_vec_ptr, ReduceDim, vector_len, shm_colind_oct_scl_ptr, barrier,
      reg_lhdata_trd_scl_imm, reg_rhdata_trd_scl_imm, reg_accumu_trd_scl_imm);

    glb_rhdata_trd_vec_ptr = OffsetCast<const float4>(glb_rhdata_trd_vec_ptr, 32 * sizeof(float));
  }

  shuffle_result(barrier.ThreadMask(), reg_accumu_trd_scl_imm);

  reg_otdata_trd_scl_imm[0] = reg_accumu_trd_scl_imm[0] * reg_spdata_trd_vec_imm.x;
  reg_otdata_trd_scl_imm[1] = reg_accumu_trd_scl_imm[1] * reg_spdata_trd_vec_imm.y;
  reg_otdata_trd_scl_imm[2] = reg_accumu_trd_scl_imm[2] * reg_spdata_trd_vec_imm.z;
  reg_otdata_trd_scl_imm[3] = reg_accumu_trd_scl_imm[3] * reg_spdata_trd_vec_imm.w;

  float *glb_otdata_trd_scl_ptr = reinterpret_cast<float*>(out + sp_row_beg + sp_N_beg) + threadIdx.x * vector_len;
  glb_otdata_trd_scl_ptr[0] = reg_otdata_trd_scl_imm[0];
  glb_otdata_trd_scl_ptr[1] = reg_otdata_trd_scl_imm[1];
  glb_otdata_trd_scl_ptr[2] = reg_otdata_trd_scl_imm[2];
  glb_otdata_trd_scl_ptr[3] = reg_otdata_trd_scl_imm[3];
}

} // namespace PowerSparse
