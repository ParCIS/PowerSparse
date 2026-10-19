#include "PowerSparse_SpMM.h"
#include "cooperative_groups.h"
#include <cooperative_groups/memcpy_async.h>

namespace PowerSparse {

namespace {

__device__ __forceinline__ void inner_pipeline(
  int *shm_colind_oct_scl_ptr, float *shm_spdata_oct_scl_ptr,
  const float4 *glb_dsdata_trd_vec_ptr, int vector_len, int N,
  PowerSparseBarrier octet_barrier, float4 (&B_fragment_thread_vec)[2][8],
  float *reg_accumu_trd_vec_imm) {

  const int *B_row_id = shm_colind_oct_scl_ptr;

  #pragma unroll
  for(int m_stage = 0; m_stage < 8; m_stage++) { // STAGE = 8
    int ds_data_offset = B_row_id[m_stage] * N;
    float4 ds_data = glb_dsdata_trd_vec_ptr[ds_data_offset / vector_len];
    B_fragment_thread_vec[0][m_stage] = ds_data;
  }
  B_row_id += 8;

  int ds_data_offsets[8];
  #pragma unroll
  for(int m_stage = 0; m_stage < 8; m_stage++) {
    ds_data_offsets[m_stage] = B_row_id[m_stage] * N;
  }

  #pragma unroll
  for(int i = 1; i < 4; i++) {
    #pragma unroll
    for(int m_stage = 0; m_stage < 8; m_stage++) {
      int ds_data_offset = ds_data_offsets[m_stage];
      float4 ds_data = glb_dsdata_trd_vec_ptr[ds_data_offset / vector_len];
      B_fragment_thread_vec[(i&1)][m_stage] = ds_data;
    }
    B_row_id += 8;

    octet_barrier.Sync();

    float reg_sp_data[8];
    int shmm_sp_data_octet_offset = (i-1)*8;
    #pragma unroll
    for(int m_stage = 0; m_stage < 8; m_stage++) {
      // The last stage has no next tile; reading it would pass the shared array.
      if (i < 3) ds_data_offsets[m_stage] = B_row_id[m_stage] * N;

      reg_sp_data[m_stage] = shm_spdata_oct_scl_ptr[shmm_sp_data_octet_offset + m_stage];
      float4 reg_ds_data = B_fragment_thread_vec[(i-1)&1][m_stage];
      float_times_float4_then_accumulate(reg_sp_data[m_stage], reg_ds_data, reg_accumu_trd_vec_imm);
    }
  }

}

} // anonymous namespace

__global__ void spmm_kernel_block(
  int group_size, int group_num, int N,
  const float* __restrict__ sp_data,
  const int * __restrict__ col_ind,
  const int* __restrict__ group_des,
  const float *B, float *C) {

  int group_id = blockIdx.x * 4 + threadIdx.y;
  int N_beg = blockIdx.y * 32;

  auto this_TB = cooperative_groups::this_thread_block();
  auto octet = cooperative_groups::tiled_partition<8>(this_TB);
  if(group_id >= group_num) return;

  int group_beg = group_id * group_size;
  int nonzeros = group_size;

  constexpr int vector_len = 4;
  constexpr int inner_pipeline_stages = 4;

  __shared__ __align__(16) float shm_spdata_tbk_scl_ptr[2 * 32 * vector_len];
  __shared__ __align__(16) int shm_colind_tbk_scl_ptr[32 * vector_len];

  float *shm_spdata_oct_scl_ptr = shm_spdata_tbk_scl_ptr + 8 * vector_len * threadIdx.y;
  int *shm_colind_oct_scl_ptr = shm_colind_tbk_scl_ptr + 8 * vector_len * threadIdx.y;

  PowerSparseBarrier octet_barrier(threadIdx.y);

  __align__(16) float reg_accumu_trd_vec_imm[4] = {};

  const float4 *glb_dsdata_trd_vec_ptr = reinterpret_cast<const float4*>(B + N_beg) + threadIdx.x;
  const float4 *glb_spdata_oct_vec_ptr = reinterpret_cast<const float4*>(sp_data + group_beg);
  const int4 *glb_colind_oct_vec_ptr = reinterpret_cast<const int4*>(col_ind + group_beg);
  float4* shm_spdata_oct_vec_ptr = reinterpret_cast<float4*>(shm_spdata_oct_scl_ptr);
  int4* shm_colind_oct_vec_ptr = reinterpret_cast<int4*>(shm_colind_oct_scl_ptr);

  cooperative_groups::memcpy_async(octet, shm_colind_oct_vec_ptr, 8, glb_colind_oct_vec_ptr, 8);
  cooperative_groups::memcpy_async(octet, shm_spdata_oct_vec_ptr, 8, glb_spdata_oct_vec_ptr, 8);
  glb_colind_oct_vec_ptr += 8;
  glb_spdata_oct_vec_ptr += 8;
  cooperative_groups::wait(octet);
  nonzeros -= 32;

  float4 B_fragment_thread_vec[2][8];

  inner_pipeline(
    shm_colind_oct_scl_ptr, shm_spdata_oct_scl_ptr, glb_dsdata_trd_vec_ptr,
    vector_len, N, octet_barrier, B_fragment_thread_vec, reg_accumu_trd_vec_imm);

  octet_barrier.Sync();

  if(nonzeros >= 32) {
    cooperative_groups::memcpy_async(octet, shm_colind_oct_vec_ptr, 8, glb_colind_oct_vec_ptr, 8);
    cooperative_groups::memcpy_async(octet, shm_spdata_oct_vec_ptr + 8 * vector_len, 8, glb_spdata_oct_vec_ptr, 8);
    glb_colind_oct_vec_ptr += 8;
    glb_spdata_oct_vec_ptr += 8;
  }

  float reg_sp_data[8];
  int shmm_sp_data_octet_offset = (inner_pipeline_stages-1)*8;
  #pragma unroll
  for(int m_stage = 0; m_stage < 8; m_stage++) {
    reg_sp_data[m_stage] = shm_spdata_oct_scl_ptr[shmm_sp_data_octet_offset + m_stage];
    float4 reg_ds_data = B_fragment_thread_vec[(inner_pipeline_stages-1)&1][m_stage];
    float_times_float4_then_accumulate(reg_sp_data[m_stage], reg_ds_data, reg_accumu_trd_vec_imm);
  }

  int shmm_col_ind_flag = 1;
  for(;nonzeros >= 32; nonzeros -= 32, shmm_col_ind_flag ^= 1) {
    cooperative_groups::wait(octet);

    shm_spdata_oct_scl_ptr = shm_spdata_tbk_scl_ptr + 32 * threadIdx.y + shmm_col_ind_flag * 128;

    inner_pipeline(
      shm_colind_oct_scl_ptr, shm_spdata_oct_scl_ptr, glb_dsdata_trd_vec_ptr,
      vector_len, N, octet_barrier, B_fragment_thread_vec, reg_accumu_trd_vec_imm);

    octet_barrier.Sync();

    if(nonzeros >= 2 * 32) {
      cooperative_groups::memcpy_async(octet, shm_colind_oct_vec_ptr, 8, glb_colind_oct_vec_ptr, 8);
      cooperative_groups::memcpy_async(octet, shm_spdata_oct_vec_ptr + (shmm_col_ind_flag^1) * 32, 8, glb_spdata_oct_vec_ptr, 8);
      glb_colind_oct_vec_ptr += 8;
      glb_spdata_oct_vec_ptr += 8;
    }

    shmm_sp_data_octet_offset = (inner_pipeline_stages-1)*8;
    #pragma unroll
    for(int m_stage = 0; m_stage < 8; m_stage++) {
      reg_sp_data[m_stage] = shm_spdata_oct_scl_ptr[shmm_sp_data_octet_offset + m_stage];
      float4 reg_ds_data = B_fragment_thread_vec[(inner_pipeline_stages-1)&1][m_stage];
      float_times_float4_then_accumulate(reg_sp_data[m_stage], reg_ds_data, reg_accumu_trd_vec_imm);
    }
  }

  int des_row_id = group_des[group_id];
  const int output_offset = des_row_id * N + N_beg;
  float *C_output_ptr = C + output_offset + threadIdx.x * 4;
  #pragma unroll
  for(int v = 0; v < vector_len; v++) {
    atomicAdd(C_output_ptr + v, reg_accumu_trd_vec_imm[v]);
  }
}

} // namespace PowerSparse
