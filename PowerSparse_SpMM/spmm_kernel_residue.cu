#include "PowerSparse_SpMM.h"

namespace PowerSparse {

__global__ void spmm_kernel_residue(
  int group_num, int N,
  const float* __restrict__ sp_data,
  const int* __restrict__ col_ind,
  const int* __restrict__ group_des,
  const float *B, float *C,
  const int* __restrict__ residue_offsets,
  const bool *residue_atomic) {

  int group_id = blockIdx.x * 4 + threadIdx.y;
  int N_beg = blockIdx.y << 5;

  if(group_id >= group_num) return;

  int group_beg = __ldg(residue_offsets + group_id);
  int nonzeros = __ldg(residue_offsets + group_id + 1) - group_beg;

  __shared__ __align__(16) float shm_spdata_tbk_scl_ptr[128];
  __shared__ __align__(16) int shm_colind_tbk_scl_ptr[128];

  float *shm_spdata_oct_scl_ptr = shm_spdata_tbk_scl_ptr + 32 * threadIdx.y;
  int *shm_colind_oct_scl_ptr = shm_colind_tbk_scl_ptr + 32 * threadIdx.y;

  PowerSparseBarrier octet_barrier(threadIdx.y); // 用于warp同步

  __align__(16) float reg_accumu_trd_vec_imm[4] = {};

  const float4 *glb_dsdata_trd_vec_ptr = reinterpret_cast<const float4*>(B + N_beg) + threadIdx.x;
  float4* shm_spdata_trd_vec_ptr = reinterpret_cast<float4*>(shm_spdata_oct_scl_ptr) + threadIdx.x;
  int4* shm_colind_trd_vec_ptr = reinterpret_cast<int4*>(shm_colind_oct_scl_ptr) + threadIdx.x;

  const float4 pad_float4_zero = make_float4(0, 0, 0, 0);
  const int4 pad_int4_zero = make_int4(0, 0, 0, 0);
  *shm_colind_trd_vec_ptr = pad_int4_zero;
  *shm_spdata_trd_vec_ptr = pad_float4_zero;

  octet_barrier.Sync();

  int sp_data_offset = group_beg + threadIdx.x;
  int _nonzeros = nonzeros;
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    if (_nonzeros <= threadIdx.x) break;
    shm_spdata_oct_scl_ptr[i * 8 + threadIdx.x] = sp_data[sp_data_offset + i * 8];
    shm_colind_oct_scl_ptr[i * 8 + threadIdx.x] = col_ind[sp_data_offset + i * 8];
    _nonzeros -= 8;
  }
  octet_barrier.Sync();

  #pragma unroll
  for(int i = 0; i < 8; i++) {
    if(nonzeros <= 0) break;
    #pragma unroll
    for(int j = 0; j < 4; j++){
      const int nnz_idx = i * 4 + j;
      int ds_data_offset = shm_colind_oct_scl_ptr[nnz_idx] * N;
      float reg_sp_data = shm_spdata_oct_scl_ptr[nnz_idx];
      float4 reg_ds_data = __ldg(glb_dsdata_trd_vec_ptr + ds_data_offset / 4);
      float_times_float4_then_accumulate(reg_sp_data, reg_ds_data, reg_accumu_trd_vec_imm);
    }
    nonzeros -= 4;
  }

  int des_row_id = group_des[group_id];
  bool atomic_flag = residue_atomic[group_id];
  const int output_offset = des_row_id * N + N_beg;
  if(atomic_flag) {
    float *C_output_ptr = C + output_offset + threadIdx.x * 4;
    #pragma unroll
    for(int v = 0; v < 4; v++) {
      atomicAdd(C_output_ptr + v, reg_accumu_trd_vec_imm[v]);
    }
  }
  else {
    float4 *C_output_ptr = reinterpret_cast<float4*>(C + output_offset) + threadIdx.x;
    *C_output_ptr = *(reinterpret_cast<const float4*>(reg_accumu_trd_vec_imm));
  }
}

} // namespace PowerSparse
