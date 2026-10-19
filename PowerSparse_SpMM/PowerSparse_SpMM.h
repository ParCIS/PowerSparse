#ifndef POWERSPARSE_SPMM_H
#define POWERSPARSE_SPMM_H

#include "../utils/matrix_utils.h"
#ifdef __CUDACC__
#include "../utils/common_utils.h"
#endif

namespace PowerSparse {

// FP32 C = A * B, with row-major B and C. Zero C before each independent call.
// Supply GroupTypeNum() blocking streams (nullptr is allowed for a single workset).
// Multiple worksets synchronize the device; a single workset uses the default stream.
void SpMM(
  const PowerSparseMatrixGPU &ps_matrix_gpu, int N,
  const float *B_gpu, float *C_gpu,
  int ps_stream_num, const cudaStream_t *ps_stream);

#ifdef __CUDACC__
// block kernel
__global__ void spmm_kernel_block(
  int group_size, int group_num, int N,
  const float* __restrict__ sp_data,
  const int * __restrict__ col_ind,
  const int* __restrict__ group_des,
  const float *B, float *C);

// residue kernel
__global__ void spmm_kernel_residue(
  int group_num, int N,
  const float* __restrict__ sp_data,
  const int* __restrict__ col_ind,
  const int* __restrict__ group_des,
  const float *B, float *C,
  const int* __restrict__ residue_offsets,
  const bool *residue_atomic);

#endif  // __CUDACC__

} // namespace PowerSparse

#endif // POWERSPARSE_SPMM_H
