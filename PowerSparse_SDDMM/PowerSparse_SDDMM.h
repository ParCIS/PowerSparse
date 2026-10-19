#ifndef POWERSPARSE_SDDMM_H
#define POWERSPARSE_SDDMM_H

#include "../utils/matrix_utils.h"
#ifdef __CUDACC__
#include "../utils/common_utils.h"
#endif

namespace PowerSparse {

// FP32 sampled (lhs * rhs^T) times sparse values; N is 32 or 128.
// Dense inputs are row-major. Output is in workset order, not original CSR order.
// Zero every output workset before each independent call. Streams follow SpMM.
void SDDMM(
  const PowerSparseMatrixGPU &ps_matrix_gpu, int N,
  const float *lhs_gpu, const float *rhs_gpu,
  float **sddmm_output_gpu,
  int ps_stream_num, const cudaStream_t *ps_stream);

#ifdef __CUDACC__
// block kernel (N = 32 and 128)
__global__ void sddmm_kernel_block_reduce32(
  int m, int n, int k,
  const int* __restrict__ row_indices,
  const int* __restrict__ column_indices,
  const float* __restrict__ values,
  const float* __restrict__ lhs_matrix,
  const float* __restrict__ rhs_matrix,
  float *out, int group_size);

__global__ void sddmm_kernel_block_reduce128(
  int m, int n, int k,
  const int* __restrict__ row_indices,
  const int* __restrict__ column_indices,
  const float* __restrict__ values,
  const float* __restrict__ lhs_matrix,
  const float* __restrict__ rhs_matrix,
  float *out, int group_size);

// residue kernel (N = 32 and 128)
__global__ void sddmm_kernel_residue_reduce32(
  int m, int k, int n,
  const int* __restrict__ row_indices,
  const int* __restrict__ row_offsets,
  const int* __restrict__ column_indices,
  const float* __restrict__ values,
  const float* __restrict__ lhs_matrix,
  const float* __restrict__ rhs_matrix,
  float *out);

__global__ void sddmm_kernel_residue_reduce128(
  int m, int k, int n,
  const int* __restrict__ row_indices,
  const int* __restrict__ row_offsets,
  const int* __restrict__ column_indices,
  const float* __restrict__ values,
  const float* __restrict__ lhs_matrix,
  const float* __restrict__ rhs_matrix,
  float *out);

#endif  // __CUDACC__

} // namespace PowerSparse

#endif // POWERSPARSE_SDDMM_H
