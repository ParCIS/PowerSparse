#include "PowerSparse_SDDMM.h"
#include "../utils/cuda_utils.h"
#include <stdexcept>

namespace PowerSparse {

void SDDMM(const PowerSparseMatrixGPU& matrix, int N, const float* lhs,
           const float* rhs, float** output, int stream_count,
           const cudaStream_t* streams) {
  if (N != 32 && N != 128)
    throw std::invalid_argument("SDDMM dispatcher expects a benchmark width of N=32 or N=128");
  ValidateDenseShape(matrix.MatrixRow(), matrix.MatrixCol(), N);
  if (!lhs || !rhs || !output || stream_count != matrix.GroupTypeNum() ||
      (stream_count > 1 && !streams))
    throw std::invalid_argument("Invalid SDDMM buffers or stream count");
  // Validate every workset before submitting any work.
  for (int s = 0; s < stream_count; ++s) {
    if (matrix.GroupNum()[s] == 0) continue;
    if (!output[s]) throw std::invalid_argument("Missing SDDMM output workset");
    if (s > 0 && GROUP_SIZE(s) / 32 > 65535)
      throw std::invalid_argument("SDDMM workset exceeds the CUDA grid-y limit");
  }
  for (int s = stream_count - 1; s >= 0; --s) {
    const int count = matrix.GroupNum()[s];
    if (count == 0) continue;
    const cudaStream_t stream = stream_count == 1 ? nullptr : streams[s];
    const dim3 grid(CEIL(count, 4), s == 0 ? 1 : GROUP_SIZE(s) / 32);
    const int cols = matrix.MatrixCol();
    const auto* rows = matrix.GroupDestination()[s];
    const auto* indices = matrix.WorksetColInd()[s];
    const auto* values = matrix.WorksetSpData()[s];
    if (s == 0) {
      if (N == 32)
        sddmm_kernel_residue_reduce32<<<grid, dim3(8, 4), 0, stream>>>(
          count, cols, N, rows, matrix.ResidueOffset(), indices, values, lhs, rhs, output[s]);
      else
        sddmm_kernel_residue_reduce128<<<grid, dim3(32, 4), 0, stream>>>(
          count, cols, N, rows, matrix.ResidueOffset(), indices, values, lhs, rhs, output[s]);
    } else {
      if (N == 32)
        sddmm_kernel_block_reduce32<<<grid, dim3(8, 4), 0, stream>>>(
          count, cols, N, rows, indices, values, lhs, rhs, output[s], GROUP_SIZE(s));
      else
        sddmm_kernel_block_reduce128<<<grid, dim3(8, 4), 0, stream>>>(
          count, cols, N, rows, indices, values, lhs, rhs, output[s], GROUP_SIZE(s));
    }
    CUDA_CALL(cudaGetLastError());
  }
  if (stream_count > 1) CUDA_CALL(cudaDeviceSynchronize());
}

}  // namespace PowerSparse
