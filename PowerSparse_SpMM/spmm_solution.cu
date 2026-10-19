#include "PowerSparse_SpMM.h"
#include "../utils/cuda_utils.h"
#include <stdexcept>

namespace PowerSparse {

void SpMM(const PowerSparseMatrixGPU& matrix, int N, const float* B, float* C,
          int stream_count, const cudaStream_t* streams) {
  if (N <= 0 || N % 32 != 0 || N / 32 > 65535)
    throw std::invalid_argument("SpMM requires N to be a positive multiple of 32, with N/32 <= 65535");
  ValidateDenseShape(matrix.MatrixRow(), matrix.MatrixCol(), N);
  if (!B || !C || stream_count != matrix.GroupTypeNum() ||
      (stream_count > 1 && !streams))
    throw std::invalid_argument("Invalid SpMM buffers or stream count");
  const dim3 block(8, 4);
  for (int s = stream_count - 1; s >= 0; --s) {
    const int count = matrix.GroupNum()[s];
    if (count == 0) continue;
    const dim3 grid(CEIL(count, 4), N / 32);
    const cudaStream_t stream = stream_count == 1 ? nullptr : streams[s];
    if (s == 0) {
      spmm_kernel_residue<<<grid, block, 0, stream>>>(
        count, N, matrix.WorksetSpData()[s], matrix.WorksetColInd()[s],
        matrix.GroupDestination()[s], B, C, matrix.ResidueOffset(), matrix.ResidueAtomic());
    } else {
      spmm_kernel_block<<<grid, block, 0, stream>>>(
        GROUP_SIZE(s), count, N, matrix.WorksetSpData()[s], matrix.WorksetColInd()[s],
        matrix.GroupDestination()[s], B, C);
    }
    CUDA_CALL(cudaGetLastError());
  }
  // Preserve the synchronization behavior of the original multi-workset API.
  if (stream_count > 1) CUDA_CALL(cudaDeviceSynchronize());
}

}  // namespace PowerSparse
