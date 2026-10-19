#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <stdexcept>

#define GROUP_SIZE(I) (32 * (1 << ((I) - 1)))
#define CEIL(X, Y) ((X) / (Y) + ((X) % (Y) != 0))
#define FLOOR(X, Y) ((X) / (Y))

namespace PowerSparse {

enum SigBitMode { PPL = 0, FFC = 1 };
enum FreqSort { FreqNoSort = 0, FreqDoSort = 1 };
enum GroupSort { GroupNoSort = 0, GroupDoSort = 1 };

// Owns the GPU worksets built from host CSR arrays. Construct and destroy on
// the same CUDA device; complete outstanding operator calls before destruction.
class PowerSparseMatrixGPU {
 public:
  PowerSparseMatrixGPU(int row_num, int col_num, int nnz_num,
                       const int* row_ptr, const int* col_ind, const float* sp_data);
  ~PowerSparseMatrixGPU() { Release(); }
  PowerSparseMatrixGPU(const PowerSparseMatrixGPU&) = delete;
  PowerSparseMatrixGPU& operator=(const PowerSparseMatrixGPU&) = delete;
  PowerSparseMatrixGPU(PowerSparseMatrixGPU&&) = delete;
  PowerSparseMatrixGPU& operator=(PowerSparseMatrixGPU&&) = delete;

  int MatrixRow() const { return _matrix_row; }
  int MatrixCol() const { return _matrix_col; }
  int MatrixNNZ() const { return _matrix_nnz; }
  int GroupTypeNum() const { return _group_type_num; }
  int GroupFineSize() const { return _group_fine_size; }
  const int* GroupNum() const { return _group_num; }
  const int* const* WorksetColInd() const { return _workset_col_ind; }
  const float* const* WorksetSpData() const { return _workset_sp_data; }
  const int* const* GroupDestination() const { return _group_des; }
  const int* ResidueOffset() const { return _residue_offset; }
  const bool* ResidueAtomic() const { return _residue_atomic; }

 private:
  void Release() noexcept {
    for (int s = 0; s < _group_type_num; ++s) {
      if (_workset_col_ind) cudaFree(_workset_col_ind[s]);
      if (_workset_sp_data) cudaFree(_workset_sp_data[s]);
      if (_group_des) cudaFree(_group_des[s]);
    }
    delete[] _group_num;
    delete[] _workset_col_ind;
    delete[] _workset_sp_data;
    delete[] _group_des;
    cudaFree(_residue_offset);
    cudaFree(_residue_atomic);
  }

  int _matrix_row, _matrix_col, _matrix_nnz;
  int _group_type_num = 0, _group_fine_size = 16;
  int* _group_num = nullptr;
  int** _workset_col_ind = nullptr;
  float** _workset_sp_data = nullptr;
  int** _group_des = nullptr;
  int* _residue_offset = nullptr;
  bool* _residue_atomic = nullptr;
  PowerSparse::SigBitMode _SigBit_mode = FFC;
  PowerSparse::FreqSort _freq_sort = FreqDoSort;
  PowerSparse::GroupSort _group_sort = GroupDoSort;
  int _ffc = 2;
};

// The original kernels use signed 32-bit dense element offsets.
inline void ValidateDenseShape(int rows, int cols, int n) {
  if (n <= 0 || int64_t(rows) * n > std::numeric_limits<int>::max() ||
      int64_t(cols) * n > std::numeric_limits<int>::max())
    throw std::invalid_argument("Dense dimensions exceed the 32-bit kernel indexing range");
}

}  // namespace PowerSparse
