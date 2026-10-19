#include "matrix_utils.h"
#include "cuda_utils.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace PowerSparse {
struct MATRIX_SETTING {
  int group_type_num;
  int group_fine_size;
  SigBitMode SigBit_mode;
  int ffc;
  FreqSort freq_sort;
  GroupSort group_sort;
};

namespace {

/**
 * @brief Analyze matrix and generate matrix_setting
 */
MATRIX_SETTING AnalyzeMatrix(
  const int row_num, const int col_num, const int nnz_num,
  const int *row_ptr, const int *col_ind) {

  MATRIX_SETTING matrix_setting;

  int dev; CUDA_CALL(cudaGetDevice(&dev));
  cudaDeviceProp prop; CUDA_CALL(cudaGetDeviceProperties(&prop, dev));
  int SM_num = prop.multiProcessorCount;
  int max_TB_per_SM = 8;  // Original workset-size heuristic, retained for both GPUs.
  int first_wave_TB = max_TB_per_SM * SM_num;
  int group_threshold = first_wave_TB * 4;

  int S = 0;  // Number of worksets/streams, including the residue workset.
  int _group_num = group_threshold + 1;
  for (; _group_num >= group_threshold; S++) {
    int group_size = GROUP_SIZE(S + 1);
    _group_num = 0;
    for (int i = 0; i < row_num; i++) {
      int row_nnz = row_ptr[i + 1] - row_ptr[i];
      _group_num += FLOOR(row_nnz, group_size);
    }
  }

  matrix_setting.group_type_num = S;
  matrix_setting.group_fine_size = 16;
  matrix_setting.SigBit_mode = FFC;
  matrix_setting.ffc = 2;
  matrix_setting.freq_sort = FreqDoSort;
  matrix_setting.group_sort = GroupDoSort;

  return matrix_setting;
}

void GenerateSigBit(
  SigBitMode mode, int ffc, int col_num, int nnz_num,
  const std::vector<int> &col_nnz,
  std::vector<int> &SigBit, int &signature_len) {

  std::vector<int> col_fq_id(col_num);
  std::iota(col_fq_id.begin(), col_fq_id.end(), 0);
  std::sort(col_fq_id.begin(), col_fq_id.end(),
    [&col_nnz](const int& col_id_1, const int& col_id_2) {
    return col_nnz[col_id_1] > col_nnz[col_id_2];
  });

  if (mode == PPL) {
    int _index = 0, _sig_pos = 0;
    int64_t _bit_num = 1;
    while (_index < col_num) {
      for (int i = 0; i < _bit_num && _index < col_num; i++) {
        SigBit[col_fq_id[_index]] = _sig_pos;
        _index += 1;
      }
      _sig_pos += 1;
      _bit_num *= 2;
    }

    signature_len = static_cast<int>(std::ceil(std::log2(double(col_num) + 1)));
  }
  else if (mode == FFC) {
    int _index = 0, _sig_pos = 0;
    int nnz_thrd = 0;
    while (_index < std::min(ffc, col_num)) {
      SigBit[col_fq_id[_index]] = _sig_pos;
      nnz_thrd += col_nnz[col_fq_id[_index]];
      _index += 1;
    }
    _sig_pos += 1;
    int nnz_temp = 0;
    while (_index < col_num) {
      nnz_temp = 0;
      const int begin = _index;
      while (_index < col_num) {
        nnz_temp += col_nnz[col_fq_id[_index]];
        if (nnz_temp < nnz_thrd || _index == begin) {
          SigBit[col_fq_id[_index]] = _sig_pos;
          _index += 1;
        }
        else {
          break;
        }
      }
      _sig_pos += 1;
    }

    signature_len = _sig_pos;
  }

  if (signature_len < 32) signature_len = 32;
  else if (signature_len % 32 < 16) signature_len = FLOOR(signature_len, 32) * 32;
  else signature_len = CEIL(signature_len, 32) * 32;
  for (int i = 0; i < col_num; i++) {
    if (SigBit[i] >= signature_len) {
      SigBit[i] = signature_len - 1;
    }
  }
}

} // anonymous namespace

void GetGroupNumArray(
  const int *row_ptr, int row_num,
  int group_type_num, int group_fine_size,
  int *group_num) {

  if (group_type_num == 1) {
    for (int i = 0; i < row_num; i++) {
      int row_nnz = row_ptr[i + 1] - row_ptr[i];
      group_num[0] += CEIL(row_nnz, group_fine_size);
    }
  }
  else {
    for (int i = 0; i < row_num; i++) {
      int row_nnz = row_ptr[i + 1] - row_ptr[i];
      for (int s = group_type_num - 1; s > 0; s--) {
        int row_group_num = FLOOR(row_nnz, GROUP_SIZE(s));
        group_num[s] += row_group_num;
        row_nnz -= GROUP_SIZE(s) * row_group_num;
      }
      if (row_nnz) {
        group_num[0] += 1;
      }
    }
  }
}

void GetGroupMetaInfo(
  const int *row_ptr, int row_num, const int *group_num,
  int group_type_num, int group_fine_size,
  std::vector<int> &h_residue_offset,
  thrust::host_vector<bool> &h_residue_atomic,
  std::vector<std::vector<int>> &h_group_des,
  std::vector<std::vector<int>> &h_group_num_prefix) {

  for (int i = 0; i < group_type_num; i++) {
    h_group_des[i].resize(group_num[i], 0);
    h_group_num_prefix[i].resize(row_num + 1, 0);
  }

  std::vector<int> group_counter(group_type_num, 0);
  if (group_type_num == 1) {
    for (int i = 0; i < row_num; i++) {
      int row_nnz = row_ptr[i + 1] - row_ptr[i];
      int row_group_num = CEIL(row_nnz, group_fine_size);
      for (int j = 0; j < row_group_num; j++, group_counter[0]++) {
        h_residue_atomic[group_counter[0]] = (row_group_num > 1) ? true : false;
        h_residue_offset[group_counter[0] + 1] = h_residue_offset[group_counter[0]] + std::min(row_nnz - j * group_fine_size, group_fine_size);
        h_group_des[0][group_counter[0]] = i;
      }
      h_group_num_prefix[0][i + 1] = h_group_num_prefix[0][i] + row_group_num;
    }
  }
  else {
    for (int i = 0; i < row_num; i++) {
      int row_nnz = row_ptr[i + 1] - row_ptr[i];
      bool res_ato = (row_nnz > 32) ? true : false;
      for (int s = group_type_num - 1; s > 0; s--) {
        int row_group_num = FLOOR(row_nnz, GROUP_SIZE(s));
        for (int j = 0; j < row_group_num; j++, group_counter[s]++) {
          h_group_des[s][group_counter[s]] = i;
        }
        row_nnz -= GROUP_SIZE(s) * row_group_num;
        h_group_num_prefix[s][i + 1] = h_group_num_prefix[s][i] + row_group_num;
      }
      if (row_nnz) {
        h_residue_atomic[group_counter[0]] = res_ato;
        h_residue_offset[group_counter[0] + 1] = h_residue_offset[group_counter[0]] + row_nnz;
        h_group_des[0][group_counter[0]] = i;
        group_counter[0] += 1;
        h_group_num_prefix[0][i + 1] = h_group_num_prefix[0][i] + 1;
      }
      else {
        h_group_num_prefix[0][i + 1] = h_group_num_prefix[0][i];
      }
    }
  }
}

struct NNZPoint {
  int row_id;
  int col_id;
  float value;
  int col_nnz;

  __host__ __device__
  bool operator<(const NNZPoint& other) const {
    if (row_id != other.row_id) {
      return row_id < other.row_id;
    }
    return col_nnz > other.col_nnz;
  }
};

struct NNZPoint2 {
  int group_id;
  int col_id;
  float value;

  __host__ __device__
  bool operator<(const NNZPoint2& other) const {
    if (group_id != other.group_id) {
      return group_id < other.group_id;
    }
    return col_id < other.col_id;
  }
};

__global__ void CopyCSRToNNZ(
  int row_num, const int *row_ptr, const int *col_ind,
  const float *sp_data, const int *col_nnz, NNZPoint *nnzs) {

  int row_id = blockIdx.x * 4 + threadIdx.y;
  int lane_id = threadIdx.x;
  if (row_id >= row_num) return;

  int row_beg = row_ptr[row_id];
  int row_end = row_ptr[row_id + 1];
  int pos = row_beg + lane_id;
  while (pos < row_end) {
    nnzs[pos].row_id = row_id;
    nnzs[pos].col_id = col_ind[pos];
    nnzs[pos].value = sp_data[pos];
    nnzs[pos].col_nnz = col_nnz[col_ind[pos]];
    pos += 32;
  }
}

typedef unsigned int _SignatureBase;

__global__ void GetResidueSignature(
  int row_num, int group_num, const int *row_ptr, int group_type_num, int group_fine_size, int uint_num,
  int *group_num_prefix, int *sigbit_array, NNZPoint *nnzs, _SignatureBase *signature) {

  int row_id = blockIdx.x * 4 + threadIdx.y;
  int lane_id = threadIdx.x;
  if (row_id >= row_num) return;
  int row_group_num = group_num_prefix[row_id + 1] - group_num_prefix[row_id];
  if (row_group_num == 0) return;

  int row_beg = row_ptr[row_id];
  int row_end = row_ptr[row_id + 1];
  int row_nnz = row_end - row_beg;
  for (int i = group_type_num - 1; i > 0; i--) {
    int row_group_num_ = row_nnz / GROUP_SIZE(i);
    row_beg += row_group_num_ * GROUP_SIZE(i);
    row_nnz -= row_group_num_ * GROUP_SIZE(i);
  }

  int group_size = (group_type_num == 1) ? group_fine_size : 32;
  for (int g = 0; g < row_group_num; g++, row_beg += group_size) {
    int group_id = group_num_prefix[row_id] + g;
    int pos = row_beg + lane_id;
    if (pos < min(row_beg + group_size, row_end)) {
      int col_id = nnzs[pos].col_id;
      int sig_pos = sigbit_array[col_id];
      int uint_index = sig_pos / 32;
      int ubit_index = 31 - (sig_pos % 32);
      size_t index = size_t(uint_index) * group_num + group_id;
      atomicOr(&signature[index], (1u << ubit_index));
    }
  }
}

__global__ void GetBlockSignature(
  int row_num, int group_num, const int *row_ptr, int group_type_num, int s, int uint_num,
  int *group_num_prefix, int *sigbit_array, NNZPoint *nnzs, _SignatureBase *signature) {

  int row_id = blockIdx.x * 4 + threadIdx.y;
  int lane_id = threadIdx.x;
  if (row_id >= row_num) return;
  int row_group_num = group_num_prefix[row_id + 1] - group_num_prefix[row_id];
  if (row_group_num == 0) return;

  int row_beg = row_ptr[row_id];
  int row_end = row_ptr[row_id + 1];
  int row_nnz = row_end - row_beg;
  for (int i = group_type_num - 1; i > s; i--) {
    int row_group_num = row_nnz / GROUP_SIZE(i);
    row_beg += row_group_num * GROUP_SIZE(i);
    row_nnz -= row_group_num * GROUP_SIZE(i);
  }

  for (int g = 0; g < row_group_num; g++, row_beg += GROUP_SIZE(s)) {
    int group_id = group_num_prefix[row_id] + g;
    int pos = row_beg + lane_id;
    while (pos < row_beg + GROUP_SIZE(s)) {
      int col_id = nnzs[pos].col_id;
      int sig_pos = sigbit_array[col_id];
      int uint_index = sig_pos / 32;
      int ubit_index = 31 - (sig_pos % 32);
      size_t index = size_t(uint_index) * group_num + group_id;
      atomicOr(&signature[index], (1u << ubit_index));
      pos += 32;
    }
  }
}

__global__ void GetSrcResidueWorkset(
  int row_num, const int *row_ptr, int group_type_num, int group_fine_size, int *residue_offset,
  int *group_num_prefix, NNZPoint *nnzs, NNZPoint2 *src_workset) {

  int row_id = blockIdx.x * 4 + threadIdx.y;
  int lane_id = threadIdx.x;
  if (row_id >= row_num) return;
  int row_group_num = group_num_prefix[row_id + 1] - group_num_prefix[row_id];
  if (row_group_num == 0) return;

  int row_beg = row_ptr[row_id];
  int row_end = row_ptr[row_id + 1];
  int row_nnz = row_end - row_beg;
  for (int i = group_type_num - 1; i > 0; i--) {
    int row_group_num_ = row_nnz / GROUP_SIZE(i);
    row_beg += row_group_num_ * GROUP_SIZE(i);
    row_nnz -= row_group_num_ * GROUP_SIZE(i);
  }

  int group_size = (group_type_num == 1) ? group_fine_size : 32;
  for (int g = 0; g < row_group_num; g++, row_beg += group_size) {
    int group_id = group_num_prefix[row_id] + g;
    int read_pos = row_beg + lane_id;
    int write_pos = residue_offset[group_id] + lane_id;
    if (read_pos < min(row_beg + group_size, row_end)) {
      src_workset[write_pos].group_id = group_id;
      src_workset[write_pos].col_id = nnzs[read_pos].col_id;
      src_workset[write_pos].value = nnzs[read_pos].value;
    }
  }
}

__global__ void GetSrcBlockWorkset(
  int row_num, const int *row_ptr, int group_type_num, int s,
  int *group_num_prefix, NNZPoint *nnzs, NNZPoint2 *src_workset) {

  int row_id = blockIdx.x * 4 + threadIdx.y;
  int lane_id = threadIdx.x;
  if (row_id >= row_num) return;
  int row_group_num = group_num_prefix[row_id + 1] - group_num_prefix[row_id];
  if (row_group_num == 0) return;

  int row_beg = row_ptr[row_id];
  int row_end = row_ptr[row_id + 1];
  int row_nnz = row_end - row_beg;
  for (int i = group_type_num - 1; i > s; i--) {
    int row_group_num = row_nnz / GROUP_SIZE(i);
    row_beg += row_group_num * GROUP_SIZE(i);
    row_nnz -= row_group_num * GROUP_SIZE(i);
  }

  for (int g = 0; g < row_group_num; g++, row_beg += GROUP_SIZE(s)) {
    int group_id = group_num_prefix[row_id] + g;
    int read_pos = row_beg + lane_id;
    int write_pos = group_id * GROUP_SIZE(s) + lane_id;
    while (read_pos < row_beg + GROUP_SIZE(s)) {
      src_workset[write_pos].group_id = group_id;
      src_workset[write_pos].col_id = nnzs[read_pos].col_id;
      src_workset[write_pos].value = nnzs[read_pos].value;
      read_pos += 32;
      write_pos += 32;
    }
  }
}

struct CompareDescending {
  const unsigned int* data;
  CompareDescending(const unsigned int* _data) : data(_data) {}
  __host__ __device__
  bool operator()(const unsigned int& a, const unsigned int& b) const {
    return data[a] > data[b];
  }
};

__global__ void GetResidueWorkset(
  int group_num, int *group_index,
  int *src_residue_offset, int *dst_residue_offset,
  int *workset_col_ind, float *workset_sp_data,
  int *group_des, int *src_group_des,
  bool *res_atomic, bool *src_res_atomic,
  NNZPoint2 *src_workset) {

  int group_id = blockIdx.x * 4 + threadIdx.y;
  int lane_id = threadIdx.x;
  if (group_id >= group_num) return;

  int src_id = group_index[group_id];
  if (lane_id == 0) {
    group_des[group_id] = src_group_des[src_id];
    res_atomic[group_id] = src_res_atomic[src_id];
  }
  int src_pos = src_residue_offset[src_id] + lane_id;
  int dst_pos = dst_residue_offset[group_id] + lane_id;

  if (src_pos < src_residue_offset[src_id + 1]) {
    workset_col_ind[dst_pos] = src_workset[src_pos].col_id;
    workset_sp_data[dst_pos] = src_workset[src_pos].value;
  }
}

__global__ void GetBlockWorkset(
  int group_num, int *group_index, int s,
  int *workset_col_ind, float *workset_sp_data,
  int *group_des, int *src_group_des,
  NNZPoint2 *src_workset) {

  int group_id = blockIdx.x * 4 + threadIdx.y;
  int lane_id = threadIdx.x;
  if (group_id >= group_num) return;

  int src_id = group_index[group_id];
  if (lane_id == 0) group_des[group_id] = src_group_des[src_id];
  int src_pos = src_id * GROUP_SIZE(s) + lane_id;
  int dst_pos = group_id * GROUP_SIZE(s) + lane_id;
  while (src_pos < (src_id + 1) * GROUP_SIZE(s)) {
    workset_col_ind[dst_pos] = src_workset[src_pos].col_id;
    workset_sp_data[dst_pos] = src_workset[src_pos].value;
    src_pos += 32;
    dst_pos += 32;
  }
}

void GetResidueOffset(
  int group_num, thrust::device_vector<int> &d_group_index,
  thrust::host_vector<int> &dst_residue_offset,
  std::vector<int> &residue_offset) {

  thrust::host_vector<int> group_index = d_group_index;
  dst_residue_offset[0] = 0;
  for (int i = 0; i < group_num; i++) {
    int src_id = group_index[i];
    int row_beg = residue_offset[src_id];
    int row_end = residue_offset[src_id + 1];
    dst_residue_offset[i + 1] = dst_residue_offset[i] + row_end - row_beg;
  }
}

/**
 * @brief: Build PowerSparseMatrixGPU from CSR Format
 * (using GPU to accelerate preprocessing)
 */
PowerSparseMatrixGPU::PowerSparseMatrixGPU(
  int row_num, int col_num, int nnz_num,
  const int *row_ptr, const int *col_ind,
  const float *sp_data)
    : _matrix_row(row_num), _matrix_col(col_num), _matrix_nnz(nnz_num) {
  if (row_num <= 0 || col_num <= 0 || nnz_num <= 0 ||
      row_num > std::numeric_limits<int>::max() - 32 ||
      nnz_num > std::numeric_limits<int>::max() - 32 ||
      !row_ptr || !col_ind || !sp_data) {
    throw std::invalid_argument("Expected a nonempty CSR matrix with 32-bit indices");
  }
  if (row_ptr[0] != 0 || row_ptr[row_num] != nnz_num)
    throw std::invalid_argument("Invalid CSR row-pointer endpoints");
  for (int r = 0; r < row_num; ++r)
    if (row_ptr[r] < 0 || row_ptr[r] > row_ptr[r + 1])
      throw std::invalid_argument("CSR row pointers must be nondecreasing");
  for (int p = 0; p < nnz_num; ++p)
    if (col_ind[p] < 0 || col_ind[p] >= col_num || !std::isfinite(sp_data[p]))
      throw std::invalid_argument("Invalid CSR column index or sparse value");
  try {
    // Copy CSR to the device before column-frequency sorting.
    thrust::device_vector<NNZPoint> d_nnzs(nnz_num);
    thrust::device_vector<int> d_row_ptr(row_num + 1);
    thrust::device_vector<int> d_col_ind(nnz_num);
    thrust::device_vector<float> d_sp_data(nnz_num);
    thrust::copy(row_ptr, row_ptr + row_num + 1, d_row_ptr.begin());
    thrust::copy(col_ind, col_ind + nnz_num, d_col_ind.begin());
    thrust::copy(sp_data, sp_data + nnz_num, d_sp_data.begin());

    MATRIX_SETTING matrix_setting = AnalyzeMatrix(
      row_num, col_num, nnz_num, row_ptr, col_ind);
    _group_type_num = matrix_setting.group_type_num;
    _group_fine_size = matrix_setting.group_fine_size;
    _SigBit_mode = matrix_setting.SigBit_mode;
    _ffc = matrix_setting.ffc;
    _freq_sort = matrix_setting.freq_sort;
    _group_sort = matrix_setting.group_sort;

    std::vector<int> col_nnz(col_num, 0);
    #pragma omp parallel for
    for (int i = 0; i < nnz_num; i++) {
      int col_id = col_ind[i];
      #pragma omp atomic
      col_nnz[col_id] += 1;
    }

    thrust::device_vector<int> d_col_nnz(col_nnz.begin(), col_nnz.end());

    CopyCSRToNNZ<<<dim3(CEIL(row_num, 4)), dim3(32, 4)>>>(
      row_num, thrust::raw_pointer_cast(d_row_ptr.data()), thrust::raw_pointer_cast(d_col_ind.data()),
      thrust::raw_pointer_cast(d_sp_data.data()), thrust::raw_pointer_cast(d_col_nnz.data()), thrust::raw_pointer_cast(d_nnzs.data()));
    CUDA_CALL(cudaGetLastError());
    if (_freq_sort) {
      thrust::sort(d_nnzs.begin(), d_nnzs.end());
    }

    // Partition rows into power-of-two block worksets and a residue workset.
    _group_num = new int[_group_type_num]();
    GetGroupNumArray(row_ptr, row_num, _group_type_num, _group_fine_size, _group_num);

    std::vector<int> h_residue_offset(_group_num[0] + 1, 0);
    thrust::host_vector<bool> h_residue_atomic(_group_num[0], false);
    std::vector<std::vector<int>> h_group_des(_group_type_num);
    std::vector<std::vector<int>> h_group_num_prefix(_group_type_num); // Groups before row r in workset s.
    GetGroupMetaInfo(
      row_ptr, row_num, _group_num, _group_type_num, _group_fine_size,
      h_residue_offset, h_residue_atomic, h_group_des, h_group_num_prefix);

    thrust::device_vector<int> d_residue_offset(h_residue_offset.begin(), h_residue_offset.end());
    thrust::device_vector<bool> d_residue_atomic(h_residue_atomic.begin(), h_residue_atomic.end());
    std::vector<thrust::device_vector<int>> d_group_des(_group_type_num);
    std::vector<thrust::device_vector<int>> d_group_num_prefix(_group_type_num);
    for (int i = 0; i < _group_type_num; i++) {
      d_group_des[i].resize(_group_num[i]);
      d_group_num_prefix[i] = thrust::device_vector<int>(
        h_group_num_prefix[i].begin(),
        h_group_num_prefix[i].end()
      );

      thrust::copy(h_group_des[i].begin(), h_group_des[i].end(), d_group_des[i].begin());
    }

    // Map columns to signature bits, then order groups by their signatures.
    std::vector<int> sigbit_array(col_num);
    int signature_len = 32;
    if (_group_sort) {
      GenerateSigBit(
        _SigBit_mode, _ffc, col_num, nnz_num,
        col_nnz, sigbit_array, signature_len);
    }
    thrust::device_vector<int> d_sigbit_array(sigbit_array.begin(),sigbit_array.end());

    std::vector<thrust::device_vector<int>> sort_group_index(_group_type_num);
    for (int i = 0; i < _group_type_num; i++) {
      sort_group_index[i].resize(_group_num[i]);
      thrust::sequence(sort_group_index[i].begin(), sort_group_index[i].end());
    }
    if (_group_sort) {
      for (int s = 0; s < _group_type_num; ++s) {
        if (_group_num[s] == 0) continue;
        thrust::device_vector<_SignatureBase> signatures(
            size_t(_group_num[s]) * (signature_len / 32), 0u);
        auto *d_signature = thrust::raw_pointer_cast(signatures.data());
        if (s == 0) {
          GetResidueSignature<<<dim3(CEIL(row_num, 4)), dim3(32, 4)>>>(
            row_num, _group_num[s], thrust::raw_pointer_cast(d_row_ptr.data()),
            _group_type_num, _group_fine_size, signature_len / 32,
            thrust::raw_pointer_cast(d_group_num_prefix[s].data()),
            thrust::raw_pointer_cast(d_sigbit_array.data()),
            thrust::raw_pointer_cast(d_nnzs.data()), d_signature);
          CUDA_CALL(cudaGetLastError());
        } else {
          GetBlockSignature<<<dim3(CEIL(row_num, 4)), dim3(32, 4)>>>(
            row_num, _group_num[s], thrust::raw_pointer_cast(d_row_ptr.data()),
            _group_type_num, s, signature_len / 32,
            thrust::raw_pointer_cast(d_group_num_prefix[s].data()),
            thrust::raw_pointer_cast(d_sigbit_array.data()),
            thrust::raw_pointer_cast(d_nnzs.data()), d_signature);
          CUDA_CALL(cudaGetLastError());
        }
        for (int word = signature_len / 32 - 1; word >= 0; --word)
          thrust::stable_sort(sort_group_index[s].begin(), sort_group_index[s].end(),
            CompareDescending(d_signature + size_t(word) * _group_num[s]));
      }
    }

    // Restore column order within each group before permuting whole groups.
    std::vector<thrust::device_vector<NNZPoint2>> src_workset(_group_type_num);
    for (int i = 0; i < _group_type_num; i++) {
      int array_size = 0;
      if (i == 0) array_size = h_residue_offset[_group_num[0]];
      else array_size = _group_num[i] * GROUP_SIZE(i);
      src_workset[i].resize(array_size);
    }
    GetSrcResidueWorkset<<<dim3(CEIL(row_num, 4)), dim3(32, 4)>>>(
      row_num, thrust::raw_pointer_cast(d_row_ptr.data()), _group_type_num, _group_fine_size, thrust::raw_pointer_cast(d_residue_offset.data()),
      thrust::raw_pointer_cast(d_group_num_prefix[0].data()),  thrust::raw_pointer_cast(d_nnzs.data()),
      thrust::raw_pointer_cast(src_workset[0].data()));
    CUDA_CALL(cudaGetLastError());
    if (_freq_sort) thrust::sort(src_workset[0].begin(), src_workset[0].end());

    for (int s = 1; s < _group_type_num; s++) {
      GetSrcBlockWorkset<<<dim3(CEIL(row_num, 4)), dim3(32, 4)>>>(
        row_num, thrust::raw_pointer_cast(d_row_ptr.data()), _group_type_num, s, thrust::raw_pointer_cast(d_group_num_prefix[s].data()),
        thrust::raw_pointer_cast(d_nnzs.data()), thrust::raw_pointer_cast(src_workset[s].data()));
      CUDA_CALL(cudaGetLastError());
      if (_freq_sort) thrust::sort(src_workset[s].begin(), src_workset[s].end());
    }

    // Materialize the final GPU worksets owned by this matrix.
    _workset_col_ind = new int*[_group_type_num]();
    _workset_sp_data = new float*[_group_type_num]();
    _group_des = new int*[_group_type_num]();
    for (int i = 0; i < _group_type_num; i++) {
      int array_size = 0;
      if (i == 0) array_size = h_residue_offset[_group_num[0]];
      else array_size = _group_num[i] * GROUP_SIZE(i);
      if (array_size == 0) continue;
      CUDA_CALL(cudaMalloc((void**)&(_workset_col_ind[i]), sizeof(int) * array_size));
      CUDA_CALL(cudaMalloc((void**)&(_workset_sp_data[i]), sizeof(float) * array_size));
      CUDA_CALL(cudaMalloc((void**)&(_group_des[i]), sizeof(int) * _group_num[i]));
    }
    CUDA_CALL(cudaMalloc((void**)&_residue_offset, sizeof(int) * (_group_num[0] + 1)));
    if (_group_num[0] > 0) CUDA_CALL(cudaMalloc((void**)&_residue_atomic, sizeof(bool) * _group_num[0]));

    thrust::host_vector<int> _residue_offset_temp(_group_num[0] + 1, 0);
    GetResidueOffset(_group_num[0], sort_group_index[0], _residue_offset_temp, h_residue_offset);
    CUDA_CALL(cudaMemcpy(_residue_offset, thrust::raw_pointer_cast(_residue_offset_temp.data()), sizeof(int) * (_group_num[0] + 1), cudaMemcpyHostToDevice));

    if (_group_num[0] > 0) GetResidueWorkset<<<dim3(CEIL(_group_num[0], 4), 1), dim3(32, 4)>>>(
      _group_num[0], thrust::raw_pointer_cast(sort_group_index[0].data()),
      thrust::raw_pointer_cast(d_residue_offset.data()), _residue_offset,
      _workset_col_ind[0], _workset_sp_data[0],
      _group_des[0], thrust::raw_pointer_cast(d_group_des[0].data()),
      _residue_atomic, thrust::raw_pointer_cast(d_residue_atomic.data()),
      thrust::raw_pointer_cast(src_workset[0].data()));
    CUDA_CALL(cudaGetLastError());

    for (int s = 1; s < _group_type_num; s++) {
      if (_group_num[s] == 0) continue;
      GetBlockWorkset<<<dim3(CEIL(_group_num[s], 4)), dim3(32, 4)>>>(
        _group_num[s], thrust::raw_pointer_cast(sort_group_index[s].data()), s,
        _workset_col_ind[s], _workset_sp_data[s],
        _group_des[s], thrust::raw_pointer_cast(d_group_des[s].data()),
        thrust::raw_pointer_cast(src_workset[s].data()));
      CUDA_CALL(cudaGetLastError());
    }

    CUDA_CALL(cudaDeviceSynchronize());
  } catch (...) {
    Release();
    throw;
  }
}

} // namespace PowerSparse
