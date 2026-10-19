#include "benchmark_utils.h"
#include "matrix_market.h"
#include "PowerSparse_SpMM/PowerSparse_SpMM.h"
#include "PowerSparse_SDDMM/PowerSparse_SDDMM.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <tuple>

#ifdef POWERSPARSE_BENCH_SDDMM
constexpr bool kSddmm = true;
#else
constexpr bool kSddmm = false;
#endif
constexpr int kWarmup = 30;
constexpr unsigned kSeed = 1;
constexpr const char* kOperator = kSddmm ? "SDDMM" : "SpMM";

namespace {
using Matrix = PowerSparse::PowerSparseMatrixGPU;
using Edge = std::tuple<int, int, float>;

struct HostWorkset {
  std::vector<int> rows, offsets, cols;
  std::vector<float> values;
};

HostWorkset ReadWorkset(const Matrix& matrix, int s, const std::vector<int>& residue) {
  HostWorkset w;
  const int groups = matrix.GroupNum()[s];
  w.rows = Download(matrix.GroupDestination()[s], groups);
  if (s == 0) w.offsets = residue;
  else {
    w.offsets.resize(size_t(groups) + 1);
    for (int g = 0; g <= groups; ++g) w.offsets[g] = g * GROUP_SIZE(s);
  }
  w.cols = Download(matrix.WorksetColInd()[s], w.offsets.back());
  w.values = Download(matrix.WorksetSpData()[s], w.offsets.back());
  return w;
}

// Compare the exact multiset of (row, column, value), including duplicates.
// This is deliberately optional: it needs extra host memory and sorting time.
void CheckPreprocessing(const CsrMatrix& csr, const Matrix& matrix,
                        const std::vector<int>& residue) {
  std::vector<Edge> expected, actual;
  expected.reserve(csr.nnz());
  actual.reserve(csr.nnz());
  for (int r = 0; r < csr.rows; ++r)
    for (int p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p)
      expected.emplace_back(r, csr.col_ind[p], csr.values[p]);
  for (int s = 0; s < matrix.GroupTypeNum(); ++s) {
    const auto w = ReadWorkset(matrix, s, residue);
    for (size_t g = 0; g < w.rows.size(); ++g)
      for (int p = w.offsets[g]; p < w.offsets[g + 1]; ++p)
        actual.emplace_back(w.rows[g], w.cols[p], w.values[p]);
  }
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  if (actual != expected) throw std::runtime_error("Preprocessing changed the sparse matrix");
}

void CheckSpmm(const CsrMatrix& csr, int n, const std::vector<float>& b, const float* output) {
  const auto actual = Download(output, size_t(csr.rows) * n);
  std::vector<double> expected(n);
  for (int r = 0; r < csr.rows; ++r) {
    std::fill(expected.begin(), expected.end(), 0.0);
    for (int p = csr.row_ptr[r]; p < csr.row_ptr[r + 1]; ++p)
      for (int j = 0; j < n; ++j)
        expected[j] += double(csr.values[p]) * b[size_t(csr.col_ind[p]) * n + j];
    for (int j = 0; j < n; ++j)
      if (!Close(actual[size_t(r) * n + j], expected[j]))
        throw std::runtime_error("SpMM mismatch at row " + std::to_string(r) +
                                 ", feature " + std::to_string(j));
  }
}

void CheckSddmm(const Matrix& matrix, const std::vector<int>& residue, int n,
                const std::vector<float>& lhs, const std::vector<float>& rhs,
                const std::vector<float*>& output) {
  for (int s = 0; s < matrix.GroupTypeNum(); ++s) {
    const auto w = ReadWorkset(matrix, s, residue);
    const auto actual = Download(output[s], w.values.size());
    for (size_t g = 0; g < w.rows.size(); ++g)
      for (int p = w.offsets[g]; p < w.offsets[g + 1]; ++p) {
        double expected = 0;
        for (int j = 0; j < n; ++j)
          expected += double(lhs[size_t(w.rows[g]) * n + j]) * rhs[size_t(w.cols[p]) * n + j];
        expected *= w.values[p];
        if (!Close(actual[p], expected))
          throw std::runtime_error("SDDMM mismatch in workset " + std::to_string(s) +
                                   ", position " + std::to_string(p));
      }
  }
}

void CsvHeader() {
  std::cout << "algorithm,operator,matrix,rows,cols,nnz,N,device,gpu,compute_capability,"
               "cuda_runtime,cuda_driver,warmup,repeats,seed,check,preprocess_ms,ms,gflops\n";
}

int Run(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--header") { CsvHeader(); return 0; }
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0] << " MATRIX.mtx N DEVICE CHECK REPEATS\n"
              << "CHECK: 0=skip, 1=verify preprocessing and compare with a CPU reference\n"
              << "SpMM and SDDMM benchmark widths are set to N=32 or 128 in main.\n";
    return 2;
  }
  const std::string path = argv[1];
  const int n = ParseInt(argv[2], "N"), device = ParseInt(argv[3], "DEVICE");
  const int check = ParseInt(argv[4], "CHECK"), repeats = ParseInt(argv[5], "REPEATS");
  if (device < 0 || (check != 0 && check != 1) || repeats <= 0)
    throw std::invalid_argument("DEVICE must be nonnegative, CHECK 0 or 1, and REPEATS positive");
  CUDA_CALL(cudaSetDevice(device));
  CUDA_CALL(cudaFree(nullptr));  // initialize the context before preprocessing timing
  cudaDeviceProp prop{};
  CUDA_CALL(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 8) throw std::runtime_error("These kernels require an Ampere or newer GPU");
  int runtime = 0, driver = 0;
  CUDA_CALL(cudaRuntimeGetVersion(&runtime));
  CUDA_CALL(cudaDriverGetVersion(&driver));
  std::cerr << kOperator << ": " << path << ", N=" << n << ", GPU=" << prop.name << '\n';
  const auto csr = ReadMatrixMarket(path);
  PowerSparse::ValidateDenseShape(csr.rows, csr.cols, n);

  CUDA_CALL(cudaDeviceSynchronize());
  const auto begin = std::chrono::steady_clock::now();
  const Matrix matrix(csr.rows, csr.cols, csr.nnz(), csr.row_ptr.data(),
                      csr.col_ind.data(), csr.values.data());
  CUDA_CALL(cudaDeviceSynchronize());
  const double preprocess_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  const auto residue = Download(matrix.ResidueOffset(), size_t(matrix.GroupNum()[0]) + 1);

  std::mt19937 rng(kSeed);
  std::vector<float> lhs;
  if (kSddmm) lhs = DenseInput(size_t(csr.rows) * n, rng);
  auto rhs = DenseInput(size_t(csr.cols) * n, rng);
  DeviceBuffer<float> d_lhs(lhs.size()), d_rhs(rhs.size());
  d_lhs.upload(lhs);
  d_rhs.upload(rhs);
  DeviceBuffer<float> dense_output(kSddmm ? 0 : size_t(csr.rows) * n);
  std::vector<DeviceBuffer<float>> sparse_output;
  std::vector<float*> output_ptrs;
  if (kSddmm) {
    sparse_output.reserve(matrix.GroupTypeNum());
    for (int s = 0; s < matrix.GroupTypeNum(); ++s) {
      const size_t count = s == 0 ? size_t(residue.back()) :
          size_t(matrix.GroupNum()[s]) * GROUP_SIZE(s);
      sparse_output.emplace_back(count);
      output_ptrs.push_back(sparse_output.back().data());
    }
  }
  Streams streams(matrix.GroupTypeNum());
  auto reset = [&] {
    dense_output.zero();
    for (auto& output : sparse_output) output.zero();
    CUDA_CALL(cudaDeviceSynchronize());
  };
  auto invoke = [&] {
    if (kSddmm)
      PowerSparse::SDDMM(matrix, n, d_lhs.data(), d_rhs.data(), output_ptrs.data(),
                         streams.size(), streams.data());
    else
      PowerSparse::SpMM(matrix, n, d_rhs.data(), dense_output.data(), streams.size(), streams.data());
  };
  reset();
  if (check) {
    CheckPreprocessing(csr, matrix, residue);
    invoke();
    CUDA_CALL(cudaDeviceSynchronize());
    if (kSddmm) CheckSddmm(matrix, residue, n, lhs, rhs, output_ptrs);
    else CheckSpmm(csr, n, rhs, dense_output.data());
    std::cerr << "Preprocessing and CPU reference checks passed (atol=rtol=1e-3).\n";
    reset();
  }
  for (int i = 0; i < kWarmup; ++i) invoke();
  CUDA_CALL(cudaDeviceSynchronize());
  reset();
  GpuTimer timer;
  timer.start();
  // As in the original harness, outputs are not cleared between timed calls.
  for (int i = 0; i < repeats; ++i) invoke();
  CUDA_CALL(cudaDeviceSynchronize());
  const double ms = double(timer.stop()) / repeats;
  if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid measured latency");
  const double gflops = 2.0 * csr.nnz() * n / (ms * 1e6);
  std::cout << std::setprecision(10) << "PowerSparse," << kOperator << ',' << Csv(path) << ','
            << csr.rows << ',' << csr.cols << ',' << csr.nnz() << ',' << n << ',' << device << ','
            << Csv(prop.name) << ',' << prop.major << '.' << prop.minor << ',' << runtime << ','
            << driver << ',' << kWarmup << ',' << repeats << ',' << kSeed << ','
            << (check ? "passed" : "skipped") << ',' << preprocess_ms << ',' << ms << ',' << gflops << '\n';
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    // Choose the feature widths evaluated by both benchmark executables here.
    if (argc == 6) {
      const int n = ParseInt(argv[2], "N");
      if (n != 32 && n != 128) {
        std::cerr << kOperator << " benchmark: N=" << n
                  << " was not selected in main; please use N=32 or N=128.\n";
        return 2;
      }
    }
    return Run(argc, argv);
  }
  catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}
