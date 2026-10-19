#include "matrix_market.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
constexpr int kIndexLimit = std::numeric_limits<int>::max() - 32;

bool Space(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}
void SkipSpace(const char*& p) { while (Space(*p)) ++p; }
bool DataLine(std::istream& in, std::string& line) {
  while (std::getline(in, line)) {
    const char* p = line.c_str();
    SkipSpace(p);
    if (*p && *p != '%') return true;
  }
  if (in.bad()) throw std::runtime_error("Error reading MatrixMarket input");
  return false;
}
long long Integer(const char*& p) {
  SkipSpace(p);
  errno = 0;
  char* end = nullptr;
  const auto n = std::strtoll(p, &end, 10);
  if (end == p || errno == ERANGE || (*end && !Space(*end)))
    throw std::runtime_error("Invalid integer in MatrixMarket input");
  p = end;
  return n;
}
void EndRecord(const char* p) {
  SkipSpace(p);
  if (*p) throw std::runtime_error("Unexpected extra fields in MatrixMarket record");
}

struct Header {
  int rows, cols, entries;
  std::string field, symmetry;
};
Header ReadHeader(std::istream& in) {
  std::string line, magic, object, format, field, symmetry, extra;
  if (!std::getline(in, line)) throw std::runtime_error("Empty MatrixMarket file");
  std::istringstream banner(line);
  if (!(banner >> magic >> object >> format >> field >> symmetry) || banner >> extra ||
      magic != "%%MatrixMarket" || object != "matrix" || format != "coordinate" ||
      (field != "pattern" && field != "real" && field != "integer") ||
      (symmetry != "general" && symmetry != "symmetric"))
    throw std::runtime_error("Expected MatrixMarket coordinate pattern/real/integer general/symmetric");
  if (!DataLine(in, line)) throw std::runtime_error("Missing MatrixMarket dimensions");
  const char* p = line.c_str();
  const auto rows = Integer(p), cols = Integer(p), entries = Integer(p);
  EndRecord(p);
  if (rows <= 0 || cols <= 0 || entries <= 0 || rows > kIndexLimit ||
      cols > kIndexLimit || entries > kIndexLimit ||
      (symmetry == "symmetric" && rows != cols))
    throw std::runtime_error("Matrix must be nonempty and fit 32-bit CSR indices");
  return {int(rows), int(cols), int(entries), field, symmetry};
}

template <typename Emit>
void ReadEntries(std::istream& in, const Header& h, Emit emit) {
  std::string line;
  for (int i = 0; i < h.entries; ++i) {
    if (!DataLine(in, line)) throw std::runtime_error("Truncated MatrixMarket input");
    const char* p = line.c_str();
    const auto row = Integer(p), col = Integer(p);
    if (row < 1 || row > h.rows || col < 1 || col > h.cols)
      throw std::runtime_error("MatrixMarket index outside the declared dimensions");
    float value = 1.0f;
    if (h.field == "integer") {
      value = static_cast<float>(Integer(p));
    } else if (h.field == "real") {
      SkipSpace(p);
      char* end = nullptr;
      const double number = std::strtod(p, &end);
      if (end == p || (*end && !Space(*end)) || !std::isfinite(number) ||
          std::abs(number) > std::numeric_limits<float>::max())
        throw std::runtime_error("Invalid finite FP32 value in MatrixMarket input");
      value = static_cast<float>(number);
      p = end;
    }
    EndRecord(p);
    emit(int(row - 1), int(col - 1), value);
    if (h.symmetry == "symmetric" && row != col)
      emit(int(col - 1), int(row - 1), value);
  }
  if (DataLine(in, line)) throw std::runtime_error("More entries than declared in MatrixMarket input");
}
}  // namespace

CsrMatrix ReadMatrixMarket(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open matrix: " + path);
  const auto h = ReadHeader(in);
  CsrMatrix matrix;
  matrix.rows = h.rows;
  matrix.cols = h.cols;
  matrix.row_ptr.assign(size_t(h.rows) + 1, 0);
  long long expanded = 0;
  ReadEntries(in, h, [&](int row, int, float) {
    if (++expanded > kIndexLimit)
      throw std::runtime_error("Expanded matrix exceeds 32-bit CSR capacity");
    ++matrix.row_ptr[row + 1];
  });
  for (int r = 0; r < h.rows; ++r) matrix.row_ptr[r + 1] += matrix.row_ptr[r];
  matrix.col_ind.resize(size_t(expanded));
  matrix.values.resize(size_t(expanded));
  auto cursor = matrix.row_ptr;

  in.clear();
  in.seekg(0);
  if (!in) throw std::runtime_error("Matrix input must be a seekable regular file");
  const auto second = ReadHeader(in);
  if (h.rows != second.rows || h.cols != second.cols || h.entries != second.entries ||
      h.field != second.field || h.symmetry != second.symmetry)
    throw std::runtime_error("Matrix header changed between reader passes");
  ReadEntries(in, h, [&](int row, int col, float value) {
    const int pos = cursor[row]++;
    if (pos >= matrix.row_ptr[row + 1])
      throw std::runtime_error("Matrix row counts changed between reader passes");
    matrix.col_ind[pos] = col;
    matrix.values[pos] = value;
  });
  for (int r = 0; r < h.rows; ++r)
    if (cursor[r] != matrix.row_ptr[r + 1])
      throw std::runtime_error("Matrix row counts changed between reader passes");
  return matrix;
}
