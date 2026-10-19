#pragma once

#include <string>
#include <vector>

struct CsrMatrix {
  int rows = 0, cols = 0;
  std::vector<int> row_ptr, col_ind;
  std::vector<float> values;
  int nnz() const { return static_cast<int>(values.size()); }
};

// Two passes over the input avoid keeping a second, full COO copy in RAM.
CsrMatrix ReadMatrixMarket(const std::string& path);
