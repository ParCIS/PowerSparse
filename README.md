# PowerSparse

FP32 CUDA-Core implementations of SpMM (node aggregation) and SDDMM (edge
messaging), with GPU-assisted sparse preprocessing.

## Build

Requires Linux, CUDA Toolkit 11.8+, CMake 3.18+, and a C++17 compiler with OpenMP.
Use CUDA architecture `80` for A800 or `90` for H100.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build --parallel
```

## Dataset

See [dataset/README.md](dataset/README.md) for the download link.

## Run

Both executables take the same arguments:

```text
powersparse_{spmm,sddmm} MATRIX.mtx N DEVICE CHECK REPEATS
```

```bash
build/bin/powersparse_spmm dataset/amazon-products.mtx 32 0 0 100
build/bin/powersparse_sddmm dataset/amazon-products.mtx 128 0 0 100
```

`DEVICE` is the CUDA-visible GPU index;
`CHECK=1` enables preprocessing and CPU reference checks (`0` skips them);
`REPEATS` is the number of timed calls.

Output is CSV, including preprocessing time, average operator time (`ms`), and
GFLOPS. Timing uses 30 warmups and excludes reading, preprocessing, and
verification. Use `--header` to print the CSV column names.

To run all `.mtx` files in a directory:

```bash
bash scripts/run_benchmarks.sh build dataset spmm 0 results/spmm.csv
bash scripts/run_benchmarks.sh build dataset sddmm 0 results/sddmm.csv
```

The batch defaults are N=32,128, 100 repeats, and CHECK=0. Set `DATASET_LIST` to a
file listing paths relative to the dataset directory to select specific inputs.

## License

This project is licensed under the [MIT License](LICENSE).
