cmake_minimum_required(VERSION 3.18)

# Set explicitly before CUDA compiler detection: A800=80, H100=90.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "Set -DCMAKE_CUDA_ARCHITECTURES=80 (A800) or 90 (H100)")
endif()
project(PowerSparse LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit 11.8 REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_library(powersparse STATIC
  utils/matrix_utils.cu
  PowerSparse_SpMM/spmm_solution.cu
  PowerSparse_SpMM/spmm_kernel_block.cu
  PowerSparse_SpMM/spmm_kernel_residue.cu
  PowerSparse_SDDMM/sddmm_solution.cu
  PowerSparse_SDDMM/sddmm_kernel_block.cu
  PowerSparse_SDDMM/sddmm_kernel_residue.cu)
target_include_directories(powersparse PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(powersparse PUBLIC CUDA::cudart OpenMP::OpenMP_CXX)
target_compile_options(powersparse PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=legacy>)
# OpenMP::OpenMP_CXX supplies host linking; nvcc needs the host compile flags too.
separate_arguments(POWERSPARSE_OPENMP_FLAGS NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS POWERSPARSE_OPENMP_FLAGS)
  target_compile_options(powersparse PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

foreach(op IN ITEMS spmm sddmm)
  add_executable(powersparse_${op} benchmarks/benchmark.cu benchmarks/matrix_market.cpp)
  target_link_libraries(powersparse_${op} PRIVATE powersparse)
  set_target_properties(powersparse_${op} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endforeach()
target_compile_definitions(powersparse_sddmm PRIVATE POWERSPARSE_BENCH_SDDMM=1)
