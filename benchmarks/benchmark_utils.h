#pragma once

#include "utils/cuda_utils.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count = 0) : count_(count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::overflow_error("Device allocation size overflow");
    if (count) CUDA_CALL(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()));
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), count_(other.count_) {
    other.data_ = nullptr;
    other.count_ = 0;
  }
  T* data() const { return data_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }
  void zero() { if (count_) CUDA_CALL(cudaMemset(data_, 0, bytes())); }
  void upload(const std::vector<T>& input) {
    if (input.size() != count_) throw std::invalid_argument("Upload size mismatch");
    if (count_) CUDA_CALL(cudaMemcpy(data_, input.data(), bytes(), cudaMemcpyHostToDevice));
  }
 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

template <typename T>
std::vector<T> Download(const T* input, size_t count) {
  std::vector<T> host(count);
  if (count) CUDA_CALL(cudaMemcpy(host.data(), input, count * sizeof(T), cudaMemcpyDeviceToHost));
  return host;
}

class Streams {
 public:
  explicit Streams(int count) : streams_(count, nullptr) {
    try {
      if (count > 1)
        for (auto& stream : streams_) CUDA_CALL(cudaStreamCreate(&stream));
    } catch (...) {
      for (auto stream : streams_) if (stream) cudaStreamDestroy(stream);
      throw;
    }
  }
  ~Streams() { for (auto stream : streams_) if (stream) cudaStreamDestroy(stream); }
  Streams(const Streams&) = delete;
  Streams& operator=(const Streams&) = delete;
  int size() const { return static_cast<int>(streams_.size()); }
  const cudaStream_t* data() const { return streams_.data(); }
 private:
  std::vector<cudaStream_t> streams_;
};

class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CALL(cudaEventCreate(&start_));
    try { CUDA_CALL(cudaEventCreate(&stop_)); }
    catch (...) { cudaEventDestroy(start_); throw; }
  }
  ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
  GpuTimer(const GpuTimer&) = delete;
  GpuTimer& operator=(const GpuTimer&) = delete;
  void start() { CUDA_CALL(cudaEventRecord(start_)); }
  float stop() {
    CUDA_CALL(cudaEventRecord(stop_));
    CUDA_CALL(cudaEventSynchronize(stop_));
    float ms = 0;
    CUDA_CALL(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
 private:
  cudaEvent_t start_{}, stop_{};
};

inline int ParseInt(const char* text, const char* name) {
  const std::string value(text);
  size_t consumed = 0;
  int number;
  try { number = std::stoi(value, &consumed); }
  catch (...) { throw std::invalid_argument(std::string("Invalid ") + name + ": " + value); }
  if (consumed != value.size()) throw std::invalid_argument(std::string("Invalid ") + name);
  return number;
}

inline std::vector<float> DenseInput(size_t count, std::mt19937& rng) {
  std::vector<float> values(count);
  for (auto& value : values) value = static_cast<float>(rng() % 3) / 10.0f;
  return values;
}

inline bool Close(float actual, double expected) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(double(actual) - expected) <= 1e-3 + 1e-3 * std::abs(expected);
}

inline std::string Csv(const std::string& input) {
  std::string result = "\"";
  for (char c : input) { if (c == '"') result += '"'; result += c; }
  return result + '"';
}
