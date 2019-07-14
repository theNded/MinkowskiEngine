//
// Created by wei on 18-11-9.
//

#pragma once

#include "../helper_cuda.h"
#include "memory_alloc.h"

#include <cassert>

template <typename T>
__global__ void ResetMemoryAllocKernel(MemoryAllocContext<T> ctx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ctx.max_capacity_) {
    ctx.value_at(i) = T(); /* This is not required. */
    ctx.addr_on_heap(i) = i;
  }
}

/**
 * Client end
 */
template <typename T> MemoryAlloc<T>::MemoryAlloc(int max_capacity) {
  max_capacity_ = max_capacity;
  gpu_context_.max_capacity_ = max_capacity;

  // This heap counter is too small, just let it live with cudaMalloc
  CHECK_CUDA(cudaMalloc(&(gpu_context_.heap_counter_), sizeof(int)));

  gpu_context_.heap_ =
      reinterpret_cast<addr_t *>(sizeof(addr_t) * max_capacity_);
  gpu_context_.data_ =
      reinterpret_cast<T *>(heap_memory_pool_.data(sizeof(T) * max_capacity_));
  // CHECK_CUDA(cudaMalloc(&(gpu_context_.heap_), sizeof(int) * max_capacity_));
  // CHECK_CUDA(cudaMalloc(&(gpu_context_.data_), sizeof(T) * max_capacity_));

  const int blocks = (max_capacity_ + 128 - 1) / 128;
  const int threads = 128;

  ResetMemoryAllocKernel<<<blocks, threads>>>(gpu_context_);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  int heap_counter = 0;
  CHECK_CUDA(cudaMemcpy(gpu_context_.heap_counter_, &heap_counter, sizeof(int),
                        cudaMemcpyHostToDevice));
}

template <typename T> MemoryAlloc<T>::~MemoryAlloc() {
  CHECK_CUDA(cudaFree(gpu_context_.heap_counter_));
  // CHECK_CUDA(cudaFree(gpu_context_.heap_));
  // CHECK_CUDA(cudaFree(gpu_context_.data_));
}

template <typename T> std::vector<int> MemoryAlloc<T>::DownloadHeap() {
  std::vector<int> ret;
  ret.resize(max_capacity_);
  CHECK_CUDA(cudaMemcpy(ret.data(), gpu_context_.heap_,
                        sizeof(int) * max_capacity_, cudaMemcpyDeviceToHost));
  return ret;
}

template <typename T> std::vector<T> MemoryAlloc<T>::DownloadValue() {
  std::vector<T> ret;
  ret.resize(max_capacity_);
  CHECK_CUDA(cudaMemcpy(ret.data(), gpu_context_.data_,
                        sizeof(T) * max_capacity_, cudaMemcpyDeviceToHost));
  return ret;
}

template <typename T> int MemoryAlloc<T>::heap_counter() {
  int heap_counter;
  CHECK_CUDA(cudaMemcpy(&heap_counter, gpu_context_.heap_counter_, sizeof(int),
                        cudaMemcpyDeviceToHost));
  return heap_counter;
}
