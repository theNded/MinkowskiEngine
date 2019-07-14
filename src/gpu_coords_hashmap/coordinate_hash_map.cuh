/*
 * Copyright 2019 Saman Ashkiani
 * Modified by Wei Dong (2019)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "coordinate_hash_map.h"
#include "memory_alloc/memory_alloc_host.cuh"
#include "slab_hash/slab_hash_host.cuh"

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
CoordinateHashMap<KeyT, D, ValueT, HashFunc>::CoordinateHashMap(
    uint32_t max_keys, const uint32_t device_idx, uint32_t keys_per_bucket,
    float expected_occupancy_per_bucket)

    : max_keys_(max_keys), cuda_device_idx_(device_idx), slab_hash_(nullptr),
      slab_list_allocator_(nullptr) {
  /* Set bucket size */
  uint32_t expected_keys_per_bucket =
      expected_occupancy_per_bucket * keys_per_bucket;
  num_buckets_ =
      (max_keys + expected_keys_per_bucket - 1) / expected_keys_per_bucket;

  /* Set device */
  int32_t cuda_device_count_ = 0;
  CHECK_CUDA(cudaGetDeviceCount(&cuda_device_count_));
  assert(cuda_device_idx_ < cuda_device_count_);
  CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

  CHECK_CUDA(cudaEventCreate(&start_));
  CHECK_CUDA(cudaEventCreate(&stop_));

  // allocate an initialize the allocator:
  key_allocator_ = std::make_shared<MemoryAlloc<KeyTD>>(max_keys_);
  value_allocator_ = std::make_shared<MemoryAlloc<ValueT>>(max_keys_);
  slab_list_allocator_ = std::make_shared<SlabAlloc>();
  slab_hash_ = std::make_shared<SlabHash<KeyT, D, ValueT, HashFunc>>(
      num_buckets_, slab_list_allocator_, key_allocator_, value_allocator_,
      cuda_device_idx_);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
CoordinateHashMap<KeyT, D, ValueT, HashFunc>::~CoordinateHashMap() {
  CHECK_CUDA(cudaSetDevice(cuda_device_idx_));

  CHECK_CUDA(cudaEventDestroy(start_));
  CHECK_CUDA(cudaEventDestroy(stop_));
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, D, ValueT, HashFunc>::Insert(KeyT *keys,
                                                           ValueT *values,
                                                           int num_keys) {
  float time;
  CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
  CHECK_CUDA(cudaEventRecord(start_, 0));
  slab_hash_->Insert(reinterpret_cast<Coordinate<KeyT, D> *>(keys), values,
                     num_keys);
  CHECK_CUDA(cudaEventRecord(stop_, 0));
  CHECK_CUDA(cudaEventSynchronize(stop_));
  CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));
  return time;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, D, ValueT, HashFunc>::Search(KeyT *query_keys,
                                                           ValueT *query_values,
                                                           uint8_t *mask,
                                                           int num_keys) {
  float time;
  CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
  CHECK_CUDA(cudaMemset(mask, 0, sizeof(uint8_t) * num_keys));
  CHECK_CUDA(cudaEventRecord(start_, 0));

  slab_hash_->Search(reinterpret_cast<Coordinate<KeyT, D> *>(query_keys),
                     query_values, mask, num_keys);

  CHECK_CUDA(cudaEventRecord(stop_, 0));
  CHECK_CUDA(cudaEventSynchronize(stop_));
  CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));
  return time;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, D, ValueT, HashFunc>::Delete(KeyT *keys,
                                                           int num_keys) {
  float time;
  CHECK_CUDA(cudaSetDevice(cuda_device_idx_));
  CHECK_CUDA(cudaEventRecord(start_, 0));

  slab_hash_->Delete(reinterpret_cast<Coordinate<KeyT, D> *>(keys), num_keys);
  CHECK_CUDA(cudaEventRecord(stop_, 0));
  CHECK_CUDA(cudaEventSynchronize(stop_));
  CHECK_CUDA(cudaEventElapsedTime(&time, start_, stop_));

  return time;
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
float CoordinateHashMap<KeyT, D, ValueT, HashFunc>::ComputeLoadFactor(
    int flag /* = 0 */) {
  return slab_hash_->ComputeLoadFactor(flag);
}
