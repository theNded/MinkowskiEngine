/*
 * Copyright 2019 Saman Ashkiani
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
 */

#pragma once

#include "slab_hash.h"
#include "slab_hash_kernel.cuh"

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
SlabHash<KeyT, D, ValueT, HashFunc>::SlabHash(
        const uint32_t num_buckets,
        const std::shared_ptr<SlabAlloc>& slab_list_allocator,
        const std::shared_ptr<MemoryAlloc<KeyTD>>& key_allocator,
        const std::shared_ptr<MemoryAlloc<ValueT>>& value_allocator,
        uint32_t device_idx)
    : num_buckets_(num_buckets),
      slab_list_allocator_(slab_list_allocator),
      key_allocator_(key_allocator),
      value_allocator_(value_allocator),
      device_idx_(device_idx),
      d_table_(nullptr) {
    assert(slab_list_allocator && key_allocator &&
           "No proper dynamic allocator attached to the slab hash.");

    int32_t devCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&devCount));
    assert(device_idx_ < devCount);

    CHECK_CUDA(cudaSetDevice(device_idx_));

    // allocating initial buckets:
    /* Let pytorch manage memory */
    size_t size = sizeof(ConcurrentSlab) * num_buckets_;
    d_table_ = static_cast<int8_t *>(table_memory_pool_.data(size));
    // CHECK_CUDA(cudaMalloc(&d_table_, sizeof(ConcurrentSlab) * num_buckets_));
    CHECK_CUDA(cudaMemset(d_table_, 0xFF, size));

    gpu_context_.Init(
            d_table_, num_buckets_, slab_list_allocator_->getContext(),
            key_allocator_->gpu_context_, value_allocator_->gpu_context_);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
SlabHash<KeyT, D, ValueT, HashFunc>::~SlabHash() {
    CHECK_CUDA(cudaSetDevice(device_idx_));

    /* Let pytorch handle this */
    // CHECK_CUDA(cudaFree(d_table_));
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void SlabHash<KeyT, D, ValueT, HashFunc>::Insert(KeyTD* keys,
                                                 ValueT* values,
                                                 uint32_t num_keys) {
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    // calling the kernel for bulk build:
    CHECK_CUDA(cudaSetDevice(device_idx_));
    InsertKernel<KeyT, D, ValueT, HashFunc><<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, keys, values, num_keys);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void SlabHash<KeyT, D, ValueT, HashFunc>::Search(KeyTD* keys,
                                                 ValueT* values,
                                                 uint8_t* founds,
                                                 uint32_t num_queries) {
    CHECK_CUDA(cudaSetDevice(device_idx_));
    const uint32_t num_blocks = (num_queries + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    SearchKernel<KeyT, D, ValueT, HashFunc><<<num_blocks, BLOCKSIZE_>>>(
            gpu_context_, keys, values, founds, num_queries);
}

template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
void SlabHash<KeyT, D, ValueT, HashFunc>::Delete(KeyTD* keys,
                                                 uint32_t num_keys) {
    CHECK_CUDA(cudaSetDevice(device_idx_));
    const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
    DeleteKernel<KeyT, D, ValueT, HashFunc>
            <<<num_blocks, BLOCKSIZE_>>>(gpu_context_, keys, num_keys);
}

/** This is a debug function so we don't wrap the memory in pytorch **/
template <typename KeyT, size_t D, typename ValueT, typename HashFunc>
double SlabHash<KeyT, D, ValueT, HashFunc>::ComputeLoadFactor(int flag = 0) {
    uint32_t* h_bucket_count = new uint32_t[num_buckets_];
    uint32_t* d_bucket_count;
    CHECK_CUDA(cudaMalloc((void**)&d_bucket_count,
                          sizeof(uint32_t) * num_buckets_));
    CHECK_CUDA(cudaMemset(d_bucket_count, 0, sizeof(uint32_t) * num_buckets_));

    const auto& dynamic_alloc = gpu_context_.getAllocatorContext();
    const uint32_t num_super_blocks = dynamic_alloc.num_super_blocks_;
    uint32_t* h_count_super_blocks = new uint32_t[num_super_blocks];
    uint32_t* d_count_super_blocks;

    CHECK_CUDA(cudaMalloc((void**)&d_count_super_blocks,
                          sizeof(uint32_t) * num_super_blocks));
    CHECK_CUDA(cudaMemset(d_count_super_blocks, 0,
                          sizeof(uint32_t) * num_super_blocks));
    //---------------------------------
    // counting the number of inserted elements:
    const uint32_t blocksize = 128;
    const uint32_t num_blocks = (num_buckets_ * 32 + blocksize - 1) / blocksize;
    bucket_count_kernel<KeyT, D, ValueT, HashFunc><<<num_blocks, blocksize>>>(
            gpu_context_, d_bucket_count, num_buckets_);
    CHECK_CUDA(cudaMemcpy(h_bucket_count, d_bucket_count,
                          sizeof(uint32_t) * num_buckets_,
                          cudaMemcpyDeviceToHost));

    int total_elements_stored = 0;
    for (int i = 0; i < num_buckets_; i++) {
        total_elements_stored += h_bucket_count[i];
    }

    if (flag) {
        printf("## Total elements stored: %d (%lu bytes).\n",
               total_elements_stored,
               total_elements_stored * (sizeof(KeyT) + sizeof(ValueT)));
    }

    // counting total number of allocated memory units:
    int num_mem_units = dynamic_alloc.NUM_MEM_BLOCKS_PER_SUPER_BLOCK_ * 32;
    int num_cuda_blocks = (num_mem_units + blocksize - 1) / blocksize;
    compute_stats_allocators<<<num_cuda_blocks, blocksize>>>(
            d_count_super_blocks, gpu_context_);

    CHECK_CUDA(cudaMemcpy(h_count_super_blocks, d_count_super_blocks,
                          sizeof(uint32_t) * num_super_blocks,
                          cudaMemcpyDeviceToHost));

    // computing load factor
    int total_mem_units = num_buckets_;
    for (int i = 0; i < num_super_blocks; i++)
        total_mem_units += h_count_super_blocks[i];

    double load_factor =
            double(total_elements_stored * (sizeof(KeyT) + sizeof(ValueT))) /
            double(total_mem_units * WARP_WIDTH * sizeof(uint32_t));

    if (d_count_super_blocks) CHECK_CUDA(cudaFree(d_count_super_blocks));
    if (d_bucket_count) CHECK_CUDA(cudaFree(d_bucket_count));
    delete[] h_bucket_count;
    delete[] h_count_super_blocks;

    return load_factor;
}
