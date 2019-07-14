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

#include "../gpu_memory_manager.hpp"
#include "coordinate_hash_map.h"

/** Mapping coordinates to indices, wrapped for MinkowskiEngine **/
template <size_t D> class CoordinateIndexer {
public:
  using KeyT = int;
  using ValueT = uint32_t;
  using HashFunc = CoordinateHashFunc<KeyT, D>;

  CoordinateIndexer(uint32_t max_keys,
                    /* CUDA device */
                    const uint32_t device_idx = 0,
                    /* Preset hash table params to estimate bucket num */
                    uint32_t keys_per_bucket = 15,
                    float expected_occupancy_per_bucket = 0.6);
  ~CoordinateIndexer();

  float Build(KeyT *keys_device, uint32_t num_keys /*, dim == D */);

  float Search(KeyT *query_keys_device, ValueT *query_values_device,
               uint8_t *mask_device, int num_keys);

  float ComputeLoadFactor(int flag = 0);

private:
  GPUMemoryManager<int8_t> index_memory_pool_;
  std::shared_ptr<CoordinateHashMap<KeyT, D, ValueT, HashFunc>> hash_map_;
};
