#ifndef GPU_MEMORY_POOL
#define GPU_MEMORY_POOL

#include "../helper_cuda.h"
#include <torch/extension.h>

/** Instantiate patter is not working well across g++ & nvcc**/
class GPUMemoryPool {
  int initial_size_ = 256;
  int device_id_ = 0;

public:
  torch::Tensor _data;

  GPUMemoryPool(int device_id = 0, int initial_size = 256)
      : initial_size_(initial_size), device_id_(device_id) {
    CHECK_CUDA(cudaGetDevice(&device_id_));
    auto options = torch::TensorOptions()
                       .dtype(torch::kInt8)
                       .device(torch::kCUDA, device_id)
                       .requires_grad(false);
    _data = torch::zeros({initial_size_}, options);
  }

  void reset() { _data.resize_({0}); }
  void resize(int new_size) { _data.resize_({new_size}); }
  int64_t size() { return _data.numel(); }
  int8_t *data() { return _data.data<int8_t>(); }
  int8_t *data(int new_size) {
    if (size() < new_size)
      resize(new_size);
    return data();
  }
};

#endif
