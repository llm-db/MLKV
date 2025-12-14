/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "dummy_var.cuh"
#include "storage_config.h"

#include <ATen/cuda/CUDAContext.h>

namespace mlkv_plus {

template <typename KeyType, typename ValueType>
DummyVar<KeyType, ValueType>::DummyVar(int64_t dim, int64_t max_hbm_for_vectors_gb, 
  bool hkv_io_by_cpu, int64_t gpu_id, bool create_if_missing, 
  int64_t gpu_init_capacity, int64_t gpu_max_capacity, 
  int64_t max_batch_size, std::string rocksdb_path):
 var_(nullptr) {
  
  mlkv_plus::StorageConfig config;
  config.dim = dim;
  dim_ = dim;
  config.max_hbm_for_vectors_gb = max_hbm_for_vectors_gb;
  config.hkv_io_by_cpu = hkv_io_by_cpu;
  config.gpu_id = gpu_id;
  config.create_if_missing = create_if_missing;
  config.hkv_init_capacity = gpu_init_capacity;
  config.hkv_max_capacity = gpu_max_capacity;
  config.max_batch_size = max_batch_size;
  config.rocksdb_path = rocksdb_path;


  var_ = std::make_unique<mlkv_plus::DB<KeyType, ValueType>>(config);


  if (var_ == nullptr) {
    throw std::runtime_error("Failed to create mlkv_plus::DB");
  }

  cudaError_t after_create = cudaGetLastError();
  if (after_create != cudaSuccess) {
      throw std::runtime_error("CUDA error at the end of DummyVar constructor: " + std::string(cudaGetErrorString(after_create)));
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();


  OperationResult result = var_->initialize(stream);

  
  if (result != OperationResult::SUCCESS) {
    throw std::runtime_error("Failed to initialize DummyVar: " + std::to_string(static_cast<int>(result)));
  }


  cudaError_t after_initialize = cudaGetLastError();
  if (after_initialize != cudaSuccess) {
    throw std::runtime_error("CUDA error at the end of DummyVar initialize: " + std::string(cudaGetErrorString(after_initialize)));
  }

}

template <typename KeyType, typename ValueType>
std::mutex* DummyVar<KeyType, ValueType>::mu() {
  return &mu_;
}

template <typename KeyType, typename ValueType>
void DummyVar<KeyType, ValueType>::Assign(const void* keys, const void* values, size_t num_keys) {
  var_->multiset(static_cast<const KeyType*>(keys), static_cast<const ValueType*>(values), num_keys);
}


template <typename KeyType, typename ValueType>
void DummyVar<KeyType, ValueType>::Read(const void* keys, void* values, void* found, size_t num_keys) {
  var_->multiget(static_cast<const KeyType*>(keys), static_cast<ValueType*>(values), static_cast<bool*>(found), num_keys);
}



template <typename KeyType, typename ValueType>
int DummyVar<KeyType, ValueType>::dim() {
  return dim_;
}

template <typename KeyType, typename ValueType>
std::string DummyVar<KeyType, ValueType>::DebugString() const {
  return "DummyVar<" + std::string(typeid(KeyType).name()) + ", " + std::string(typeid(ValueType).name()) + ">";
}





// explicit instance the template
template class DummyVar<int64_t, float>;



TORCH_LIBRARY_FRAGMENT(libmlkvplus_torch, m) {

  m.class_<mlkv_plus::DummyVar<int64_t, float>>("DummyVar")
      .def(torch::init<int64_t, int64_t, bool, int64_t, bool, int64_t, int64_t, int64_t, std::string>());


  // Main factory function - returns custom class object
  m.def("create_dummy_var(int dim, int max_hbm_for_vectors_gb, bool hkv_io_by_cpu, int gpu_id, bool create_if_missing, int gpu_init_capacity, int gpu_max_capacity, int max_batch_size, str rocksdb_path, Tensor ensure_device) -> Any");

  // Operations for int32 key type
  m.def("assign(__torch__.torch.classes.libmlkvplus_torch.DummyVar dummy_var, Tensor indices, Tensor values) -> bool");
  m.def("read(__torch__.torch.classes.libmlkvplus_torch.DummyVar dummy_var, Tensor indices) -> Tensor");
}



}  // namespace mlkv_plus
