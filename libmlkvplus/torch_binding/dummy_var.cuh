#pragma once
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


#include <memory>
#include <string>
#include <mutex>
#include <cuda_runtime.h>
#include "mlkv_plus.cuh"

#include <torch/custom_class.h>


namespace mlkv_plus {

template <typename KeyType, typename ValueType>
class DummyVar : public torch::CustomClassHolder {
 public:
  DummyVar(int64_t dim, int64_t max_hbm_for_vectors_gb, bool hkv_io_by_cpu, 
    int64_t gpu_id, bool create_if_missing, int64_t gpu_init_capacity, int64_t gpu_max_capacity, 
    int64_t max_batch_size, std::string rocksdb_path);
  ~DummyVar() = default;

  std::string DebugString() const;  // Remove override keyword
  std::mutex* mu();  // Use std::mutex instead of TensorFlow mutex

  void Assign(const void *keys, const void *values, size_t num_keys);

  void Read(const void *keys, void *values, void *found, size_t num_keys);

  int dim();


  



 private:
  std::unique_ptr<mlkv_plus::DB<KeyType, ValueType>> var_;
  std::mutex mu_; 
  int dim_;

  void check_var();
};







} 
