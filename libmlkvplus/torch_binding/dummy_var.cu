#include "dummy_var.cuh"
#include "storage_config.h"

#include <ATen/cuda/CUDAContext.h>
#include <nlohmann/json.hpp>

namespace mlkv_plus {

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
DummyVar<KeyType, ValueType>::DummyVar(std::string json_config):
 var_(nullptr) {
  
  mlkv_plus::StorageConfig config;
  
  // Parse JSON config
  try {
    nlohmann::json j = nlohmann::json::parse(json_config);
    
    // Parse all config fields from JSON
    if (j.contains("dim")) {
      config.dim = j["dim"].get<size_t>();
      dim_ = config.dim;
    } else {
      throw std::runtime_error("JSON config must contain 'dim' field");
    }
    
    if (j.contains("max_hbm_for_vectors_gb")) {
      config.max_hbm_for_vectors_gb = j["max_hbm_for_vectors_gb"].get<size_t>();
    }
    
    if (j.contains("hkv_io_by_cpu")) {
      config.hkv_io_by_cpu = j["hkv_io_by_cpu"].get<bool>();
    }
    
    if (j.contains("gpu_id")) {
      config.gpu_id = j["gpu_id"].get<int>();
    }
    
    if (j.contains("create_if_missing")) {
      config.create_if_missing = j["create_if_missing"].get<bool>();
    }
    
    if (j.contains("hkv_init_capacity")) {
      config.hkv_init_capacity = j["hkv_init_capacity"].get<size_t>();
    }
    
    if (j.contains("hkv_max_capacity")) {
      config.hkv_max_capacity = j["hkv_max_capacity"].get<size_t>();
    }
    
    if (j.contains("max_batch_size")) {
      config.max_batch_size = j["max_batch_size"].get<size_t>();
    }
    
    if (j.contains("rocksdb_path")) {
      config.rocksdb_path = j["rocksdb_path"].get<std::string>();
    }
    
    if (j.contains("enable_gds_log")) {
      config.enable_gds_log = j["enable_gds_log"].get<bool>();
    }
    
    if (j.contains("enable_gds_get_from_sst")) {
      config.enable_gds_get_from_sst = j["enable_gds_get_from_sst"].get<bool>();
    }
    
    if (j.contains("disableWAL")) {
      config.disableWAL = j["disableWAL"].get<bool>();
    }
    
    if (j.contains("force_skip_memtable")) {
      config.force_skip_memtable = j["force_skip_memtable"].get<bool>();
    }
    
    if (j.contains("rocksdb_use_direct_reads")) {
      config.rocksdb_use_direct_reads = j["rocksdb_use_direct_reads"].get<bool>();
    }
    
  } catch (const nlohmann::json::parse_error& e) {
    throw std::runtime_error("Failed to parse JSON config: " + std::string(e.what()));
  } catch (const nlohmann::json::type_error& e) {
    throw std::runtime_error("JSON config type error: " + std::string(e.what()));
  }

  // Set CUDA device if gpu_id is specified
  if (config.gpu_id >= 0) {
    cudaError_t err = cudaSetDevice(config.gpu_id);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to set CUDA device: " + std::string(cudaGetErrorString(err)));
    }
  }

  var_ = std::make_unique<mlkv_plus::DB<KeyType, ValueType>>(config);

  if (var_ == nullptr) {
    throw std::runtime_error("Failed to create mlkv_plus::DB");
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  OperationResult result = var_->initialize(stream);

  
  if (result != OperationResult::SUCCESS) {
    throw std::runtime_error("Failed to initialize DummyVar: " + std::to_string(static_cast<int>(result)));
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA error after initialize: " + std::string(cudaGetErrorString(err)));
  }

}

template <typename KeyType, typename ValueType>
std::string DummyVar<KeyType, ValueType>::DebugString() const {
  return "DummyVar<" + std::string(typeid(KeyType).name()) + ", " + std::string(typeid(ValueType).name()) + ">";
}





// explicit instance the template
template class DummyVar<int64_t, float>;



TORCH_LIBRARY_FRAGMENT(libmlkvplus_torch, m) {

  m.class_<mlkv_plus::DummyVar<int64_t, float>>("DummyVar")
      .def(torch::init<std::string>());


  // Main factory function - returns custom class object
  // Note: create_dummy_var is kept for backward compatibility but not recommended
  m.def("create_dummy_var_from_json(str json_config, Tensor ensure_device) -> Any");

  // Operations for int32 key type
  m.def("assign(__torch__.torch.classes.libmlkvplus_torch.DummyVar dummy_var, Tensor indices, Tensor values) -> bool");
  m.def("read(__torch__.torch.classes.libmlkvplus_torch.DummyVar dummy_var, Tensor indices) -> Tensor");
}



}  // namespace mlkv_plus
