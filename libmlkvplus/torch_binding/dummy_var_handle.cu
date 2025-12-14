#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "dummy_var.cuh"
#include "storage_config.h"

#include <ATen/Operators.h>

#include <torch/all.h>
#include <torch/library.h>

namespace mlkv_plus {

// Factory function for creating DummyVar with type dispatch
template<typename KeyType, typename ValueType>
c10::IValue create_dummy_var(
    int64_t dim,
    int64_t max_hbm_for_vectors_gb,
    bool hkv_io_by_cpu,
    int64_t gpu_id,
    bool create_if_missing,
    int64_t gpu_init_capacity,
    int64_t gpu_max_capacity,
    int64_t max_batch_size,
    std::string rocksdb_path,
    torch::Tensor ensure_device
    ) {

    // Ensure we're on a CUDA device
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for creating DummyVar");
    
    // Explicitly initialize CUDA context for the target device
    // This is crucial when PyTorch hasn't done any GPU operations yet
    cudaError_t err = cudaSetDevice(gpu_id);
    TORCH_CHECK(err == cudaSuccess, "Failed to set CUDA device: ", cudaGetErrorString(err));

    // Create DummyVar using make_intrusive
    auto dummy_var = c10::make_intrusive<mlkv_plus::DummyVar<KeyType, ValueType>>(
        dim, max_hbm_for_vectors_gb, hkv_io_by_cpu, gpu_id, create_if_missing, gpu_init_capacity, gpu_max_capacity, max_batch_size, rocksdb_path);

    cudaError_t after_create = cudaGetLastError();
    if (after_create != cudaSuccess) {
        throw std::runtime_error("CUDA error after DummyVar creation: " + std::string(cudaGetErrorString(after_create)));
    }

    return c10::IValue(dummy_var);
}

// Add CPU fallback to redirect to CUDA implementation
TORCH_LIBRARY_IMPL(libmlkvplus_torch, CatchAll, m) {
    m.impl("create_dummy_var", [](
        int64_t dim,
        int64_t max_hbm_for_vectors_gb,
        bool hkv_io_by_cpu,
        int64_t gpu_id,
        bool create_if_missing,
        int64_t gpu_init_capacity,
        int64_t gpu_max_capacity,
        int64_t max_batch_size,
        std::string rocksdb_path,
        torch::Tensor ensure_device
        ) {
        
        // Ensure CUDA is available
        TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for creating DummyVar");

        // Call the CUDA implementation directly
        return create_dummy_var<int64_t, float>(dim, max_hbm_for_vectors_gb, hkv_io_by_cpu, gpu_id, create_if_missing, gpu_init_capacity, gpu_max_capacity, 
            max_batch_size, rocksdb_path, ensure_device);
    });
}


TORCH_LIBRARY_IMPL(libmlkvplus_torch, CUDA, m) {
    // Register CUDA implementations
    m.impl("create_dummy_var", &create_dummy_var<int64_t, float>);
}


} // namespace mlkv_plus

