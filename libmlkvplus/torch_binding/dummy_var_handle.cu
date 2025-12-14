#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <nlohmann/json.hpp>

#include "dummy_var.cuh"
#include "storage_config.h"

#include <ATen/Operators.h>

#include <torch/all.h>
#include <torch/library.h>

namespace mlkv_plus {

// Factory function for creating DummyVar from JSON config
template<typename KeyType, typename ValueType>
c10::IValue create_dummy_var_from_json(
    std::string json_config,
    torch::Tensor ensure_device
    ) {

    // Ensure we're on a CUDA device
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for creating DummyVar");
    
    // Parse JSON to get gpu_id for device setting
    int gpu_id = 0;
    try {
        nlohmann::json j = nlohmann::json::parse(json_config);
        if (j.contains("gpu_id")) {
            gpu_id = j["gpu_id"].get<int>();
        }
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse JSON config for gpu_id: " + std::string(e.what()));
    }
    
    // Explicitly initialize CUDA context for the target device
    // This is crucial when PyTorch hasn't done any GPU operations yet
    cudaError_t err = cudaSetDevice(gpu_id);
    TORCH_CHECK(err == cudaSuccess, "Failed to set CUDA device: ", cudaGetErrorString(err));

    // Create DummyVar using make_intrusive with JSON config
    auto dummy_var = c10::make_intrusive<mlkv_plus::DummyVar<KeyType, ValueType>>(json_config);

    return c10::IValue(dummy_var);
}

// Add CPU fallback to redirect to CUDA implementation
TORCH_LIBRARY_IMPL(libmlkvplus_torch, CatchAll, m) {
    
    m.impl("create_dummy_var_from_json", [](
        std::string json_config,
        torch::Tensor ensure_device
        ) {
        
        // Ensure CUDA is available
        TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for creating DummyVar");

        // Call the CUDA implementation directly
        return create_dummy_var_from_json<int64_t, float>(json_config, ensure_device);
    });
}


TORCH_LIBRARY_IMPL(libmlkvplus_torch, CUDA, m) {
    // Register CUDA implementations
    m.impl("create_dummy_var_from_json", &create_dummy_var_from_json<int64_t, float>);
}


} // namespace mlkv_plus

