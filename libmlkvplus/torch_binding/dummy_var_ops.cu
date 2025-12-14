#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <mutex>

#include "dummy_var.cuh"

#include <ATen/Operators.h>

#include <torch/all.h>
#include <torch/library.h>

namespace mlkv_plus {


using KeyType = int64_t;
using ValueType = float;

bool assign(c10::intrusive_ptr<mlkv_plus::DummyVar<KeyType, ValueType>> dummy_var, torch::Tensor indices, torch::Tensor values) {
    // Clear any pending CUDA errors before starting
    cudaGetLastError();
    
    // Validate input parameters
    TORCH_CHECK(values.device().type() == torch::kCUDA, 
               "values must be on CUDA device for mlkv_plus operations, got ", values.device());
    TORCH_CHECK(indices.device().type() == torch::kCUDA, 
               "indices must be on CUDA device for mlkv_plus operations, got ", indices.device());
    
    // Acquire exclusive lock for thread safety
    std::lock_guard<std::mutex> lock(*dummy_var->mu());
    
    // Get number of indices
    int64_t N = indices.numel();
    
    // Call the Assign method directly on the custom class object
    dummy_var->Assign(indices.data_ptr<KeyType>(), values.data_ptr<ValueType>(), N);
    
    // Synchronize device to ensure all operations complete
    cudaError_t sync_err = cudaDeviceSynchronize();
    
    // Get and clear any CUDA errors that occurred during the operation
    cudaError_t last_err = cudaGetLastError();
    
    TORCH_CHECK(sync_err == cudaSuccess, 
                "CUDA sync error after assign: ", cudaGetErrorString(sync_err));
    TORCH_CHECK(last_err == cudaSuccess, 
                "CUDA error after assign: ", cudaGetErrorString(last_err));
    
    return true;
}




torch::Tensor read(c10::intrusive_ptr<mlkv_plus::DummyVar<KeyType, ValueType>> dummy_var, torch::Tensor indices) {

    // Clear any pending CUDA errors before starting
    cudaGetLastError();

    // check if the indices is on CUDA
    if (indices.device().type() == torch::kCPU) {
        indices = indices.to(torch::kCUDA);
    }

    // get current cuda device
    int current_device = at::cuda::current_device();
    
    // Acquire lock for thread safety (read operation)
    std::lock_guard<std::mutex> lock(*dummy_var->mu());
    
    // Get number of indices
    int64_t N = indices.numel();
    
    auto output = torch::empty({N, dummy_var->dim()}, 
                                torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(torch::kCUDA, current_device));

    // Allocate found tensor with shape {N} on the same device as indices
    auto found = torch::empty({N}, torch::TensorOptions()
                                .dtype(torch::kBool)
                                .device(torch::kCUDA, current_device));

    // Call the SparseRead method directly on the custom class object
    dummy_var->Read(indices.data_ptr<KeyType>(), output.data_ptr(), found.data_ptr<bool>(), N);
    
    // Synchronize device to ensure all operations complete before returning
    cudaError_t sync_err = cudaDeviceSynchronize();
    
    // Get and clear any CUDA errors that occurred during the operation
    cudaError_t last_err = cudaGetLastError();
    
    TORCH_CHECK(sync_err == cudaSuccess, 
                "CUDA sync error after read: ", cudaGetErrorString(sync_err));
    TORCH_CHECK(last_err == cudaSuccess, 
                "CUDA error after read: ", cudaGetErrorString(last_err));
    
    return output;
}


TORCH_LIBRARY_IMPL(libmlkvplus_torch, CatchAll, m) {
    m.impl("assign", [](
        c10::intrusive_ptr<mlkv_plus::DummyVar<KeyType, ValueType>> dummy_var,
        torch::Tensor indices,
        torch::Tensor values) -> bool {
            
        // Ensure CUDA is available
        TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for mlkv_plus operations");
        
        // Call the CUDA implementation directly
        return mlkv_plus::assign(dummy_var, indices, values);
    });
    
    m.impl("read", [](
        c10::intrusive_ptr<mlkv_plus::DummyVar<KeyType, ValueType>> dummy_var,
        torch::Tensor indices) -> torch::Tensor {
            
        // Ensure CUDA is available
        TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for SOK operations");
        
        // Call the CUDA implementation directly
        return mlkv_plus::read(dummy_var, indices);
    });
}

TORCH_LIBRARY_IMPL(libmlkvplus_torch, CUDA, m) {
    // Register CUDA implementations for different key types
    m.impl("assign", &assign);
    m.impl("read", &read);
}



} // namespace mlkv_plus

