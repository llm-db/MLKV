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


#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <memory>
#include <string>
#include <mutex>

#include "variable/binding/dummy_var.h"

#include <Python.h> 
#include <ATen/Operators.h>

#include <torch/all.h>
#include <torch/library.h>

namespace sok {


template<typename KeyType, typename ValueType>
bool assign_impl(c10::intrusive_ptr<sok::DummyVar<KeyType, ValueType>> dummy_var, torch::Tensor indices, torch::Tensor values) {
    // Validate input parameters
    TORCH_CHECK(indices.device() == values.device(), 
               "indices and values must be on the same device. Got indices on ", 
               indices.device(), " and values on ", values.device());
    TORCH_CHECK(indices.device().type() == torch::kCPU, 
               "indices must be on CPU device for SOK operations, got ", indices.device());
               
    // Set CUDA device context
    cudaStream_t stream = dummy_var->stream();
    
    // Acquire exclusive lock for thread safety
    std::lock_guard<std::mutex> lock(*dummy_var->mu());
    
    // Get number of indices
    int64_t N = indices.numel();
    
    // Call the Assign method directly on the custom class object
    dummy_var->Assign(indices.data_ptr<KeyType>(), values.data_ptr<ValueType>(), N, stream);
    
    return true;
}


template<typename KeyType, typename ValueType>
torch::Tensor sparse_read_impl(c10::intrusive_ptr<sok::DummyVar<KeyType, ValueType>> dummy_var, torch::Tensor indices) {
    // Validate input parameters
    TORCH_CHECK(indices.device().type() == torch::kCUDA, 
               "indices must be on CUDA device for SOK operations, got ", indices.device());
    
    // Set CUDA device context
    cudaStream_t stream = dummy_var->stream();

    // get current cuda device
    int current_device = at::cuda::current_device();
    
    // Acquire lock for thread safety (read operation)
    std::lock_guard<std::mutex> lock(*dummy_var->mu());
    
    // Get number of indices
    int64_t N = indices.numel();
    
    // Allocate output tensor with shape {N, var->cols()} on the same device as indices
    auto output = torch::empty({N, dummy_var->cols()}, 
                              torch::TensorOptions()
                                .dtype(torch::kFloat32)
                                .device(torch::kCUDA, current_device));


    // Debug:
    std::cout << "N:" << N << std::endl;
    std::cout << "cols:" << dummy_var->cols() << std::endl;
    // print the indices
    std::cout << "indices:" << indices << std::endl;


    std::cout << "output" << output << std::endl;


    // Call the SparseRead method directly on the custom class object
    dummy_var->SparseRead(indices.data_ptr(), output.data_ptr(), N, stream);


    std::cout << "output:" << output << std::endl;
    
    return output;
}


// Overloaded assign functions for different key types
bool assign_int32(c10::intrusive_ptr<sok::DummyVar<int32_t, float>> dummy_var, torch::Tensor indices, torch::Tensor values) {
    return assign_impl<int32_t, float>(dummy_var, indices, values);
}

bool assign_int64(c10::intrusive_ptr<sok::DummyVar<int64_t, float>> dummy_var, torch::Tensor indices, torch::Tensor values) {
    return assign_impl<int64_t, float>(dummy_var, indices, values);
}

// Overloaded sparse_read functions for different key types
torch::Tensor sparse_read_int32(c10::intrusive_ptr<sok::DummyVar<int32_t, float>> dummy_var, torch::Tensor indices) {
    return sparse_read_impl<int32_t, float>(dummy_var, indices);
}

torch::Tensor sparse_read_int64(c10::intrusive_ptr<sok::DummyVar<int64_t, float>> dummy_var, torch::Tensor indices) {
    return sparse_read_impl<int64_t, float>(dummy_var, indices);
}


TORCH_LIBRARY_IMPL(sok, CatchAll, m) {
    m.impl("assign_int64", [](
        c10::intrusive_ptr<sok::DummyVar<int64_t, float>> dummy_var,
        torch::Tensor indices,
        torch::Tensor values) -> bool {
            
        // Ensure CUDA is available
        TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for SOK operations");
        
        // Call the CUDA implementation directly
        return sok::assign_impl<int64_t, float>(dummy_var, indices, values);
    });
    
    m.impl("sparse_read_int64", [](
        c10::intrusive_ptr<sok::DummyVar<int64_t, float>> dummy_var,
        torch::Tensor indices) -> torch::Tensor {
            
        // Ensure CUDA is available
        TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for SOK operations");
        
        // Call the CUDA implementation directly
        return sok::sparse_read_impl<int64_t, float>(dummy_var, indices);
    });
}

TORCH_LIBRARY_IMPL(sok, CUDA, m) {
    // Register CUDA implementations for different key types
    m.impl("assign_int32", &assign_int32);
    m.impl("assign_int64", &assign_int64);
    m.impl("sparse_read_int32", &sparse_read_int32);
    m.impl("sparse_read_int64", &sparse_read_int64);
}

} // namespace sok

