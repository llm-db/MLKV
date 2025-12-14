#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <memory>
#include <string>

#include "variable/binding/dummy_var.h"

#include <Python.h> 
#include <ATen/Operators.h>


#include <torch/all.h>
#include <torch/library.h>

namespace sok {

// Helper to create DummyVar and return as intrusive_ptr
template<typename KeyType, typename ValueType>
c10::intrusive_ptr<sok::DummyVar<KeyType, ValueType>> create_dummy_var_impl(
    const std::string& container,
    const std::string& shared_name,
    const std::vector<int64_t>& shape,
    const torch::Tensor& dummy_device,
    const std::string& var_type = "hybrid",
    const std::string& initializer = "random",
    const std::string& config = "{}"
    ) {
    
    // Validate shape (must be rank 2)
    TORCH_CHECK(shape.size() == 2, "Shape must be rank 2, got rank ", shape.size());
    TORCH_CHECK(shape[1] > 0, "Shape[1] must be > 0, got ", shape[1]);
    
    int64_t rows = shape[0] <= 0 ? -1 : shape[0];  // -1 for dynamic rows
    int64_t cols = shape[1];
    
    // Ensure we're on a CUDA device
    TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for creating DummyVar");
    
    // Set the current CUDA device and get stream
    int device_id = dummy_device.device().index();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_id).stream();
    
    // Create DummyVar using make_intrusive
    auto dummy_var = c10::make_intrusive<sok::DummyVar<KeyType, ValueType>>(
        rows, cols, var_type, initializer, config, container, shared_name, stream);
    
    // Verify DummyVar was created successfully
    TORCH_CHECK(dummy_var != nullptr, "Failed to create DummyVar");
    TORCH_CHECK(dummy_var->get_var() != nullptr, "DummyVar internal variable is nullptr");
    
    return dummy_var;
}

// Factory function for creating DummyVar with type dispatch
c10::IValue create_dummy_var(
    const std::string& container,
    const std::string& shared_name,
    const std::vector<int64_t>& shape,
    int64_t key_type,
    int64_t value_type,
    const torch::Tensor& dummy_device,
    const std::string& var_type,
    const std::string& initializer,
    const std::string& config
    ) {
    
    TORCH_CHECK(value_type == 0, "Only float value type is currently supported");
    
    if (key_type == 0) {
        auto dummy_var = create_dummy_var_impl<int32_t, float>(
            container, shared_name, shape, dummy_device, var_type, initializer, config);
        return c10::IValue(dummy_var);
    } else if (key_type == 1) {
        auto dummy_var = create_dummy_var_impl<int64_t, float>(
            container, shared_name, shape, dummy_device, var_type, initializer, config);
        return c10::IValue(dummy_var);
    } else {
        TORCH_CHECK(false, "Unsupported key_type: ", key_type, ". Use 0 for int32, 1 for int64");
    }
}

// These helper functions are no longer needed since we use intrusive_ptr directly


// Add CPU fallback to redirect to CUDA implementation
// TORCH_LIBRARY_IMPL(sok, CatchAll, m) {
//     m.impl("create_dummy_var", [](
//         const std::string& container,
//         const std::string& shared_name,
//         const std::vector<int64_t>& shape,
//         int64_t key_type,
//         int64_t value_type,
//         const std::string& var_type,
//         const std::string& initializer,
//         const std::string& config) -> torch::Tensor {
            
//         // Ensure CUDA is available
//         TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for SOK operations");
        
//         // Call the CUDA implementation directly
//         return sok::create_dummy_var(container, shared_name, shape, key_type, value_type, var_type, initializer, config);
//     });
// }


TORCH_LIBRARY_IMPL(sok, CUDA, m) {
    // Register CUDA implementations
    m.impl("create_dummy_var", &create_dummy_var);
}

} // namespace sok

