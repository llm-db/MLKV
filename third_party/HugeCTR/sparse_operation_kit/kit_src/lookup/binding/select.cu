#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <memory>
#include <string>
#include <vector>

#include "lookup/impl/select_kernel.h"


#include <ATen/Operators.h>

#include <torch/all.h>
#include <torch/library.h>
#include <Python.h> 

namespace sok {
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dist_select(
        const torch::Tensor& indices,
        int64_t num_splits) {
    
      // Copy input to CUDA if it’s on CPU
      torch::Tensor indices_dev = indices;
      if (indices.device().type() == torch::kCPU) {
        // send to the *current* CUDA device
        int cur = at::cuda::current_device();  
        indices_dev = indices_dev.to(torch::Device(torch::kCUDA, cur));
      }
    
      // All the rest lives on indices_dev.device()
      auto device = indices_dev.device();
      int64_t num_keys = indices_dev.numel();
    
      TORCH_CHECK(indices_dev.dim() == 1, "indices must be 1-dimensional");
      TORCH_CHECK(num_splits > 0, "num_splits must be positive");
      TORCH_CHECK(
          indices_dev.scalar_type() == torch::kInt32 ||
          indices_dev.scalar_type() == torch::kInt64,
          "indices must be int32 or int64");
    
      // allocate exactly like TF: one output, one order, one splits
      auto output_dev = torch::empty_like(indices_dev);
      auto order_dev  = torch::empty({num_keys},
                             torch::dtype(torch::kInt32).device(device));
      auto splits_dev = torch::empty({num_splits},
                             torch::dtype(torch::kInt32).device(device));
    
      // temp buffers
      auto output_buffer = torch::empty(
          {num_keys * num_splits},
          torch::dtype(indices_dev.scalar_type()).device(device));
      auto order_buffer = torch::empty(
          {num_keys * num_splits},
          torch::dtype(torch::kInt32).device(device));
    
      // grab the current CUDA stream
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
      // launch exactly as before, but all ptrs point into GPU memory
      if (indices_dev.scalar_type() == torch::kInt64) {
        static SelectLauncher<int64_t> launcher;
        static bool initialized = false;
        if (!initialized) {
          launcher.initialize(num_splits);
          initialized = true;
        }
        launcher(indices_dev.data_ptr<int64_t>(), num_keys,
                 output_dev.data_ptr<int64_t>(),
                 output_buffer.data_ptr<int64_t>(),
                 order_dev.data_ptr<int32_t>(),
                 order_buffer.data_ptr<int32_t>(),
                 splits_dev.data_ptr<int32_t>(),
                 num_splits, stream);
      } else {
        static SelectLauncher<int32_t> launcher;
        static bool initialized = false;
        if (!initialized) {
          launcher.initialize(num_splits);
          initialized = true;
        }
        launcher(indices_dev.data_ptr<int32_t>(), num_keys,
                 output_dev.data_ptr<int32_t>(),
                 output_buffer.data_ptr<int32_t>(),
                 order_dev.data_ptr<int32_t>(),
                 order_buffer.data_ptr<int32_t>(),
                 splits_dev.data_ptr<int32_t>(),
                 num_splits, stream);
      }
    

      torch::Tensor output = output_dev;
      torch::Tensor order  = order_dev;
      torch::Tensor splits = splits_dev;
    
      return {output, order, splits};
    }
    
TORCH_LIBRARY_FRAGMENT(sok, m) {
  m.def("dist_select(Tensor indices, int num_splits) -> (Tensor output, Tensor order, Tensor splits)");
}


// Add CPU fallback to redirect to CUDA implementation
TORCH_LIBRARY_IMPL(sok, CatchAll, m) {
  m.impl("dist_select", [](const torch::Tensor& indices, int64_t num_splits) {
      // Ensure CUDA is available
      TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for dist_select operation");
      
      // Call the CUDA implementation directly
      return sok::dist_select(indices, num_splits);
  });
}

// Register CUDA implementation
TORCH_LIBRARY_IMPL(sok, CUDA, m) {
    m.impl("dist_select", &dist_select);
}

} // namespace sok

