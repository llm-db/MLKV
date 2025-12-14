#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>

#include <memory>
#include <string>
#include <vector>

#include "lookup/impl/reorder_kernel.h"

#include <torch/all.h>
#include <torch/library.h>

#include <Python.h> 

namespace sok {
torch::Tensor reorder(const torch::Tensor &embedding, const torch::Tensor &order) {
  TORCH_CHECK(embedding.is_cuda(), "embedding must be a CUDA tensor");
  TORCH_CHECK(order.is_cuda(), "order must be a CUDA tensor");
  TORCH_CHECK(embedding.dim() == 2, "embedding must be a 2D tensor");
  TORCH_CHECK(order.dim() == 1, "order must be a 1D tensor");
  TORCH_CHECK(embedding.size(0) == order.size(0),
              "embedding and order must have the same size in dimension 0");
  TORCH_CHECK(order.scalar_type() == torch::kInt32, "order must be an int32 tensor");
  TORCH_CHECK(embedding.scalar_type() == torch::kFloat32,
              "embedding must be float32 (torch.float32)");

  auto output = torch::zeros_like(embedding);

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  // Only support float32
  
  sok::ReorderLauncher<float> launcher;
  launcher.initialize();
  launcher(embedding.const_data_ptr<float>(), embedding.size(0), embedding.size(1),
            order.const_data_ptr<int32_t>(), output.data_ptr<float>(), stream);


  return output;
}


TORCH_LIBRARY_FRAGMENT(sok, m) {
  m.def("reorder(Tensor embedding, Tensor order) -> Tensor");
}

// Add CPU fallback to redirect to CUDA implementation
TORCH_LIBRARY_IMPL(sok, CatchAll, m) {
  m.impl("reorder",
         [](const torch::Tensor &embedding, const torch::Tensor &order) -> torch::Tensor {
           TORCH_CHECK(torch::cuda::is_available(), "CUDA is required for reorder operation");
           return sok::reorder(embedding, order);
         });
}

// Register CUDA implementation
TORCH_LIBRARY_IMPL(sok, CUDA, m) { m.impl("reorder", &reorder); }

}  // namespace sok


