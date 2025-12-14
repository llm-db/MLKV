#include <torch/all.h>
#include <torch/library.h>
#include "variable/binding/dummy_var.h"


namespace sok {

// Register custom classes with PyTorch
TORCH_LIBRARY_FRAGMENT(sok, m) {

    m.class_<DummyVar<int32_t, float>>("DummyVarInt32Float");

    m.class_<DummyVar<int64_t, float>>("DummyVarInt64Float");


    // Main factory function - returns custom class object
    m.def("create_dummy_var(str container, str shared_name, int[] shape, int key_type, int value_type, Tensor dummy_device, str var_type=\"hybrid\", str initializer=\"random\", str config=\"{}\") -> Any");

    // Operations for int32 key type
    m.def("assign_int32(__torch__.torch.classes.sok.DummyVarInt32Float dummy_var, Tensor indices, Tensor values) -> bool");
    m.def("sparse_read_int32(__torch__.torch.classes.sok.DummyVarInt32Float dummy_var, Tensor indices) -> Tensor");
    
    // Operations for int64 key type
    m.def("assign_int64(__torch__.torch.classes.sok.DummyVarInt64Float dummy_var, Tensor indices, Tensor values) -> bool");
    m.def("sparse_read_int64(__torch__.torch.classes.sok.DummyVarInt64Float dummy_var, Tensor indices) -> Tensor");

}


}