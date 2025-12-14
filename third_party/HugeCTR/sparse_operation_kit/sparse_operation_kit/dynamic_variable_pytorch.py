#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import json
from sparse_operation_kit.communication import num_gpus


dynamic_variable_count = 0


class DynamicVariable(object):
    """
    Abbreviated as ``sok.DynamicVariable``.

    A variable that allocates memory dynamically. It is backed by a PyTorch custom C++ class.

    Parameters
    ----------
    dimension: int
        The last dimension of this variable(that is, the embedding vector
        size of embedding table).

    initializer: string
        a string to specify how to initialize this variable.
        Currently, only support "random" or string of a float
        value(meaning const initializer). Default value is "random".

    var_type: string
        a string to specify to use DET("hbm") or HKV("hybrid") as the backend.
        default value is "hybrid".

        If use DET as the backend, DET will retain two key values as the empty value and reclaim value for the hash table.
        If the input key is the same as these two values, the program will crash.
        If the key type is signed, the empty value = std::numeric_limits::max(), and the reclaim value = std::numeric_limits::min().
        If the key type is unsigned, the empty value = std::numeric_limits::max(), and the reclaim value = std::numeric_limits::max() - 1.

        If use HKV as the backend, only support torch.int64 as key_type.
        If use HKV as the backend, please set init_capacity and max_capacity value equal to 2 powers.

    key_type: dtype
        specify the data type of indices. This variable is dynamically allocated and contains
        a hash table inside it. So the data type of indices must be
        specified to construct the hash table. Default value is torch.int64.

    dtype: dtype
        specify the data type of values. Default value is torch.float32.

    Example
    -------
    .. code-block:: python

        import torch
        import sparse_operation_kit as sok

        sok.init()
        v = sok.DynamicVariable(dimension=3, initializer="13")

        indices = torch.tensor([0, 1, 2**40], dtype=torch.int64)

        embedding = v.sparse_read(indices)
        print("embedding:", embedding)
    """

    def __init__(
        self,
        dimension,
        initializer=None,
        var_type=None,
        name=None,
        trainable=True,  # This will be used to set requires_grad on read tensors
        key_type=None,
        dtype=None,
        mode=None,
        **kwargs,
    ):
        self._key_type = key_type if key_type is not None else torch.int64
        self._dtype = dtype if dtype is not None else torch.float32
        self._dimension = dimension
        self._mode = mode
        self._config = json.dumps(kwargs)
        self._config_dict = kwargs
        if var_type == "hybrid" and self._key_type != torch.int64:
            raise NotImplementedError("only key_type torch.int64 is supported in HKV backend")
        if name is None:
            global dynamic_variable_count
            self._name = "sok_dynamic_Variable_" + str(dynamic_variable_count)
            dynamic_variable_count += 1
        else:
            self._name = name

        var_type = "hybrid" if var_type is None else var_type
        self._var_type = var_type
        self._initializer = "" if initializer is None else str(initializer)

        # PyTorch requires integer types for key_type and value_type in the op
        # 0: int32, 1: int64 for key_type
        # 0: float32 for value_type
        key_type_enum = -1
        if self._key_type == torch.int32:
            key_type_enum = 0
        elif self._key_type == torch.int64:
            key_type_enum = 1
        else:
            raise ValueError(f"Unsupported key_type: {self._key_type}")

        value_type_enum = -1
        if self._dtype == torch.float32:
            value_type_enum = 0
        else:
            raise ValueError(f"Unsupported dtype: {self._dtype}")

        shape = [-1, dimension]
        
        # This tensor's device is used to select the GPU for the variable
        dummy_device = torch.empty(0, device=f"cuda:{torch.cuda.current_device()}")

        self._handle = torch.ops.sok.create_dummy_var(
            "DummyVariableContainer",
            self._name,
            shape,
            key_type_enum,
            value_type_enum,
            dummy_device,
            var_type=self._var_type,
            initializer=self._initializer,
            config=self._config,
        )
        self.shape = torch.Size(shape)
        self.dtype = self._dtype
        self.trainable = trainable

    def __repr__(self):
        return "<sok.DynamicVariable '%s' shape=%s dtype=%s>" % (
            self._name,
            self.shape,
            self.dtype,
        )

    @property
    def size(self):
        # Not implemented in C++ bindings
        pass

    @property
    def dimension(self):
        return self._dimension

    @property
    def key_type(self):
        return self._key_type

    @property
    def handle_dtype(self):
        return self._dtype

    @property
    def backend_type(self):
        return self._var_type

    @property
    def config_dict(self):
        return self._config_dict

    @property
    def target_gpu(self):
        if self._mode is not None and self._mode.startswith("localized"):
            target_gpu = int(self._mode.split(":")[1])
            if target_gpu >= num_gpus():
                raise RuntimeError(
                    f"There are only {num_gpus()} GPU(s), cannot put embedding table on "
                    f"{target_gpu}-th(zero-indexed) GPU."
                )
            return target_gpu
        return -1

    @property
    def mode(self):
        return self._mode

    @property
    def num_gpus(self):
        return num_gpus()

    @property
    def initializer_str(self):
        return self._initializer
    
    def key_map(self, indices):
        return indices

    # -------------------------------------------------------------------------
    # Methods supported
    # -------------------------------------------------------------------------
    def sparse_read(self, indices, name=None):
        if indices.dtype != self.key_type:
            indices = indices.to(self.key_type)
            
        if indices.device.type != "cuda":
            indices = indices.to(torch.device('cuda'))

        output = torch.ops.sok.sparse_read_int64(self._handle, indices)
        return output
        
    def assign(self, indices, values):
        """
        Assigns values to specified indices.
        """
        if indices.dtype != self.key_type:
            indices = indices.to(self.key_type)
        if values.dtype != self.dtype:
            values = values.to(self.dtype)

        if self.key_type == torch.int32:
            return torch.ops.sok.assign_int32(self._handle, indices, values)
        elif self.key_type == torch.int64:
            return torch.ops.sok.assign_int64(self._handle, indices, values)
        else:
            raise TypeError(f"Unsupported key_type for assign: {self.key_type}")


    # -------------------------------------------------------------------------
    # Methods not supported by C++ bindings yet
    # -------------------------------------------------------------------------

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        # Not implemented in C++ bindings
        pass

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        # Not implemented in C++ bindings
        pass

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        # Not implemented in C++ bindings
        pass
        
    def to_static(self, indices):
        # TF-specific logic, not applicable to PyTorch
        pass

    def to_dynamic(self):
        # TF-specific logic, not applicable to PyTorch
        pass


def export(var):
    """
    Abbreviated as ``sok.export``.

    Export the indices and value tensor from the given variable.
    This functionality is not implemented in the provided C++ bindings.
    """
    # Not implemented in C++ bindings
    pass


def assign(var, indices, values):
    """
    Abbreviated as ``sok.assign``.

    Assign the indices and value tensor to the target variable.
    """
    if isinstance(var, DynamicVariable):
        return var.assign(indices, values)
    else:
        raise TypeError(f"var must be a DynamicVariable, but got {type(var)}")
