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

import os
import sys
import torch

try:
    # The C++ extension is built as a python module `kit_src.sok`.
    # Importing it registers the custom ops.
    import kit_src.sok
    print("[SOK INFO] SOK PyTorch extension loaded.")
    # print location
    print(f"kit_src.sok location: {kit_src.sok.__file__}")
except ImportError as e:
    # Provide a helpful error message if the extension is not built.
    raise ImportError(
        "[SOK ERROR] Failed to import 'kit_src.sok'. "
        "Please build and install the extension by running "
        "`python setup.py install` from the 'sparse_operation_kit' directory. "
        f"Original error: {e}"
    )

from sparse_operation_kit._version import __version__

import sparse_operation_kit.communication
from sparse_operation_kit.communication import set_comm_tool

# Import from the PyTorch-specific module
from sparse_operation_kit.dynamic_variable_pytorch import DynamicVariable, assign, export
from sparse_operation_kit.lookup import all2all_dense_embedding


def init(comm_tool="horovod"):
    """
    Abbreviated as ``sok.init``.

    This function is used to do the initialization of SparseOperationKit (SOK).

    SOK will leverage all available GPUs for current CPU process. Please set
    `CUDA_VISIBLE_DEVICES` to specify which GPU(s) are used in this process.

    Currently, these API only support ``horovod`` as the communication
    tool, so ``horovod.init`` must be called before initializing SOK.

    Parameters
    ----------
    comm_tool: string
            a string to specify which communication tool to use. Default value is "horovod".

    Returns
    -------
    None

    Example
    -------
    .. code-block:: python

        import torch
        import horovod.torch as hvd
        import sparse_operation_kit as sok

        hvd.init()
        # Pin GPU to be used to process local rank (one GPU per process)
        torch.cuda.set_device(hvd.local_rank())

        sok.init()

    """
    set_comm_tool(comm_tool)
    print("[SOK INFO] Initialize finished, communication tool: " + comm_tool)


def filter_variables(vars):
    """
    When using dynamic variables, it is necessary to use a SOK-compatible optimizer
    to update these variables. This API filters out SOK variables from all parameters.

    Parameters
    ----------
    vars: A list of PyTorch parameters.

    Returns
    -------
    sok_vars: A list of SOK variables.
    other_vars: A list of variables that do not belong to SOK.

    Example
    -------
    .. code-block:: python

        import torch
        import horovod.torch as hvd
        import sparse_operation_kit as sok

        hvd.init()
        torch.cuda.set_device(hvd.local_rank())
        sok.init()

        # Note: This is a conceptual example. 
        # The SOK variables are not nn.Parameter and might need special handling in optimizers.
        param1 = torch.nn.Parameter(torch.rand(3, 3))
        sok_var = sok.DynamicVariable(dimension=3, var_type="hbm", initializer="13")

        # The list passed to filter_variables might be model.parameters()
        all_vars = [param1, sok_var] 
        
        sok_vars, other_vars = sok.filter_variables(all_vars)
        assert len(sok_vars) == 1
        assert len(other_vars) == 1

        print("[SOK INFO] filter_variables test passed")
    """
    sok_vars, other_vars = [], []
    for v in vars:
        if isinstance(v, DynamicVariable):
            sok_vars.append(v)
        else:
            other_vars.append(v)
    return sok_vars, other_vars
