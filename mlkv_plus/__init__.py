import os
import sys
import torch


# The C++ extension is built as a python module `kit_src.sok`.
# Importing it registers the custom ops.
import mlkv_plus.libmlkvplus_torch
import mlkv_plus.sok
print("[MLKV PLUS INFO] MLKV Plus PyTorch extension loaded.")
# print location
print(f"mlkv_plus.libmlkvplus_torch location: {mlkv_plus.libmlkvplus_torch.__file__}")
print(f"mlkv_plus.sok location: {mlkv_plus.sok.__file__}")


from mlkv_plus.communication import set_comm_tool

# Import from the PyTorch-specific module
from mlkv_plus.db import MLKVPlusDB
from mlkv_plus.dist import all2all_dense_embedding, dist_assign


def init(comm_tool="horovod"):
    """
    Abbreviated as ``mlkv_plus.init``.

    This function is used to do the initialization of MLKV Plus.

    MLKV Plus will leverage all available GPUs for current CPU process. Please set
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
    print("[MLKV PLUS INFO] Initialize finished, communication tool: " + comm_tool)