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

from mlkv_plus.communication import alltoall, rank

from mlkv_plus.db import MLKVPlusDB
from mlkv_plus import sok


def all2all_dense_embedding(param: MLKVPlusDB, indices: torch.Tensor):
    # Filter key
    selected_indices, order, splits = torch.ops.sok.dist_select(indices, num_splits=param.num_gpus)
    
    # All-to-all of indices
    ex_indices, rsplits = alltoall(selected_indices, splits)
    
    ex_indices = param.key_map(ex_indices)
    
    # Local lookup
    embeddings = param.read(ex_indices)

    # All-to-all of embedding vectors
    ex_embeddings, _ = alltoall(embeddings, rsplits)
    
    # Reorder of embedding vectors
    ex_embeddings = torch.ops.sok.reorder(ex_embeddings, order)

    return ex_embeddings



def dist_assign(param: MLKVPlusDB, keys: torch.Tensor, values: torch.Tensor):
    
    # print("Debug: dist_assign")
    # print("------------------")
    mask = (keys.remainder(param.num_gpus) == rank())
    
    local_keys = keys[mask].contiguous()
    local_values = values[mask].contiguous()

    param.assign(local_keys, local_values)
    
    # print("------------------")
    
    
    
    
    
    
    