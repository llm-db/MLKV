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

import numpy as np
from itertools import chain

import torch


from sparse_operation_kit.communication import rank
from sparse_operation_kit.communication import num_ranks
from sparse_operation_kit.communication import id_in_rank
from sparse_operation_kit.communication import num_gpus
from sparse_operation_kit.communication import alltoall
from sparse_operation_kit.communication import allreduce
from sparse_operation_kit.communication import allgather


from sparse_operation_kit.dynamic_variable_pytorch import DynamicVariable
import sys


def all2all_dense_embedding(param, indices):
    # Filter key
    selected_indices, order, splits = torch.ops.sok.dist_select(indices, num_splits=param.num_gpus)

    print(f"selected_indices={selected_indices}")
    print(f"order={order}")
    print(f"splits={splits}")

    # All-to-all of indices
    ex_indices, rsplits = alltoall(selected_indices, splits)
    
    print(f"ex_indices={ex_indices}")
    print(f"rsplits={rsplits}")
    
    
    ex_indices = param.key_map(ex_indices)
    
    print(f"ex_indices={ex_indices}")

    # Local lookup
    embeddings = param.sparse_read(ex_indices)
    
    
    print(f"embeddings={embeddings}")

    # All-to-all of embedding vectors
    ex_embeddings, _ = alltoall(embeddings, rsplits)

    # Reorder of embedding vectors
    ex_embeddings = torch.ops.sok.reorder(ex_embeddings, order)

    return ex_embeddings