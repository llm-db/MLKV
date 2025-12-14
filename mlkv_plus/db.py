import torch
import json
from mlkv_plus.communication import num_gpus, num_ranks, rank, is_comm_tool_set

import warnings


class MLKVPlusDB:
    def __init__(self,
                 dim: int,
                 max_hbm_for_vectors_gb: int,
                 create_if_missing: bool,
                 gpu_init_capacity: int,
                 gpu_max_capacity: int,
                 max_batch_size: int,
                 hkv_io_by_cpu: bool = False,
                 rocksdb_path: str = None,
                 gpu_id: int = None):
        self._dim = dim
        self._max_hbm_for_vectors_gb = max_hbm_for_vectors_gb
        self._hkv_io_by_cpu = hkv_io_by_cpu
        # Get current device if not specified
        self._gpu_id = gpu_id if gpu_id is not None else torch.cuda.current_device()
        self._create_if_missing = create_if_missing
        self._gpu_init_capacity = gpu_init_capacity
        self._gpu_max_capacity = gpu_max_capacity
        self._max_batch_size = max_batch_size
        
    
    
        num_ranks_ = 1
        rank_ = 0
        
        if is_comm_tool_set():
            num_ranks_ = num_ranks()
            rank_ = rank()
            
            
        self._rocksdb_path = rocksdb_path if rocksdb_path else f"/tmp/mlkv_plus_rocksdb_{rank_}"
        
        
        if self._gpu_id != torch.cuda.current_device():
            warnings.warn(f"GPU ID {self._gpu_id} is not the current device {torch.cuda.current_device()}, but it is required to be the current device for MLKV Plus to work correctly.")
        
        
        self._db = torch.ops.libmlkvplus_torch.create_dummy_var(
            dim=self._dim,
            max_hbm_for_vectors_gb=self._max_hbm_for_vectors_gb,
            hkv_io_by_cpu=self._hkv_io_by_cpu,
            gpu_id=self._gpu_id,
            create_if_missing=self._create_if_missing,
            gpu_init_capacity=self._gpu_init_capacity,
            gpu_max_capacity=self._gpu_max_capacity,
            max_batch_size=self._max_batch_size * num_ranks_,
            rocksdb_path=self._rocksdb_path,
            ensure_device=torch.tensor([self._gpu_id], dtype=torch.int64, device=torch.device('cuda', self._gpu_id))
        )
        
        
    def __repr__(self):
        return f"MLKVPlusDB(dim={self._dim}, max_hbm_for_vectors_gb={self._max_hbm_for_vectors_gb}, hkv_io_by_cpu={self._hkv_io_by_cpu}, gpu_id={self._gpu_id}, create_if_missing={self._create_if_missing}, gpu_init_capacity={self._gpu_init_capacity}, gpu_max_capacity={self._gpu_max_capacity})"
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def target_gpu(self):
        return self.gpu_id
    
    
    @property
    def num_gpus(self):
        return num_gpus()
    
    
    def key_map(self, keys: torch.Tensor):
        return keys
    
    
    
    def initialize(self):
        torch.ops.libmlkvplus_torch.initialize(self._db)
    
    
    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def assign(self, keys: torch.Tensor, values: torch.Tensor):
        torch.ops.libmlkvplus_torch.assign(self._db, keys, values)
        
        
        
    def read(self, keys: torch.Tensor):
        return torch.ops.libmlkvplus_torch.read(self._db, keys)
    