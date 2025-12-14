import torch
import json
from mlkv_plus.communication import num_gpus, num_ranks, rank, is_comm_tool_set

import warnings


class MLKVPlusDB:
    def __init__(self,
                 dim: int,
                 max_hbm_for_vectors_gb: int,
                 gpu_init_capacity: int,
                 gpu_max_capacity: int,
                 max_batch_size: int,
                 create_if_missing: bool = True,
                 hkv_io_by_cpu: bool = False,
                 rocksdb_path: str = None,
                 gpu_id: int = None,
                 enable_gds_log: bool = False,
                 enable_gds_get_from_sst: bool = False,
                 disableWAL: bool = False,
                 force_skip_memtable: bool = False,
                 rocksdb_use_direct_reads: bool = False
                 ):
        """
        Initialize MLKVPlusDB from individual parameters.
        Parameters are internally converted to JSON and passed to the JSON-based create function.
        
        Args:
            dim: Dimension of the vectors
            max_hbm_for_vectors_gb: Maximum HBM size for vectors in GB
            create_if_missing: Whether to create RocksDB if missing
            gpu_init_capacity: Initial GPU capacity
            gpu_max_capacity: Maximum GPU capacity
            max_batch_size: Maximum batch size
            hkv_io_by_cpu: Whether to use CPU for HKV I/O
            rocksdb_path: Path to RocksDB (default: /tmp/mlkv_plus_rocksdb_{rank})
            gpu_id: GPU ID (default: current device)
            json_config: Optional JSON config string (if provided, other parameters are ignored)
        """

        # Convert parameters to JSON config
        # Get current device if not specified
        if gpu_id is None:
            gpu_id = torch.cuda.current_device()
        
        num_ranks_ = 1
        rank_ = 0
        
        if is_comm_tool_set():
            num_ranks_ = num_ranks()
            rank_ = rank()
        
        # Build config dictionary from parameters
        config_dict = {
            "dim": dim,
            "max_hbm_for_vectors_gb": max_hbm_for_vectors_gb if max_hbm_for_vectors_gb is not None else 0,
            "hkv_io_by_cpu": hkv_io_by_cpu,
            "gpu_id": gpu_id,
            "create_if_missing": create_if_missing,
            "hkv_init_capacity": gpu_init_capacity if gpu_init_capacity is not None else 64 * 1024 * 1024,
            "hkv_max_capacity": gpu_max_capacity if gpu_max_capacity is not None else 64 * 1024 * 1024,
            "max_batch_size": (max_batch_size if max_batch_size is not None else 1048576) * num_ranks_,
            "rocksdb_path": rocksdb_path if rocksdb_path else f"/tmp/mlkv_plus_rocksdb_{rank_}",
            "enable_gds_log": enable_gds_log,
            "enable_gds_get_from_sst": enable_gds_get_from_sst,
            "disableWAL": disableWAL,
            "force_skip_memtable": force_skip_memtable,
            "rocksdb_use_direct_reads": rocksdb_use_direct_reads
        }
        
        # Extract values for internal use
        self._dim = config_dict.get("dim")
        if self._dim is None:
            raise ValueError("Config must contain 'dim' field")
        
        self._max_hbm_for_vectors_gb = config_dict["max_hbm_for_vectors_gb"]
        self._hkv_io_by_cpu = config_dict["hkv_io_by_cpu"]
        self._gpu_id = config_dict["gpu_id"]
        self._create_if_missing = config_dict["create_if_missing"]
        self._gpu_init_capacity = config_dict["hkv_init_capacity"]
        self._gpu_max_capacity = config_dict["hkv_max_capacity"]
        self._max_batch_size = config_dict["max_batch_size"]
        self._rocksdb_path = config_dict["rocksdb_path"]
        self._enable_gds_log = config_dict["enable_gds_log"]
        self._enable_gds_get_from_sst = config_dict["enable_gds_get_from_sst"]
        self._disableWAL = config_dict["disableWAL"]
        self._force_skip_memtable = config_dict["force_skip_memtable"]
        self._rocksdb_use_direct_reads = config_dict["rocksdb_use_direct_reads"]
        
        
        if self._gpu_id != torch.cuda.current_device():
            warnings.warn(f"GPU ID {self._gpu_id} is not the current device {torch.cuda.current_device()}, but it is required to be the current device for MLKV Plus to work correctly.")
        
        # Always use JSON-based create function
        json_config_str = json.dumps(config_dict)
        self._db = torch.ops.libmlkvplus_torch.create_dummy_var_from_json(
            json_config=json_config_str,
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
    