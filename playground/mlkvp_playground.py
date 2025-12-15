# torchrun --nproc_per_node=4 dist_mlkvp_playground.py

import torch
import mlkv_plus

torch.cuda.set_device(0)

keys = torch.tensor([1, 2, 3], dtype=torch.int64, device=torch.device('cuda', torch.cuda.current_device()))

values = torch.tensor([[11,12,13,14,15], [21,22,23,24,25], [31,32,33,34,35]], dtype=torch.float32, device=torch.device('cuda', torch.cuda.current_device()))

db = mlkv_plus.MLKVPlusDB(
    dim=5,
    max_hbm_for_vectors_gb=2,
    create_if_missing=True,
    gpu_init_capacity=1024,
    gpu_max_capacity=4096,
    max_batch_size=1048576,
    enable_gds_log=False,
    enable_gds_get_from_sst=False,
    disableWAL=False,
    force_skip_memtable=False,
    rocksdb_use_direct_reads=False
)

print(f"db: {db}")

db.assign(keys, values)
torch.cuda.synchronize()  # Sync to catch any CUDA errors from assign
print("assign completed successfully")

results = db.read(keys)
torch.cuda.synchronize()  # Sync to catch any CUDA errors from read
print(f"results shape: {results.shape}")
print(f"results: {results}")


try:
    torch.tensor([1, 2, 3], dtype=torch.int64, device=torch.device('cuda', torch.cuda.current_device()))
except Exception as e:
    print(f"Error: {e}")

