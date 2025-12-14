import torch

import mlkv_plus


# mlkv_plus.init(comm_tool="torch_dist")

# keys = torch.tensor([1, 2, 3], dtype=torch.int64, device=torch.device('cuda', torch.cuda.current_device()))

# values = torch.tensor([[11,12,13,14,15], [21,22,23,24,25], [31,32,33,34,35]], dtype=torch.float32, device=torch.device('cuda', torch.cuda.current_device()))

db = mlkv_plus.MLKVPlusDB(
    dim=15,
    max_hbm_for_vectors_gb=2,
    create_if_missing=True,
    gpu_init_capacity=100000,
    gpu_max_capacity=500000,
    max_batch_size=10000,
    hkv_io_by_cpu=False
)

# db.initialize()


# db.assign(keys, values)

# print(f"db: {db}")

# if communication.rank() == 0:
#     keys = torch.tensor([1, 2, 3], dtype=torch.int64, device=torch.device('cuda', torch.cuda.current_device()))
#     values = torch.tensor([[11,12,13,14,15], [21,22,23,24,25], [31,32,33,34,35]], dtype=torch.float32, device=torch.device('cuda', torch.cuda.current_device()))
# else:
#     keys = torch.zeros(3, dtype=torch.int64, device=torch.device('cuda', torch.cuda.current_device()))
#     values = torch.zeros((3, 5), dtype=torch.float32, device=torch.device('cuda', torch.cuda.current_device()))
    

# communication.broadcast(keys, root=0)
# communication.broadcast(values, root=0)

# print(f"values: {values}")

# mlkv_plus.dist_assign(db, keys, values)

# communication.barrier()
# print(f"Rank {communication.rank()}: Barrier passed")



# result = mlkv_plus.all2all_dense_embedding(db, keys)


# print(f"Final result rank {communication.rank()}: {result}")



