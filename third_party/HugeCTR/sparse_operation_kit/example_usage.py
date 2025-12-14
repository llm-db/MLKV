# import json
# from kit_src import sok
# import torch
# import numpy as np


# # print the sok location
# print("sok location: ", sok.__file__)

# # Ensure CUDA is available and set as current device
# assert torch.cuda.is_available(), "CUDA is required for this example"

# print("torch.cuda.device_count(): ", torch.cuda.device_count())

# hkv_config = {
#     "device_id": 0,
#     "init_capacity": 1 << 10,
#     "max_capacity":  1 << 19,
# }


# db = torch.ops.sok.create_dummy_var(
#     container="test_container",
#     shared_name="test_table",
#     shape=[1000, 16],
#     key_type=1,
#     value_type=0,
#     dummy_device=torch.empty(0, dtype=torch.int64, device=torch.device('cuda', 0)),
#     var_type="hybrid",
#     initializer="random",
#     config=json.dumps(hkv_config)
# )
# print(f"DummyVar created successfully!")

    

# keys = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64)
# values = torch.randn(10, 16, dtype=torch.float32)

# print(f"Created keys on: {keys.device}")
# print(f"Created values on: {values.device}")

# print("Attempting assign operation...")
# print(f"keys device: {keys.device}, shape: {keys.shape}, dtype: {keys.dtype}")
# print(f"values device: {values.device}, shape: {values.shape}, dtype: {values.dtype}")

# result = torch.ops.sok.assign_int64(db, keys, values)
# print(f"Assign successful: {result}")





# # test sparse_read
# keys_gpu = torch.tensor([1, 2, 3, 1,2,3], dtype=torch.int64, device=torch.device('cuda', 0))
# read_result = torch.ops.sok.sparse_read_int64(db, keys_gpu)
# print(read_result)

# read_result_again = torch.ops.sok.sparse_read_int64(db, keys_gpu)
# print("read_result_again:", read_result_again)

# read_result_again = torch.ops.sok.sparse_read_int64(db, keys_gpu)
# print("read_result_again:", read_result_again)

# # validate the sparse_read result
# ground_truth = values.to(read_result.device)

# assert torch.allclose(read_result, ground_truth)
# print(f"Sparse read result is correct!")


# # test dist_select
# # Use the same device as db for consistency
# indices = torch.tensor([1,2,3,4,5,6,7,8,9,10],
#                        dtype=torch.int64,
#                        device=torch.device('cuda', 0))

# output, order, splits = torch.ops.sok.dist_select(indices, 2)
# print(output, order, splits)




import torch
import sparse_operation_kit as sok
import horovod.torch as hvd


# Initialize Horovod
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
torch.cuda.set_device(hvd.local_rank())


sok.init()

db = sok.DynamicVariable(
    dimension=16,
    var_type="hybrid",
    initializer="0",
    init_capacity=1 << 10,
    max_capacity=1 << 19
)



if hvd.rank() == 0:
    keys = torch.tensor([2, 4, 6], dtype=torch.int64)
    values = torch.randn(3, 16, dtype=torch.float32)
elif hvd.rank() == 1:
    keys = torch.tensor([1, 3, 5], dtype=torch.int64)
    values = torch.randn(3, 16, dtype=torch.float32)
else:
    keys = torch.tensor([], dtype=torch.int64)
    values = torch.tensor([], dtype=torch.float32)

sok.assign(db, keys, values)

hvd.allreduce(torch.tensor(0.0))


# test all2all_dense_embedding
indices = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64, device=torch.device('cuda', torch.cuda.current_device()))


result = sok.all2all_dense_embedding(db, indices)

hvd.allreduce(torch.tensor(0.0))
print(f"Final result rank {hvd.rank()}: {result}")