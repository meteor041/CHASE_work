# test_nccl.py
import os
import torch
import torch.distributed as dist

os.environ["RANK"] = os.environ.get("RANK", "0")
os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")
os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "3")

dist.init_process_group(backend="nccl", init_method="env://")
print(f"✅ NCCL initialized! Rank: {dist.get_rank()}, Device: {torch.cuda.current_device()}")
