#!/usr/bin/env -S uv run

import os

os.environ["NCCL_P2P_DISABLE"] = "0"
os.environ["NCCL_DEBUG"] = "INFO"

import torch
import torch.distributed as dist
import datetime


def run():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=30))  # type: ignore[attr-defined]
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {local_rank}] Initialized. Testing P2P...")
    tensor = torch.ones(1024 * 1024 * 100, device=device) * (
        local_rank + 1
    )  # 100MB tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)  # type: ignore[attr-defined]

    print(f"[Rank {local_rank}] P2P Success! Tensor[0] = {tensor[0].item()}")
    dist.destroy_process_group()  # type: ignore[attr-defined]


if __name__ == "__main__":
    run()
