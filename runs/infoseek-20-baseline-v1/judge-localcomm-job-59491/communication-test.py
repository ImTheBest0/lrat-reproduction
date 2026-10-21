import json
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path

def worker(rank):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=90))
    print(f"Communication test: rank {rank} Gloo initialized.", flush=True)
    value = torch.tensor([float(rank + 1)])
    dist.all_reduce(value)
    assert value.item() == 3.0, "Gloo reduction failed."
    gpu_group = dist.new_group(ranks=[0, 1], backend="nccl", timeout=timedelta(seconds=90))
    value = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    dist.all_reduce(value, group=gpu_group)
    torch.cuda.synchronize()
    assert value.item() == 3.0, "PyTorch NCCL reduction failed."
    print(f"Communication test: rank {rank} PyTorch NCCL passed.", flush=True)
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    communicator = PyNcclCommunicator(group=dist.group.WORLD, device=rank)
    value = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    reduced = communicator.all_reduce(value)
    torch.cuda.synchronize()
    assert reduced.item() == 3.0, "vLLM NCCL reduction failed."
    print(f"Communication test: rank {rank} vLLM NCCL passed.", flush=True)
    dist.barrier()
    dist.destroy_process_group(gpu_group)
    dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    # One node only: the local loopback address is reachable by both workers.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(listener.getsockname()[1])
    mp.spawn(worker, nprocs=2, join=True)
    Path(sys.argv[1]).write_text(json.dumps({"passed": True, "ranks": 2,
        "tests": ["Gloo", "PyTorch NCCL", "vLLM PyNcclCommunicator"],
        "environment": {key: os.environ.get(key) for key in (
            "VLLM_HOST_IP", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME",
            "NCCL_IB_DISABLE", "NCCL_CUMEM_ENABLE")}}, indent=2) + "\n")
    print("Two-GPU communication test passed.", flush=True)
