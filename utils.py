import torch
import torch.distributed as dist


def setup(rank, world_size):
    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank
    )
    #torch.cuda.set_device(rank)
    print(f"Rank: {rank} has initialized its process group with world size {world_size}")

def cleanup():
    rank = dist.get_rank()
    dist.destroy_process_group()
    print(f"Rank {rank} cleaned up.")

