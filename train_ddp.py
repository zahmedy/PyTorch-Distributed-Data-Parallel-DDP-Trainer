import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SimpleCNN
from data import get_dataloader
from utils import setup, cleanup

import argparse
import os
from pathlib import Path


def ddp_train(rank, world_size, epochs, batch_size):
    """
    Main function for parallel distrbuted model training 

    rank:           current process rank
    world_size:     number of process per training
    epochs:         number of training loops
    batch_size:     size of chunks to run per GPU
    """

    ##################
    ##    Set up    ##
    ##################

    setup(rank=rank, world_size=world_size)     # connect the process to the group and set the correct GPU.
    print(f"Rank {rank} initialized")
    model = SimpleCNN().to(rank)                # core model
    model = DDP(model,                          # Distrbuited Data Parallel wrapper, which selects the device
                device_ids=[rank], 
                output_device=rank)
    print(f"Rank {rank} model wrapped in DDP")  
    dataloader, sampler = get_dataloader(batch_size,   # chops MINST dataset to equal sizes for world_size
                                         world_size, 
                                         rank)
    print(f"Rank {rank}: Dataloader ready")
    loss_fn = nn.CrossEntropyLoss()                 
    opt = optim.SGD(model.parameters(),lr=0.01)
    model.train()                                   # turns on batchnorm + dropout
    print(f"Rank {rank}: Loss and Optimizer ready")

    ##################
    ## Pre-Training ##
    ##################

    flag = torch.tensor([0], dtype=torch.int64)
    latest_file = None

    if rank == 0:   # only for first process to prevent overwritting ... etc
        dir = Path("./checkpoints")
        if dir.exists():
            files = [f for f in dir.iterdir() if f.is_file()]
            if files:
                latest_file = max(files, key=lambda f: f.stat().st_ctime)
                flag[0] = 1 
        else:
            os.mkdir("./checkpoints") # Make sure folder exist
    
    dist.broadcast(flag, src=0)                 # Tell everyone about checkpoint
    dist.barrier()                          # wait until folder is created
    epoch_tensor = torch.tensor([0], 
                                dtype=torch.int64)
    num_key_tensor = torch.tensor([0], 
                                  dtype=torch.int64)   
    num_keys = 0

    if flag[0]:
        if rank ==0:
            # Load checkpoint
            ckpt = torch.load(str(latest_file), map_location="cpu")
            model_state = ckpt["model"]
            optimizer_state = ckpt["optimizer"]
            start_epoch = ckpt["epoch"] + 1
            epoch_tensor[0] = start_epoch

            # keys
            num_keys = len(model_state.keys())
            num_key_tensor[0] = num_keys        # All ranks know num of keys

            # sort keys to insure consistency among all ranks(Processes)
            keys = sorted(model_state.keys())
            for key in keys:
                shape = model_state[key].shape
            
    # Broadcast epoch and number of keys
    dist.broadcast(epoch_tensor, src=0)
    dist.broadcast(num_key_tensor, src=0)
    
    all_keys = []
    for i in range(num_keys):

        # 1. Each rank creates key_len_tensor = [0]
        key_len_tensor = torch.tensor([0], dtype=torch.int64)

        # 2. Rank 0 fills the key_len_tensor
        if rank == 0:
            key = keys[i]
            key_bytes = key.encode("utf-8")
            key_len_tensor[0] = len(key_bytes)
        
        # 3. Broadcast key length
        dist.broadcast(key_len_tensor)

        # 4. All ranks allocate recv_buffer = uint8[length]
        recv_buffer = torch.zeros(key_len_tensor.item(), dtype=torch.uint8)

        # 5. Rank 0 copies bytes into recv_buffer
        if rank == 0:
            recv_buffer[:] = torch.tensor(list(key_bytes), dtype=torch.uint8)

        # 6. Broadcast recv_buffer
        dist.broadcast(recv_buffer, src=0)

        # 7. All ranks decode to string
        key_str = bytes(recv_buffer.tolist()).decode("utf-8")

        # 8. Append to all_keys
        all_keys.append(key_str)

    start_epoch = int(epoch_tensor.item())


    ##################
    ##   Training   ##
    ################## 

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)    # for shuffling data for each epoch
        print(f"Rank {rank}: Starting epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(rank)    # assign data to correct GPU
            labels = labels.to(rank)    

            opt.zero_grad()     # clear resduial gradients from prev runs
            output = model(images)       # forward pass throught the network for current rank 
            loss = loss_fn(output, labels)    # calculate crossEntropy loss of current weights
            loss.backward()     # compute gradient 
            opt.step()          # update weights
            if rank == 0:
                print(f"Epoch: {epoch} Loss: {loss}")

        if rank == 0:
            torch.save({
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": opt.state_dict()
            },f"./checkpoints/epoch_{epoch}.pt")


    cleanup()


def main():
    parser = argparse.ArgumentParser(description='DDP Training Script')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    spawn(ddp_train, 
             args=(world_size, args.epochs, args.batch_size), 
             nprocs=world_size)


if __name__ == "__main__":
    main()