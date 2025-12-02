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
    model = SimpleCNN()                         # core model
    model = DDP(model)                          # Distrbuited Data Parallel wrapper, which selects the device
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

    # If NO checkpoint exists → start from scratch
    if flag.item() == 0:
        start_epoch = 0
        print(f"Rank {rank}: No checkpoint found, starting from scratch.")
        # go directly to TRAINING loop
        # do NOT run all the key/shape/value resume code
        # so you skip model.load_state_dict(...)
        goto_training = True
    else:
        goto_training = False

    epoch_tensor = torch.tensor([0], 
                                dtype=torch.int64)
    num_key_tensor = torch.tensor([0], 
                                  dtype=torch.int64)   
    num_keys = 0

    if goto_training == False:
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

        all_shapes = []
        for key in all_keys:
            # 1. Create dimension tensor
            dim_count_tensor = torch.tensor([0], dtype=torch.int64)

            # 2. Rank 0 gets shape tuple for this key and fills dim_count_tensor
            if rank == 0:
                shape = model_state[key].shape
                dim_count_tensor[0] = len(shape)

            # 3. Broadcast dims
            dist.broadcast(dim_count_tensor, src=0)

            # 4. All ranks allocate shape_tensor = zeros(dim_count)
            shape_tensor = torch.zeros(dim_count_tensor.item(), dtype=torch.int64)

            # 5. Rank 0 fills shape_tensor (one dimension per entry)
            if rank == 0:
                shape_tensor[:] = torch.tensor(list(shape), dtype=torch.int64)

            # 6. Broadcast shape_tensor
            dist.broadcast(shape_tensor)

            # 7. All ranks store shape
            all_shapes.append(tuple(shape_tensor.tolist()))


        # Final stage
        loaded_state = {}
        for key, shape in zip(all_keys, all_shapes):

            # 1. Compute number of elements
            numel = model_state[key].numel()

            # 2. Allocate a 1-D float32 buffer on ALL ranks
            value_buffer = torch.zeros(numel, dtype=torch.float32)

            # 3. Rank 0 flattens the tensor
            if rank == 0:
                flat_value = model_state[key].reshape(-1)
                value_buffer[:] = flat_value
            
            # 4. Broadcast buffer
            dist.broadcast(value_buffer, src=0)

            # 5. Reshape it back
            value = value_buffer.view(shape)

            loaded_state["module." + key] = value
        
        model.load_state_dict(loaded_state)
        start_epoch = int(epoch_tensor.item())


    ##################
    ##   Training   ##
    ################## 

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)    # for shuffling data for each epoch
        print(f"Rank {rank}: Starting epoch {epoch}")
        for batch_idx, (images, labels) in enumerate(dataloader):
            opt.zero_grad()     # clear resduial gradients from prev runs
            output = model(images)       # forward pass throught the network for current rank 
            loss = loss_fn(output, labels)    # calculate crossEntropy loss of current weights
            loss.backward()     # compute gradient 
            opt.step()          # update weights
            if rank == 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

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

    world_size = 2 # can't use torch.cuda.device_count() on Apple Mac Silicon
    spawn(ddp_train, 
             args=(world_size, args.epochs, args.batch_size), 
             nprocs=world_size)


if __name__ == "__main__":
    main()