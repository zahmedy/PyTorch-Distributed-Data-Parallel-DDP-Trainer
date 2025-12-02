import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SimpleCNN
from data import get_dataloader
from utils import setup, cleanup

import argparse


def ddp_train(rank, world_size, epochs, batch_size):
    """
    Main function for parallel distrbuted model training 

    rank:           current process rank
    world_size:     number of process per training
    epochs:         number of training loops
    batch_size:     size of chunks to run per GPU
    """
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

    # Training loop 
    for epoch in range(epochs):
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

    cleanup()


def main():
    parser = argparse.ArgumentParser(description='DDP Training Script')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(ddp_train, 
             args=(world_size, args.epochs, args.batch_size), 
             nprocs=world_size)


if __name__ == "__main__":
    main()