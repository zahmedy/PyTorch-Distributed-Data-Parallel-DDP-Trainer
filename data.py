import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler


def get_dataloader(batch_size, world_size, rank):
    """
    Get dataset as a Tensor then create sampler for each process then create
    the dataloader 
    """
    dataset = datasets.MNIST(root="./data", 
                        train=True, 
                        transform=transforms.ToTensor(), 
                        download=True)
    
    sampler = DistributedSampler(dataset, 
                                 num_replicas=world_size, 
                                 rank=rank, 
                                 shuffle=True)
    
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            sampler=sampler, 
                            num_workers=2, 
                            pin_memory=True)

    return dataloader, sampler