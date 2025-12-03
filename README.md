# PyTorch Distributed Data Parallel Trainer

Small, self contained reference that shows how to train a PyTorch model with Distributed Data Parallel (DDP), resume from checkpoints, and keep the codebase approachable for beginners.

```mermaid

flowchart TD

    A[Start torchrun --nproc_per_node=2] --> B[Spawn Processes: Rank 0 and Rank 1]

    B --> C[Initialize Process Group]

    C --> D1[Rank 0 Loads Dataset MNIST]
    C --> D2[Rank 1 Loads Dataset MNIST]

    D1 --> E1[Rank 0 Builds Model]
    D2 --> E2[Rank 1 Builds Model]

    E1 --> F1[Rank 0 Wraps Model in DDP]
    E2 --> F2[Rank 1 Wraps Model in DDP]

    F1 --> G1[Rank 0 Forward Pass]
    F2 --> G2[Rank 1 Forward Pass]

    G1 --> H1[Rank 0 Backward Pass]
    G2 --> H2[Rank 1 Backward Pass]

    H1 --> I[DDP Gradient Synchronization All-Reduce]
    H2 --> I

    I --> J1[Optimizer Step on Rank 0]
    I --> J2[Optimizer Step on Rank 1]

    J1 --> K1[Rank 0 Saves Checkpoint]
    J2 --> K2[Rank 1 Does Not Save]

    K1 --> L[Cleanup Process Group]

    L --> M[End Training]


```

## What this repo demonstrates
- Distributed setup with `torch.multiprocessing.spawn` and proper backend selection for CPU/GPU.
- Simple CNN on MNIST with per-rank distributed samplers.
- Automatic checkpoint discovery/resume (latest file in `./checkpoints`).
- CLI flags for epochs, batch size, and world size so you can scale up or run a quick local demo.

## Repo layout
```
├── train_ddp.py    # main entrypoint (run with torchrun)
├── model.py        # simple CNN classifier
├── data.py         # MNIST dataset + distributed sampler
├── utils.py        # process group setup/cleanup helpers
└── checkpoints/    # populated at runtime
```

## Quickstart
1) Create an environment and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Launch DDP training (CPU example, 2 processes):
```bash
torchrun --nproc_per_node=2 train_ddp.py --epochs 3 --batch_size 64
```
> MNIST is downloaded automatically to `./data` on first run.
> Prefer plain Python? The script will spawn workers itself: `python train_ddp.py --world_size 2`.

3) Resume training: rerun the same command — the latest checkpoint in `./checkpoints` will be loaded automatically.

## Common tweaks
- GPUs available: `torchrun --nproc_per_node=4 train_ddp.py --world_size 4`
- Different dataset location: edit `data.py`'s `datasets.MNIST(root=...)`.
- Fresh start: delete the `checkpoints/` directory before launching.

## Notes
- Backend switches to `nccl` when CUDA is available, otherwise defaults to `gloo`.
- On Apple Silicon this example runs in CPU mode; adjust `world_size` to match how many processes you want to spawn.
