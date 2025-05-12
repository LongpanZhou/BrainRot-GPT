import os
import time
import torch
import tiktoken
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader import FineWeb
from model import GPT, GPTConfig

# Device configuration
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Hyperparameters
BATCH_SIZE = 64                             # Batch size
SEQUENCE_LENGTH = 512                       # Sequence length
STEPS = 20000                               # Total training steps
LEARNING_RATE = 1e-3                        # Learning rate             <- Your model might explode due to higher lr (This might because of Mixed Precision)
WARMUP_STEPS = STEPS // 1000                # Warmup steps
EVAL_EVERY_STEPS = 100                      # Evaluate every N steps
EVAL_ITERS = 10                             # Number of evaluation iterations
SAVE_EVERY_STEPS = STEPS//10                # Save every N steps
SAVE_DIR = "./checkpoints"                  # Directory to save checkpoints
CUDA_ENABLED = (device == "cuda")           # Use CUDA

# Optimizations
if CUDA_ENABLED:
    torch.backends.cudnn.benchmark = True               # Enable cuDNN auto-tuner
    torch.backends.cuda.matmul.allow_tf32 = True        # Enable TensorFloat-32 for matmul
    torch.backends.cudnn.allow_tf32 = True              # Enable TensorFloat-32 for cuDNN
    torch.set_float32_matmul_precision("medium")        # Set float32 matmul precision to medium

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")              # Tokenizer (same as dataloader.py)

# DDP setup -> please ignore this if you are not using DDP
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# Main
def main(rank, world_size, ddp, train_dataset, val_dataset):
    try:
        # Setup DDP if enabled
        if ddp:
            setup_ddp(rank, world_size)

        # Set device
        local_device = torch.device(f"cuda:{rank}")

        # Model
        model = GPT(GPTConfig(
            # block_size= 512,     # sequence length
            # vocab_size= 100258,  # using cl100k_base tokenizer
            # n_layers= 36,        # depth
            # n_embd= 1280,        # hidden size
            # n_head= 20,          # attention heads
            dropout= 0.1,          # dropout rate
            # bias= True,          # use bias
            # GQA= True,           # GQA model
        )).to(local_device)

        # Compile model if CUDA is enabled
        if CUDA_ENABLED:
            model = torch.compile(model, mode="max-autotune", fullgraph=True)

        # DDP model wrapping
        if ddp:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            dist.barrier()

        # Optimizer and Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=CUDA_ENABLED)
        scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-6)
        scaler = GradScaler(enabled=CUDA_ENABLED)

        # Precompute random indices
        train_idx = np.random.randint(0, len(train_dataset), size=STEPS)
        val_idx = np.random.randint(0, len(val_dataset), size=STEPS*EVAL_ITERS)
        v_idx = 0

        # Training Loop
        for step in range(STEPS):
            # Get random batch (each process gets different batch)
            x, y = train_dataset[train_idx[step]]
            x, y = x.to(local_device, non_blocking=True), y.to(local_device, non_blocking=True)

            # Train step
            model.train()
            optimizer.zero_grad()

            # Forward pass
            with torch.autocast(device_type="cuda", enabled=CUDA_ENABLED):
                _, loss = model(x, y)

            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # EVALUATE
            if step % EVAL_EVERY_STEPS == 0:
                if ddp:
                    dist.barrier()  # Synchronize before evaluation

                if rank == 0:       # Only rank 0 does evaluation
                    start = time.time()
                    model.eval()
                    val_losses = []

                    # Evaluate on validation set
                    with torch.no_grad():
                        for t in range(EVAL_ITERS):
                            x, y = val_dataset[val_idx[v_idx]]
                            x, y = x.to(local_device, non_blocking=True), y.to(local_device, non_blocking=True)
                            with torch.autocast(device_type="cuda", enabled=CUDA_ENABLED):
                                _, loss = model(x, y)
                            val_losses.append(loss.item())
                            v_idx += 1

                    # Calculate average validation loss and perplexity
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    perplexity = torch.exp(torch.tensor(avg_val_loss))

                    # Print information
                    end = time.time()
                    print(end - start)
                    print(f"Step: {step}, Train Loss: {loss.item()}, Val Loss: {avg_val_loss}, "
                          f"Perplexity: {perplexity.item()}, Grad Norm: {grad_norm.item()}, lr: {scheduler.get_last_lr()[0]}")

                    # Save checkpoint
                    if step % SAVE_EVERY_STEPS == 0 and rank == 0:
                        torch.save({
                            'model_state_dict': model.module.state_dict() if ddp else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': step,
                        }, 'checkpoints/last_ckpt.pth')

                    # Generate text
                    start = time.time()
                    with torch.no_grad():
                        text = "Hello, how are you doing today? I hope you're having a great day!"
                        tokens = enc.encode(text)
                        out = model.module.generate(
                            torch.tensor([tokens]).to(local_device),
                            max_new_tokens=SEQUENCE_LENGTH,
                            temperature=1.0,
                            end_token=enc.eot_token,
                        ) if ddp else model.generate(
                            torch.tensor([tokens]).to(local_device),
                            max_new_tokens=SEQUENCE_LENGTH,
                            temperature=1.0,
                            end_token=enc.eot_token,
                        )
                    end = time.time()
                    print(end-start)
                    print(enc.decode(out[0].tolist()))

    # Handle exceptions
    except Exception as e:
        print(f"Rank {rank}: Error occurred: {str(e)}")
        raise
    # Ensure all processes exit
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    ddp = False  # Set this to False for single GPU training
    print("Distributed Data Parallel (DDP) mode:", ddp)
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="train")
    val_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="test")

    if ddp:
        GPU_IDS = [0, 1, 2, 3]  # Set your GPU IDs here
        world_size = len(GPU_IDS)

        # Check if the requested GPUs are available
        if world_size > torch.cuda.device_count():
            raise ValueError(f"Requested {world_size} GPUs, but only {torch.cuda.device_count()} available")

        # DDP initialization
        print(f"Starting DDP with {world_size} GPUs: {GPU_IDS}")
        mp.set_start_method('spawn', force=True)
        mp.spawn(
            main,
            args=(world_size, ddp, train_dataset, val_dataset),
            nprocs=world_size,
            join=True
        )
    else:
        print("Starting single GPU training")
        main(0, 1, ddp=False, train_dataset=train_dataset, val_dataset=val_dataset)