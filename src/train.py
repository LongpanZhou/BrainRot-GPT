import os
import torch
import random
import tiktoken
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
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

# Optimizations
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

# Hyperparameters
BATCH_SIZE = 32
SEQUENCE_LENGTH = 512
STEPS = 200000
LEARNING_RATE = 6e-4
WARMUP_STEPS = STEPS // 1000
EVAL_EVERY_STEPS = 100
EVAL_ITERS = 10
SAVE_EVERY_STEPS = 10000
SAVE_DIR = "./checkpoints"

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def main(rank, world_size, ddp, train_dataset=None, val_dataset=None):
    try:
        # Setup DDP if enabled
        if ddp:
            setup_ddp(rank, world_size)
        local_device = torch.device(f"cuda:{rank}")

        # Model
        model = GPT(GPTConfig(dropout=0.1)).to(local_device)
        if device == "cuda":
            model = torch.compile(model, mode="default", fullgraph=False)
        if ddp:
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            dist.barrier()

        # Optimizer and Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-6)
        scaler = GradScaler(enabled=(device == "cuda"))

        # Training Loop
        for step in range(STEPS):
            # Get random batch (each process gets different batch)
            train_idx = random.randint(0, len(train_dataset) - 1)
            x, y = train_dataset[train_idx]
            x, y = x.to(local_device, non_blocking=True), y.to(local_device, non_blocking=True)

            # TRAIN
            model.train()
            optimizer.zero_grad()

            # Forward pass
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
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

                if rank == 0:  # Only rank 0 does evaluation
                    model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for _ in range(EVAL_ITERS):
                            val_idx = random.randint(0, len(val_dataset) - 1)
                            x, y = val_dataset[val_idx]
                            x, y = x.to(local_device, non_blocking=True), y.to(local_device, non_blocking=True)
                            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                                _, loss = model(x, y)
                            val_losses.append(loss.item())
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    perplexity = torch.exp(torch.tensor(avg_val_loss))
                    print(f"Step: {step}, Train Loss: {loss.item()}, Val Loss: {avg_val_loss}, "
                          f"Perplexity: {perplexity.item()}, Grad Norm: {grad_norm.item()}")

                    # Save checkpoint
                    if step % SAVE_EVERY_STEPS == 0 and rank == 0:
                        torch.save(model.module.state_dict() if ddp else model.state_dict(), 'last_ckpt.pt')

                    # Generate text
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
                        print(f"Generated text: {enc.decode(out[0].tolist())}")

    except Exception as e:
        print(f"Rank {rank}: Error occurred: {str(e)}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    ddp = True  # Set this to False for single GPU training
    print("Distributed Data Parallel (DDP) mode:", ddp)

    os.makedirs(SAVE_DIR, exist_ok=True)
    train_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="train")
    val_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="test")

    if ddp:
        GPU_IDS = [0, 1, 2, 3]  # Set your GPU IDs here
        world_size = len(GPU_IDS)

        if world_size > torch.cuda.device_count():
            raise ValueError(f"Requested {world_size} GPUs, but only {torch.cuda.device_count()} available")

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