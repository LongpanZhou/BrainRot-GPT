import os
import sys
import time
import torch
import tiktoken
import logging
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader import FineWeb
from model_ import GPT, GPTConfig

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
LEARNING_RATE = 1e-3                       # Learning rate
WARMUP_STEPS = STEPS // 1000                # Warmup steps
EVAL_EVERY_STEPS = 100                      # Evaluate every N steps
EVAL_ITERS = 10                             # Number of evaluation iterations
SAVE_EVERY_STEPS = STEPS//10                # Save every N steps
SAVE_DIR = "./checkpoints"                  # Directory to save checkpoints
LOG_DIR = "./logs"                          # Directory to save logs
CUDA_ENABLED = (device == "cuda")           # Use CUDA
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16 # Default dtype for CUDA

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")              #same as dataloader.py

# Setup logging
def setup_logging(ddp):
    # Configure logging
    log_idx = max([int(f.split(".")[0]) for f in os.listdir(LOG_DIR) if f.endswith(".log")], default=-1) + 1
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(LOG_DIR, f"{log_idx}.log"), mode='a'),
        ]
    )

    # Start info
    logging.info(f"{datetime.now()}")
    logging.info(f"Distributed Data Parallel (DDP) mode: {ddp}")
    logging.info(f"Using device: {device}")
    logging.info("===== Training Configuration =====")
    logging.info(f"BATCH_SIZE         = {BATCH_SIZE}")
    logging.info(f"SEQUENCE_LENGTH    = {SEQUENCE_LENGTH}")
    logging.info(f"STEPS              = {STEPS}")
    logging.info(f"LEARNING_RATE      = {LEARNING_RATE}")
    logging.info(f"WARMUP_STEPS       = {WARMUP_STEPS}")
    logging.info(f"EVAL_EVERY_STEPS   = {EVAL_EVERY_STEPS}")
    logging.info(f"EVAL_ITERS         = {EVAL_ITERS}")
    logging.info(f"SAVE_EVERY_STEPS   = {SAVE_EVERY_STEPS}")
    logging.info(f"SAVE_DIR           = {SAVE_DIR}")
    logging.info(f"LOG_DIR            = {LOG_DIR}")
    logging.info(f"CUDA_ENABLED       = {CUDA_ENABLED}")
    logging.info("===================================")

# DDP setup -> please ignore this if you are not using DDP
def setup_ddp(rank, world_size, GPU_ID):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(GPU_ID)

# Main
def main(rank, world_size, GPU_IDs, ddp, train_dataset, val_dataset, master_device):
    try:
        # Set current GPU_ID
        GPU_ID = GPU_IDs[rank]

        # Setup DDP if enabled
        if ddp:
            setup_ddp(rank, world_size, GPU_ID)

        # Set device
        local_device = torch.device(f"cuda:{GPU_ID}")
        is_master = True if world_size == 1 else (GPU_ID == master_device)

        # Setup logging
        if is_master: setup_logging(ddp)

        # Model
        model = GPT(GPTConfig(
            block_size = SEQUENCE_LENGTH,     # sequence length
            # vocab_size = 100258,  # using cl100k_base tokenizer
            # n_layers = 36,        # depth
            # n_embd = 1280,        # embedding size
            # n_head = 20,          # attention heads
            # dropout = 0.1,        # dropout rate
            # bias = True,          # bias
            # GQA = False,          # GQA
            # ROPE = True,          # ROPE
        )).to(dtype=DTYPE, device=local_device)

        # Compile model if CUDA is enabled
        if CUDA_ENABLED:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        # DDP model wrapping
        if ddp:
            model = DDP(model, device_ids=[GPU_ID], find_unused_parameters=True)
            dist.barrier()

        # Optimizer and Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=CUDA_ENABLED)
        scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=1e-5)

        # Precompute random indices
        train_idx = np.random.randint(0, len(train_dataset), size=STEPS)

        if is_master:
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
            with torch.autocast(device_type="cuda", enabled=CUDA_ENABLED, dtype=DTYPE):
                _, loss = model(x, y)

            # Backward pass with mixed precision
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # ONLY MASTER
            if is_master:
                # Evaluate
                if step % EVAL_EVERY_STEPS == 0:
                    # Evaluation Mode
                    model.eval()
                    val_losses = []

                    # Evaluate on validation set
                    # start = time.time()
                    with torch.no_grad():
                        for t in range(EVAL_ITERS):
                            x, y = val_dataset[val_idx[v_idx]]
                            x, y = x.to(local_device, non_blocking=True), y.to(local_device, non_blocking=True)
                            _, loss = model(x, y)
                            val_losses.append(loss.item())
                            v_idx += 1

                    # Calculate average validation loss and perplexity
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    perplexity = torch.exp(torch.tensor(avg_val_loss))

                    # Display information
                    # end = time.time()
                    # logging.info(end - start)
                    logging.info(f"Step: {step}, Train Loss: {loss.item()}, Val Loss: {avg_val_loss}, "
                                 f"Perplexity: {perplexity.item()}, Grad Norm: {grad_norm.item()}, lr: {scheduler.get_last_lr()[0]}")

                    # Generate text
                    # start = time.time()
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
                    # end = time.time()
                    # logging.info(end-start)
                    logging.info(enc.decode(out[0].tolist()))

                # Save checkpoint
                if step % SAVE_EVERY_STEPS == 0:
                    torch.save({
                        'step': step,
                        'model_state_dict': model.module.state_dict() if ddp else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, 'checkpoints/last_ckpt.pth')

    # Handle exceptions
    except Exception as e:
        logging.error(f"GPU {GPU_ID}: Error occurred: {str(e)}")
        raise
    # Ensure all processes exit
    finally:
        # Destroy DDP process group
        if dist.is_initialized():
            dist.destroy_process_group()
        # Cleanup CUDA memory
        if CUDA_ENABLED:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set this to False for single GPU training, True for DDP
    ddp = False if torch.cuda.device_count() == 1 else True # or False

    # Create directories
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load datasets
    train_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="train")
    val_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="test")

    start = time.time()
    # Main function
    if ddp:
        GPU_IDS = [0, 1, 2, 3]  # Set your GPU IDs here
        world_size = len(GPU_IDS)
        master_device = GPU_IDS[0]

        # Check if the requested GPUs are available
        assert world_size <= torch.cuda.device_count(), f"Requested {world_size} GPUs, but only {torch.cuda.device_count()} available"
        assert world_size != 0, f"No GPUs available"

        # DDP initialization
        print(f"Starting DDP with {world_size} GPUs: {GPU_IDS}")
        mp.set_start_method('spawn', force=True)
        mp.spawn(
            main,
            args=(world_size, GPU_IDS, ddp, train_dataset, val_dataset, master_device),
            nprocs=world_size,
            join=True
        )
    else:
        print("Starting single GPU training")
        main(0, 1, [0], ddp=False, train_dataset=train_dataset, val_dataset=val_dataset, master_device=0)
    end = time.time()

    logging.info(f"Total time: {end - start} seconds")