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
from sms.sms import sms_client
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.parallel import DistributedDataParallel as DDP

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
BATCH_SIZE = 48                             # Batch size
SEQUENCE_LENGTH = 1024                      # Sequence length
STEPS = 30000                               # Total training steps
LEARNING_RATE = 1.2e-3                        # Learning rate
EVAL_EVERY_STEPS = 100                      # Evaluate every N steps
EVAL_ITERS = 25                             # Number of evaluation iterations
SAVE_EVERY_STEPS = STEPS//10                # Save every N steps
SAVE_DIR = "./checkpoints"                  # Directory to save checkpoints
SAVE = False                                # Save Model Toggle
LOG_DIR = "./logs"                          # Directory to save logs
TIME_TO_FETCH = 5                           # Time to fetch commands (in mins)
CUDA_ENABLED = (device == "cuda")           # Use CUDA

# Tokenizer
enc = tiktoken.get_encoding("cl100k_base")              # same as dataloader.py

# Setup logging
def setup_logging(ddp, model, GPU_IDs):
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
    logging.info(f"{__file__}")
    logging.info(f"===== General Information  =====")
    logging.info(f"Distributed Data Parallel (DDP) mode: {ddp}")
    logging.info(f"Using device: {device}")
    logging.info(f"Total tokens: {min(ddp * len(GPU_IDs),1) * BATCH_SIZE * STEPS * SEQUENCE_LENGTH / 1e6}M")
    logging.info(f"Parameters: {model.get_num_params() / 1e6}M")
    logging.info("===== Training Configuration =====")
    logging.info(f"BATCH_SIZE         = {BATCH_SIZE}")
    logging.info(f"SEQUENCE_LENGTH    = {SEQUENCE_LENGTH}")
    logging.info(f"STEPS              = {STEPS}")
    logging.info(f"LEARNING_RATE      = {LEARNING_RATE}")
    logging.info(f"EVAL_EVERY_STEPS   = {EVAL_EVERY_STEPS}")
    logging.info(f"EVAL_ITERS         = {EVAL_ITERS}")
    logging.info(f"SAVE_EVERY_STEPS   = {SAVE_EVERY_STEPS}")
    logging.info(f"SAVE               = {SAVE}")
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
def main(rank, world_size, GPU_IDs, ddp, train_dataset, val_dataset):
    try:
        # Set current GPU_ID
        GPU_ID = GPU_IDs[rank]

        # Set device
        local_device = torch.device(f"cuda:{GPU_ID}")
        is_master = (world_size == 1) or (rank == 0)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Model
        model = GPT(GPTConfig(
            # block_size = SEQUENCE_LENGTH,     # sequence length
            vocab_size = 100588,    # using cl100k_base tokenizer (scaled to 100588)
            # n_layers = 24,        # depth
            # n_embd = 1024,        # embedding size
            # n_head = 16,          # attention heads
            dropout = 0.1,          # dropout rate
            # bias = True,          # bias
            # GQA_factor = 4,       # GQA factor (Set to 1 for no GQA)
        )).to(device=local_device, dtype=dtype)

        # Setup logging
        if is_master: setup_logging(ddp,model,GPU_IDs)

        # Compile model if CUDA is enabled
        if CUDA_ENABLED: model = torch.compile(model)

        # Setup DDP
        if ddp:
            setup_ddp(rank, world_size, GPU_ID)
            model = DDP(model, device_ids=[GPU_ID], find_unused_parameters=True)
            dist.barrier()

        # Optimizer and Scheduler
        raw_model = model.module if ddp else model
        optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=LEARNING_RATE)
        scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, total_steps=STEPS, anneal_strategy='cos', pct_start=0.04)

        # Precompute random indices
        train_idx = np.random.randint(0, len(train_dataset), size=STEPS)
        val_idx = np.random.randint(0, len(val_dataset), size=STEPS*EVAL_ITERS) if is_master else None
        v_idx = 0

        last_checked = start = time.time()
        # Training Loop
        for step in range(STEPS):
            # Get random batch (each process gets different batch)
            x, y = train_dataset[train_idx[step]]
            x, y = x.to(local_device, non_blocking=True), y.to(local_device, non_blocking=True)

            # Forward pass
            _, loss = model(x, y)

            # Backward pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # ONLY MASTER
            if is_master:
                # Evaluate
                if step % EVAL_EVERY_STEPS == 0:
                    # Evaluation Mode
                    model.eval()
                    val_loss = 0

                    # Evaluate on validation set
                    # start = time.time()
                    with torch.no_grad():
                        for t in range(EVAL_ITERS):
                            x, y = val_dataset[val_idx[v_idx]]
                            x, y = x.to(local_device, non_blocking=True), y.to(local_device, non_blocking=True)
                            _, loss_ = model(x, y)
                            val_loss += loss_.item()
                            v_idx += 1

                    # Calculate average validation loss and perplexity
                    avg_val_loss = val_loss / EVAL_ITERS
                    perplexity = torch.exp(torch.tensor(avg_val_loss))

                    # Display information
                    # end = time.time()
                    # logging.info(end - start)
                    logging.info(f"Step: {step}, Val Loss: {avg_val_loss}, Perplexity: {perplexity.item()}, Grad Norm: {grad_norm.item()}, lr: {scheduler.get_last_lr()[0]}")

                    # Generate text
                    # start = time.time()
                    with torch.no_grad():
                        text = "Hello, how are you doing today? I hope you're having a great day!"
                        tokens = enc.encode(text)
                        out = raw_model.generate(
                            torch.tensor([tokens]).to(local_device),
                            max_new_tokens=SEQUENCE_LENGTH-len(tokens)-1,
                            temperature=1.0,
                            end_token=enc.eot_token
                        )
                    model.train()
                    # end = time.time()
                    # logging.info(end-start)
                    try: logging.info(enc.decode(out[0].tolist()))
                    except Exception as e: logging.error(f"Void token: {e}")

                    # Fetch commands
                    if time.time() - last_checked > TIME_TO_FETCH * 60:
                        if sms_client.fetch_message_received("Status"):
                            sms_client.send_message(f"Time:{time.time()-start}s Step: {step}, Val Loss: {avg_val_loss}, Perplexity: {perplexity.item()}")
                        last_checked = time.time()

                # Save checkpoint
                if SAVE and step % SAVE_EVERY_STEPS == 0 and step != 0:
                    torch.save({
                        'step': step,
                        'model_state_dict': raw_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, 'checkpoints/last_ckpt.pth')

        # Save final checkpoint
        if is_master:
            if SAVE:
                torch.save({
                    'step': STEPS,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, 'checkpoints/final_ckpt.pth')

    # Handle exceptions
    except Exception as e:
        logging.error(f"GPU {GPU_IDs[rank]}: Error occurred: {str(e)}")
        raise
    # Ensure all processes exit
    finally:
        # Clear CUDA memory
        if CUDA_ENABLED: torch.cuda.empty_cache()
        # Destroy DDP process group
        if rank==0 and dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    # Set this to False for single GPU training, True for DDP
    ddp = torch.cuda.device_count() > 1 and False # True or False

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

        # Check if the requested GPUs are available
        assert world_size <= torch.cuda.device_count(), f"Requested {world_size} GPUs, but only {torch.cuda.device_count()} available"
        assert world_size > 1, f"DDP requires more than one GPU, but only {world_size} GPUs were requested"

        # DDP initialization
        print(f"Starting DDP with {world_size} GPUs: {GPU_IDS}")
        mp.set_start_method('spawn', force=True)
        mp.spawn(
            main,
            args=(world_size, GPU_IDS, ddp, train_dataset, val_dataset),
            nprocs=world_size,
            join=True
        )
    else:
        print("Starting single GPU training")
        main(0, 1, [2], ddp=False, train_dataset=train_dataset, val_dataset=val_dataset)
    end = time.time()
    logging.info(f"Total time: {end - start} seconds")
    sms_client.send_message(f"Training completed in {end-start} seconds.")