# This script is used to test the performance of the model on the optimizations
import numpy as np
import torch
import random

from dataloader import FineWeb
from model import GPT, GPTConfig

# Best Configuration for performance

# Hyperparameters
BATCH_SIZE = 64                             # Batch size
SEQUENCE_LENGTH = 512                       # Sequence length
STEPS = 20000                               # Total training steps
LEARNING_RATE = 1e-3                        # Learning rate

# Model
model = GPT(GPTConfig(vocab_size=100288, bias=False)).to(device='cuda')
model = torch.compile(model)
optimizer = model.configure_optimizers(1e-8,1e-3)

# Load datasets
train_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="train")
val_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="test")

train_idx = np.random.randint(0, len(train_dataset), size=50)

# Accurate timing
total_time = 0.0
for i in range(20):
    idx = random.randint(0, len(train_dataset) - 1)
    x, y = train_dataset[idx]
    x, y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)  # ms
    if i != 0 and i != 1:
        total_time += elapsed_time
    print(f"Forward + backward pass time: {elapsed_time:.3f} ms")

print(f"Average time (excluding first run): {total_time / 18:.3f} ms")