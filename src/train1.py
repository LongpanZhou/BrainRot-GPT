import os
import random
import torch
import tiktoken
import time

from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from dataloader import FineWeb
from model2 import GPT, GPTConfig

# Device Setup
device = "cuda:1"
print(f"Using device: {device}")

# Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

# Hyperparameters
BATCH_SIZE = 32
SEQUENCE_LENGTH = 512
STEPS = 200000
LEARNING_RATE = 6e-4 * 3
WARMUP_STEPS = 200  # Number of steps for warm-up
EVAL_EVERY_STEPS = 100
EVAL_ITERS = 10
SAVE_EVERY_STEPS = 10000
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataloaders
train_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="train")
val_dataset = FineWeb(B=BATCH_SIZE, T=SEQUENCE_LENGTH, split="test")

# Model
model = GPT(GPTConfig(dropout=0.1)).to(device)
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

#Test
enc = tiktoken.get_encoding("cl100k_base")
text = "Hello, how are you doing today? I hope you're having a great day!"
tokens = enc.encode(text)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / WARMUP_STEPS))
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=STEPS - WARMUP_STEPS, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_STEPS])
scaler = GradScaler()

print('--------------------------------Training--------------------------------')
# Training Loop
for step in range(STEPS):
    train_idx = random.randint(0, len(train_dataset) - 1)
    x, y = train_dataset[train_idx]
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    model.train()
    optimizer.zero_grad()

    with torch.autocast(device_type=device):
        _, loss = model(x, y)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    if step % EVAL_EVERY_STEPS == 0:
        model.eval()
        val_losses = []
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(EVAL_ITERS):
                val_idx = random.randint(0, len(val_dataset) - 1)
                x, y = val_dataset[val_idx]
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type=device):
                    _, loss = model(x, y)
                val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        perplexity = torch.exp(torch.tensor(avg_val_loss))
        end = time.perf_counter()
        print((end - start) * 1000)
        print(f"Step: {step}, Train Loss: {loss.item()}, Val Loss: {avg_val_loss}, Perplexity: {perplexity.item()}, Grad Norm: {grad_norm.item()}, LR: {optimizer.param_groups[0]['lr']}")

        with torch.no_grad():
            out = model.generate(
                torch.tensor([tokens]).to(device),
                max_new_tokens=SEQUENCE_LENGTH,
                temperature=0.8,
                top_k=10,
            )
            print(enc.decode(out[0].tolist()))