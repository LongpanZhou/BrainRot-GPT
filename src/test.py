import torch
import tiktoken

from model import GPT, GPTConfig

device = "cuda"

# Optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

# Hyperparameters
BATCH_SIZE = 32
SEQUENCE_LENGTH = 512

# Model
model = GPT(GPTConfig(dropout=0.1)).to(device)
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

state_dict = torch.load('./checkpoints/last_ckpt.pth', map_location='cuda')
model.load_state_dict(state_dict['model_state_dict'])

# Encoder
enc = tiktoken.get_encoding("cl100k_base")
text = "Hello, how are you doing today? I hope you're having a great day!"
tokens = enc.encode(text)

model.eval()

with torch.no_grad():
    # Measure inference time
    for _ in range(10):
        out = model.generate(
            torch.tensor([tokens]).to(device),
            max_new_tokens=SEQUENCE_LENGTH,
            temperature=0.8,
            top_k=10,
        )
        print(enc.decode(out[0].tolist()))