import torch
import tiktoken

from model import GPT, GPTConfig

# Device configuration
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# Optimizations
if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

# Hyperparameters
NUM_TIMES = 10
SEQUENCE_LENGTH = 512

# Model
model = GPT(GPTConfig()).to(device)
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

state_dict = torch.load('./checkpoints/last_ckpt.pth', map_location='cuda')
model.load_state_dict(state_dict['model_state_dict'])

# Encoder
enc = tiktoken.get_encoding("cl100k_base")
text = "Hello, how are you doing today? I hope you're having a great day!"
tokens = enc.encode(text)

# Generation
model.eval()
with torch.no_grad():
    # Measure inference time
    for _ in range(NUM_TIMES):
        out = model.generate(
            torch.tensor([tokens]).to(device),
            max_new_tokens=SEQUENCE_LENGTH,
            temperature=1.0,
            end_token=enc.eot_token,
        )
        print(enc.decode(out[0].tolist()))