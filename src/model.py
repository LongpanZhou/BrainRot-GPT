import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024          # The maximum context length
    vocab_size: int = 100258        # The size of the vocabulary (same as the tokenizer)
    n_layers: int = 12              # The number of transformer blocks
    n_embd: int = 768               # The size of the embedding dimension
    n_head: int = 12                # The number of attention heads
    dropout: float = 0.0            # The dropout probability
    bias: bool = True               # Whether to use bias

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"

        # Variables
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.bias = config.bias
        self.dropout = config.dropout

        # Linear Projections
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and torch.cuda.is_available()

        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # Input shape: (B, T, C)
        B, T, C = x.size()

        # Set Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #( B, n_head, T, C // n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #( B, n_head, T, C // n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #( B, n_head, T, C // n_head)

        # Flash attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        # Regular attention
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # Layers
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Feedforward pass
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Layers
        self.attn_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CasualSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # Residual connection
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Initialize the model
        print("Creating Model...")
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),             # Token embedding
                wpe = nn.Embedding(self.config.block_size, self.config.n_embd),             # Positional encoding
                drop = nn.Dropout(self.config.dropout),                                     # Dropout
                h = nn.ModuleList([Block(config) for _ in range(self.config.n_layers)]),    # Transformer blocks
                ln_f = nn.LayerNorm(self.config.n_embd, bias=self.config.bias)              # Final normalization
            )
        )

        # I don't get how their weights is the same between nn.Embedding and nn.Linear, but the abstract idea.
        # You use the same projection input -> embedding and embedding -> output.
        # Meaning same embedding weights for both to keep it consistent. f(x) & f^-1(x)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=self.config.bias)
        self.transformer.wte.weight = self.lm_head.weight
        self.c_proj = nn.Linear(4 * self.config.n_embd, self.config.n_embd, bias=self.config.bias)

    def forward(self, idx, targets=None):
        # Batch size and sequence length
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # Token Embedding and positional encoding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx) # (B, T, C)
        pos_emb = self.transformer.wpe(pos) # (1, T, C)

        # Add token and positional embeddings with dropouts
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final normalization
        x = self.transformer.ln_f(x)

        # Logits and Loss
        if targets is not None:
            # Training mode
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode (e.g., during autoregressive generation)
            logits = self.lm_head(x[:, [-1], :])  # Only compute logits for last token
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, end_token=None):
        # Batch size and sequence length
        B, T = idx.size()

        # Preallocate output tensor -> Optimization
        output = torch.zeros(B, T+max_new_tokens, dtype=idx.dtype, device=idx.device)
        output[:, :T] = idx

        # Generation loop
        for _ in range(max_new_tokens):
            # Get the last T tokens, feed them to the model, and get the logits
            idx_cond = output[:, max(0, T - self.config.block_size):T]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to the output
            output[:, T] = idx_next[:, 0]
            T+=1

            # Check for end token
            if end_token == idx_next:
                break

        # Update the output tensor up to last token
        return output[:, :T]