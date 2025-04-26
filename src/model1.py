import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 100277 #I used cl100k here
    n_layers: int = 12
    n_embd: int = 768
    n_head: int = 12
    dropout: float = 0.0
    bias: bool = True

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_groups = 4  # Number of groups for GQA
        self.head_dim = self.n_embd // self.n_head  # 64
        self.bias = config.bias
        self.dropout = config.dropout
        self.c_attn_q = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)  # Query: [B, T, n_embd]
        self.c_attn_kv = nn.Linear(self.n_embd, 2 * self.n_groups * self.head_dim, bias=self.bias)  # K,V: [B, T, 2 * n_groups * head_dim]
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_attn_q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # [B, n_head, T, head_dim]
        k, v = self.c_attn_kv(x).split(self.n_groups * self.head_dim, dim=2)  # Each: [B, T, n_groups * head_dim]
        k = k.view(B, T, self.n_groups, self.head_dim).transpose(1, 2)  # [B, n_groups, T, head_dim]
        v = v.view(B, T, self.n_groups, self.head_dim).transpose(1, 2)  # [B, n_groups, T, head_dim]
        k = k.repeat(1, self.n_head // self.n_groups, 1, 1)  # [B, n_head, T, head_dim]
        v = v.repeat(1, self.n_head // self.n_groups, 1, 1)  # [B, n_head, T, head_dim]
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.silu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp_norm = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CasualSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        attn_out = self.attn(self.attn_norm(x))
        mlp_out = self.mlp(self.mlp_norm(x))
        return x + attn_out + mlp_out


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        print("Creating Model...")
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
                ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx) # (B, T, C)
        pos_emb = self.transformer.wpe(pos) # (1, T, C)

        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss