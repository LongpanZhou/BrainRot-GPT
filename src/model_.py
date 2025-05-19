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
    GQA_factor = 1                  # Factor of GQA

# Code stolen from: https://docs.pytorch.org/torchtune/stable/_modules/torchtune/modules/kv_cache.html#KVCache
class KVCache(nn.Module):
    """
    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_kv_heads (int): number of key/value heads.
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
            self,
            batch_size: int,
            max_seq_len: int,
            num_kv_heads: int,
            head_dim: int,
            dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "cache_pos", torch.arange(0, cache_shape[2]), persistent=False
        )
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_pos -= self.size

    @property
    def size(self) -> int:
        return self.cache_pos[0].item()

    def update(self, k_val: torch.Tensor, v_val: torch.Tensor):
        """Update KV cache with the new ``k_val``, ``v_val`` and return the updated cache.

        Note:
            When updating the KV cache, it is assumed that subsequent updates should update key-value
            positions in consecutive sequence positions. If you wish to update cache values which have
            already been filled, use ``.reset()``, which will reset the cache to the zero-th position.

        Args:
            k_val (torch.Tensor): Current key tensor with shape [B, H, S, D]
            v_val (torch.Tensor): Current value tensor with shape [B, H, S, D]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated key and value cache tensors, respectively.

        Raises:
            ValueError: if the batch size of the new key (or value) tensor is greater than the batch size
                used during cache setup.

        Note:
            This function will raise an ``AssertionError`` if the sequence length of ``k_val``
                is longer than the maximum cache sequence length.

        """
        bsz, _, seq_len, _ = k_val.shape
        if bsz > self.k_cache.shape[0]:
            raise ValueError(
                f"The current cache has been setup with a batch size of {self.k_cache.shape[0]}"
                f", but found new key tensors with batch size {k_val.shape[0]}!"
            )

        assert (self.cache_pos[0] + seq_len) <= self.k_cache.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, self.cache_pos[:seq_len]] = k_val
        v_out[:, :, self.cache_pos[:seq_len]] = v_val

        self.cache_pos.add_(seq_len)

        return k_out, v_out

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Variables
        self.block_size = config.block_size
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.h_dim = config.n_embd // config.n_head
        self.kv_head = config.n_head // config.GQA_factor
        self.bias = config.bias
        self.dropout = config.dropout
        self.GQA_factor = config.GQA_factor

        # Key/Value cache
        self.KV_cache = KVCache(1, self.block_size, self.kv_head, self.h_dim, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

        # Linear Projections
        assert self.n_embd % self.GQA_factor == 0, "Embedding dimension must be divisible by GQA factor"

        self.c_attn_q = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
        self.c_attn_k = nn.Linear(self.n_embd, self.n_embd//self.GQA_factor, bias=self.bias)
        self.c_attn_v = nn.Linear(self.n_embd, self.n_embd//self.GQA_factor, bias=self.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(self.dropout, inplace=True)
        self.resid_dropout = nn.Dropout(self.dropout, inplace=True)

        # Flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and torch.cuda.is_available()

        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, cache_enable):
        # Input shape: (B, T, C)
        B, T, C = x.size()

        # Set Q, K, V
        q = self.c_attn_q(x).view(B, T, self.n_head, self.h_dim).transpose(1, 2)    #(B, n_head, T, h_dim)
        k = self.c_attn_k(x).view(B, T, self.kv_head, self.h_dim).transpose(1, 2)   #(B, n_head//GQA_factor, T, h_dim)
        v = self.c_attn_v(x).view(B, T, self.kv_head, self.h_dim).transpose(1, 2)   #(B, n_head//GQA_factor, T, h_dim)

        # Cache enable
        if cache_enable:
            k, v = self.KV_cache.update(k, v)

        # Repeat K and V for GQA
        if self.GQA_factor > 1:
            k = k.repeat_interleave(self.GQA_factor, dim=1)
            v = v.repeat_interleave(self.GQA_factor, dim=1)

        # Flash attention
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
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
        self.gate_c_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.down_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.up_c_proj = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(config.dropout, inplace=True)

    def forward(self, x):
        # Feedforward pass
        return self.dropout(self.down_c_proj(self.silu(self.up_c_proj(x)) * self.gate_c_proj(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Layers
        # self.attn_norm = nn.RMSNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        # self.mlp_norm = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.n_embd = config.n_embd

    def forward(self, x, cache_enable):
        # Residual connection
        x = x + self.attn(F.rms_norm(x, normalized_shape=[self.n_embd]), cache_enable=cache_enable)
        x = x + self.mlp(F.rms_norm(x, normalized_shape=[self.n_embd]))
        return x

    def clear_cache(self):
        self.attn.KV_cache.reset()

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert self.config.n_embd % self.config.n_head == 0, "Embedding dimension must be divisible by number of heads"

        # Initialize the model
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),                                 # Token embedding
                drop = nn.Dropout(self.config.dropout, inplace=True),                                           # Dropout
                h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layers)]),        # Transformer blocks
                # ln_f = nn.RMSNorm(self.config.n_embd)                                                         # Final normalization
            )
        )

        # I don't get how their weights is the same between nn.Embedding and nn.Linear, but the abstract idea.
        # You use the same projection input -> embedding and embedding -> output.
        # Meaning same embedding weights for both to keep it consistent. f(x) & f^-1(x)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters()) - self.transformer.wte.weight.numel()

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    def clear_cache(self):
        for block in self.transformer.h:
            block.clear_cache()

    def forward(self, x, targets=None, cache_enable=False):
        # Batch size and sequence length
        B, T = x.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # Token Embedding
        tok_emb = self.transformer.wte(x) # (B, T, C)

        # Add token and positional embeddings with dropouts
        x = self.transformer.drop(tok_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, cache_enable=cache_enable)

        # Final normalization
        x = F.rms_norm(x, normalized_shape=[self.config.n_embd])

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

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, end_token=None, cache_enable=False):
        # Batch size and sequence length
        B, T = x.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # Preallocate output tensor -> Optimization
        output = torch.zeros(B, T+max_new_tokens, dtype=x.dtype, device=x.device)
        output[:, :T] = x

        # Generation loop
        for i in range(T,T+max_new_tokens):
            # Get the last T tokens, feed them to the model, and get the logits
            if cache_enable:
                logits, _ = self(x, targets=None, cache_enable=cache_enable)

            logits2, _ = self(output[:, max(0, i - self.config.block_size):i])
            print(logits.shape,logits2.shape)

            for a, b in zip(logits, logits2):
                print(a, b)

            # Temperature scaling
            logits = logits[:, -1, :] / temperature

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to the output
            x = output[:, i] = x_next[:, 0]
            x = x.unsqueeze(0)

            # Check for end token
            if x_next >= end_token:
                break

        # Return the generated sequence
        self.clear_cache()
        return output[:, :i+1]