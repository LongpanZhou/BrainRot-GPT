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
    GQA: bool = True                # Whether to use GQA (Grouped Query Attention)
    GQA_factor = 4                  # The factor for GQA
    ROPE: bool = True               # Whether to use ROPE (Rotary Positional Embedding)


# Code stolen from https://github.com/pytorch/torchtune/blob/main/torchtune/modules/position_embeddings.py
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verification)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
            self,
            dim: int,
            max_seq_len: int = 4096,
            base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
            self, x: torch.Tensor, *, input_pos: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
                ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

#Code stolen from https://github.com/pytorch/torchtune/blob/main/torchtune/modules/kv_cache.py
class KVCache(nn.Module):
    """
    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_heads (int): number of heads. We take num_heads instead of num_kv_heads because
            the cache is created after we've expanded the key and value tensors to have the
            same shape as the query tensor. See attention.py for more details
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
            self,
            batch_size: int,
            max_seq_len: int,
            num_heads: int,
            head_dim: int,
            dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, num_heads, max_seq_len, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.batch_size = batch_size

    def reset(self) -> None:
        """Reset the cache to zero."""
        self.k_cache.zero_()
        self.v_cache.zero_()


    def update(self, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor):
        """Update KV cache with the new k_val, v_val and return the updated cache.

        Args:
            input_pos (Tensor): Current position tensor with shape [S]
            k_val (Tensor): Current key tensor with shape [B, H, S, D]
            v_val (Tensor): Current value tensor with shape [B, H, S, D]

        Raises:
            ValueError: if ``input_pos`` is longer than the maximum sequence length

        Returns:
            Tuple[Tensor, Tensor]: Updated KV cache with key first
        """
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Variables
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.h_dim = config.n_embd // config.n_head
        self.bias = config.bias
        self.dropout = config.dropout
        self.GQA = config.GQA
        self.GQA_factor = config.GQA_factor
        self.ROPE = config.ROPE

        # Linear Projections
        if self.GQA:
            assert self.n_embd % self.GQA_factor == 0, "Embedding dimension must be divisible by GQA factor"
            KV_dim = self.n_embd // self.GQA_factor
            self.c_attn_q = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)
            self.c_attn_k = nn.Linear(self.n_embd, KV_dim, bias=self.bias)
            self.c_attn_v = nn.Linear(self.n_embd, KV_dim, bias=self.bias)
        else:
            self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=self.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=self.bias)

        # Rotary Positional Embeddings
        if self.ROPE: self.rope = RotaryPositionalEmbeddings(dim=self.h_dim,max_seq_len=config.block_size,base=10000)

        # Regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and torch.cuda.is_available()

        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, use_cache=False, cache=None):
        # Input shape: (B, T, C)
        B, T, C = x.size()

        # Set Q, K, V
        if self.GQA:
            q = self.c_attn_q(x).view(B, T, self.n_head, self.h_dim).transpose(1, 2) #( B, n_head, T, h_dim)
            k = self.c_attn_k(x).view(B, T, self.n_head // self.GQA_factor, self.h_dim).transpose(1, 2) #( B, n_head//GQA_factor, T, h_dim)
            v = self.c_attn_v(x).view(B, T, self.n_head // self.GQA_factor, self.h_dim).transpose(1, 2) #( B, n_head//GQA_factor, T, h_dim)
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.h_dim).transpose(1, 2) #( B, n_head, T, h_dim)
            k = k.view(B, T, self.n_head, self.h_dim).transpose(1, 2) #( B, n_head, T, h_dim)
            v = v.view(B, T, self.n_head, self.h_dim ).transpose(1, 2) #( B, n_head, T, h_dim)

        # ROPE
        if self.ROPE:
            q = self.rope(q)
            k = self.rope(k)

        # Flash attention
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True, enable_gqa=self.GQA)
        # Regular attention
        else:
            if self.GQA:
                k = k.repeat_interleave(q.size(-3)//k.size(-3), -3)
                v = v.repeat_interleave(q.size(-3)//v.size(-3), -3)

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
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Feedforward pass
        return self.dropout(self.down_c_proj(self.silu(self.up_c_proj(x)) * self.gate_c_proj(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Layers
        self.attn_norm = nn.RMSNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.mlp_norm = nn.RMSNorm(config.n_embd)
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
        assert self.config.n_embd % self.config.n_head == 0, "Embedding dimension must be divisible by number of heads"

        # Initialize the model
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),             # Token embedding
                wpe = nn.Embedding(self.config.block_size, self.config.n_embd),             # Positional encoding
                drop = nn.Dropout(self.config.dropout),                                     # Dropout
                h = nn.ModuleList([Block(config) for _ in range(self.config.n_layers)]),    # Transformer blocks
                ln_f = nn.RMSNorm(self.config.n_embd)                                       # Final normalization
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
        return sum(p.numel() for p in self.parameters()) - self.transformer.wpe.weight.numel()

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        return torch.optim.RAdam(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

    def forward(self, x, targets=None, use_cache=False):
        # Batch size and sequence length
        B, T = x.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        # Token Embedding and positional encoding
        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        tok_emb = self.transformer.wte(x) # (B, T, C)
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

    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=1.0, end_token=None):
        # Batch size and sequence length
        B, T = x.size()

        # Preallocate output tensor -> Optimization
        output = torch.zeros(B, T+max_new_tokens, dtype=x.dtype, device=x.device)
        output[:, :T] = x

        # Generation loop
        for _ in range(max_new_tokens):
            # Get the last T tokens, feed them to the model, and get the logits
            logits, _ = self(output[:, max(0, T - self.config.block_size):T], targets=None, use_cache=True)
            logits = logits[:, -1, :] / temperature

            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to the output
            output[:, T] = x_next[:, 0]
            T+=1

            # Check for end token
            if end_token == x_next:
                break

        # Update the output tensor up to last token
        return output[:, :T]