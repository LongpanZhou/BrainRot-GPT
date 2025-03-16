import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_embed, num_heads=n_head, dropout=0.1, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.1)
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, attn_mask=None):
        # Pass attn_mask to MultiheadAttention
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.ffn(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed=512, n_head=8, n_layers=12, block_size=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        # Use ModuleList instead of Sequential to allow custom forward logic
        self.blocks = nn.ModuleList([TransformerBlock(n_embed, n_head) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        # Create causal mask (True for positions to attend to, False for masked positions)
        causal_mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        attn_mask = ~causal_mask  # Invert for PyTorch: True means attend, False means mask

        # Pass through each TransformerBlock with the attention mask
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='mean')
        return logits, loss

    def generate(self, idx, max_new_tokens=50, eos_token_id=None, temperature=1.0, top_k=50):
        self.eval()
        input_ids = idx

        with torch.no_grad():
            for i in range(max_new_tokens):
                input_ids_cond = input_ids[:, -self.block_size:]
                logits, _ = self.forward(input_ids_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                    probs = F.softmax(top_k_logits, dim=-1)
                    next_token_id = top_k_indices.gather(-1, torch.multinomial(probs, num_samples=1))
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token_id], dim=1)
                if eos_token_id is not None and next_token_id.item() == eos_token_id:
                    break
            return input_ids