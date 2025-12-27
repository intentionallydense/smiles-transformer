import torch as t
import torch.nn as nn
import torch.nn.functional as F
import einops
import math


class Attention(nn.Module):
    """
    Multi-head attention with separate W_Q, W_K, W_V, W_O matrices.
    Stores attention weights for interpretability.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Parameter(t.empty(n_heads, d_model,  self.d_head ))
        self.W_K = nn.Parameter(t.empty(n_heads, d_model,  self.d_head )) 
        self.W_V = nn.Parameter(t.empty(n_heads, d_model,  self.d_head )) 
        self.W_O = nn.Parameter(t.empty(n_heads,  self.d_head , d_model))
        self.b_Q = nn.Parameter(t.zeros(n_heads,  self.d_head ))
        self.b_K = nn.Parameter(t.zeros(n_heads,  self.d_head ))
        self.b_V = nn.Parameter(t.zeros(n_heads,  self.d_head ))
        self.b_O = nn.Parameter(t.zeros(d_model))

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

        self.last_attn_weights = None
    
    def forward(self, x: t.Tensor, key_attention_mask: t.Tensor = None):
        """
        x: (batch, seq, d_model)
        key_attention_mask: (batch, seq). 1 for real, 0 for padding
        out: (batch, seq, d_model)
        """
        q = einops.einsum(
            x, self.W_Q,
            "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head"
        ) + self.b_Q

        k = einops.einsum(
            x, self.W_K,
            "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head"
        ) + self.b_K

        v = einops.einsum(
            x, self.W_V,
            "batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head"
        ) + self.b_V

        attn_scores = einops.einsum(
            q,k,
            "batch seqQ n_heads d_head, batch seqK n_heads d_head -> batch n_heads seqQ seqK"
        )

        attn_scores_scaled = attn_scores / self.d_head**0.5
        mask = key_attention_mask.unsqueeze(1).unsqueeze(2)

        masked_attn_scores = attn_scores_scaled.masked_fill(mask == 0, float('-inf'))
        pattern = masked_attn_scores.softmax(-1)
        self.last_attn_weights = pattern.detach()

        z = einops.einsum(
            v, pattern,
            "batch seqK n_heads d_head, batch n_heads seqQ seqK -> batch seqQ n_heads d_head"
        )

        result = einops.einsum(
            z, self.W_O,
            "batch seqQ n_heads d_head, n_heads d_head d_model -> batch seqQ d_model"
        ) + self.b_O

        return result

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()
        self.W_in = nn.Parameter(t.empty(d_model, d_mlp)) 
        self.W_out = nn.Parameter(t.empty(d_mlp, d_model)) 

        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.gelu = nn.GELU()

        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)
    
    def forward(self, x):
        """
        x: (batch, seq, d_model)
        out: (batch, seq, d_model)
        """
        hidden = F.gelu(x @ self.W_in + self.b_in)
        return hidden @ self.W_out + self.b_out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.mlp = MLP(d_model, d_mlp)

    def forward(self, x, key_attention_mask):
        """
         x: (batch, seq, d_model)
         key_attention_mask: (batch, seq)
         out: (batch, seq, d_model)
         """
        resid_mid = self.attn(self.ln1(x), key_attention_mask) + x
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_mlp, n_layers, max_len):
        super().__init__()
        self.W_E = nn.Parameter(t.empty((vocab_size, d_model)))
        nn.init.normal_(self.W_E)

        self.W_pos = nn.Parameter(t.empty((max_len, d_model)))
        nn.init.normal_(self.W_pos)

        self.transformerblocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_mlp) for i in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.reg_head = nn.Linear(d_model, 1)
    
    def forward(self, x, key_attention_mask):
        embed = self.W_E[x]
        batch, seq_len = x.shape
        pos_embed = einops.repeat(
            self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch
        )
        residual = embed + pos_embed
        for block in self.transformerblocks:
            residual = block(residual,key_attention_mask)

        residual_after_ln = self.ln_final(residual)
        mask = key_attention_mask.unsqueeze(-1)
        masked_residual = mask*residual_after_ln
        summed = masked_residual.sum(dim=1)
        lengths = key_attention_mask.sum(dim=1, keepdim=True)
        pooled = summed / lengths        
        return self.reg_head(pooled)
