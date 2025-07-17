import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Defines a Transformer block that forms the building block of transformer-based
    architectures. This includes multi-head self-attention, feed-forward layers,
    layer normalization, and dropout for residual connections.
    """
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super().__init__()
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        # Multi-head self-attention (batch_first allows input as (batch, seq, embed))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        # Layer normalization
        self.ln2 = nn.LayerNorm(embed_dim)
        # Two-layer feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # expand (e.g., 4x hidden size)
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)   # project back to hidden size
        )
        # Dropout layer for residual connections
        self.dropout = nn.Dropout(dropout_rate)
        # Causal mask will be provided from outside (for efficiency, handled in GPTModel)

    def forward(self, x, attn_mask=None):
        # x: (batch, seq_len, embed_dim) - 3-dimentional input data. One training example is (seq_len, embed_dim)
        # LayerNorm + Self-Attention
        y = self.ln1(x)
        attn_output, _ = self.attn(y, y, y, attn_mask=attn_mask)  # self-attention
        x = x + self.dropout(attn_output)       # Residual connection
        # LayerNorm + Feed-Forward
        y2 = self.ln2(x)
        ff_output = self.ff(y2)
        x = x + self.dropout(ff_output)         # Residual connection
        return x

class GPTModel(nn.Module):
    """
    Defines a Generative Pre-trained Transformer (GPT) model architecture.
    This class implements a GPT-style model, which includes token embeddings,
    learned positional encodings, a stack of Transformer blocks, and a final
    projection layer to produce logits for prediction. The model also applies
    causal masking, which prevents tokens from attending to future positions in
    the sequence.
    """
    def __init__(self, vocab_size, max_len, embed_dim, num_layers, num_heads, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        # Token embeddings and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))  # learned positional encodings
        self.drop = nn.Dropout(dropout_rate)
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout_rate) for _ in range(num_layers)
        ])
        # Final layer normalization (GPT-2 style uses LN here)
        self.ln_final = nn.LayerNorm(embed_dim)
        # Output projection to vocabulary size
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Register a causal mask buffer (lower triangular matrix) for use in attention
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()  # True where future tokens should be masked
        self.register_buffer("attn_mask", mask)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_len, "Sequence length exceeds model capacity"
        # Get token embeddings and add positional embeddings
        tok_embed = self.token_emb(input_ids)                              # shape: (batch, seq_len, embed_dim)
        pos_embed = self.pos_emb[:, :seq_len, :]                           # shape: (1, seq_len, embed_dim)
        x = self.drop(tok_embed + pos_embed)                               # add token & position, then dropout
        # Apply each Transformer block
        for block in self.blocks:
            x = block(x, attn_mask=self.attn_mask[:seq_len, :seq_len])
        x = self.ln_final(x)                                               # final layer norm
        logits = self.head(x)                                              # logits for each token position
        return logits
