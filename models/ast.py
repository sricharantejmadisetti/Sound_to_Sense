import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionTransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer that returns attention weights for explainability.
    """
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, return_attn=False):
        # Self attention
        attn_out, attn_weights = self.self_attn(x, x, x, need_weights=True)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feed forward
        ff_out = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        if return_attn:
            return x, attn_weights
        return x

class BabyCryAST(nn.Module):
    """
    AST (Audio Spectrogram Transformer) Model.
    Processes Mel-spectrogram of shape (batch, 1, 128, 256).
    Uses 6 Transformer Encoder layers, with custom support for attention extraction.
    """
    def __init__(self, num_classes=5, embed_dim=256, nhead=8, num_layers=6, dim_feedforward=1024):
        super(BabyCryAST, self).__init__()
        
        # Patch extraction parameters
        self.patch_size = (16, 16)
        self.stride = (10, 10)
        self.embed_dim = embed_dim
        
        # Project patches: Input (128, 256) -> Outputs (grid_h, grid_w)
        # grid_h = (128 - 16) // 10 + 1 = 12
        # grid_w = (256 - 16) // 10 + 1 = 25
        self.grid_h = (128 - self.patch_size[0]) // self.stride[0] + 1
        self.grid_w = (256 - self.patch_size[1]) // self.stride[1] + 1
        self.num_patches = self.grid_h * self.grid_w
        
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=self.patch_size,
            stride=self.stride
        )
        
        # CLS Token and Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            AttentionTransformerEncoderLayer(embed_dim, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize parameters
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x, return_attn=False):
        # Input shape: (batch, 1, 128, 256)
        # 1. Patch projection
        patches = self.patch_embed(x) # (batch, embed_dim, grid_h, grid_w)
        patches = patches.flatten(2).transpose(1, 2) # (batch, num_patches, embed_dim)
        
        # 2. Prepend CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch, 1, embed_dim)
        x = torch.cat((cls_tokens, patches), dim=1) # (batch, num_patches + 1, embed_dim)
        
        # 3. Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Pass through Transformer encoder layers
        # Store final layer attention weights if requested
        final_attn = None
        for i, layer in enumerate(self.layers):
            if return_attn and i == len(self.layers) - 1:
                x, final_attn = layer(x, return_attn=True)
            else:
                x = layer(x)
                
        # 5. Classification head on CLS token
        cls_out = x[:, 0] # (batch, embed_dim)
        logits = self.fc(cls_out)
        
        if return_attn:
            return logits, final_attn
        return logits
