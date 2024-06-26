import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torch.nn as nn
from utils import *
from dimensions import *

class ConformerConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0
    ):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange(CONFORMER_FREQUENCY_REARRANGEMENT),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange(CONFORMER_TIME_REARRANGEMENT),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=CONFORMER_DIMENSION_HEADS,
        heads=CONFORMER_HEADS,
        ff_mult=CONFORMER_FEED_FORWARD_MULTIPLIER,
        conv_expansion_factor=CONFORMER_CONV_EXPANSION_FACTOR,
        conv_kernel_size=CONFORMER_CONV_KERNEL_SIZE,
        attn_dropout=CONFORMER_ATTENTION_DROPOUT,
        ff_dropout=CONFORMER_FEED_FORWARD_DROPOUT,
        conv_dropout=CONFORMER_CONV_DROPOUT
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=CONFORMER_CON_CAUSAL_ENABLED_FLAG,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x
