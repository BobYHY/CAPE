import torch
import torch.nn.functional as F
from torch import nn, einsum
from entmax import Sparsemax, Entmax15
from einops import rearrange


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

        # self.selector = nn.Softmax(dim=-1)
        # self.selector = Entmax15(dim=-1)
        self.selector = Sparsemax(dim=-1)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.selector(sim)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


# transformer


class PrompterTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 embedding_dim,
                 output_dim,
                 depth,
                 heads,
                 dim_head,
                 attn_dropout,
                 ff_dropout):
        super().__init__()
        self.embeds = nn.Embedding(input_dim, embedding_dim)
        self.layers = nn.ModuleList([])

        # transformer layers
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(
                    PreNorm(embedding_dim, Attention(embedding_dim, heads=heads, dim_head=dim_head,
                                                     dropout=attn_dropout))),
                Residual(PreNorm(embedding_dim, FeedForward(embedding_dim, dropout=ff_dropout))),
            ]))

        self.output_layer = nn.Sequential(nn.Flatten(-2, -1),
                                          nn.Dropout(p=0.5),
                                          nn.Linear(48 * embedding_dim, output_dim))

    def forward(self, x):
        x = self.embeds(x) # (B, token_dims) -> (B, token_dims, embedding_dim)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return self.output_layer(x)

