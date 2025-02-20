# Tnl: https://arxiv.org/pdf/2405.17381

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor


class TnlAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        norm_type: str = "layernorm",
        layer_idx: int = 0,
        num_layers: int = 12,
        causal: bool = True,
    ):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.output_gate = nn.Sequential(
            nn.Linear(embed_dim, self.head_dim, bias=bias),
            nn.Linear(self.head_dim, embed_dim, bias=bias),
        )
        self.layer_idx = layer_idx
        slope_rate = _build_slope_tensor(self.num_heads)
        slope_rate = slope_rate * (1 - layer_idx / (num_layers - 1) + 1e-5)
        self.register_buffer("slope_rate", slope_rate, persistent=False)

    def forward(
        self,
        x,
        **kwargs,
    ):
        # x: b n d
        # linear map
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # act
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        q, k, v = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", d=self.head_dim),
            [q, k, v],
        )

        output = lightning_attn_func(q, k, v, self.slope_rate)

        # reshape
        output = rearrange(output, "... h n d -> ... n (h d)")

        output = self.norm(output)

        output_gate = F.sigmoid(self.output_gate(x))
        output = output * output_gate

        # outproj
        output = self.o_proj(output)

        return output


if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.bfloat16
    tnl_attn = TnlAttention(embed_dim=1024, num_heads=8).to(device).to(dtype)
    x = torch.randn(2, 1024, 1024, device=device, dtype=dtype).requires_grad_()
    output = tnl_attn(x)
    print(output.shape)
