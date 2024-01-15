import torch

from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor

dtype = torch.bfloat16
device = torch.device("cuda")
b, h, n, d, e = 2, 12, 2048, 64, 64

q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
k = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
v = torch.randn((b, h, n, e), dtype=dtype, device=device).requires_grad_()
s = _build_slope_tensor(h).to(q.device).to(torch.float32)

o = lightning_attn_func(q, k, v, s)

print(o.shape)
