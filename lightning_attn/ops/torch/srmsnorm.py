# Copyright (c) 2024 Doraemonzzz
# CREDITS: This comes almost as-is from the Triton layer norm tutorial
# https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py

import torch
import torch.nn as nn


class SimpleRMSNormTorch(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output
