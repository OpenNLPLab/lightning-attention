import math

import pytest
import torch

from lightning_attn.ops import lightning_attn2, linear_attn


def get_params():
    array = [
        (6, 8, 256, 128, 64),
        (6, 8, 512, 128, 64),
        (6, 8, 1024, 128, 64),
        (6, 8, 2048, 128, 64),
        (6, 8, 4096, 128, 64),
        (6, 8, 8192, 128, 64),
        (6, 8, 2048, 32, 64),
        (6, 8, 2048, 64, 64),
        (6, 12, 2048, 128, 64),
        (6, 16, 2048, 128, 64),
        (6, 20, 2048, 128, 64),
        (1, 8, 2048, 128, 64),
        (2, 8, 2048, 128, 64),
        (3, 8, 2048, 128, 64),
        (6, 8, 913, 128, 64),
        (6, 8, 513, 128, 64),
        (6, 8, 1213, 128, 64),
        (6, 8, 2048, 16, 64),
    ]

    return array


def _build_slope_tensor(n_attention_heads: int):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    # h, 1, 1
    slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
        n_attention_heads, 1, 1
    )

    return slopes


@pytest.mark.parametrize("b, h, n, d, e", get_params())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_lightning2(b, h, n, d, e, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device) / 10).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device) / 10
    s = _build_slope_tensor(h).to(q.device).to(torch.float32)

    # forward
    o_ref = linear_attn(q, k, v, s)
    o = lightning_attn2(q, k, v, s)

    # backward
    o_ref.backward(do, retain_graph=True)
    dq_ref, q.grad = q.grad.clone(), None
    dk_ref, k.grad = k.grad.clone(), None
    dv_ref, v.grad = v.grad.clone(), None

    o.backward(do, retain_graph=True)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None

    print(torch.norm(o - o_ref))
    print(torch.norm(dq - dq_ref))
    print(torch.norm(dk - dk_ref))
    print(torch.norm(dv - dv_ref))
    assert False
