import pytest
import torch

from lightning_attn.ops import lightning_attn2_no_decay, linear_attn


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


@pytest.mark.parametrize("b, h, n, d, e", get_params())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_lightning2(b, h, n, d, e, dtype):
    torch.manual_seed(2024)
    device = torch.device("cuda")
    q = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
    k = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
    v = (torch.randn((b, h, n, e), dtype=dtype, device=device) / 10).requires_grad_()
    do = torch.randn((b, h, n, e), dtype=dtype, device=device) / 10
    s = None

    # forward
    o_ref = linear_attn(q, k, v, s)
    o = lightning_attn2_no_decay(q, k, v)

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
