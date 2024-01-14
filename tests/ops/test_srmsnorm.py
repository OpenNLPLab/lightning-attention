import pytest
import torch
from torch.testing import assert_close

from lightning_attn.ops import SimpleRMSNorm, SimpleRMSNormTorch


def get_params():
    array = []
    for b in [1, 2, 4, 8]:
        for n in [127, 128, 256, 257, 1024, 1025]:
            for d in [768, 1024]:
                array.append((b, n, d))

    return array


@pytest.mark.parametrize("b, n, d", get_params())
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_srmsnorm(b, n, d, dtype):
    torch.manual_seed(2024)
    atol = 5e-2
    rtol = 1e-2
    device = torch.device("cuda")
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()
    dy = torch.randn((b, n, d), dtype=dtype, device=device)
    srms_torch = SimpleRMSNormTorch(d)
    srms_triton = SimpleRMSNorm(d)

    # forward
    y_ref = srms_torch(x)
    y = srms_triton(x)

    # backward
    y_ref.backward(dy, retain_graph=True)
    dx_ref, x.grad = x.grad.clone(), None

    y.backward(dy, retain_graph=True)
    dx, x.grad = x.grad.clone(), None

    # test
    assert_close(y, y_ref, atol=atol, rtol=rtol)
    assert_close(dx, dx_ref, atol=atol, rtol=rtol)
