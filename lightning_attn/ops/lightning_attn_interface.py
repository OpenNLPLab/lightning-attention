import math

import torch.nn.functional as F

from .triton import lightning_attn2, lightning_attn2_no_decay, lightning_attn2_parallel


def is_support(dim):
    return 16 % dim


def next_power_of_2(n):
    return 2 ** (int(math.ceil(math.log(n, 2))))


def lightning_attn_func(q, k, v, s=None, variant="chunk_loop"):
    if s is None:
        fn = lightning_attn2_no_decay
    else:
        if variant == "parallel":
            fn = lightning_attn2_parallel
        elif variant == "chunk_loop":
            fn = lightning_attn2
    b, h, n, d = q.shape
    e = v.shape[-1]
    assert is_support(d) and is_support(e)

    # pad v's feature dim to power of 2
    e_pad = next_power_of_2(e)
    need_pad = e_pad != e
    if need_pad:
        v = F.pad(v, (0, e_pad - e))

    if d > 128:
        # split over head
        if d % 64 == 0:
            m = 64
        elif d % 32 == 0:
            m = 32
        elif d % 16 == 0:
            m = 16
        arr = [m * i for i in range(d // m + 1)]
        if arr[-1] != d:
            arr.append(d)
        n = len(arr)
        o = 0
        for i in range(n - 1):
            start = arr[i]
            end = arr[i + 1]
            q1 = q[..., start:end]
            k1 = k[..., start:end]
            if s != None:
                o += fn(q1, k1, v, s)
            else:
                o += fn(q1, k1, v)
    else:
        if s != None:
            o = fn(q, k, v, s)
        else:
            o = fn(q, k, v)

    if need_pad:
        o = o[:, :, :, :e]

    return o
