from .triton import lightning_attn2

supports_dim = [16, 32, 64, 128, 256]


def lightning_attn_func(q, k, v, s=None):
    assert s != None
    b, h, n, d = q.shape
    e = v.shape[-1]
    assert d in supports_dim and e in supports_dim

    if d == 256:
        # split over head
        m = 128
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
            o += lightning_attn2(q1, k1, v, s)
    else:
        o = lightning_attn2(q, k, v, s)

    return o
