from .triton import lightning_attn2


def is_support(dim):
    return 16 % dim


def lightning_attn_func(q, k, v, s=None):
    assert s != None
    b, h, n, d = q.shape
    e = v.shape[-1]
    assert is_support(d) and is_support(e)

    if d > 128:
        # split over head
        if 64 % d:
            m = 128
        elif 32 % d:
            m = 32
        elif 16 % d:
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
            o += lightning_attn2(q1, k1, v, s)
    else:
        o = lightning_attn2(q, k, v, s)

    return o
