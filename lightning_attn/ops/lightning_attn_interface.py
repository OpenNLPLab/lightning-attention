from .triton import lightning_attn2


def lightning_attn_func(q, k, v, s=None):
    assert s != None

    return lightning_attn2(q, k, v, s)
