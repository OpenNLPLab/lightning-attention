import torch


def get_mask(n, slope=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    # -n, ..., -2, -1, 0
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, : i + 1] = -torch.flip(y, [0])

    return torch.exp(mask)


def get_full_mask(n, slopes):
    arr = []
    for slope in slopes:
        arr.append(get_mask(n, slope.item()))
    mask = torch.stack(arr, dim=0)

    return mask


def linear_attn(q, k, v, s):
    b, h, n, d = q.shape
    mask = get_full_mask(n, s).to(q.device).to(torch.float32)
    qk = torch.matmul(q, k.transpose(2, 3))
    qk = (qk.to(torch.float32) * mask).to(q.dtype)
    o = torch.matmul(qk, v)

    return o
