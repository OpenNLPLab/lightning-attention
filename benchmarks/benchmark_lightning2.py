import os

import numpy as np
import torch
import triton
from einops import rearrange

from lightning_attn.ops import lightning_attn2, lightning_attn2_no_decay
from lightning_attn.utils import _build_slope_tensor, get_memory

try:
    from flash_attn import flash_attn_func

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

try:
    import xformers.ops as xops

    HAS_XFORMERS = True
except BaseException:
    HAS_XFORMERS = False

b, h, n, d, e = 4, 32, 4096, 128, 128
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def flash_wrapper(q, k, v, causal=True):
    q, k, v = map(lambda x: rearrange(x, "b h n d -> b n h d"), [q, k, v])
    o = flash_attn_func(q, k, v, causal=causal)
    o = rearrange(o, "b n h d -> b h n d")

    return o


def xformer_wrapper(q, k, v, causal=True):
    q, k, v = map(lambda x: rearrange(x, "b h n d -> b n h d"), [q, k, v])
    if causal:
        o = xops.memory_efficient_attention(
            q, k, v, attn_bias=xops.LowerTriangularMask()
        )
    else:
        o = xops.memory_efficient_attention(q, k, v)
    o = rearrange(o, "b n h d -> b h n d")

    return o


##### speed benchmark

speed_configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time(ms)",
        line_arg="provider",
        line_vals=["lightning2", "lightning2_no_decay"]
        + (["flash2"] if HAS_FLASH else [])
        + (["xformers"] if HAS_XFORMERS else []),
        line_names=["Lightning2", "Lightning2NoDecay"]
        + (["Flash2"] if HAS_FLASH else [])
        + (["Xformers"] if HAS_XFORMERS else []),
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"lightning2-speed_{mode}-batch{b}-head{h}-qk_dim{d}-v_dim{e}-dtype_{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
        },
    )
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
]


@triton.testing.perf_report(speed_configs)
def bench_speed(b, h, n, d, dtype, device, mode, provider):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    s = _build_slope_tensor(h).to(q.device).to(torch.float32)

    if provider == "lightning2":
        fn = lambda: lightning_attn2(q, k, v, s)
    elif provider == "lightning2_no_decay":
        fn = lambda: lightning_attn2_no_decay(q, k, v)
    elif provider == "flash2":
        fn = lambda: flash_wrapper(q, k, v)
    else:
        fn = lambda: xformer_wrapper(q, k, v)

    if mode == "bwd":
        o = fn()
        do = torch.randn((b, h, n, e), dtype=dtype, device=device)
        fn = lambda: o.backward(do, retain_graph=True)

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


##### memory benchmark
memory_configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 16)],
        xlabel="Sequence Length",
        ylabel="Memory(mb)",
        line_arg="provider",
        line_vals=["lightning2", "lightning2_no_decay"]
        + (["flash2"] if HAS_FLASH else [])
        + (["xformers"] if HAS_XFORMERS else []),
        line_names=["Lightning2", "Lightning2NoDecay"]
        + (["Flash2"] if HAS_FLASH else [])
        + (["Xformers"] if HAS_XFORMERS else []),
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"lightning2-memory_{mode}-batch{b}-head{h}-qk_dim{d}-v_dim{e}-dtype_{dtype_name}",
        args={
            "b": b,
            "h": h,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
        },
    )
    for mode in ["fwd", "bwd"]
    for dtype_name in ["bf16"]
]


@triton.testing.perf_report(memory_configs)
def bench_memory(b, h, n, d, dtype, device, mode, provider):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    rep = 20
    q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
    s = _build_slope_tensor(h).to(q.device).to(torch.float32)

    if provider == "triton":
        fn = lambda: lightning_attn2(q, k, v, s)
    elif provider == "lightning2_no_decay":
        fn = lambda: lightning_attn2_no_decay(q, k, v)
    elif provider == "flash":
        fn = lambda: flash_wrapper(q, k, v)
    else:
        fn = lambda: xformer_wrapper(q, k, v)

    if mode == "bwd":
        o = fn()
        do = torch.randn((b, h, n, e), dtype=dtype, device=device)
        fn = lambda: o.backward(do, retain_graph=True)

    try:
        torch.cuda.reset_peak_memory_stats(device)
        mb_arr = []
        for _ in range(rep):
            fn()
            mb_arr.append(get_memory(device))
        mb = np.mean(mb_arr)
    except:
        mb = -1

    return mb


save_path = "stat/lightning2"
os.makedirs(save_path, exist_ok=True)
bench_speed.run(save_path=save_path, print_data=True)
bench_memory.run(save_path=save_path, print_data=True)
