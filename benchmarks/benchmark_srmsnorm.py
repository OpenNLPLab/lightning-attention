import os

import numpy as np
import torch
import triton

from lightning_attn.ops import SimpleRMSNorm, SimpleRMSNormTorch

b, n, d = 12, 8192, 2048
device = torch.device("cuda")

dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

##### speed benchmark

speed_configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time (ms)",
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"srms-speed-{mode}-batch{b}-dim{d}-dtype-{dtype_name}",
        args={
            "b": b,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
        },
    )
    for mode in ["fwd", "bwd"]
    for dtype_name in ["fp32", "fp16", "bf16"]
]


@triton.testing.perf_report(speed_configs)
def bench_speed(b, n, d, dtype, device, mode, provider):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()

    if provider == "triton":
        module = SimpleRMSNorm(d)
    else:
        module = SimpleRMSNormTorch(d)

    fn = lambda: module(x)
    if mode == "bwd":
        y = fn()
        dy = torch.randn((b, n, d), dtype=dtype, device=device)
        fn = lambda: y.backward(dy, retain_graph=True)

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


##### memory benchmark
memory_configs = [
    triton.testing.Benchmark(
        x_names=["n"],
        x_vals=[2**i for i in range(9, 16)],
        xlabel="Sequence Length",
        ylabel="Execution Time (ms)",
        line_arg="provider",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[
            ("red", "-"),
            ("orange", "-"),
            ("green", "-"),
            ("blue", "-"),
            ("black", "-"),
        ],
        plot_name=f"srms-memory-{mode}-batch{b}-dim{d}-dtype-{dtype_name}",
        args={
            "b": b,
            "d": d,
            "dtype": dtype_map[dtype_name],
            "device": device,
            "mode": mode,
        },
    )
    for mode in ["fwd", "bwd"]
    for dtype_name in ["fp32", "fp16", "bf16"]
]


def get_memory(device):
    mb_used = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    torch.cuda.reset_peak_memory_stats(device)

    return mb_used


@triton.testing.perf_report(memory_configs)
def bench_memory(b, n, d, dtype, device, mode, provider):
    torch.manual_seed(2024)
    assert mode in ["fwd", "bwd"]
    rep = 20
    x = torch.randn((b, n, d), dtype=dtype, device=device).requires_grad_()

    if provider == "triton":
        module = SimpleRMSNorm(d)
    else:
        module = SimpleRMSNormTorch(d)

    fn = lambda: module(x)
    if mode == "bwd":
        y = fn()
        dy = torch.randn((b, n, d), dtype=dtype, device=device)
        fn = lambda: y.backward(dy, retain_graph=True)

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


save_path = "stat/srmsnorm"
os.makedirs(save_path, exist_ok=True)
bench_speed.run(save_path=save_path, print_data=True)
bench_memory.run(save_path=save_path, print_data=True)
