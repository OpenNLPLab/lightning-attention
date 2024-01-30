# Lightning Attention

<p align="center">
💻 <a href="https://github.com/OpenNLPLab/lightning-attention" target="_blank">GitHub </a> •
💬 <a href="https://discord.gg/JEU3nTcWKC" target="_blank">Discord</a> •
💬 <a href="./images/contact_me_qr.png" target="_blank">WeChat</a>
</p>

## Introduction
This repository provides the official implementation of Lightning Attention 1/2 Algorithm.

- [Lightning Attention-1](https://arxiv.org/abs/2307.14995)
- [Lightning Attention-2](https://arxiv.org/abs/2401.04658)


## Installation
```
pip install lightning_attn
```
The code has been test under the following environment:
```
triton                   2.0.0
triton-nightly           2.1.0.dev20230728172942
```
You can use the following command to install:
```
pip install triton==2.0.0
pip install triton-nightly==2.1.0.dev20230728172942 --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/
```

## How to use lightning attention
```
import torch

from lightning_attn.ops import lightning_attn_func
from lightning_attn.utils import _build_slope_tensor

dtype = torch.bfloat16
device = torch.device("cuda")
b, h, n, d, e = 2, 12, 2048, 192, 192

q = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
k = torch.randn((b, h, n, d), dtype=dtype, device=device).requires_grad_()
v = torch.randn((b, h, n, e), dtype=dtype, device=device).requires_grad_()
s = _build_slope_tensor(h).to(q.device).to(torch.float32)

o = lightning_attn_func(q, k, v, s)

print(o.shape)

loss = o.sum()
loss.backward()

```

## Benchmark
```
lightning2-speed_fwd-batch4-head32-qk_dim128-v_dim128-dtype_bf16:
         n  Lightning2      Flash2    Xformers
0    512.0    0.351540    0.094412    0.127568
1   1024.0    0.585876    0.232286    0.375690
2   2048.0    1.134238    0.754831    1.297325
3   4096.0    2.240815    2.740033    4.804503
4   8192.0    4.414397   10.392551   18.329409
5  16384.0    8.832678   40.573997   71.699486
6  32768.0   17.661427  162.895615  286.869446

lightning2-speed_bwd-batch4-head32-qk_dim128-v_dim128-dtype_bf16:
         n  Lightning2      Flash2     Xformers
0    512.0    1.169621    0.397422     0.797627
1   1024.0    2.334296    0.957989     2.027344
2   2048.0    4.657026    2.739919     5.976820
3   4096.0    9.307817    8.891191    19.931032
4   8192.0   18.617611   31.986572    72.536194
5  16384.0   37.212578  121.685730   276.402618
6  32768.0   74.594788  470.666473  1075.611450

lightning2-speed_fwd-batch4-head32-qk_dim128-v_dim128-dtype_bf16:
         n   Lightning2       Flash2     Xformers
0    512.0    64.000488    64.250977    64.250488
1   1024.0   128.000488   128.500977   128.500488
2   2048.0   256.000488   257.000977   257.000488
3   4096.0   512.000488   514.000977   514.000488
4   8192.0  1024.000488  1028.000977  1028.000488
5  16384.0  2048.000488  2056.000977  2056.000488
6  32768.0  4096.000488  4112.000977  4112.000488

lightning2-speed_bwd-batch4-head32-qk_dim128-v_dim128-dtype_bf16:
         n    Lightning2        Flash2      Xformers
0    512.0    173.600488    206.100977    270.100977
1   1024.0    347.200488    412.200977    540.200977
2   2048.0    694.400488    824.400977   1080.400977
3   4096.0   1388.800488   1648.800977   2160.800977
4   8192.0   2777.600488   3297.600977   4321.600977
5  16384.0   5555.200488   6595.200977   8643.200977
6  32768.0  11110.400488  13190.400977  17286.400977
```

## Todo

- [ ] Add support for lightning attention parallel version.
- [ ] Add support for linear attention with no decay.
- [ ] Add support for linear attention with data dependent decay.
- [ ] Add block size for 3090.
- [ ] Add efficient version to deal with not power of 2 feature dim.


## Citation
If you find our work useful, please cite the following papers:
```
@misc{qin2024transnormerllm,
      title={TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer},
      author={Zhen Qin and Dong Li and Weigao Sun and Weixuan Sun and Xuyang Shen and Xiaodong Han and Yunshen Wei and Baohong Lv and Xiao Luo and Yu Qiao and Yiran Zhong},
      year={2024},
      eprint={2307.14995},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{qin2024lightning,
      title={Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models},
      author={Zhen Qin and Weigao Sun and Dong Li and Xuyang Shen and Weixuan Sun and Yiran Zhong},
      year={2024},
      eprint={2401.04658},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```

## Acknowledgment

Thanks for [sustcsonglin](https://github.com/sustcsonglin) and [yzhangcs](https://github.com/yzhangcs) for the helpful discussions. You may also find [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) useful.
