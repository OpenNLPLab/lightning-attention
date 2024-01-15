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
pip install lightning_attn2
```

## How to use lightning attention
```
from lightning_attn.ops import lightning_attn2
from lightning_attn.utils import _build_slope_tensor

b, h, n, d = 2, 12, 2048, 64

q = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
k = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
v = (torch.randn((b, h, n, e), dtype=dtype, device=device) / 10).requires_grad_()
s = _build_slope_tensor(h).to(q.device).to(torch.float32)

o = lightning_attn2(q, k, v, s)
```

## Benchmark
```
lightning2-speed_fwd-batch4-head32-qk_dim128-v_dim128-dtype_bf16:
         n     Triton       Flash    Xformers
0    512.0   0.351782    0.092664    0.125723
1   1024.0   0.600033    0.231678    0.375830
2   2048.0   1.137833    0.748262    1.294125
3   4096.0   2.236332    2.744572    4.825598
4   8192.0   4.462824   10.487619   18.328091
5  16384.0   8.816148   40.497246   72.606529
6  32768.0  17.641855  162.978043  285.161804

lightning2-speed_bwd-batch4-head32-qk_dim128-v_dim128-dtype_bf16:
         n     Triton       Flash     Xformers
0    512.0   1.166269    0.396368     0.796550
1   1024.0   2.330066    0.955281     2.023791
2   2048.0   4.648997    2.739009     5.981194
3   4096.0   9.282595    8.892038    19.932423
4   8192.0  18.524307   31.988714    72.275970
5  16384.0  37.076271  120.425346   275.637726
6  32768.0  74.256706  470.944183  1078.135376

lightning2-speed_fwd-batch4-head32-qk_dim128-v_dim128-dtype_bf16:
         n       Triton        Flash     Xformers
0    512.0    64.000488    64.250977    64.250488
1   1024.0   128.000488   128.500977   128.500488
2   2048.0   256.000488   257.000977   257.000488
3   4096.0   512.000488   514.000977   514.000488
4   8192.0  1024.000488  1028.000977  1028.000488
5  16384.0  2048.000488  2056.000977  2056.000488
6  32768.0  4096.000488  4112.000977  4112.000488

lightning2-speed_bwd-batch4-head32-qk_dim128-v_dim128-dtype_bf16:
         n        Triton         Flash      Xformers
0    512.0    173.600488    206.100977    270.100977
1   1024.0    347.200488    412.200977    540.200977
2   2048.0    694.400488    824.400977   1080.400977
3   4096.0   1388.800488   1648.800977   2160.800977
4   8192.0   2777.600488   3297.600977   4321.600977
5  16384.0   5555.200488   6595.200977   8643.200977
6  32768.0  11110.400488  13190.400977  17286.400977
```

## Citation
If you wish to cite our work, please use the following reference:
```
@article{qin2023scaling,
  title={Scaling transnormer to 175 billion parameters},
  author={Qin, Zhen and Li, Dong and Sun, Weigao and Sun, Weixuan and Shen, Xuyang and Han, Xiaodong and Wei, Yunshen and Lv, Baohong and Yuan, Fei and Luo, Xiao and others},
  journal={arXiv preprint arXiv:2307.14995},
  year={2023}
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
