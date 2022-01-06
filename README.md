# Block Sparse Attention 研究总结

本人近半年来对Block Sparse Attention（块稀疏注意力）的研究总结（持续更新中）。按时间顺序，主要分为如下三部分：

- [x] [PyTorch 自定义 CUDA 算子——以矩阵乘法为例](./cuda_matmul.ipynb)
- [ ] 基于 [Triton](https://github.com/openai/triton) 的 Block Sparse Attention 及踩过的坑
- [ ] PyTorch 自定义基于 CUDA 的 Block Sparse Attention 算子

## 环境
- Ubuntu 20.04
- CUDA 11.3
- PyTorch 1.10.0+cu113
- Triton 1.1.1