# 官方 SageAttention 2++ CUDA 插件

AF3 的 `--triangle_attention_kernel=mod.triattn.sageattention2pp_cuda:attention`
选项直接调用 SageAttention v2.2.0 的 CUDA 扩展，使用 `fp32+fp16` 累加模式。
对应的 GPU 架构为 Ada `sm_89` 或 Blackwell `sm_120`。CUDA 工具链需 12.8
或更新版本，并准备匹配的 CUDA 版 PyTorch 和 Triton。

```bash
.venv/bin/python -m pip install setuptools wheel packaging ninja
.venv/bin/python -m pip install --no-build-isolation \
  'sageattention @ git+https://github.com/thu-ml/SageAttention.git@eb615cf6cf4d221338033340ee2de1c37fbdba4a'
```

交叉编译时可用 `TORCH_CUDA_ARCH_LIST=8.9` 指定 Ada 目标。生成的 wheel 需要在
对应架构的 GPU 上运行。多 GPU 环境可通过 `CUDA_VISIBLE_DEVICES` 让 JAX 和
PyTorch 使用同一张卡。

AF3 的每个查询分块都会访问全部有效 key。插件按 mask 筛选 K/V，并把该分块的
bias 写入 K 的额外通道。Q 的 one-hot 通道选择对应的 bias，缩放后的点积等于
`scale * Q @ K.T + bias`，随后由 CUDA kernel 执行量化注意力。全遮挡行广播
整个 key 轴上的 `mean(V)`。JAX 通过主机回调将输入交给 PyTorch，再接收输出。

在 AF3 仓库根目录执行以下全量门控命令，使用新的输出目录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -B tests/triattn/verify.py --all \
  --triangle_attention_kernel=mod.triattn.sageattention2pp_cuda:attention \
  --output-dir=/tmp/triattn-sageattention2pp-cuda
```
