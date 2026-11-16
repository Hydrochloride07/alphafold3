# Triangle Attention 单次精度回放

`verify.py` 在 gate 之前验证一个已保存的 attention chunk。默认读取 theta ending 的完整事件并调用 Tokamax XLA。回放使用 AF3 的 Python 环境与已有张量抓取，无需加载模型权重。

在 AF3 仓库根目录运行回放，给每次运行选择一个新的输出目录。

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B tests/triattn/verify.py \
  --output-dir /tmp/af3-triattn-xla
```

回放可以直接加载推理使用的同一个实现。以下命令运行 [wrapper 附带的纯 JAX 示例](../../mod/triattn/README.md)。

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -B tests/triattn/verify.py \
  --triangle_attention_kernel=mod.triattn.example_attention:attention \
  --output-dir /tmp/af3-triattn-example
```

## 选择事件

回放默认读取 `tests/triattn/data/` 中的 30 个事件。NPZ 文件合计约 921 MiB，随仓库检出。用 `--capture-root ../data/af3_capture` 可读取原始抓取目录。三个输入均为 seed 1、recycle 10、block 16。

| `--input` | 每方向的事件数 | 每事件的 Q/K/V 形状 |
| --- | ---: | --- |
| `theta` | 1 | `[128,128,4,32]` |
| `beta` | 4 | `[128,512,4,32]` |
| `demo` | 10 | `[1,2560,4,32]` |

`--direction` 取 `starting` 或 `ending`。`--i N` 选择包含全局索引 `N` 的已保存事件，默认索引为 0。例如 `--input beta --direction starting --i 128` 回放位置 128 所在的整个 chunk。demo 仅保存了 `0, 256, …, 2304` 这十个索引。用 `--all` 回放两个方向的全部 30 个事件，共 1,300 个 `i`。

回放按位恢复 BF16 并保留 padding、chunk 和方向布局。每个 NPZ 触发一次 JIT kernel 调用。可用 `--flash_attention_implementation=triton` 或 `cudnn` 在支持的 GPU 上验证 Tokamax 的其他后端。使用 GPU 时按实际环境设定 JAX 设备。

## 读取结果

独立 NumPy FP64 参考只沿 query 分片，稳定 softmax 始终使用完整的有效 key 集合。全遮挡的 `i` 用广播后的 `mean(V)` 作参考。

- `errors.csv` 按事件、`i` 和 head 列出 `capture_vs_fp64` 与 `candidate_vs_fp64`，包含相对 L2、最大绝对误差、最坏 query/channel 和 mask 分组。默认单事件有 1,024 行，全量有 10,400 行。
- `run_summary.json` 汇总分组最坏值、容差超限、运行环境、失败和源文件的前后 SHA-256。

默认记录数值误差。传入 `--max-relative-l2`、`--max-abs` 或两者时，脚本逐 `i`、逐 head 检查本次输出与 FP64 的差异。参考范数为零且误差非零时，相对 L2 留空，启用相对 L2 限制时视为超限。运行失败、出现非有限值、源文件变化或超限均返回非零。
