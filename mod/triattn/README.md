# AlphaFold 3 Triangle Attention Kernel 接口

用 `--triangle_attention_kernel=MODULE:FUNCTION` 加载 Python 可导入的实现。这个参数优先于 `--flash_attention_implementation`。省略时，推理使用后者选择的 Tokamax 后端，默认是 Triton。自定义实现须支持所选的 JAX 设备。

在 AF3 仓库根目录运行以下命令，将输入、模型和输出路径换成实际路径。

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python run_alphafold.py \
  --json_path=/path/to/input.json \
  --model_dir=/path/to/models \
  --output_dir=/path/to/predictions \
  --triangle_attention_kernel=mod.triattn.example_attention:attention
```

## 函数契约

```python
def attention(*, query, key, value, bias, mask, scale) -> jax.Array:
    ...
```

| 参数 | 形状和语义 |
| --- | --- |
| `query`, `key`, `value` | JAX 浮点张量 `[I, N, H, D]`。`I` 是当前 chunk 大小，`N` 含 padding，`H` 是 head 数，`D` 是每个 head 的维数。 |
| `bias` | JAX 浮点张量 `[1, H, N, N]`，沿 `I` 广播。最后两轴依次对应 query 和 key。 |
| `mask` | JAX 布尔张量 `[I, 1, 1, N]`，`True` 表示有效 key。 |
| `scale` | Python `float`。推理取 `1 / sqrt(D)`，回放取抓取时记录的值。 |
| 返回值 | gate 之前的 JAX 张量，形状和 dtype 与 `query` 相同。 |

对有效 key 计算 `softmax_key(scale * Q @ K.T + bias) @ V`。对全遮挡的 `i`，将整个 key 轴上 `V` 的均值广播给该 `i` 的所有 query。

函数须支持 `jax.jit` 和不同的 chunk 大小，包括尾块。推理可能传入 BF16 或 FP32。starting 和 ending 都使用同一函数，输入已经排成对应方向的算子布局。启动时验证模块和函数，wrapper 在 tracing 时检查返回值。插件执行或编译出错时，推理错误中会带上当前插件名。

[example_attention.py](example_attention.py) 用纯 JAX 实现接口。它用 FP32 计算，最后转换回输入 dtype。它显式构造 `[I, H, N, N]` 的 logits，适合用小抓取验证接入。[精度回放指南](../../tests/triattn/README.md) 给出了用同一插件回放真实输入的命令。
