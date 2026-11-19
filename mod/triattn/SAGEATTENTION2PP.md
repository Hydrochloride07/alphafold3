# SageAttention2++ triangle attention

用 AF3 的 `--triangle_attention_kernel` 参数选用插件。两个方向都通过同一个入口，接收各自的张量布局。用 `--flash_attention_implementation` 为模型的其他注意力算子选择 Tokamax 后端。在 AF3 仓库根目录运行下面的命令。

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python run_alphafold.py \
  --json_path=/path/to/input.json \
  --model_dir=/path/to/models \
  --output_dir=/path/to/predictions \
  --triangle_attention_kernel=mod.triattn.sageattention2pp:attention
```

可用以下设置做消融实验。

| 配置 | 命令行设置 |
| --- | --- |
| AF3 基线 | 省略 `--triangle_attention_kernel` |
| 带残差补偿的 SageAttention2++ | `--triangle_attention_kernel=mod.triattn.sageattention2pp:attention` |
| 单项量化消融 | `--triangle_attention_kernel=mod.triattn.sageattention2pp:attention_uncorrected` |

## 数值实现

- Q/K 去均值后按 token 量化到 INT8。INT32 点积反量化后恢复 Q 均值对应的分数项，再应用 `scale` 并加上 FP32 pair bias。
- 分块在线 softmax 使用有效 key 计算 FP32 分母。P 按块、V 按通道量化到 E4M3。JAX 用可移植点积计算每组 32 个 key 的 FP8 乘积和，每步舍入到 FP16，每两步将结果存入 FP32 累加。
- 精度模式累加 Q/K 各两项、P/V 各三项的交叉乘积。每项量化上一项留下的残差。单项消融使用相同分块、偏置、mask 和累加顺序。
- 全遮挡行在原始 key 轴上计算 `mean(V)` 并广播。返回值恢复输入 dtype。

该版本用 JAX 实现 [SageAttention2++ 的 8+8 数值路径](https://arxiv.org/abs/2505.21136)及 AF3 的残差补偿，使用 FP32 点积模拟 FP8 MMA 的部分积，精度验收在本机 CPU 完成。

## 本地精度回放

运行 [冻结精度门](../../tests/triattn/ACCURACY_GATE.md)。每次提供新的输出目录。

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -B tests/triattn/verify.py --all \
  --triangle_attention_kernel=mod.triattn.sageattention2pp:attention \
  --output-dir=/tmp/triattn-sageattention2pp
```
