# Triangle Attention 新 kernel 精度准入

本文件是给接手 agent 的验收规范。使用本目录的 [verify.py](verify.py) 和冻结测试数据，用同一个 CPU NumPy FP64 参考为冻结抓取输出和候选 kernel 打分。[回放 README](README.md) 说明数据路径。[推理 wrapper 契约](../../mod/triattn/README.md) 定义候选插件接口。

## 数据与硬门槛

- 全量使用 `--all`：seed 1、recycle 10、block 16，覆盖 theta、beta、demo 的 starting / ending。每次运行必须有 **30 个事件、1,300 个 i、10,400 条误差记录**，其中 5,200 条为本次执行输出的 `candidate_vs_fp64`，另 5,200 条为已保存输出的 `capture_vs_fp64`。检查键 `(input, direction, source_event, i, head, comparison)` 唯一，并核对两种比较的事件及行覆盖完全相同。
- 每事件以原形状调用一次：theta `[128,128,4,32]`、beta `[128,512,4,32]`、demo `[1,2560,4,32]`。保留 padding、bias、mask 和 scale；ending 直接使用抓取布局。不补采非对齐位置、不拼接 demo 的单 i 数据、不重跑完整 JAX 推理。
- 任一运行失败、样本缺失或重复、输入哈希变化、shape/dtype 不符合 wrapper 契约，或出现 NaN/Inf，直接拒绝准入。全遮挡的 i 单独与整条 key 轴上的广播 `mean(V)` 比较；该路径同样必须符合语义和数值门槛。

## 冻结误差与通过判据

`data/accuracy_baseline.json` 保存抓取输出相对 FP64 的分组最坏误差和事件哈希。按 `(input, direction, mask_group)` 分组，分别处理有效 key 与全遮挡行。每次回放直接用冻结误差计算门槛。

| 指标 | 每组门槛 |
| --- | --- |
| 逐 i/head 相对 L2 | `min(0.01, 1.2 × 抓取组内最坏相对 L2)` |
| 最大绝对误差 | `1.2 × 抓取组内最大绝对误差` |

相对 L2 使用比例，`0.01` 即 1%。参考范数为零且差异非零时只检查绝对误差。每个已回放组都须通过，报告保留最坏样本、中位数、P95 和超限位置。单事件结果覆盖本次选择的事件，完整验收使用 `--all`。

## 生成基线与运行门控

数据更新时，在 AF3 仓库根目录预处理全部抓取输出，再将生成的 `accuracy_baseline.json` 放到 `tests/triattn/data/`。预处理只执行 CPU NumPy FP64 参考。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -B tests/triattn/verify.py \
  --prepare-accuracy-baseline --output-dir /tmp/triattn-baseline
```

回放默认读取这份基线，通过 `--accuracy-baseline` 可指定另一份预处理文件。下例验证完整抓取覆盖。

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -B tests/triattn/verify.py --all \
  --triangle_attention_kernel=your_module:attention \
  --output-dir /tmp/triattn-candidate
```

终端与 `run_summary.json` 默认输出 `accuracy_gate`。门控失败时退出码为非零。demo 只覆盖单 i 事件，且当前抓取使用 BF16。
