# 真实算子张量抓取

`--capture_dir` 在真实推理的 JIT 可执行程序中插入主机写盘 callback，默认全量
导出目标算子的每次执行。不开启时不插入 callback。此功能仅面向推理，不支持
对带抓取的计算求梯度。数据保留 padding、原始分块与广播维度，不是抽样或
重新生成的数据。开启后设备传输和磁盘占用可能很大，不能用该次运行测性能。

## 运行

准备已有 MSA/模板的 AF3 输入和合法模型权重，沿用原来的设备及 XLA flags：

```bash
bash scripts/native run \
  --json_path=/path/input.json --model_dir=/path/models \
  --output_dir=.local/output/capture --run_data_pipeline=false \
  --capture_dir=.local/captures
```

可选 `--capture_config=/path/filters.json` 使用下列 JSON 格式（每个字段是非空
列表；省略的字段不限制；各字段之间为 AND、列表内为 OR）：

```json
{
  "operator": ["attention", "triangle_multiplication"],
  "stage": ["trunk"],
  "recycle": [0, 1],
  "layer": [0, 3]
}
```

还支持 `diffusion_step`、`sample`、`template` 的非负整数列表及 `direction`
字符串列表。所有索引从 0 开始。动态索引过滤在 callback 中完成：减少保存的
文件，不保证减少设备到主机的传输。未指定配置时没有隐含上限或截断。

`stage` 为 `conditioning`、`trunk`、`msa`、`template`、`diffusion` 或
`confidence`。`operator` 为 `attention`、`triangle_multiplication`、
`diffusion_global_attention`、`diffusion_local_attention`。后两者按共享实现
命名：Pairformer 的 single attention 也会调用 global attention 实现，
conditioning 阶段也会调用 local attention；用 stage 区分，勿只依赖 operator。

## 内容和计算边界

- `attention`：Tokamax `dot_product_attention` 调用边界上的实际
  `query/key/value/bias/mask/result`，result 位于 reshape、gate、projection 之前。
  保留所请求的 `implementation`，不把它误称为自动选择后的实际 kernel 名。
  方向为 `starting`/`ending`，对应未转置/转置的 grid attention。
- `triangle_multiplication`：原始模块 `act/mask`、投影和 gating 后用于 einsum
  的实际 `a/b`、`contraction_result` 以及完整模块 `result`。方向为
  `incoming/outgoing`；JSON 同时保存实际 channel-first `equation` 和模块方程。
- `diffusion_global_attention`：float32 转换后的实际 q/k、实际 v、原始 `mask`、
  `mask_bias`、可为空的 `pair_logits`、二者合并的有效 `bias`，以及
  reshape/gate/projection 之前的 `result`。
- `diffusion_local_attention`：同上，原始 mask 分别保存为 `mask_query/mask_key`，
  通用 `mask` 为 null。完整保留 subset 维度，并保存 `queries_to_keys_indices`、
  `queries_to_keys_mask` 和 `queries_to_keys_input_shape`。方向为 encoder/decoder。

Diffusion 仍执行原来的 `einsum + softmax`。精确复现时按原始顺序计算
`einsum(query * scale, key) + mask_bias`，再加 `pair_logits`，softmax 后转到 v
的 dtype 再与 v 相乘。有效 bias 是辅助信息，将两次加法合并可能改变舍入。
Local mask bias 也原样保留，不修正或替换原模型的 mask 公式。

## 目录和格式

每次 CLI 运行生成独立 `run-<UUID>` 目录：

```text
manifest.json
 events.jsonl
 events/<event UUID>.npz
 events/<event UUID>.json
```

每条 events.jsonl 指向一对文件。NPZ 默认不压缩，JSON 最后原子提交；`.tmp`
或未出现在 index 中的文件不代表完成事件。callback 可能并行、乱序，UUID 和
文件列表顺序不表示执行顺序。运行退出前等待 effects barrier，写入错误会令
运行失败。突然终止时 manifest 可能仍为 running，event_count 可能尚未更新。

事件 JSON 包含 `schema_version`、`event_id`、`metadata` 和 `tensors`：

- `metadata`：`input_name/seed/invocation_id`，`operator/module_path/stage/direction`，
  以及适用的 `recycle/layer/template/diffusion_step/sample` 与各自 `*_count`。
  `recycle=0` 是首次 trunk pass；总 pass 数为模型 num_recycles + 1。
- Global diffusion 层同时记录 `super_block/block` 及其 count，`layer` 是展开后的
  层号。模块完整路径可区分不同 attention 堆栈。
- `chunk_start/chunk_end` 是实际分片的半开区间，`chunk_axis/chunk_total` 表示
  当前分片的轴和总长度；嵌套分片记录当前最内层，sample 等身份另行记录。
- 不适用的索引为 null；实现相关信息包括 `scale/implementation/layout` 或
  `equation/module_equation`。
- `tensors` 映射每个数组名到 `shape/dtype/storage_dtype`；原始 None 映射到 null，
  NPZ 中没有占位数组。bfloat16 按 uint16 原始位存储，禁止当作普通整数使用。

manifest 记录运行状态、筛选配置、事件总数、index 文件和每次输入/seed 的
调用状态。不同种子使用动态 invocation_id，可复用同一 JIT executable，
不会将前一次输入的身份固定进编译结果。

## 读取与验证

```python
from alphafold3.model.capture import load_event
record, tensors = load_event('/path/run-xxx/events/xxx.json')
query = tensors['query']  # 自动恢复 bfloat16，None 仍为 None
```

```bash
.venv/bin/python scripts/inspect_capture.py .local/captures/run-xxx
.venv/bin/python scripts/inspect_capture.py /path/event.json
JAX_PLATFORMS=cpu .venv/bin/python -m pytest src/alphafold3/model/capture_test.py -q
```

小尺寸 CPU 测试覆盖实际算子、Haiku 参数兼容、循环索引与保存可靠性。
真实权重和 GPU 全模型抓取还需在本机普通终端或服务器验证；CPU 模块测试
不能代替该项验收。

## 单个 Pairformer block 的首轮 trunk 采集

`scripts/capture_trunk.py` 为本机 `demo_cut.json` 提供两阶段入口：
先在 CPU 上复用序列/修饰精确匹配的已有 MSA 和模板并重新特征化，再使用
真实权重运行 GPU trunk。不会调用 MSA/template 搜索管线。

```bash
JAX_PLATFORMS=cpu .venv/bin/python scripts/capture_trunk.py \
  --prepare-only --output-dir=/new/attempt-directory

JAX_PLATFORMS=cuda,cpu TF_FORCE_UNIFIED_MEMORY=true \
XLA_CLIENT_MEM_FRACTION=4.0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_FLAGS='--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_autotune_level=0' \
AF3_CAPTURE_COMPRESS=1 AF3_CAPTURE_MIN_FREE_GB=10 \
.venv/bin/python scripts/capture_trunk.py \
  --run-existing --transition-chunk-size=128 --output-dir=/new/attempt-directory
```

运行需清除继承的 `PYTHONPATH/PYTHONHOME/LD_LIBRARY_PATH` 及旧 allocator
覆盖；GPU 执行需设备可访问。入口验证 managed-memory 指针、真实权重哈希及
prepared 输入/特征哈希。`num_recycles=0` 计算初始轮全部 48 blocks，随后停止，
不运行 diffusion/confidence/distogram。只采集内部 `recycle=0,layer=0` 的五类
算子，旧选择接口对应 `pairformer_blocks=[1]`。不开启采集时默认模型行为不变。

此输入共 1257 tokens，padding=1280、attention chunk=128 时预期 23 个事件、
140 个数组、原始数组 7.4365 GiB。4 GiB GPU 上，`XLA_CLIENT_MEM_FRACTION=2.0`
的初始尝试在 GEMM 自动调优阶段 OOM；上述配置提高池上限并关闭该自动调优，
不删减输入、MSA、模型层或 attention 分片。是否成功以状态和验证报告为准。
若整块 transition 引发严重换页，`--transition-chunk-size=128` 将其按独立行
分片计算，避免完整 6.25 GiB 中间矩阵；不改变 attention 的 chunk、输入或层数。
FP32/BF16 回归检查参数一致及数值误差；实际分片配置记录在 state.json 中。

运行成功后，按 `archives`、`numerical`、`finalize` 顺序调用
`scripts/validate_trunk_capture.py OUTPUT --phase=PHASE`。完整性检查读取全部
数组；数值复算分别检查 triangle contraction 的三个 channel、两个 attention
方向首个分片的三个 i，以及 single attention 全部 heads。报告明确覆盖范围。
可读文件名中的 recycle/block 均从 0 开始；原 UUID 仍保留在 `event_id` 中。

## 多输入 trunk 与稀疏 i 采集

`capture_trunk.py --prepare-only` 支持 `--num-recycles=10 --capture-recycles=10`
和 `--capture-blocks=1,16,32`。执行 11 轮完整 trunk，写盘只限选定轮和 block；
`--filename-block-base=1` 将内部 layer 0/15/31 命名为 block 01/16/32。
`run_trunk_batch.py` 顺序运行独立 attempt 并验收，避免多个 GPU 进程竞争显存。

`--capture-attention-i-step=256` 只对 trunk 的 grid attention 导出全局 i 为
256 整数倍的 Q/K/V/mask/result。计算分片和模型输出不变，bias 保留完整矩阵，
triangle multiplication 与 single attention 的采集不变。当前实现要求采集
步长至少等于计算分片大小，保证每分片最多一个采集行。JSON 的 `chunk_start/end`
仍是计算分片边界，`sampled_i` 是实际导出的全局 i，Q/K/V/result 首维长度为 1。
设备端在传输前合并 i、recycle 和 layer 条件；不合条件的分片不进入主机 writer。
