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
