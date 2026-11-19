# AlphaFold 3 Triangle Attention 中的 SageAttention2++ 数值实现

本文对应 `mod.triattn.sageattention2pp:attention`，即已经通过本地全量精度门的 JAX 插件。它在 softmax 前把 FP32 pair bias 加入分数，在 QK 和 PV 两个阶段补偿输入量化误差。Q、K 各用 **1 个 INT8 主项 + 1 个 INT8 残差项**，P、V 各用 **1 个 E4M3 主项 + 2 个 E4M3 残差项**。输出按 AF3 接口转换回 query 的 dtype。[JAX 实现](sageattention2pp.py#L189)

## 1. 从 AF3 的算子定义出发

Q、K、V 都采用 `[I, N, H, D]`。`I` 是当前 triangle attention chunk 的大小，`N` 是含模型 padding 的序列长度，`H` 是 head 数，`D` 是每个 head 的通道数。bias 的形状是 `[1, H, N, N]`，在 `I` 轴上广播。mask 的形状是 `[I, 1, 1, N]`，决定每个 `i` 能看到哪些 key。starting 和 ending 各自传入对应方向已经排好的张量，插件内使用相同的公式。[插件接口](README.md)

固定一个 `i` 和一个 head，令 $a$ 表示 query 位置，$b$ 表示 key 位置，$M$ 表示有效 key 集合，$\alpha$ 表示 AF3 传入的 scale。目标是计算

$$
s_{ab}=\alpha\,q_a k_b^{\mathsf T}+B_{ab},
\qquad
o_a=\frac{\sum_{b\in M} e^{s_{ab}}v_b}{\sum_{b\in M}e^{s_{ab}}}.
$$

$\alpha$ 通常取 $1/\sqrt D$。bias 已经处在 logit 的尺度上，实现先按 $\alpha$ 缩放点积，再以 FP32 加入 $B_{ab}$，然后把无效 key 的 logit 设为 $-\infty$。这样每个 $B_{ab}$ 都能直接改变对应 key 的 softmax 权重，Q/K 量化只作用于点积分支。[偏置与 mask](sageattention2pp.py#L104)

计算按 query 最多 128 行、key 每块 64 行切分。key 尾块补零并将新增位置标为无效，query 尾块的补齐结果在返回前裁掉。实现按 `i` 遍历、按 head 向量化，在线累加每个 query 的输出，只在块内生成当前的 QK 分数和指数权重。原有 bias 张量仍按 AF3 接口传入。[分块与输出布局](sageattention2pp.py#L147)

全遮挡时，AF3 的 wrapper 契约规定输出为 $\frac{1}{N}\sum_b v_b$，并沿 query 轴广播。这个分支直接走 FP32 均值，使用调用时的完整 key 轴，包括 AF3 本身的 padding，索引到原始长度 $N$ 为止。[全遮挡分支](sageattention2pp.py#L178)

## 2. Q/K 去均值及其分数修正

去均值让量化尺度更多地反映 token 之间的变化。对 K，在整个有效 key 集合上按通道求一次均值 $\mu_K$。对 Q，在当前 query 块按通道求均值 $\mu_Q$。记

$$
k'_b=k_b-\mu_K,\qquad q'_a=q_a-\mu_Q.
$$

K 去均值给同一个 query 的所有有效分数平移了相同的量。softmax 对同行常数平移不变，因此可以直接使用 $q_a{k'_b}^{\mathsf T}$。Q 去均值带来的变化随 key 而变，需要在反量化之后恢复

$$
q_a{k'_b}^{\mathsf T}
=
q'_a{k'_b}^{\mathsf T}+\mu_Q{k'_b}^{\mathsf T}.
$$

第一项走 INT8 点积。第二项用原先的 FP32 $K'$ 计算，每个 query 块只需一行修正结果，然后广播给该块的 query。两项相加后才乘 $\alpha$ 并加 bias。这里的 **Q 均值修正** 来自去均值的代数展开。下一节的 **量化残差** 来自低精度舍入，处理的是另一个误差来源。[Q/K 去均值和修正](sageattention2pp.py#L71) · [SageAttention2 论文](https://proceedings.mlr.press/v267/zhang25ae.html)

## 3. 残差补偿如何保留低位信息

### 3.1 每一项重新量化上一步留下的差

设 $X$ 是当前要量化的张量，$r_0=X$。对每个尺度组反复执行三步。$\ell$ 是分量编号，$R$ 是该张量的量化范围上限，$\mathcal Q$ 表示转换成目标格式。

$$
\begin{aligned}
s_\ell &= \frac{\max_{\text{组内}}\lvert r_\ell\rvert}{R},\\
\widehat X_\ell &= \mathcal Q\!\left(\operatorname{clip}(r_\ell/s_\ell,-R,R)\right),\\
r_{\ell+1} &= r_\ell-s_\ell\widehat X_\ell.
\end{aligned}
$$

每项保存低精度值 $\widehat X_\ell$ 和 FP32 尺度 $s_\ell$。全零组使用尺度 1，所以量化值和后续残差都为 0。对 INT8，$\mathcal Q$ 用最近整数舍入，再转换成有符号整数。对 E4M3，$\mathcal Q$ 由 `float8_e4m3fn` 转换完成。第 $m$ 项之后，原输入可以写成

$$
X = \sum_{\ell=0}^{m-1} s_\ell\widehat X_\ell+r_m
$$

这里的等式按实数运算写出，实现中的尺度和减法使用 FP32。残差项用自己的幅度选择尺度，通常能把首轮舍入后幅值很小的差值铺开到目标格式的可表示范围。残差可能为正或负，分别修正主项的低估或高估。[量化实现](sageattention2pp.py#L20)

一个 INT8 例子是向量 $(127,\;1.30)$。首轮尺度是 1，存储 $(127,\;1)$，剩下 $(0,\;0.30)$。第二轮尺度是 $0.30/127$，存储 $(0,\;127)$。反量化后两项相加得到 $(127,\;1.30)$。这个例子只有一个非零残差值，因此第二项可完整恢复它。对一般向量，第二项仍会舍入，最后留下比第一轮更小的误差。在理想舍入模型中，尺度组最大幅值为 $A$ 时，一轮 INT8 的组内最大绝对误差至多为 $A/254$，因为量化步长为 $A/127$，最近整数舍入至多留下半步。

### 3.2 每个分量用什么尺度

| 操作数 | 格式 | 每项尺度的共享范围 | 单项范围 | 精度入口的总项数 |
| --- | --- | --- | ---: | ---: |
| 去均值后的 Q | INT8 | 每个 query token 的 $D$ 个通道 | $[-127,127]$ | 2 |
| 去均值后的 K | INT8 | 每个 key token 的 $D$ 个通道 | $[-127,127]$ | 2 |
| 当前块的指数权重 $\widetilde P$ | E4M3FN | 单个 head 的当前 query × key 分块 | $[-224,224]$ | 3 |
| V | E4M3FN | 单个 head 的每个 key 块、每个输出通道 | $[-4.5,4.5]$ | 3 |

INT8 在每个尺度组内使用固定步长。E4M3 的 8 位由 1 位符号、4 位指数和 3 位小数构成，可表示值的间距随幅值变化。将首轮的差值单独缩放再量化，相当于为这些差值分配另一组 E4M3 表示。V 按通道定尺度，一个通道的峰值只决定该通道的量化范围。$\widetilde P$ 的主项来自非负指数权重，后续残差项可以取正值或负值。[尺度与分量数](sageattention2pp.py#L63) · [权重分量](sageattention2pp.py#L120)

Q/K 的每 token 尺度由张量轴定义。K、V 的分量在同一 `i` 和 head 的各个 query 块之间复用，Q 的分量在当前 query 块的各个 key 块之间复用。[分块实现](sageattention2pp.py#L71)

### 3.3 点积为什么要算交叉项

把反量化后的 Q/K 主项和残差项分别记为 $Q_0,Q_1$、$K_0,K_1$。要计算的是两个近似操作数的点积，因此有

$$
(Q_0+Q_1)(K_0+K_1)^{\mathsf T}
=
Q_0K_0^{\mathsf T}
+Q_1K_0^{\mathsf T}
+Q_0K_1^{\mathsf T}
+Q_1K_1^{\mathsf T}.
$$

$Q_1K_0^{\mathsf T}$ 和 $Q_0K_1^{\mathsf T}$ 分别补回 Q 侧和 K 侧首轮量化误差的主要贡献，$Q_1K_1^{\mathsf T}$ 补回两侧残差的乘积。附带尺度之外，真正送入每个点积的是 INT8 值。每项先得到 INT32 点积，再乘对应 query 和 key 的尺度，最后把四项累加到 FP32 分数。[INT8 交叉点积](sageattention2pp.py#L35)

P/V 使用相同的展开。三个 P 分量与三个 V 分量全配对，因此每个 key 块有九项 $\widetilde P_\ell V_j$。先对各项执行对应的低精度乘累规则，再反量化并合并。从实数运算看，这等于先重建两个近似操作数再矩阵乘。从实际的有限精度实现看，各项还分别承受 FP16 部分累加的舍入。增加操作数分量会缩小剩余量化误差，FP16 累加、FP32 运算及最终输出转换也各自影响误差。[E4M3 交叉点积](sageattention2pp.py#L45)

## 4. 在线 softmax 以及 FP16 / FP32 两级累加

### 4.1 分母保留 FP32 权重，分子使用量化权重

对每条 query，遍历 key 块时维护三个量。$m$ 是已处理 key 的最大 logit，$L$ 是以 $m$ 为零点的指数和，$U$ 是以同一零点计算的加权 V 和。初值为 $(-\infty,0,0)$。对新块 $T$ 中的有效 key，更新为

$$
\begin{aligned}
m' &= \max\!\left(m,\ \max_{b\in T}s_{ab}\right),\\
\rho &= e^{m-m'},\qquad \widetilde p_b=e^{s_{ab}-m'},\\
L' &= \rho L+\sum_{b\in T}\widetilde p_b,\\
U' &= \rho U+\operatorname{PV}_{\mathrm{quant}}(\widetilde p_T,V_T).
\end{aligned}
$$

$\rho$ 把之前的分子和分母换算到新的零点。最后返回 $U/L$。这里量化的 $\widetilde p_b$ 是尚未除以 $L$ 的指数权重。softmax 分母直接从 FP32 指数权重求和，分子中的 PV 采用 E4M3 分量和 FP16 部分累加。QK 的误差则在 softmax 之前进入 logits，因此会同时影响指数权重、分母和分子。[在线更新](sageattention2pp.py#L114)

全无效的 key 块直接跳过。个别 query 在当前块的所有 logit 都为 $-\infty$，且前面的块还没有有限 logit 时，指数计算暂用零作为减数，得到零权重并保留运行最大值 $-\infty$。后续出现有限 logit 时即可正常开始更新。[空块与稳定指数](sageattention2pp.py#L114)

### 4.2 为什么设成 32 + 32，范围上限为什么是 224 和 4.5

对一对 E4M3 分量，把 64 个 key 分成两组。以 $S_0,S_1$ 表示每组 32 个乘积之和，$R_{16}$ 表示舍入到 FP16。代码实现的顺序是

$$
A_0=R_{16}(S_0),\qquad
A_1=R_{16}(A_0+S_1).
$$

一对分量的 $A_1$ 转回 FP32 后乘 P 的块尺度和 V 的通道尺度，再加入该块的 FP32 部分和。FP32 在线状态 $U$ 负责跨块累加。由于每个量化 P 值的幅值至多是 224，每个量化 V 值的幅值至多是 4.5，两组共 64 个乘积的绝对和上界是

$$
64\times224\times4.5=64512<65504,
$$

其中 65504 是 FP16 的最大有限值。首组的上界 32256 也可以在 FP16 中精确表示，首步舍入仍留有同样的溢出余量。这个范围约束对每一对主项或残差项都成立，因为每一项独立使用相同的归一化上限。[SageAttention2++ §3.1–3.2](https://arxiv.org/pdf/2505.21136) · [累加代码](sageattention2pp.py#L45)

JAX 版先将已量化的 E4M3 数据转换为 FP32 做每组 32 个 key 的点积，两次更新实际转换到 FP16，再转回 FP32。这样可以在本机 CPU 上验证 E4M3 量化和两个 FP16 舍入边界。原论文在 GPU 上用 FP8 MMA 实现这一累加策略，GPU 指令的内部运算顺序和本地性能属于另外的测量范围。仓库中的 `sageattention2pp_cuda` 是独立的 CUDA 插件，使用额外 Q/K 通道编码 bias，此处回放和残差分量的数据均对应 JAX 入口。[CUDA 插件说明](SAGEATTENTION2PP_CUDA.md)

## 5. 残差分量数如何由回放结果决定

用同一份 theta ending 抓取的有有效 key 组比较三种配置。表中的最大值都是逐 `i/head` 对同一个 NumPy FP64 参考测量后，再在该组内取最坏值。分量数包括主项。

| Q/K 各自项数 | P/V 各自项数 | 最坏相对 L2 | 最大绝对误差 | 该组门控 |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 3.71874% | 0.3079294 | 560 条指标超限 |
| 2 | 2 | 0.22277% | 0.0215983 | 2 条最大绝对误差超限 |
| 2 | 3 | 0.21153% | 0.0174285 | 通过 |
| — | — | 0.25203% | 0.0187095 | 该组门槛 |

单项配置保留了相同的公式、偏置位置、mask 和累加精度，用于同时关闭 Q/K 和 P/V 的残差补偿。两项 P/V 配置的相对 L2 已经达标，个别输出元素的误差仍超限。最终保留第三个 P/V 分量，使这一组的最大绝对误差也进入门槛。按每块交叉乘积计，精度入口比单项配置多做三项 QK 和八项 PV，换取更低的回放误差。[单项消融原始报告](/tmp/af3-sage2pp-ablation.vXjEjX/replay/run_summary.json) · [两项试验原始报告](/tmp/af3-sage2pp-two-part.wtVxxB/replay/run_summary.json) · [最终原始报告](/tmp/af3-sage2pp-final.EAI90z/replay/run_summary.json)

### 全量结果

本机 CPU 回放覆盖 30 个事件、1,300 个 `i`，共 10,400 条记录，其中候选输出与冻结抓取输出各有 5,200 条。所有 10 个分组通过。相对 L2 门槛为 1% 与冻结抓取组内最坏相对 L2 的 1.2 倍取小，绝对误差门槛为冻结抓取组内最大绝对误差的 1.2 倍。下表每格为「候选最坏值 / 分组门槛」。[精度门定义](../../tests/triattn/ACCURACY_GATE.md)

| 输入 | 方向 | mask 分组 | 相对 L2 | 最大绝对误差 |
| --- | --- | --- | ---: | ---: |
| theta | starting | 有效 key | 0.21231% / 0.25304% | 0.0248775 / 0.0298530 |
| theta | starting | 全遮挡 | 0.20612% / 0.24734% | 0.0263672 / 0.0316406 |
| theta | ending | 有效 key | 0.21153% / 0.25203% | 0.0174285 / 0.0187095 |
| theta | ending | 全遮挡 | 0.19676% / 0.23611% | 0.0297852 / 0.0357422 |
| beta | starting | 有效 key | 0.20066% / 0.23584% | 0.0317370 / 0.0370779 |
| beta | starting | 全遮挡 | 0.22803% / 0.27363% | 0.0275879 / 0.0331055 |
| beta | ending | 有效 key | 0.20951% / 0.25095% | 0.0329430 / 0.0374399 |
| beta | ending | 全遮挡 | 0.21233% / 0.25480% | 0.0300293 / 0.0360352 |
| demo | starting | 有效 key | 0.17476% / 0.40189% | 0.0350203 / 0.0693219 |
| demo | ending | 有效 key | 0.17566% / 0.96953% | 0.0282893 / 0.1346108 |

全遮挡分支直接求均值，因此 beta starting 全遮挡组的 0.22803% 来自均值计算和 BF16 输出转换，与 Q/K 和 P/V 的残差分量数无关。有有效 key 的各组中，最坏相对 L2 为 0.21231%。这些结果对应 BF16 冻结抓取和算子级 CPU 回放。[完整分组统计](/tmp/af3-sage2pp-final.EAI90z/replay/run_summary.json)

在 AF3 仓库根目录运行下列命令可复现全量报告。将模块入口改为 `mod.triattn.sageattention2pp:attention_uncorrected`，并将 `--all` 改为 `--input theta --direction ending`，可复现表中的单项消融。每次使用新的输出目录。[运行说明](SAGEATTENTION2PP.md#本地精度回放)

```bash
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -B tests/triattn/verify.py --all \
  --triangle_attention_kernel=mod.triattn.sageattention2pp:attention \
  --output-dir=/tmp/triattn-sageattention2pp-numerics
```
