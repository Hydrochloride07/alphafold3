# 原生 AF3 开发与 profiling

需要导出真实算子输入输出时，参见[张量抓取说明](tensor_capture.md)。

本方案使用当前 checkout 的 `uv.lock` 和 Python 3.12，不依赖 Docker。
本机为 Debian 13、约 16 GB RAM、GTX 1650 Mobile；完整推理以服务器为主。
代理看不到 GPU 设备，GPU 检查请在本机终端执行。

## 安装和检查

在仓库根目录执行：

```bash
bash scripts/native setup
bash scripts/native check
bash scripts/native run --help
bash scripts/native test
```

若官方 PyPI 下载缓慢，可使用已测速的镜像：

```bash
NATIVE_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/native setup
```

该路径先导出锁文件中的固定版本和哈希，以 `--require-hashes` 安装依赖，
再安装 editable 项目，最后运行 `uv sync --locked` 检查环境。它不更改
`uv.lock` 或项目的默认 registry。构建工具的依赖与普通安装一样由构建系统解析。

需要联网，以及 C++20 编译器、CMake >=3.28、zlib 开发头文件和 Python
开发头文件。uv 会提供 Python 3.12 及 Python 构建依赖，CMake 构建还会下载
第三方源码。默认限制两个并行编译任务；低内存时设置
`CMAKE_BUILD_PARALLEL_LEVEL=1`。所有 uv 管理命令建议通过脚本运行；直接
调用 uv 时先设置 `UV_CACHE_DIR="$PWD/.local/uv-cache"` 和
`UV_PYTHON_INSTALL_DIR="$PWD/.local/python"`。

脚本以 editable 模式安装项目：Python 修改下一次运行生效；C++ 修改执行
`bash scripts/native rebuild`，随后重新生成 CCD。`/.local/` 和 `/.venv/`
已在 `.gitignore` 排除。`build_data` 可能在源码包内产生 CCD 文件，保留它们
供本地运行；这些生成的数据也已在 `.gitignore` 中排除。

脚本清除 `PYTHONPATH`、`PYTHONHOME`、`LD_LIBRARY_PATH`，避免 Conda 与系统
CUDA 库污染。若服务器必须依赖自定义动态库，需按服务器配置调整这部分。
项目锁定 JAX CUDA 12 依赖，系统 CUDA 13.4 的存在不代表要升级项目 JAX。

## 无权重的模块验证

```bash
bash scripts/native benchmark --backend cpu --trace
bash scripts/native gpu-check
bash scripts/native benchmark --backend gpu --trace --output-dir .local/profiles/gpu-layernorm
```

benchmark 使用 AF3 自身的 LayerNorm、固定随机种子和 float32 数据，校验与
NumPy 参考实现的一致性。输出 `timing.json`；首次调用包含编译，后续计时
均等待设备完成，trace 在计时循环之外单独采集。该结果只代表此模块，不能
代替完整模型的性能。`trace` 目录中的 `perfetto_trace.json.gz` 可上传到
Perfetto。JAX trace 与 Nsight 建议分次采集，减少互相干扰。

## 本机与服务器的完整推理

服务器复制源码及锁文件，在服务器重新运行 setup，不复制 `.venv`。
权重通过 `--model_dir` 指定，不放进 Git。使用已准备 MSA 和模板的 AF3 JSON，
避免本机下载全量数据库；若使用空 MSA/模板，应明确这是不同输入条件。

```bash
bash scripts/native gpu-check
export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false'
bash scripts/native run \
  --json_path=/absolute/path/precomputed_input.json \
  --model_dir=/absolute/path/models \
  --output_dir=.local/output/baseline \
  --run_data_pipeline=false \
  --jax_compilation_cache_dir=.local/jax-cache
```

GPU 7.x（包括本机 GTX 1650 的架构）必须改用下面的设置，并给运行命令加上
`--flash_attention_implementation=xla`：

```bash
export XLA_FLAGS='--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'
```

本机 GPU 的显存及主机内存有限，不保证完整模型能够运行。先做模块实验；
遇到 OOM 保留诊断，不自动降低模型配置。CPU 推理需同时传入
`--jax_backend=cpu --flash_attention_implementation=xla`，也可能超出本机资源。

完整推理加 `--profile_dir=.local/profiles/inference` 可采集 JAX trace，
每次模型调用生成独立子目录，包含编译、计算与回传标注。输入设备传输在
采集开始之前，但异步传输可能延续到采样区间；trace 不应被解释为纯计算时间。
默认不采样，保持原有执行路径。该选项不自动重复模型或隐藏首次编译。

Nsight Systems 示例（先 setup，再直接使用项目 Python，继承相同运行环境）：

```bash
mkdir -p .local/profiles
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH \
  nsys profile --trace=cuda,nvtx,osrt --sample=none \
  -o .local/profiles/af3-system \
  .venv/bin/python run_alphafold.py \
  --json_path=/absolute/path/precomputed_input.json \
  --model_dir=/absolute/path/models \
  --output_dir=.local/output/nsys --run_data_pipeline=false
```

时间线中区分 CPU 编译、GPU 执行和数据传输。编译缓存不等于进程内预热；
稳态对比使用相同输入形状、种子和配置，记录 GPU、驱动、Git revision 及
`uv.lock`。先定位热点，再使用 `ncu --kernel-name <筛选表达式>` 针对目标
kernel 采样；GPU 性能计数器权限不足时保留报错，由服务器管理员配置。
本方案不修改驱动或系统性能计数器权限，也不提交服务器任务。

## 验收

- `check` 确认解释器、源码导入路径、C++ 扩展及 CCD。
- 小型输入处理测试通过；LayerNorm 校验通过并生成计时文件及 trace。
- GPU 验证在普通终端执行，CPU 结果不作为 GPU 性能证据。
- 服务器使用真实权重完成小输入推理，生成有效 GPU 时间线后，才算完成
  服务器阶段验收。服务器地址和权重路径未提供前，仅准备可复用的命令。
