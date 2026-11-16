"""Durable sequential capture/validation queue; state and logs survive the chat."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from capture_trunk import ROOT, dump, now, sha


def execute(command, log, environment, state, save):
  with log.open('ab', buffering=0) as stream:
    process = subprocess.Popen(command, cwd=ROOT, env=environment,
                               stdin=subprocess.DEVNULL, stdout=stream,
                               stderr=subprocess.STDOUT)
    state['active_child_pid'] = process.pid
    state['active_command'] = command
    save()
    code = process.wait()
  state['last_process_exit_code'] = code
  state.pop('active_child_pid', None)
  save()
  if code:
    raise RuntimeError(f'Exit {code}; see {log}')


def complete_case(out):
  state = json.loads((out / 'state.json').read_text())
  expected = json.loads((out / 'expected.json').read_text())
  assert state['status'] == 'complete'
  memory = [json.loads(line) for line in (out / 'runtime_memory.jsonl').read_text().splitlines()]
  state['runtime_memory_summary'] = {
      'gpu_peak_MiB': max(x.get('gpu', {}).get('used_MiB', 0) for x in memory),
      'minimum_available_host_GiB': min(x['host']['MemAvailable'] for x in memory) / 2**20,
      'maximum_process_swap_GiB': max(x['process'].get('VmSwap', 0) for x in memory) / 2**20}
  state['validation_reports'] = {name: {'path': str(out / name), 'sha256': sha(out / name)}
      for name in ('full_tensor_validation.json', 'coverage_after_readable_names.json',
                   'value_validation.json', 'input_provenance.json')}
  dump(out / 'state.json', state)
  shutil.copy2(ROOT / 'scripts/validate_trunk_capture.py', out / 'validate_trunk_capture_source.py')
  i_step = expected.get('capture_attention_i_step')
  sampling_note = (f'Triangle attention 仅保存每方向全局 i={expected["selected_attention_i"]}；'
                   'Q/K/V/mask/result 只保留这些行，pair bias 保留完整矩阵。'
                   '模型计算全部 i，其他算子保持全量采集。\n\n' if i_step else '')
  attention_check = '首个保存的 i' if i_step else '首分片三个 i'
  (out / 'README.md').write_text(f'''# {state['input_name']}：recycle 10，blocks 1/16/32（已完成）

完整计算初始轮加 10 次 recycle，共 11×48 blocks；只采集 `recycle=10`、内部 `layer=0/15/31`，文件名 block 采用一基编号 1/16/32。seed=1，实际 token={state['actual_tokens']}，padding={state['bucket']}。

{sampling_note}MSA/模板复用 data 中序列和修饰精确匹配的已有 prepared 输入，逐链来源与哈希见 `input_provenance.json`。重新特征化；不运行搜索管线。模型内部 MSA/template 计算保留；diffusion/confidence/distogram 均不执行，无结构文件。

算子目录：`{Path(state['capture_run']).relative_to(out)}`，索引为 `events.jsonl`。共 {expected['events']} 个事件、{expected['arrays']} 个数组；原始 {state['captured_raw_array_bytes']/2**30:.4f} GiB，无损 NPZ {state['captured_npz_bytes']/2**30:.4f} GiB。另存 `trunk_outputs.npz`，包含最终 pair/single embeddings 和 target_feat。

全部数组通过解压、CRC、存储/逻辑 dtype、shape、NPY 长度、有限值和采集覆盖检查。每个选定 block 的五类算子代表数值复算通过，最大相对 L2 误差 {state['numerical_max_relative_l2']*100:.3f}%。triangle 检查三个 channel，grid attention 检查{attention_check}的全部 heads/query/key，single attention 检查全部 heads/query/key；不是所有事件全部元素的数值重执行。

GPU 使用 Unified Memory offload，池倍数 4.0，关闭预分配/GEMM 自动调优；transition 行分片=128，attention 使用默认分片。运行约 {state['elapsed_seconds']/60:.2f} 分钟（含编译、I/O、保存），退出码 0。日志、源码快照、配置、哈希和内存采样均保留。

BF16 按 uint16 位模式存储，用 `alphafold3.model.capture.load_event(path)` 恢复。`stage=trunk` 的 `diffusion_global_attention` 元数据对应共享实现的 single attention。
''')


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--batch-dir', type=Path, required=True)
  parser.add_argument('--resume', action='store_true', help='Resume only a queue waiting for a storage decision.')
  parser.add_argument('--wait-for-prior', action='store_true', help='Wait for the previous supervisor to reach its storage pause.')
  args = parser.parse_args()
  root = args.batch_dir.resolve()
  state = json.loads((root / 'batch_state.json').read_text())
  continuation = root / 'continuation_state.json'
  if args.wait_for_prior:
    if not args.resume:
      parser.error('--wait-for-prior requires --resume')
    dump(continuation, {'status': 'waiting_for_prior', 'pid': os.getpid(), 'started_at': now()})
    while state['status'] in ('queued', 'running'):
      time.sleep(30)
      state = json.loads((root / 'batch_state.json').read_text())
    if state['status'] != 'waiting_storage_decision':
      dump(continuation, {'status': 'not_started', 'prior_status': state['status'], 'finished_at': now()})
      return
    queue = json.loads((root / 'queue_config.json').read_text())
    free = shutil.disk_usage(root).free
    needed = queue['resume_min_free_bytes']
    if free < needed:
      dump(continuation, {'status': 'blocked_low_space', 'free_bytes': free,
                          'needed_bytes': needed, 'finished_at': now()})
      return
    for name in queue['resume_enable']:
      queue['enabled'][name] = True
    dump(root / 'queue_config.json', queue)
    state['scope']['capture_attention_i_steps'] = queue.get('capture_attention_i_steps', {})
    dump(continuation, {'status': 'starting', 'pid': os.getpid(), 'free_bytes': free})
  allowed = ('waiting_storage_decision',) if args.resume else ('queued',)
  if state['status'] not in allowed:
    raise ValueError('Batch is not queued; do not start a duplicate supervisor')
  state.update(status='running', supervisor_pid=os.getpid())
  state.setdefault('started_at', now())
  if args.resume:
    state['resumed_at'] = now()

  def save():
    state['updated_at'] = now()
    dump(root / 'batch_state.json', state)

  save()
  if args.wait_for_prior:
    dump(continuation, {'status': 'running', 'pid': os.getpid(), 'started_at': now()})
  environment = os.environ.copy()
  for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH',
              'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_ALLOCATOR'):
    environment.pop(key, None)
  environment.update(TF_FORCE_UNIFIED_MEMORY='true', XLA_CLIENT_MEM_FRACTION='4.0',
      XLA_PYTHON_CLIENT_PREALLOCATE='false',
      XLA_FLAGS='--xla_disable_hlo_passes=custom-kernel-fusion-rewriter --xla_gpu_autotune_level=0',
      AF3_CAPTURE_COMPRESS='1', AF3_CAPTURE_MIN_FREE_GB='10', OPENBLAS_NUM_THREADS='4')
  cpu = {**environment, 'JAX_PLATFORMS': 'cpu'}
  gpu = {**environment, 'JAX_PLATFORMS': 'cuda,cpu'}
  state['environment'] = {k: gpu[k] for k in ('JAX_PLATFORMS', 'TF_FORCE_UNIFIED_MEMORY',
      'XLA_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_FLAGS',
      'AF3_CAPTURE_COMPRESS', 'AF3_CAPTURE_MIN_FREE_GB')}
  save()
  try:
    for item in state['cases']:
      if item['status'] == 'complete':
        continue
      config = json.loads((root / 'queue_config.json').read_text())
      name = item['name']
      if not config['enabled'].get(name, False):
        item['status'] = 'waiting_storage_decision'
        save()
        continue
      if shutil.disk_usage(root).free < 10 * 1024**3:
        raise OSError('10 GiB disk reserve reached before next case')
      out = Path(item['output_dir'])
      out.parent.mkdir(parents=True, exist_ok=True)
      state['active_case'] = name
      item['status'] = 'preparing'
      save()
      command = [sys.executable, '-u', str(ROOT / 'scripts/capture_trunk.py'),
          '--prepare-only', '--input=' + item['input'], '--reuse-input=' + item['reuse_input'],
          '--output-dir=' + str(out), '--num-recycles=10', '--capture-recycles=10',
          '--capture-blocks=1,16,32', '--filename-block-base=1']
      step = config.get('capture_attention_i_steps', {}).get(name)
      if step is not None:
        command.append(f'--capture-attention-i-step={step}')
      execute(command, root / (name + '_prepare.log'), cpu, state, save)
      item['status'] = 'running'
      item['state'] = str(out / 'state.json')
      save()
      command = [sys.executable, '-u', str(ROOT / 'scripts/capture_trunk.py'),
          '--run-existing', '--transition-chunk-size=128', '--output-dir=' + str(out)]
      execute(command, out / 'inference.log', gpu, state, save)
      item['status'] = 'validating'
      save()
      for phase in ('archives', 'numerical', 'finalize'):
        command = [sys.executable, '-u', str(ROOT / 'scripts/validate_trunk_capture.py'),
                   str(out), '--phase=' + phase]
        execute(command, out / (phase + '_validation.log'), cpu, state, save)
      complete_case(out)
      item.update(status='complete', finished_at=now(), readme=str(out / 'README.md'))
      save()
    state.pop('active_case', None)
    state['status'] = ('complete' if all(c['status'] == 'complete' for c in state['cases'])
                       else 'waiting_storage_decision')
    state['finished_at'] = now()
    save()
    if args.wait_for_prior:
      dump(continuation, {'status': state['status'], 'pid': os.getpid(), 'finished_at': now()})
  except BaseException as error:
    state.update(status='failed', error=repr(error), finished_at=now())
    for item in state['cases']:
      if item['name'] == state.get('active_case') and item['status'] != 'complete':
        item['status'] = 'failed'
        item['error'] = repr(error)
    save()
    if args.wait_for_prior:
      dump(continuation, {'status': 'failed', 'pid': os.getpid(), 'error': repr(error), 'finished_at': now()})
    traceback.print_exc()
    raise


if __name__ == '__main__':
  main()
