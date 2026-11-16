"""Drain committed AF3 archives from a quota-limited Slurm home directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

from capture_trunk import ROOT, dump, now


POLL = '''
import json, subprocess, sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
def read(path):
 return json.loads(path.read_text()) if path.is_file() else None
s=read(root/'out/state.json') or {}
run=Path(s['capture_run']) if 'capture_run' in s else None
rows=[]
if run:
 assert run.is_relative_to(root/'out/tensors')
 for line in (run/'events.jsonl').read_text().splitlines(keepends=True):
  if line.endswith('\\n'): rows.append(json.loads(line))
direct=sys.argv[3]=='direct'
result=read(root/'job_result.json')
if direct:
 import os
 try: os.kill(int(sys.argv[2]),0); alive=True
 except ProcessLookupError: alive=False
 q='i1 GPU process alive' if alive else 'i1 GPU process exited'
 a=(f'{sys.argv[2]}|COMPLETED|0:0' if result and result['exit_code']==0 else
    f'{sys.argv[2]}|FAILED|1:0' if not alive and not result else '')
else:
 a=subprocess.run(['sacct','-X','-nP','-j',sys.argv[2],'-o','JobIDRaw,State,ExitCode'],text=True,capture_output=True).stdout.strip()
 q=subprocess.run(['squeue','-h','-j',sys.argv[2],'-o','%T|%R'],text=True,capture_output=True).stdout.strip()
print(json.dumps(dict(case=s.get('status'),run=run.name if run else None,rows=rows,
 result=result,slurm=q,accounting=a)))
'''


def member_digests(path):
  with zipfile.ZipFile(path) as archive:
    result = {}
    for name in archive.namelist():
      with archive.open(name) as member:
        result[name] = hashlib.file_digest(member, 'sha256').hexdigest()
    return result


def combine_recovery(first, second, second_run, state):
  previous = json.loads((first / 'remote_failure/state.json').read_text())
  previous_exit = json.loads((first / 'remote_failure/job_result.json').read_text())
  assert previous_exit['exit_code'] == 124
  for key in ('prepared_sha256', 'features_sha256', 'weights_sha256', 'runner_sha256',
              'base_git_revision', 'attention_implementation', 'gpu', 'seed',
              'bucket', 'num_recycles', 'blocks_per_pass'):
    assert previous[key] == state[key], f'Recovery execution differs: {key}'
  first_run = first / 'tensors' / Path(previous['capture_run']).name

  def records(run):
    result = []
    for line in (run / 'events.jsonl').read_text().splitlines():
      row = json.loads(line)
      meta = json.loads((run / row['json']).read_text())['metadata']
      key = tuple(meta.get(field) for field in ('recycle', 'layer', 'operator', 'direction', 'sampled_i'))
      result.append((key, row, run))
    return result

  old, new = records(first_run), records(second_run)
  assert len(old) == 47 and len(new) == 23
  old_index, new_index = {k: r for k, r, _ in old}, {k: r for k, r, _ in new}
  assert len(old_index) == 47 and len(new_index) == 23
  overlap = old_index.keys() & new_index.keys()
  assert overlap == {(10, 31, 'triangle_multiplication', 'outgoing', None)}
  anchor = next(iter(overlap))
  digests = member_digests(first_run / old_index[anchor]['npz'])
  assert digests == member_digests(second_run / new_index[anchor]['npz']), 'Recovery overlap is not bit-identical'
  combined = second / 'combined'
  combined.mkdir(exist_ok=False)
  run = combined / 'tensors' / ('run-' + uuid.uuid4().hex)
  (run / 'events').mkdir(parents=True)
  selected = old + [record for record in new if record[0] not in overlap]
  assert len(selected) == 69
  with (run / 'events.jsonl').open('w') as index:
    for _, row, origin in selected:
      os.link(origin / row['npz'], run / row['npz'])
      record = json.loads((origin / row['json']).read_text())
      record['metadata']['source_capture_run'] = str(origin)
      dump(run / row['json'], record)
      index.write(json.dumps(row) + '\n')
  proof = {'initial_run': str(first_run), 'initial_exit': 124,
           'replay_run': str(second_run), 'replay_exit': 0,
           'initial_events_used': 47, 'replay_events_used': 22,
           'overlap_identity': list(anchor), 'overlap_member_sha256': digests,
           'overlap_bit_identical': True, 'trunk_outputs_from_replay': True}
  dump(combined / 'recovery_provenance.json', proof)
  manifest = json.loads((second_run / 'manifest.json').read_text())
  manifest.update(event_count=69, recovery_provenance=proof,
                  filters=json.loads((first / 'capture_config.json').read_text()))
  dump(run / 'manifest.json', manifest)
  for name in ('expected.json', 'capture_config.json'):
    shutil.copy2(first / name, combined / name)
  for name in ('input.json', 'input_provenance.json', 'features_seed-1.npz',
               'features_seed-1.json', 'trunk_outputs.npz', 'source.diff',
               'bundle_provenance.json', 'cluster_state.json'):
    os.link(second / name, combined / name)
  state.update(capture_run=str(run), capture_event_count=69, capture_config=manifest['filters'],
               recovery_provenance=str(combined / 'recovery_provenance.json'))
  dump(combined / 'state.json', state)
  return combined, run


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--ssh-config', type=Path, required=True)
  parser.add_argument('--remote', required=True)
  parser.add_argument('--job-id', required=True)
  parser.add_argument('--direct', action='store_true', help='Collect a user-authorized direct i1 run, keyed by PID.')
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--local-batch-state', type=Path, required=True)
  parser.add_argument('--download-only', action='store_true', help='Download a verified execution without running local archive validation yet.')
  parser.add_argument('--recover-from', type=Path, help='Combine the timed-out initial run with a bit-checked block-32 replay.')
  args = parser.parse_args()
  assert re.fullmatch(r'[a-zA-Z0-9_/-]+', args.remote) and '..' not in args.remote
  assert args.job_id.isdigit()
  out = args.output_dir.resolve()
  expected_events = json.loads((out / 'expected.json').read_text())['events']
  ssh = ['ssh', '-T', '-F', str(args.ssh_config.resolve())]
  peer = 'flashbias-scc'
  rsync = ['rsync', '-a', '--timeout=600', '--partial-dir=.rsync-partial', '-e', shlex.join(ssh)]
  status = {'job_id': args.job_id, 'remote': args.remote, 'status': 'waiting', 'started_at': now()}
  status_file = out / 'collector_state.json'

  def save():
    status['updated_at'] = now()
    dump(status_file, status)

  save()
  consecutive_errors = 0
  try:
    while True:
      try:
        reply = subprocess.check_output(ssh + [peer, shlex.join(['python3.12', '-c', POLL, args.remote, args.job_id,
            'direct' if args.direct else 'slurm'])], text=True, timeout=60)
        progress = json.loads(reply)
        status.update(slurm=progress['slurm'], accounting=progress['accounting'], remote_case=progress['case'])
        rows = progress['rows']
        run_name = progress['run']
        if run_name:
          assert re.fullmatch(r'run-[0-9a-f]{32}', run_name)
          run = out / 'tensors' / run_name
          run.mkdir(parents=True, exist_ok=True)
          pending = []
          for row in rows:
            for extension in ('json', 'npz'):
              rel = row[extension]
              assert rel == f'events/{row["event_id"]}.{extension}'
              assert re.fullmatch(r'events/[0-9a-f]{32}\.(json|npz)', rel)
              if not (run / rel).is_file():
                pending.append(rel)
          if pending:
            # rsync verifies each transfer before removing that committed source.
            subprocess.run(rsync + ['--checksum', '--remove-source-files', '--from0', '--files-from=-',
                f'{peer}:{args.remote}/out/tensors/{run_name}/', str(run) + '/'],
                input=b'\0'.join(x.encode() for x in pending) + b'\0', check=True)
          collected = [row for row in rows if all((run / row[k]).is_file() for k in ('npz', 'json'))]
          status['collected_events'] = len(collected)
          index_tmp = run / 'events.jsonl.tmp'
          index_tmp.write_text(''.join(json.dumps(row) + '\n' for row in collected))
          index_tmp.replace(run / 'events.jsonl')
        result = progress['result']
        if result:
          assert result['job_id'] == args.job_id
        if result and result['exit_code']:
          raise RuntimeError(f"Remote runner exited {result['exit_code']}")
        job_rows = [line.split('|') for line in progress['accounting'].splitlines()]
        terminal = next((x for x in job_rows if x[0] == args.job_id and len(x) >= 3), None)
        if terminal and terminal[1].split()[0].rstrip('+') in {
            'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}:
          raise RuntimeError(f'Remote run failed: {terminal}')
        if result and result['exit_code'] == 0 and terminal and terminal[1] == 'COMPLETED' and terminal[2] == '0:0':
          assert run_name and status['collected_events'] == len(rows) == expected_events
          subprocess.run(rsync + ['--exclude=/tensors/', f'{peer}:{args.remote}/out/', str(out) + '/'], check=True)
          subprocess.run(rsync + [f'{peer}:{args.remote}/out/tensors/{run_name}/manifest.json',
              f'{peer}:{args.remote}/out/tensors/{run_name}/events.jsonl', str(run) + '/'], check=True)
          break
        status['status'] = 'collecting' if run_name else ('waiting_for_i1' if args.direct else 'waiting_for_slurm')
        consecutive_errors = 0
        status.pop('last_transfer_error', None)
        save()
        time.sleep(20 if run_name else 60)
      except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        consecutive_errors += 1
        status.update(status='retrying_transfer', last_transfer_error=repr(error))
        save()
        if consecutive_errors >= 10:
          raise
        time.sleep(30)
    state = json.loads((out / 'state.json').read_text())
    shutil.copy2(out / 'state.json', out / 'cluster_state.json')
    state.update(capture_run=str(run), cluster_job_id=args.job_id, cluster_remote=args.remote,
                 cluster_execution='i1_direct' if args.direct else 'slurm')
    dump(out / 'state.json', state)
    if args.recover_from:
      status['status'] = 'checking_recovery_overlap'
      save()
      out, run = combine_recovery(args.recover_from.resolve(), out, run, state)
      status['final_output_dir'] = str(out)
    if args.download_only:
      status.update(status='download_complete', finished_at=now())
      save()
      return
    # Do not compete for host RAM with the local managed-memory AF3 job.
    while json.loads(args.local_batch_state.read_text())['status'] in ('queued', 'running'):
      status['status'] = 'download_complete_waiting_for_local_ram'
      save()
      time.sleep(60)
    status['status'] = 'validating'
    save()
    env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'OPENBLAS_NUM_THREADS': '4'}
    for phase in ('archives', 'numerical', 'finalize'):
      with (out / f'{phase}_validation.log').open('ab') as log:
        subprocess.run([sys.executable, str(ROOT / 'scripts/validate_trunk_capture.py'), str(out), '--phase=' + phase],
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    final = json.loads((out / 'state.json').read_text())
    label = 'i1 进程 PID' if args.direct else 'Slurm 作业'
    replay_note = ('首次尝试超时后，用相同输入、权重和软件重算完整 trunk，仅补抓 block 32。'
                   '重合算子全部 NPY 成员逐字节一致，合并索引含首次 47 个和复跑 22 个事件；'
                   '两次来源及最终 embeddings 来源见 `recovery_provenance.json`。\n\n' if args.recover_from else '')
    (out / 'README.md').write_text(f'''# demo：集群 trunk 采集（已验收）

{label} {args.job_id}，GPU {final['gpu']}。完整执行 11×48 个 trunk blocks，最终采集覆盖 recycle 10、block 1/16/32。关闭搜索和 diffusion/confidence/distogram。

{replay_note}Triangle attention 计算全部 i，计算分片 32，只保存每方向 i=0,256,…,2304 的 Q/K/V/mask/result，保留完整 bias；triangle multiplication 与 single attention 不抽样。

69 个事件、420 个数组，原始 51.8154 GiB，压缩 NPZ {final['captured_npz_bytes']/2**30:.3f} GiB。全部完整性检查和代表数值复算通过，最大相对 L2 误差 {final['numerical_max_relative_l2']*100:.3f}%。最终输出为 `trunk_outputs.npz`。

张量索引：`{run.relative_to(out)}/events.jsonl`。输入与已验证数值特征来源见 `input_provenance.json`；集群推理状态见 `cluster_state.json`，本地验收状态见 `state.json`。
''')
    status.update(status='complete', finished_at=now())
    save()
  except BaseException as error:
    status.update(status='failed', error=repr(error), finished_at=now())
    save()
    raise


if __name__ == '__main__':
  main()
