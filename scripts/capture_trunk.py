"""Prepare an inference-only input, then capture selected trunk passes/blocks.

Run --prepare-only on CPU, followed by --run-existing on the GPU. No search
pipeline is called. Existing attempts are never overwritten.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUCKETS = (128, 256, 384, 512, 768, 1024, 1280, 1536, 2048, 2560,
           3072, 3584, 4096, 4608, 5120)


def storage_estimate(tokens, bucket, recycles, blocks, num_recycles, attention_i_step=None):
  chunk = 128 if bucket <= 1536 else 32
  if attention_i_step is not None and attention_i_step < chunk:
    raise ValueError('Capture i-step must be at least the attention compute chunk size')
  sampled_i = list(range(0, bucket, attention_i_step)) if attention_i_step else None
  chunks = len(sampled_i) if sampled_i is not None else (bucket + chunk - 1) // chunk
  exported_rows = len(sampled_i) if sampled_i is not None else bucket
  executions = len(recycles) * len(blocks)
  mul = (10 * 128 + 2) * bucket**2
  attention = (8 * 128 + 1) * exported_rows * bucket + 2 * chunks * 4 * bucket**2
  single = 12 * bucket * 384 + 6 * 16 * bucket**2 + 6 * bucket
  counts = [('triangle_multiplication', 'outgoing', executions),
            ('triangle_multiplication', 'incoming', executions),
            ('attention', 'starting', executions * chunks),
            ('attention', 'ending', executions * chunks),
            ('diffusion_global_attention', None, executions)]
  return {'actual_tokens': tokens, 'padded_tokens': bucket,
          'num_recycles': num_recycles, 'trunk_passes': num_recycles + 1,
          'blocks_per_pass': 48, 'selected_recycles': recycles,
          'selected_layers': [b - 1 for b in blocks],
          'capture_attention_i_step': attention_i_step, 'selected_attention_i': sampled_i,
          'attention_chunk_size': chunk, 'events': executions * (3 + 2 * chunks),
          'arrays': executions * (20 + 12 * chunks),
          'raw_array_bytes': executions * (2 * mul + 2 * attention + single),
          'final_trunk_raw_bytes': 4 * (bucket**2 * 128 + bucket * (384 + 447)),
          'operators': [{'operator': op, 'direction': direction, 'events': count}
                        for op, direction, count in counts]}


def now():
  return datetime.datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()


def sha(path):
  h = hashlib.sha256()
  with Path(path).open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024**2), b''):
      h.update(block)
  return h.hexdigest()


def dump(path, value):
  path = Path(path)
  tmp = path.with_suffix(path.suffix + '.tmp')
  tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
  tmp.replace(path)


def prepare(args):
  import numpy as np
  from alphafold3.common import folding_input
  from alphafold3.constants import chemical_components
  from alphafold3.data import featurisation
  from alphafold3.model.components import utils

  out = args.output_dir
  out.mkdir(parents=True, exist_ok=False)
  original = json.loads(args.input.read_text())
  tokens = sum(len(chain.get('sequence', '')) *
               (len(chain['id']) if isinstance(chain['id'], list) else 1)
               for entry in original['sequences'] for chain in entry.values())
  bucket = next(b for b in BUCKETS if b >= tokens)
  recycles = sorted(set(int(v) for v in args.capture_recycles.split(',')))
  blocks = sorted(set(int(v) for v in args.capture_blocks.split(',')))
  if any(r < 0 or r > args.num_recycles for r in recycles):
    raise ValueError('Selected recycle exceeds configured execution')
  if any(b < 1 or b > 48 for b in blocks):
    raise ValueError('Pairformer blocks must use one-based indices 1..48')
  reused = json.loads(args.reuse_input.read_text())
  prepared = json.loads(json.dumps(original))
  provenance = []
  for entry in prepared['sequences']:
    kind, chain = next(iter(entry.items()))
    if kind != 'protein':
      continue
    matches = [s[kind] for s in reused['sequences'] if kind in s
               and s[kind]['sequence'] == chain['sequence']
               and s[kind].get('modifications', []) == chain.get('modifications', [])]
    if not matches:
      raise ValueError(f'No exact MSA/template source for chain {chain["id"]}')
    fields = [k for k in ('unpairedMsa', 'pairedMsa', 'templates') if chain.get(k) is None]
    for key in fields:
      if matches[0].get(key) is None:
        raise ValueError(f'Missing source field {key}')
      if any(m.get(key) != matches[0][key] for m in matches):
        raise ValueError('Ambiguous preprocessing sources')
      chain[key] = matches[0][key]
    provenance.append({'chain_ids': chain['id'], 'fields': fields,
                       'source': str(args.reuse_input),
                       'source_sha256': sha(args.reuse_input),
                       'source_chain_ids': matches[0]['id'],
                       'sequence_sha256': hashlib.sha256(chain['sequence'].encode()).hexdigest()})
  stripped = json.loads(json.dumps(prepared))
  for entry, source in zip(stripped['sequences'], original['sequences'], strict=True):
    kind = next(iter(entry))
    for key in ('unpairedMsa', 'pairedMsa', 'templates'):
      if key not in source[kind]:
        entry[kind].pop(key, None)
  assert stripped == original, 'Original non-preprocessing input changed'
  dump(out / 'input.json', prepared)
  history = ROOT.parent / 'data/af3_capture/prepared_provenance.json'
  origin = json.loads(history.read_text())
  dump(out / 'input_provenance.json', {
      'input': str(args.input), 'input_sha256': sha(args.input),
      'prepared_sha256': sha(out / 'input.json'), 'reused': provenance,
      'upstream_provenance': origin, 'upstream_provenance_sha256': sha(history),
      'original_fields_preserved': True, 'search_pipeline_executed': False})
  config = {'stage': ['trunk'], 'pairformer_blocks': blocks, 'pairformer_recycles': recycles}
  if args.capture_attention_i_step is not None:
    config['pairformer_attention_i_step'] = [args.capture_attention_i_step]
  dump(out / 'capture_config.json', config)
  dump(out / 'expected.json', storage_estimate(tokens, bucket, recycles, blocks, args.num_recycles,
                                               args.capture_attention_i_step))
  state = {'status': 'featurising', 'started_at': now(), 'input': str(args.input),
           'input_name': prepared['name'], 'seed': 1, 'num_recycles': args.num_recycles,
           'trunk_passes': args.num_recycles + 1, 'blocks_per_pass': 48, 'bucket': bucket,
           'expected_tokens': tokens, 'filename_block_base': args.filename_block_base,
           'search_pipeline_executed': False, 'diffusion_executed': False,
           'confidence_executed': False, 'distogram_executed': False,
           'capture_config': config, 'branch': subprocess.check_output(
               ['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()}
  assert state['branch'] == '数据抓取'
  dump(out / 'state.json', state)
  fold = folding_input.Input.from_json((out / 'input.json').read_text())
  assert list(fold.rng_seeds) == [1]
  batches = featurisation.featurise_input(
      fold, chemical_components.Ccd(user_ccd=fold.user_ccd),
      buckets=BUCKETS, verbose=True)
  batch = utils.remove_invalidly_typed_feats(batches[0])
  descriptors = {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in batch.items()}
  np.savez_compressed(out / 'features_seed-1.npz', **batch)
  with np.load(out / 'features_seed-1.npz', allow_pickle=False) as arrays:
    assert set(arrays.files) == set(batch)
    for k, v in batch.items():
      np.testing.assert_array_equal(arrays[k], v)
  assert int(np.asarray(batch['seq_mask']).sum()) == tokens
  assert batch['seq_mask'].shape == (bucket,)
  state.update(status='prepared', prepared_at=now(), actual_tokens=tokens,
               prepared_sha256=sha(out / 'input.json'),
               features_sha256=sha(out / 'features_seed-1.npz'))
  dump(out / 'features_seed-1.json', descriptors)
  dump(out / 'state.json', state)
  print('Prepared:', out, flush=True)


def managed_probe(device):
  import jax
  import jax.numpy as jnp
  buffer = jax.device_put(jnp.ones((1024,), dtype=jnp.float32), device)
  buffer.block_until_ready()
  driver = ctypes.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
  driver.cuInit.argtypes = [ctypes.c_uint]
  driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
  assert driver.cuInit(0) == 0
  managed = ctypes.c_uint()
  status = driver.cuPointerGetAttribute(ctypes.byref(managed), 8,
                                        buffer.unsafe_buffer_pointer())
  assert status == 0 and managed.value == 1, (status, managed.value)
  return {'attribute': 'CU_POINTER_ATTRIBUTE_IS_MANAGED', 'query_status': status,
          'buffer_is_managed': True}


def monitor_memory(out, stopped):
  with (out / 'runtime_memory.jsonl').open('a', buffering=1) as stream:
    while not stopped.is_set():
      record = {'time': now(), 'pid': os.getpid()}
      for filename, key in (('/proc/self/status', 'process'), ('/proc/meminfo', 'host')):
        lines = Path(filename).read_text().splitlines()
        record[key] = {k: int(v.split()[0]) for line in lines if ':' in line
                       for k, v in [line.split(':', 1)]
                       if k in {'VmRSS', 'VmSwap', 'MemAvailable', 'SwapFree'}}
      try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'], text=True, timeout=5)
        used, free, utilization = map(int, result.strip().split(','))
        record['gpu'] = {'used_MiB': used, 'free_MiB': free, 'utilization_percent': utilization}
      except Exception as error:
        record['gpu_error'] = repr(error)
      stream.write(json.dumps(record) + '\n')
      stopped.wait(5)


def run(args):
  import jax
  import numpy as np
  from absl import flags
  from alphafold3.model import capture
  import run_alphafold

  if not flags.FLAGS.is_parsed():
    flags.FLAGS(['capture_trunk'])
  out = args.output_dir
  state = json.loads((out / 'state.json').read_text())
  if state['status'] != 'prepared':
    raise ValueError(f'Attempt is not prepared: {state["status"]}')
  assert sha(out / 'input.json') == state['prepared_sha256']
  assert sha(out / 'features_seed-1.npz') == state['features_sha256']
  bundle = json.loads(args.bundle_provenance.read_text()) if args.bundle_provenance else None
  if bundle is None:
    assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT,
                                   text=True).strip() == '数据抓取'
  else:
    assert bundle['branch'] == '数据抓取'
  assert shutil.disk_usage(out).free > 10 * 1024**3
  state.update(status='initialising_gpu', pid=os.getpid(), inference_started_at=now(),
               environment={k: os.environ.get(k) for k in (
                   'JAX_PLATFORMS', 'TF_FORCE_UNIFIED_MEMORY', 'XLA_CLIENT_MEM_FRACTION',
                   'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_FLAGS', 'AF3_CAPTURE_COMPRESS',
                   'AF3_CAPTURE_MIN_FREE_GB')})
  dump(out / 'state.json', state)
  start = time.monotonic()
  stopped = threading.Event()
  monitor = threading.Thread(target=monitor_memory, args=(out, stopped), daemon=True)
  monitor.start()
  try:
    weights = ROOT.parent / 'data/af3.bin'
    state['weights'] = str(weights)
    state['weights_sha256'] = sha(weights)
    assert state['weights_sha256'] == 'f4e06602897983e5d0212699c3c572e395354408b8739f62bc5fee086627d151'
    device = jax.devices('gpu')[0]
    state['gpu'] = device.device_kind
    state['offload_verified'] = managed_probe(device) if not args.native_gpu else None
    if bundle is None:
      state['base_git_revision'] = subprocess.check_output(
          ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
      (out / 'source.diff').write_bytes(subprocess.check_output(['git', 'diff'], cwd=ROOT))
    else:
      state['base_git_revision'] = bundle['base_git_revision']
      for name, digest in bundle['source_sha256'].items():
        assert sha(ROOT / name) == digest, f'Staged source changed: {name}'
      shutil.copy2(args.bundle_provenance, out / 'bundle_provenance.json')
      shutil.copy2(args.bundle_provenance.with_name('source.diff'), out / 'source.diff')
    shutil.copy2(__file__, out / 'capture_trunk_source.py')
    state['runner_sha256'] = sha(__file__)
    dump(out / 'state.json', state)
    print('GPU:', state['gpu'], 'managed:', state['offload_verified'], flush=True)
    state['attention_implementation'] = args.attention_implementation
    config = run_alphafold.make_model_config(
        flash_attention_implementation=args.attention_implementation,
        num_recycles=state['num_recycles'])
    config.stop_after_trunk = True
    if args.transition_chunk_size is not None:
      if args.transition_chunk_size < 1:
        raise ValueError('Transition chunk size must be positive')
      config.global_config.pair_transition_shard_spec = ((None, args.transition_chunk_size),)
    state['pair_transition_shard_spec'] = config.global_config.pair_transition_shard_spec
    state['pair_attention_chunk_size'] = config.global_config.pair_attention_chunk_size
    dump(out / 'state.json', state)
    runner = run_alphafold.ModelRunner(config, device, weights.parent)
    _ = runner.model_params
    with np.load(out / 'features_seed-1.npz', allow_pickle=False) as archive:
      batch = {k: archive[k] for k in archive.files}
    descriptors = json.loads((out / 'features_seed-1.json').read_text())
    assert set(batch) == set(descriptors)
    for k, v in batch.items():
      assert list(v.shape) == descriptors[k]['shape'] and str(v.dtype) == descriptors[k]['dtype']
    state['actual_tokens'] = int(batch['seq_mask'].sum())
    assert state['actual_tokens'] == state.get('expected_tokens', 1257)
    assert batch['seq_mask'].shape == (state['bucket'],)
    sink = capture.CaptureSession(out / 'tensors', state['capture_config'])
    state.update(status='running', capture_run=str(sink.path))
    dump(out / 'state.json', state)
    with sink.activate():
      result = runner.run_inference(batch, jax.random.PRNGKey(1),
                                    input_name=state['input_name'], seed=1)
    arrays = {k: result[k] for k in ('pair_embeddings', 'single_embeddings', 'target_feat')}
    assert set(result) == {*arrays, '__identifier__'}
    assert arrays['pair_embeddings'].shape == (state['bucket'], state['bucket'], 128)
    assert arrays['single_embeddings'].shape == (state['bucket'], 384)
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(out / 'trunk_outputs.npz', **arrays)
    with np.load(out / 'trunk_outputs.npz', allow_pickle=False) as stored:
      for k, v in arrays.items():
        np.testing.assert_array_equal(stored[k], v)
    state.update(status='inference_complete', finished_at=now(),
                 elapsed_seconds=time.monotonic() - start, process_exit_code=0,
                 capture_event_count=sink.manifest['event_count'],
                 outputs={k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()},
                 outputs_sha256=sha(out / 'trunk_outputs.npz'))
    dump(out / 'state.json', state)
    print('Inference complete:', state['elapsed_seconds'], 'seconds', flush=True)
  except BaseException as error:
    state.update(status='failed', finished_at=now(), error=repr(error),
                 elapsed_seconds=time.monotonic() - start, process_exit_code=1)
    dump(out / 'state.json', state)
    traceback.print_exc()
    raise
  finally:
    stopped.set()
    monitor.join(timeout=10)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--input', type=Path, default=ROOT.parent / 'data/processed/demo_cut.json')
  parser.add_argument('--reuse-input', type=Path, default=ROOT.parent / 'data/af3_capture/prepared/demo.json')
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--transition-chunk-size', type=int,
                      help='Bound row-independent transition working memory; attention chunks are unchanged.')
  parser.add_argument('--num-recycles', type=int, default=0)
  parser.add_argument('--capture-recycles', default='0', help='Comma-separated zero-based recycle indices.')
  parser.add_argument('--capture-blocks', default='1', help='Comma-separated one-based Pairformer blocks.')
  parser.add_argument('--capture-attention-i-step', type=int,
                      help='Export only globally strided i rows of trunk grid attention; leave computation intact.')
  parser.add_argument('--filename-block-base', type=int, choices=(0, 1), default=0)
  parser.add_argument('--bundle-provenance', type=Path,
                      help='Verified source manifest for a cluster source snapshot without Git.')
  parser.add_argument('--native-gpu', action='store_true', help='Use a GPU without managed-memory offload.')
  parser.add_argument('--attention-implementation', choices=('xla', 'triton', 'xla_chunked'), default='xla')
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument('--prepare-only', action='store_true')
  mode.add_argument('--run-existing', action='store_true')
  args = parser.parse_args()
  prepare(args) if args.prepare_only else run(args)


if __name__ == '__main__':
  main()
