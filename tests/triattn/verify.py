#!/usr/bin/env python3
"""Replay AF3 triangle attention captures against a NumPy FP64 reference."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys

from absl import flags
from alphafold3.model.components import triangle_attention
import jax
import jax.numpy as jnp
import numpy as np


RECYCLE, LAYER = 10, 15
QKV = ('query', 'key', 'value')
DIRECTIONS = {'starting': 'pair_attention1', 'ending': 'pair_attention2'}
BATCH = 'trunk_recycle10_blocks1_16_32/batch-01'
DEFAULT_CAPTURE_ROOT = Path(__file__).resolve().parent / 'data'
CAPTURE = 'capture_vs_fp64'
CANDIDATE = 'candidate_vs_fp64'


@dataclass(frozen=True)
class CaptureRun:
  attempt: str
  run_id: str
  tokens: int
  chunk: int
  sample_step: int | None = None


RUNS = {
    'theta': CaptureRun(
        f'{BATCH}/theta/attempt-01', 'run-e83d93cb6a3244a38738e08f4d6583ad',
        128, 128,
    ),
    'beta': CaptureRun(
        'beta_trunk_offload/attempt-01', 'run-dddc2d948b7a4fbeb5b8c07d48aa87dc',
        512, 128,
    ),
    'demo': CaptureRun(
        f'{BATCH}/demo-cluster/attempt-04', 'run-059dbb116bd240fba1eb240118bb587c',
        2560, 32, 256,
    ),
}
BOUNDARIES = {
    'theta': {31, 32, 33, 69, 70},
    'beta': {127, 128, 129, 255, 256, 257, 393, 394},
    'demo': set(),
}
CSV_FIELDS = (
    'input', 'direction', 'capture_backend', 'source_event', 'chunk_start',
    'chunk_end', 'i', 'i_local', 'head', 'key_count', 'mask_group', 'boundary',
    'comparison', 'relative_l2', 'max_abs', 'worst_query', 'worst_channel',
    'nonfinite_count',
)


@dataclass(frozen=True)
class Event:
  path: Path
  record: dict
  record_digest: str
  indices: tuple[int, ...]
  token_count: int


def discover(root, names, directions):
  """Selects the accepted runs and validates each requested direction."""
  events, manifests = [], {}
  for name in names:
    run = RUNS[name]
    run_dir = (root / run.attempt / 'tensors' / run.run_id).resolve(strict=True)
    index = run_dir / 'events.jsonl'
    index_bytes = index.read_bytes()
    manifests[index] = hashlib.sha256(index_bytes).hexdigest()
    groups = {direction: [] for direction in directions}
    seen_ids, seen_paths = set(), set()
    for line in index_bytes.decode('utf-8').splitlines():
      row = json.loads(line)
      sidecar = (run_dir / row['json']).resolve(strict=True)
      path = (run_dir / row['npz']).resolve()
      if (sidecar.parent != run_dir / 'events'
          or path.with_suffix('.json') != sidecar or path.suffix != '.npz'):
        raise ValueError(f'Invalid manifest paths: {sidecar}')
      record_bytes = sidecar.read_bytes()
      record = json.loads(record_bytes)
      meta = record['metadata']
      direction = meta.get('direction')
      if (meta.get('stage'), meta.get('operator'), meta.get('recycle'),
          meta.get('layer')) != ('trunk', 'attention', RECYCLE, LAYER):
        continue
      if direction not in groups:
        continue
      if (
          record.get('schema_version') != 1
          or record.get('event_id') != row['event_id']
          or meta.get('input_name') != name
          or meta.get('seed') != 1
          or meta.get('module_path') != f'trunk_pairformer/{DIRECTIONS[direction]}'
          or meta.get('layout') != '...qhc'
          or meta.get('chunk_axis') != 0
          or meta.get('chunk_total') != run.tokens
          or not math.isfinite(meta['scale'])
          or meta['scale'] <= 0
          or not meta.get('implementation')
      ):
        raise ValueError(f'Unexpected event metadata: {sidecar}')
      start, end = meta['chunk_start'], meta['chunk_end']
      if (
          not 0 <= start < end <= run.tokens
          or start % run.chunk
          or end != min(start + run.chunk, run.tokens)
      ):
        raise ValueError(f'Invalid compute interval: {sidecar}')
      sampled_i = meta.get('sampled_i')
      if run.sample_step is None:
        if sampled_i is not None:
          raise ValueError(f'Unexpected sparse event: {sidecar}')
        indices = tuple(range(start, end))
      else:
        if (
            type(sampled_i) is not int
            or not start <= sampled_i < end
            or meta.get('i_step') != run.sample_step
        ):
          raise ValueError(f'Invalid sampled_i: {sidecar}')
        indices = (sampled_i,)
      if path in seen_paths or row['event_id'] in seen_ids:
        raise ValueError(f'Duplicate event: {path}')
      seen_ids.add(row['event_id'])
      seen_paths.add(path)
      groups[direction].append(Event(
          path, record, hashlib.sha256(record_bytes).hexdigest(), indices, run.tokens
      ))
    for direction, group in groups.items():
      group.sort(key=lambda event: event.indices[0])
      observed = tuple(i for event in group for i in event.indices)
      expected = tuple(range(0, run.tokens, run.sample_step or 1))
      if observed != expected:
        raise ValueError(f'{name}/{direction}: missing, overlapping or unexpected i')
      events.extend(group)
  return events, manifests


def load_event(event):
  n, batch = event.token_count, len(event.indices)
  shapes = {name: (batch, n, 4, 32) for name in (*QKV, 'result')}
  shapes.update(bias=(1, 4, n, n), mask=(batch, 1, 1, n))
  arrays = {}
  with np.load(event.path, allow_pickle=False) as archive:
    for name, shape in shapes.items():
      value = archive[name]
      dtype = 'bool' if name == 'mask' else 'uint16'
      logical = 'bool' if name == 'mask' else 'bfloat16'
      descriptor = dict(shape=list(shape), dtype=logical, storage_dtype=dtype)
      if (
          value.shape != shape
          or value.dtype.name != dtype
          or event.record['tensors'][name] != descriptor
      ):
        raise ValueError(f'{event.path}: invalid {name} shape or dtype')
      value = value.astype(dtype, copy=False)
      if dtype == 'uint16' and np.any((value & np.uint16(0x7f80)) == 0x7f80):
        raise ValueError(f'{event.path}: nonfinite {name}')
      arrays[name] = value
  return arrays


def decode_bfloat16(bits):
  return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def reference_head(query, key, value, bias, keep, scale, query_tile=256):
  """Evaluates stable FP64 softmax across all valid keys for each query tile."""
  if not keep.any():
    return np.broadcast_to(value.mean(axis=0, dtype=np.float64), query.shape)
  q = query.astype(np.float64)
  k = key[keep].astype(np.float64)
  v = value[keep].astype(np.float64)
  out = np.empty(q.shape, dtype=np.float64)
  for start in range(0, len(q), query_tile):
    tile = slice(start, start + query_tile)
    logits = q[tile] @ k.T
    logits *= scale
    logits += bias[tile][:, keep]
    logits -= logits.max(axis=-1, keepdims=True)
    np.exp(logits, out=logits)
    logits /= logits.sum(axis=-1, keepdims=True)
    out[tile] = logits @ v
  return out


def metrics(expected, actual):
  nonfinite = ~np.isfinite(actual)
  count = int(np.count_nonzero(nonfinite))
  if count:
    row, col = np.unravel_index(nonfinite.argmax(), actual.shape)
    relative_l2, max_abs = None, None
  else:
    gap = np.abs(actual.astype(np.float64) - expected)
    row, col = np.unravel_index(gap.argmax(), gap.shape)
    error_l2 = float(np.linalg.norm(gap))
    ref_l2 = float(np.linalg.norm(expected))
    relative_l2 = error_l2 / ref_l2 if ref_l2 else (None if error_l2 else 0.0)
    max_abs = float(gap[row, col])
  return dict(
      relative_l2=relative_l2, max_abs=max_abs, worst_query=int(row),
      worst_channel=int(col), nonfinite_count=count,
  )


def make_replayer(implementation, kernel_spec):
  @functools.partial(jax.jit, static_argnames=('scale',))
  def run(query, key, value, bias, mask, *, scale):
    q, k, v, b = (
        jax.lax.bitcast_convert_type(bits, jnp.bfloat16)
        for bits in (query, key, value, bias)
    )
    return triangle_attention.attention(
        q, k, v, bias=b, mask=mask, scale=scale, implementation=implementation,
        kernel_spec=kernel_spec,
    )

  return run


def replay(event, run):
  raw = load_event(event)
  meta = event.record['metadata']
  scale = float(meta['scale'])
  # One JIT invocation on the complete stored event, including the ending layout.
  result = np.asarray(jax.device_get(run(
      **{name: raw[name] for name in (*QKV, 'bias', 'mask')}, scale=scale,
  )), dtype=np.float32)
  q, k, v, bias, captured = (
      decode_bfloat16(raw[name]) for name in (*QKV, 'bias', 'result')
  )
  for batch_index, i in enumerate(event.indices):
    keep = raw['mask'][batch_index, 0, 0]
    key_count = int(keep.sum())
    common = dict(
        input=meta['input_name'], direction=meta['direction'],
        capture_backend=meta['implementation'], source_event=str(event.path),
        chunk_start=meta['chunk_start'], chunk_end=meta['chunk_end'], i=i,
        i_local=i - meta['chunk_start'], key_count=key_count,
        mask_group='has_keys' if key_count else 'all_masked',
        boundary=i in BOUNDARIES[meta['input_name']],
    )
    for head in range(q.shape[-2]):
      at = (batch_index, slice(None), head)
      ref = reference_head(q[at], k[at], v[at], bias[0, head], keep, scale)
      for comparison, output in ((CAPTURE, captured), (CANDIDATE, result)):
        yield dict(
            **common, head=head, comparison=comparison, **metrics(ref, output[at])
        )


def summarize(rows):
  groups = {}
  for row in rows:
    key = (row['comparison'], row['input'], row['direction'], row['mask_group'])
    group = groups.setdefault(key, dict(
        zip(('comparison', 'input', 'direction', 'mask_group'), key),
        head_rows=0, nonfinite_count=0, undefined_relative_l2=0,
        worst_relative_l2=None, worst_max_abs=None,
    ))
    group['head_rows'] += 1
    group['nonfinite_count'] += row['nonfinite_count']
    group['undefined_relative_l2'] += row['relative_l2'] is None
    for field in ('relative_l2', 'max_abs'):
      worst = 'worst_' + field
      previous = group[worst]
      if row[field] is not None and (previous is None or row[field] > previous[field]):
        group[worst] = row
  return list(groups.values())


def sha256(path):
  with path.open('rb') as handle:
    return hashlib.file_digest(handle, 'sha256').hexdigest()


def nonnegative_number(text):
  value = float(text)
  if not math.isfinite(value) or value < 0:
    raise argparse.ArgumentTypeError('expected a finite nonnegative number')
  return value


def exceeds(error, limit):
  return limit is not None and (error is None or error > limit)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--capture-root', type=Path, default=DEFAULT_CAPTURE_ROOT)
  parser.add_argument('--input', choices=tuple(RUNS), help='Default: theta')
  parser.add_argument('--direction', choices=tuple(DIRECTIONS), help='Default: ending')
  parser.add_argument(
      '--i', type=int, help='Replay the event containing this i (default: 0)'
  )
  parser.add_argument('--all', action='store_true', help='Replay all 30 accepted events')
  parser.add_argument('--output-dir', required=True, type=Path)
  parser.add_argument(
      '--triangle_attention_kernel', metavar='MODULE:FUNCTION',
      help='Use the same JAX plugin as run_alphafold.py',
  )
  parser.add_argument(
      '--flash_attention_implementation', choices=('triton', 'cudnn', 'xla'),
      default='xla', help='Tokamax backend when no plugin is specified (default: xla)',
  )
  parser.add_argument('--max-relative-l2', type=nonnegative_number)
  parser.add_argument('--max-abs', type=nonnegative_number)
  args = parser.parse_args(argv)
  if args.all and any(x is not None for x in (args.input, args.direction, args.i)):
    parser.error('--all cannot be combined with --input, --direction or --i')
  # Tokamax reads Abseil flags lazily. The replay CLI has parsed its own flags.
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(sys.argv[:1])

  implementation = args.flash_attention_implementation
  kernel_spec = args.triangle_attention_kernel
  backend_name = kernel_spec if kernel_spec is not None else f'tokamax:{implementation}'
  candidate = {'name': backend_name}
  try:
    if kernel_spec is not None:
      triangle_attention.load_kernel(kernel_spec)
      module = sys.modules[kernel_spec.partition(':')[0]]
      module_file = getattr(module, '__file__', None)
      if module_file and Path(module_file).is_file():
        candidate.update(
            module_file=module_file, source_sha256=sha256(Path(module_file))
        )
    root = args.capture_root.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
      raise ValueError('Output and capture directories must be separate')
    names = tuple(RUNS) if args.all else (args.input or 'theta',)
    directions = tuple(DIRECTIONS) if args.all else (args.direction or 'ending',)
    events, manifests = discover(root, names, directions)
    if not args.all:
      i = 0 if args.i is None else args.i
      events = [event for event in events if i in event.indices]
      if not events:
        raise ValueError(f'{names[0]}/{directions[0]}: no saved event for i={i}')
    digests = dict(manifests)
    for event in events:
      digests[event.path] = sha256(event.path)
      digests[event.path.with_suffix('.json')] = event.record_digest
    paths = sorted(digests)
    before = {str(path): digests[path] for path in paths}
    device = jax.devices()[0]
    output.mkdir(parents=True, exist_ok=False)
  except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
    parser.error(str(exc))

  run = make_replayer(implementation, kernel_spec)
  print(
      f'Replaying {len(events)} event(s) using {backend_name} on {device}',
      flush=True,
  )
  failures, all_rows = [], []
  with (output / 'errors.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator='\n')
    writer.writeheader()
    for index, event in enumerate(events, 1):
      try:
        rows = list(replay(event, run))
      except Exception as exc:
        error = f'{backend_name}: {type(exc).__name__}: {exc}'
        failures.append(dict(event=str(event.path), error=error))
        print(f'FAILED {event.path.name}: {error}', file=sys.stderr, flush=True)
        continue
      writer.writerows(rows)
      handle.flush()
      all_rows.extend(rows)
      print(
          f'[{index}/{len(events)}] {event.path.name}: {len(rows)} comparisons',
          flush=True,
      )

  after = {}
  for path in paths:
    try:
      after[str(path)] = sha256(path)
    except OSError as exc:
      after[str(path)] = None
      failures.append(dict(event=str(path), error=f'Cannot rehash source: {exc}'))
  groups = summarize(all_rows)
  violations = [
      row for row in all_rows
      if row['comparison'] == CANDIDATE and (
          exceeds(row['relative_l2'], args.max_relative_l2)
          or exceeds(row['max_abs'], args.max_abs)
      )
  ]
  report = dict(
      recycle=RECYCLE, block=LAYER + 1, candidate=candidate,
      reference='NumPy FP64 stable softmax; all-masked: broadcast mean(V)',
      quality_gate=dict(
          max_relative_l2=args.max_relative_l2, max_abs=args.max_abs,
          violation_count=len(violations),
          first_violation=violations[0] if violations else None,
      ),
      environment=dict(
          platform=platform.platform(), python=sys.version, numpy=np.__version__,
          jax=jax.__version__, tokamax=importlib.metadata.version('tokamax'),
          device=str(device), device_kind=device.device_kind,
          variables={name: os.environ.get(name) for name in (
              'JAX_PLATFORMS', 'XLA_FLAGS', 'CUDA_VISIBLE_DEVICES',
              'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
          )},
      ),
      event_count=len(events), i_count=sum(len(event.indices) for event in events),
      metric_rows=len(all_rows),
      coverage=dict(Counter(
          f'{event.record["metadata"]["input_name"]}/'
          f'{event.record["metadata"]["direction"]}'
          for event in events
      )),
      source_hashes_before=before, source_hashes_after=after,
      inputs_unchanged=before == after, failures=failures, groups=groups,
  )
  (output / 'run_summary.json').write_text(
      json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8'
  )
  nonfinite = sum(group['nonfinite_count'] for group in groups)
  print(
      f'{len(events)} events, {len(all_rows)} metric rows, {len(failures)} failures, '
      f'{nonfinite} nonfinite values, {len(violations)} tolerance violations, '
      f'inputs unchanged={before == after}',
      flush=True,
  )
  return int(bool(failures or nonfinite or violations or before != after))


if __name__ == '__main__':
  raise SystemExit(main())
