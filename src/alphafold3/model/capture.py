"""Opt-in, inference-only runtime tensor capture (never used for timing)."""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import shutil
from pathlib import Path
import threading
import time
import uuid
import zipfile

import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
import numpy as np

_CONTEXT = contextvars.ContextVar('af3_capture', default={})
_INDEX_FIELDS = (
  'recycle',
  'recycle_count',
  'layer',
  'layer_count',
  'template',
  'template_count',
  'diffusion_step',
  'diffusion_step_count',
  'sample',
  'sample_count',
  'super_block',
  'super_block_count',
  'block',
  'block_count',
  'chunk_start',
  'chunk_end',
  'chunk_axis',
  'chunk_total',
)
_FILTER_FIELDS = {
  'operator',
  'stage',
  'recycle',
  'layer',
  'diffusion_step',
  'sample',
  'template',
  'direction',
}
_SELECTION_FIELDS = {
    'diffusion_steps', 'pairformer_blocks', 'pairformer_recycles',
    'pairformer_attention_i_step',
}


def session():
  return _CONTEXT.get().get('session')


def enabled():
  return session() is not None


def current(key, default=None):
  return _CONTEXT.get().get(key, default)


@contextlib.contextmanager
def scope(**metadata):
  token = _CONTEXT.set({**_CONTEXT.get(), **metadata})
  try:
    yield
  finally:
    _CONTEXT.reset(token)


def layer_stack(count, *, stage=None, index='layer', with_per_layer_inputs=False):
  """Attach scan-carried indices without changing Haiku stack names or RNGs."""

  def decorate(fn):
    ordinary = hk.experimental.layer_stack(
      count, with_per_layer_inputs=with_per_layer_inputs
    )(fn)

    def call(*args):
      if not enabled():
        return ordinary(*args)
      metadata = {f'{index}_count': count}
      if stage is not None:
        metadata['stage'] = stage
      if with_per_layer_inputs:

        def indexed(carry, *ys):
          i, x = carry
          with scope(**{index: i}):
            y, zs = fn(x, *ys)
          return (i + 1, y), zs

        with scope(
          **metadata,
          initializing=(_CONTEXT.get().get('initializing', False) or hk.running_init()),
        ):
          (_, out), zs = hk.experimental.layer_stack(count, with_per_layer_inputs=True)(
            indexed
          )((jnp.int32(0), args[0]), *args[1:])
        return out, zs
      else:

        def indexed(i, xs):
          with scope(**{index: i}):
            y = fn(*xs)
          return i + 1, (y,) if len(xs) == 1 else y

        with scope(
          **metadata,
          initializing=(_CONTEXT.get().get('initializing', False) or hk.running_init()),
        ):
          _, out = hk.experimental.layer_stack(count)(indexed)(jnp.int32(0), args)
        return out[0] if len(args) == 1 else out

    return call

  return decorate


def emit(operator, tensors, **metadata):
  """Export actual array operands through an effectful runtime callback."""
  context = _CONTEXT.get()
  sink = context.get('session')
  if sink is None or context.get('initializing') or hk.running_init():
    return
  context = {k: v for k, v in context.items() if k not in ('session', 'initializing')}
  context.update(metadata, operator=operator)
  context.setdefault('module_path', hk.experimental.current_name())
  # Local attention stores its gather layout alongside its operands.
  extras = context.pop('layout_tensors', {})
  tensors = {**extras, **tensors}
  sample_gate = None
  step = sink.pairformer_attention_i_step
  if step is not None and context.get('stage') == 'trunk' and operator == 'attention':
    size = tensors['query'].shape[0]
    if step < size:
      raise ValueError('Capture i-step must be at least the attention compute chunk size')
    start = context.setdefault('chunk_start', 0)
    context.setdefault('chunk_end', start + size)
    context.setdefault('chunk_total', size)
    context.setdefault('chunk_axis', 0)
    offset = (-start) % step
    if isinstance(offset, int):
      if offset >= size:
        return
    else:
      sample_gate = (jnp.int32(offset < size), (1,))
    context.update(i_step=step, sampled_i=start + offset)
    # Only exported operands are sliced. The caller retains the full result.
    for name in ('query', 'key', 'value', 'mask', 'result'):
      value = tensors[name]
      if value is not None:
        if value.shape[0] != size:
          raise ValueError(f'Unexpected i-axis size for {name}')
        tensors[name] = jax.lax.dynamic_slice_in_dim(value, offset, 1, axis=0)
  dynamic = {
    k: v
    for k, v in context.items()
    if isinstance(v, (jax.Array, jax.core.Tracer, np.ndarray))
  }
  static = {k: v for k, v in context.items() if k not in dynamic}
  keys = tuple(k for k, value in tensors.items() if value is not None)
  absent = tuple(k for k, value in tensors.items() if value is None)
  dynamic_keys = tuple(dynamic)

  # Known static filters can avoid the callback and its transfers entirely.
  if any(k not in dynamic and static.get(k) not in allowed
         for k, allowed in sink.filters.items()):
    return

  def write(*values):
    attrs = {
      **static,
      **{k: np.asarray(v).item()
         for k, v in zip(dynamic_keys, values[:len(dynamic_keys)])},
    }
    sink.write(attrs, dict(zip(keys, values[len(dynamic_keys):])), absent)

  def export(*values):
    io_callback(
      write, None, *values, ordered=False
    )

  values = (*dynamic.values(), *(tensors[k] for k in keys))
  selection = sink.selection(context.get('stage'))
  dynamic_selection = [] if sample_gate is None else [sample_gate]
  for name, allowed in selection.items():
    count = context.get(name + '_count')
    if isinstance(count, int) and max(allowed) >= count:
      raise ValueError(f'Capture selection {name} exceeds model count {count}')
    if name in dynamic:
      dynamic_selection.append((context[name], allowed))
    elif context.get(name) not in allowed:
      return
  if not dynamic_selection:
    export(*values)
    return

  # Gate the effect before transfers. Sequential callback batching keeps IO
  # outside unsupported vmap-of-cond; model operations stay vectorised.
  @jax.custom_batching.sequential_vmap
  def export_selected(selected_values, *values):
    selected = jnp.bool_(True)
    for value, (_, allowed) in zip(selected_values, dynamic_selection):
      selected &= jnp.any(value == jnp.asarray(allowed))
    jax.lax.cond(selected, lambda: export(*values), lambda: None)

  export_selected(tuple(v for v, _ in dynamic_selection), *values)


def _atomic_json(path, value):
  tmp = path.with_suffix(path.suffix + '.tmp')
  with tmp.open('w') as stream:
    json.dump(value, stream, indent=2, allow_nan=False)
    stream.write('\n')
  os.replace(tmp, path)


class CaptureSession:
  """One run, with thread-safe atomic events and a streamed event index."""

  def __init__(self, root, config=None):
    if config is None:
      config = {}
    allowed_fields = _FILTER_FIELDS | _SELECTION_FIELDS
    if not isinstance(config, dict) or set(config) - allowed_fields:
      raise ValueError(f'Capture filters must use {sorted(allowed_fields)}')
    for key, values in config.items():
      if not isinstance(values, list) or not values:
        raise ValueError(f'Capture filter {key} must be a nonempty list')
      expected = str if key in {'operator', 'stage', 'direction'} else int
      if any(type(v) is not expected or (expected is int and v < 0) for v in values):
        raise ValueError(f'Invalid values for capture filter {key}: {values}')
      if key in {'diffusion_steps', 'pairformer_blocks'} and any(v < 1 for v in values):
        raise ValueError(f'{key} uses positive, one-based numbers')
      if key == 'pairformer_attention_i_step' and (len(values) != 1 or values[0] < 1):
        raise ValueError('pairformer_attention_i_step requires one positive stride')
    self.config = config
    self.filters = {k: v for k, v in config.items() if k in _FILTER_FIELDS}
    self.diffusion_steps = (
      tuple(sorted({v - 1 for v in config['diffusion_steps']}))
      if 'diffusion_steps' in config else None
    )
    self.pairformer_blocks = (
      tuple(sorted({v - 1 for v in config['pairformer_blocks']}))
      if 'pairformer_blocks' in config else None
    )
    self.pairformer_recycles = (
      tuple(sorted(set(config['pairformer_recycles'])))
      if 'pairformer_recycles' in config else None
    )
    self.pairformer_attention_i_step = config.get('pairformer_attention_i_step', [None])[0]
    self.compressed = os.environ.get('AF3_CAPTURE_COMPRESS') == '1'
    self.min_free_bytes = int(float(os.environ.get('AF3_CAPTURE_MIN_FREE_GB', '0')) * 1024**3)
    self.wait_free_seconds = int(os.environ.get('AF3_CAPTURE_WAIT_FREE_SECONDS', '0'))
    level = os.environ.get('AF3_CAPTURE_COMPRESS_LEVEL')
    self.compress_level = int(level) if level is not None else None
    if self.compress_level is not None and not 0 <= self.compress_level <= 9:
      raise ValueError('AF3_CAPTURE_COMPRESS_LEVEL must be between 0 and 9')
    self.path = Path(root) / ('run-' + uuid.uuid4().hex)
    self.path.mkdir(parents=True)
    (self.path / 'events').mkdir()
    self._lock = threading.Lock()
    self._error = None
    self.manifest = {
      'schema_version': 1,
      'status': 'running',
      'event_count': 0,
      'event_index': 'events.jsonl',
      'filters': config,
      'invocations': [],
      'jax_version': jax.__version__,
      'npz_compressed': self.compressed,
      'min_free_bytes': self.min_free_bytes,
    }
    (self.path / 'events.jsonl').touch()
    self._save()

  def _save(self):
    _atomic_json(self.path / 'manifest.json', self.manifest)

  def selection(self, stage):
    if stage == 'diffusion' and self.diffusion_steps is not None:
      return {'diffusion_step': self.diffusion_steps}
    if stage == 'trunk':
      return {k: v for k, v in (
        ('layer', self.pairformer_blocks),
        ('recycle', self.pairformer_recycles),
      ) if v is not None}
    return {}

  def begin(self, input_name=None, seed=None):
    with self._lock:
      invocation = len(self.manifest['invocations'])
      self.manifest['invocations'].append(
        {
          'invocation_id': invocation,
          'input_name': input_name,
          'seed': seed,
          'status': 'running',
        }
      )
      self._save()
    return invocation

  def finish(self, invocation, failed=False):
    try:
      jax.effects_barrier()
      if self._error:
        raise RuntimeError('Capture writer failed') from self._error
    except BaseException:
      failed = True
      raise
    finally:
      with self._lock:
        self.manifest['invocations'][invocation]['status'] = (
          'failed' if failed else 'complete'
        )
        self._save()

  def write(self, metadata, tensors, absent=()):
    if any(metadata.get(k) not in values for k, values in self.filters.items()):
      return
    if any(metadata.get(k) not in values
           for k, values in self.selection(metadata.get('stage')).items()):
      return
    with self._lock:
      try:
        invocation = self.manifest['invocations'][int(metadata['invocation_id'])]
        metadata = {
          **dict.fromkeys((*_INDEX_FIELDS, 'stage', 'direction')),
          **invocation, **metadata,
        }
        metadata.pop('status', None)
        event_id = uuid.uuid4().hex
        stem = self.path / 'events' / event_id
        descriptors = dict.fromkeys(absent)
        storage = {}
        for name, value in tensors.items():
          value = np.asarray(value)
          dtype = str(value.dtype)
          storage[name] = value.view(np.uint16) if dtype == 'bfloat16' else value
          descriptors[name] = {
            'shape': list(value.shape),
            'dtype': dtype,
            'storage_dtype': str(storage[name].dtype),
          }
        required = self.min_free_bytes
        if self.wait_free_seconds:
          # Allow one archive plus ZIP overhead while the collector drains
          # previously committed archives. Never release the writer lock here.
          raw = sum(v.nbytes for v in storage.values())
          required += raw + raw // 500 + 1024**2
          deadline = time.monotonic() + self.wait_free_seconds
          while shutil.disk_usage(self.path).free < required and time.monotonic() < deadline:
            time.sleep(5)
        if shutil.disk_usage(self.path).free < required:
          raise OSError('Capture disk reserve reached; no events were dropped silently.')
        with stem.with_suffix('.npz.tmp').open('wb') as stream:
          if self.compressed and self.compress_level is not None:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=self.compress_level) as archive:
              for name, value in storage.items():
                with archive.open(name + '.npy', 'w', force_zip64=True) as member:
                  np.lib.format.write_array(member, value, allow_pickle=False)
          else:
            save = np.savez_compressed if self.compressed else np.savez
            save(stream, **storage)
        os.replace(stem.with_suffix('.npz.tmp'), stem.with_suffix('.npz'))
        record = {
          'schema_version': 1,
          'event_id': event_id,
          'metadata': metadata,
          'tensors': descriptors,
        }
        _atomic_json(stem.with_suffix('.json'), record)
        with (self.path / 'events.jsonl').open('a') as stream:
          stream.write(
            json.dumps(
              {
                'event_id': event_id,
                'json': f'events/{event_id}.json',
                'npz': f'events/{event_id}.npz',
              }
            )
            + '\n'
          )
        self.manifest['event_count'] += 1
      except Exception as error:
        self._error = error
        raise

  @contextlib.contextmanager
  def activate(self):
    failed = False
    try:
      with scope(session=self):
        yield self
    except BaseException:
      failed = True
      raise
    finally:
      try:
        jax.effects_barrier()
      except BaseException:
        failed = True
        raise
      finally:
        with self._lock:
          for invocation in self.manifest['invocations']:
            if invocation['status'] == 'running':
              invocation['status'] = 'failed'
          failed = failed or any(
            i['status'] == 'failed' for i in self.manifest['invocations']
          )
          self.manifest['status'] = 'failed' if failed or self._error else 'complete'
          self._save()


def load_event(path):
  """Load an event without pickle and restore bfloat16 from its exact bits."""
  path = Path(path)
  record = json.loads(path.with_suffix('.json').read_text())
  tensors = {}
  with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
    for name, spec in record['tensors'].items():
      if spec is None:
        tensors[name] = None
      else:
        value = arrays[name]
        if spec['dtype'] == 'bfloat16':
          import ml_dtypes

          value = value.view(ml_dtypes.bfloat16)
        if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype']:
          raise ValueError(f'Invalid tensor descriptor: {name}')
        tensors[name] = value
  return record, tensors
