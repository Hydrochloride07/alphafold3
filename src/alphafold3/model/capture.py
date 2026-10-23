"""Opt-in, inference-only runtime tensor capture (never used for timing)."""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
from pathlib import Path
import threading
import uuid

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
  context.update(
    metadata, operator=operator, module_path=hk.experimental.current_name()
  )
  # Local attention stores its gather layout alongside its operands.
  extras = context.pop('layout_tensors', {})
  tensors = {**extras, **tensors}
  dynamic = {
    k: v
    for k, v in context.items()
    if isinstance(v, (jax.Array, jax.core.Tracer, np.ndarray))
  }
  static = {k: v for k, v in context.items() if k not in dynamic}
  keys = tuple(k for k, value in tensors.items() if value is not None)
  absent = tuple(k for k, value in tensors.items() if value is None)
  dynamic_keys = tuple(dynamic)

  def write(*values):
    attrs = {
      **static,
      **{
        k: np.asarray(v).item()
        for k, v in zip(dynamic_keys, values[: len(dynamic_keys)])
      },
    }
    sink.write(attrs, dict(zip(keys, values[len(dynamic_keys) :])), absent)

  io_callback(
    write, None, *dynamic.values(), *(tensors[k] for k in keys), ordered=False
  )


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
    if not isinstance(config, dict) or set(config) - _FILTER_FIELDS:
      raise ValueError(f'Capture filters must use {sorted(_FILTER_FIELDS)}')
    for key, values in config.items():
      if not isinstance(values, list) or not values:
        raise ValueError(f'Capture filter {key} must be a nonempty list')
      expected = str if key in {'operator', 'stage', 'direction'} else int
      if any(type(v) is not expected or (expected is int and v < 0) for v in values):
        raise ValueError(f'Invalid values for capture filter {key}: {values}')
    self.config = config
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
    }
    (self.path / 'events.jsonl').touch()
    self._save()

  def _save(self):
    _atomic_json(self.path / 'manifest.json', self.manifest)

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
    if any(metadata.get(k) not in values for k, values in self.config.items()):
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
        with stem.with_suffix('.npz.tmp').open('wb') as stream:
          np.savez(stream, **storage)
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
