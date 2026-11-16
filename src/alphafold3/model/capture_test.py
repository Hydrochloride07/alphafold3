"""CPU integration tests for capture, loops, dtypes and original parameters."""

import json
from types import SimpleNamespace
from unittest import mock

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from absl import flags

from alphafold3.model import capture
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import mapping
from alphafold3.model.network import diffusion_head
from alphafold3.model.network import diffusion_transformer as dt
from alphafold3.model.network import modules

GC = model_config.GlobalConfig(
  flash_attention_implementation='xla',
  bfloat16='none',
  final_init='linear',
  pair_attention_chunk_size=((None, 2),),
)
KEY = jax.random.PRNGKey(37)


@pytest.fixture(autouse=True)
def parse_absl_flags():
  # Tokamax reads Abseil flags lazily; do not let it parse pytest's CLI.
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(['capture_test'])


def records(sink):
  return [
    capture.load_event(sink.path / row['json'])
    for row in map(json.loads, (sink.path / 'events.jsonl').read_text().splitlines())
  ]


def assert_tree_equal(a, b, *, exact=False):
  assert jax.tree.structure(a) == jax.tree.structure(b)
  for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
    if exact:
      np.testing.assert_array_equal(x, y)
    else:
      np.testing.assert_allclose(x, y, rtol=2e-5, atol=2e-5)


def run_capture(tmp_path, fn, *args, config=None, calls=1):
  baseline = hk.transform(fn)
  params = baseline.init(KEY, *args)
  expected = jax.jit(baseline.apply)(params, KEY, *args)
  sink = capture.CaptureSession(tmp_path, config)

  def instrumented(invocation, *xs):
    with capture.scope(
      session=sink, invocation_id=invocation, initializing=hk.running_init()
    ):
      return fn(*xs)

  captured = hk.transform(instrumented)
  with sink.activate():
    captured_params = captured.init(KEY, jnp.int32(0), *args)
    assert_tree_equal(params, captured_params, exact=True)
    assert sink.manifest['event_count'] == 0  # No init exports.
    apply = jax.jit(captured.apply)
    for i in range(calls):
      invocation = sink.begin(f'input-{i}', 100 + i)
      actual = apply(params, KEY, jnp.int32(invocation), *args)
      jax.block_until_ready(actual)
      sink.finish(invocation)
      assert_tree_equal(expected, actual)
  assert json.loads((sink.path / 'manifest.json').read_text())['status'] == 'complete'
  assert 'io_callback' not in str(jax.make_jaxpr(baseline.apply)(params, KEY, *args))
  return sink, records(sink)


def test_grid_attention_chunks_and_recycles(tmp_path):
  class Stack(hk.Module):
    def __call__(self, x):
      def layer(x):
        return modules.GridSelfAttention(
          modules.GridSelfAttention.Config(num_head=2), GC, transpose=False, name='grid'
        )(x, jnp.ones(x.shape[:2]))

      return capture.layer_stack(2, stage='trunk')(layer)(x)

  def net(x):
    stack = Stack(name='trunk')

    def recycle(i, x):
      with capture.scope(recycle=i, recycle_count=2):
        return stack(x)

    if hk.running_init():
      return recycle(0, x)
    return hk.fori_loop(0, 2, recycle, x)

  x = jax.random.normal(KEY, (5, 5, 8))
  sink, rows = run_capture(tmp_path, net, x, calls=2)
  assert len(rows) == 2 * 2 * 2 * 3
  assert {(r['metadata']['recycle'], r['metadata']['layer']) for r, _ in rows} == {
    (r, l) for r in range(2) for l in range(2)
  }
  assert {
    (r['metadata']['chunk_start'], r['metadata']['chunk_end']) for r, _ in rows
  } == {(0, 2), (2, 4), (4, 5)}
  assert {r['metadata']['seed'] for r, _ in rows} == {100, 101}
  for record, tensors in rows:
    q, k, v = (jnp.asarray(tensors[n]) for n in ('query', 'key', 'value'))
    import tokamax

    expected = tokamax.dot_product_attention(
      q,
      k,
      v,
      bias=jnp.asarray(tensors['bias']),
      mask=jnp.asarray(tensors['mask']),
      implementation='xla',
    )
    np.testing.assert_allclose(expected, tensors['result'], rtol=2e-5, atol=2e-5)
    assert tensors['mask'].dtype == np.bool_
    assert record['metadata']['layer_count'] == 2
    assert record['metadata']['stage'] == 'trunk'
  assert sink.manifest['event_count'] == len(rows)


@pytest.mark.parametrize('transpose,step', [(False, 4), (True, 3)])
def test_sampled_grid_attention_preserves_full_computation(tmp_path, monkeypatch, transpose, step):
  def net(x):
    def body(i, value):
      with capture.scope(stage='trunk', recycle=i, recycle_count=2, layer=0, layer_count=1):
        indices = jnp.arange(value.shape[0])
        mask = (indices[:, None] + 2 * indices[None, :]) % 3 != 0
        return modules.GridSelfAttention(
            modules.GridSelfAttention.Config(num_head=2), GC, transpose=transpose, name='grid'
        )(value, mask)
    return body(0, x) if hk.running_init() else hk.fori_loop(0, 2, body, x)

  x = jax.random.normal(KEY, (5, 5, 8))
  _, dense = run_capture(tmp_path, net, x)
  writes = []
  original_write = capture.CaptureSession.write

  def counted_write(self, metadata, *args, **kwargs):
    writes.append(metadata)
    return original_write(self, metadata, *args, **kwargs)

  monkeypatch.setattr(capture.CaptureSession, 'write', counted_write)
  _, sparse = run_capture(tmp_path, net, x, config={
      'stage': ['trunk'], 'pairformer_recycles': [1], 'pairformer_blocks': [1],
      'pairformer_attention_i_step': [step]})
  assert len(writes) == len(sparse) == 2
  assert sorted(r['metadata']['sampled_i'] for r, _ in sparse) == list(range(0, 5, step))
  for record, tensors in sparse:
    m = record['metadata']
    assert m['i_step'] == step and m['recycle'] == 1 and m['layer'] == 0
    ref = next(t for r, t in dense if r['metadata']['recycle'] == 1 and
               r['metadata']['chunk_start'] == m['chunk_start'])
    offset = m['sampled_i'] - m['chunk_start']
    for name in ('query', 'key', 'value', 'mask', 'result'):
      np.testing.assert_array_equal(tensors[name], ref[name][offset:offset + 1])
    np.testing.assert_array_equal(tensors['bias'], ref['bias'])


@pytest.mark.parametrize('equation', ['ikc,jkc->ijc', 'kjc,kic->ijc'])
@pytest.mark.parametrize('use_glu', [False, True])
def test_triangle_inputs_and_contraction(tmp_path, equation, use_glu):
  def net(x, mask):
    return modules.TriangleMultiplication(
      modules.TriangleMultiplication.Config(equation=equation, use_glu_kernel=use_glu),
      GC,
      name='triangle',
    )(x, mask)

  x = jax.random.normal(KEY, (3, 3, 4))
  mask = jnp.eye(3)
  _, rows = run_capture(tmp_path, net, x, mask)
  assert len(rows) == 1
  record, tensors = rows[0]
  np.testing.assert_array_equal(tensors['act'], x)
  np.testing.assert_array_equal(tensors['mask'], mask)
  np.testing.assert_allclose(
    np.einsum(record['metadata']['equation'], tensors['a'], tensors['b']),
    tensors['contraction_result'],
    rtol=2e-5,
    atol=2e-5,
  )


def check_manual_attention(tensors, scale):
  q, k, v = (jnp.asarray(tensors[n]) for n in ('query', 'key', 'value'))
  logits = jnp.einsum('...qhc,...khc->...hqk', q * scale, k) + jnp.asarray(
    tensors['mask_bias']
  )
  if tensors['pair_logits'] is not None:
    logits += jnp.asarray(tensors['pair_logits'])
  weights = jax.nn.softmax(logits, -1).astype(v.dtype)
  result = jnp.einsum('...hqk,...khc->...qhc', weights, v)
  np.testing.assert_allclose(result, tensors['result'], rtol=2e-5, atol=2e-5)


def test_global_nested_blocks(tmp_path):
  config = dt.Transformer.Config(
    num_blocks=4, super_block_size=2, attention=dt.SelfAttentionConfig(num_head=2)
  )

  def net(x):
    with capture.scope(stage='diffusion', sample=0, diffusion_step=1):
      return dt.Transformer(config, GC)(
        x, jnp.ones(3), jnp.ones_like(x), jnp.ones((3, 3, 4))
      )

  _, rows = run_capture(tmp_path, net, jax.random.normal(KEY, (3, 8)))
  assert len(rows) == 4
  assert {
    (r['metadata']['super_block'], r['metadata']['block'], r['metadata']['layer'])
    for r, _ in rows
  } == {(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)}
  for r, t in rows:
    check_manual_attention(t, r['metadata']['scale'])


@pytest.mark.parametrize('direction', ['encoder', 'decoder'])
def test_local_layout(tmp_path, direction):
  config = dt.CrossAttTransformer.Config(
    num_blocks=2,
    num_intermediate_factor=2,
    attention=dt.CrossAttentionConfig(num_head=2, key_dim=8, value_dim=8),
  )
  gather = atom_layout.GatherInfo(
    gather_idxs=jnp.array([[0, 1, 2], [1, 2, 3]]),
    gather_mask=jnp.ones((2, 3), bool),
    input_shape=jnp.array([2, 2]),
  )

  def net(x):
    with capture.scope(stage='diffusion'):
      return dt.CrossAttTransformer(config, GC, name='atom_transformer_' + direction)(
        x,
        jnp.ones((2, 2)),
        gather,
        jnp.ones((2, 3)),
        jnp.ones_like(x),
        jnp.ones((2, 3, 8)),
        jnp.ones((2, 2, 3, 4)),
      )

  _, rows = run_capture(tmp_path, net, jax.random.normal(KEY, (2, 2, 8)))
  assert len(rows) == 2
  for r, t in rows:
    assert r['metadata']['direction'] == direction
    assert t['query'].shape[:2] == (2, 2)
    np.testing.assert_array_equal(t['queries_to_keys_indices'], gather.gather_idxs)
    check_manual_attention(t, r['metadata']['scale'])


@pytest.mark.parametrize('selected_steps', [None, [1, 5, 10, 100, 200]])
def test_diffusion_scan_vmap_indices(tmp_path, monkeypatch, selected_steps):
  batch = SimpleNamespace(
    predicted_structure_info=SimpleNamespace(atom_mask=jnp.ones((2, 2)))
  )
  steps = 3 if selected_steps is None else 200
  config = diffusion_head.SampleConfig(num_samples=2, steps=steps)
  writes = []
  original_write = capture.CaptureSession.write

  def counted_write(self, metadata, *args, **kwargs):
    writes.append(metadata)
    return original_write(self, metadata, *args, **kwargs)

  monkeypatch.setattr(capture.CaptureSession, 'write', counted_write)

  def net():
    if selected_steps is not None:
      for stage in ('conditioning', 'template', 'msa', 'trunk', 'confidence'):
        with capture.scope(stage=stage):
          capture.emit('identity', {'result': jnp.ones(1)})

    def denoise(positions, level):
      # A real self-attention call inside the original diffusion sampler.
      x = positions.reshape(4, 3)
      dt.self_attention(
        x, jnp.ones(4), None, dt.SelfAttentionConfig(num_head=1), GC, name='denoise'
      )
      return positions * 0.9

    with capture.scope(stage='diffusion'):
      return diffusion_head.sample(denoise, batch, hk.next_rng_key(), config)

  filters = {'diffusion_steps': selected_steps} if selected_steps else None
  _, rows = run_capture(tmp_path, net, config=filters)
  diffusion_rows = [(r, t) for r, t in rows if r['metadata']['stage'] == 'diffusion']
  wanted = range(steps) if selected_steps is None else [v - 1 for v in selected_steps]
  assert len(diffusion_rows) == len(wanted) * 2
  # Rejected steps must not even reach the host writer.
  assert len(writes) == len(rows)
  assert {
    (r['metadata']['diffusion_step'], r['metadata']['sample']) for r, _ in diffusion_rows
  } == {(i, j) for i in wanted for j in range(2)}
  assert all(r['metadata']['diffusion_step_count'] == steps for r, _ in diffusion_rows)
  if selected_steps is not None:
    assert {r['metadata']['stage'] for r, _ in rows} == {
      'conditioning', 'template', 'msa', 'trunk', 'confidence', 'diffusion'
    }
    assert len(rows) == 15
  for r, t in diffusion_rows:
    assert t['pair_logits'] is None
    check_manual_attention(t, r['metadata']['scale'])


def test_template_confidence_and_filter(tmp_path):
  def net(x):
    def template(carry, i):
      with capture.scope(template=i, template_count=2):
        y = capture.layer_stack(2, stage='template')(lambda x: emit_identity(x))(x)
      return carry + y, None

    def confidence(x, i):
      with capture.scope(sample=i, sample_count=3):
        return capture.layer_stack(2, stage='confidence')(lambda x: emit_identity(x))(x)

    out, _ = hk.scan(template, jnp.zeros_like(x), jnp.arange(2))
    return out, mapping.sharded_map(confidence, in_axes=(0, 0))(
      jnp.stack([x] * 3), jnp.arange(3)
    )

  def emit_identity(x):
    capture.emit('attention', dict(query=x, bias=None, mask=None, result=x))
    return x

  _, rows = run_capture(
    tmp_path,
    net,
    jnp.ones((2, 2)),
    config={'stage': ['confidence'], 'sample': [1], 'layer': [1]},
  )
  assert len(rows) == 1
  assert rows[0][0]['metadata']['sample'] == 1
  assert rows[0][1]['mask'] is None


def test_bfloat16_and_repeated_runs(tmp_path):
  def net(x):
    capture.emit(
      'attention', dict(query=x, result=x, mask=jnp.array([True, False]), bias=None)
    )
    return x

  x = jnp.array([1.5, float('nan')], dtype=jnp.bfloat16)
  # No floating comparison on NaN: directly exercise original bits in the writer.
  paths = []
  for _ in range(2):
    sink = capture.CaptureSession(tmp_path)
    with sink.activate():
      i = sink.begin('bf16', 0)
      f = hk.transform(lambda x: net(x))
      with capture.scope(invocation_id=i):
        params = f.init(KEY, x)
        f.apply(params, KEY, x)
      sink.finish(i)
    paths.append(sink.path)
    _, t = records(sink)[0]
    np.testing.assert_array_equal(
      np.asarray(x).view(np.uint16), t['query'].view(np.uint16)
    )
  assert paths[0] != paths[1]


def test_write_failure_marks_run_failed(tmp_path):
  sink = capture.CaptureSession(tmp_path)
  with pytest.raises(OSError, match='disk full'):
    with sink.activate():
      i = sink.begin('failed', 0)
      with mock.patch.object(np, 'savez', side_effect=OSError('disk full')):
        sink.write(
          {'invocation_id': i, 'operator': 'attention'}, {'result': np.ones(1)}
        )
  assert json.loads((sink.path / 'manifest.json').read_text())['status'] == 'failed'
  assert sink.manifest['event_count'] == 0
  assert not list((sink.path / 'events').glob('*.json'))


@pytest.mark.parametrize(
  'config', [{'typo': [0]}, {'layer': []}, {'layer': [-1]}, {'stage': [1]},
             {'diffusion_steps': [0]}, {'diffusion_steps': [True]},
             {'diffusion_steps': []}]
)
def test_invalid_filters(tmp_path, config):
  with pytest.raises(ValueError):
    capture.CaptureSession(tmp_path, config)


def test_pairformer_shared_attention_and_triangle(tmp_path):
  config = modules.PairFormerIteration.Config(num_layer=1)
  config.pair_attention.num_head = 2
  config.single_attention = dt.SelfAttentionConfig(num_head=2)
  config.single_transition = modules.TransitionBlock.Config()
  config.triangle_multiplication_incoming.use_glu_kernel = False
  config.triangle_multiplication_outgoing.use_glu_kernel = False

  def net(pair, single):
    def block(xs):
      p, s = xs
      return modules.PairFormerIteration(
        config, GC, with_single=True, name='pairformer'
      )(p, jnp.ones(p.shape[:2]), s, jnp.ones(s.shape[:1]))

    return capture.layer_stack(2, stage='trunk')(block)((pair, single))

  _, rows = run_capture(
    tmp_path, net, jax.random.normal(KEY, (3, 3, 8)), jax.random.normal(KEY, (3, 8))
  )
  assert {r['metadata']['operator'] for r, _ in rows} == {
    'attention',
    'triangle_multiplication',
    'diffusion_global_attention',
  }
  assert {r['metadata']['layer'] for r, _ in rows} == {0, 1}


def test_model_runner_cached_invocation_identity(tmp_path, monkeypatch):
  # The runner must pass invocation identity as a dynamic executable argument.
  import run_alphafold as af3

  class FakeModel:
    def __init__(self, config):
      pass

    def __call__(self, batch):
      y = batch['x'] * 2
      capture.emit('attention', dict(query=batch['x'], result=y, bias=None, mask=None))
      return {'value': y}

  monkeypatch.setattr(af3.model, 'Model', FakeModel)
  runner = af3.ModelRunner(None, jax.devices('cpu')[0], '.')
  runner.model_params = {'__meta__': {'__identifier__': np.array([1], np.uint8)}}
  x = {'x': np.ones((2, 3), np.float32)}
  runner.run_inference(x, KEY)  # Compile the disabled executable first.
  sink = capture.CaptureSession(tmp_path)
  with sink.activate():
    for i in range(2):
      result = runner.run_inference(x, KEY, input_name=f'name-{i}', seed=i)
      np.testing.assert_array_equal(result['value'], 2 * x['x'])
    assert runner._model.func._cache_size() == 1
  rows = records(sink)
  assert len(rows) == 2
  assert {
    (r['metadata']['input_name'], r['metadata']['seed'], r['metadata']['invocation_id'])
    for r, _ in rows
  } == {('name-0', 0, 0), ('name-1', 1, 1)}
  runner.run_inference(x, KEY)  # Return safely to an executable without callbacks.


@pytest.mark.parametrize('trunk_only', [False, True])
@pytest.mark.parametrize('num_recycles', [0, 1])
def test_real_model_recycle_and_confidence_control_flow(tmp_path, monkeypatch, trunk_only, num_recycles):
  # Exercise Model.__call__ itself, substituting tiny networks for the huge model.
  from alphafold3.model import model as af3_model
  config = af3_model.Model.Config(num_recycles=num_recycles, stop_after_trunk=trunk_only)

  batch = SimpleNamespace(
    num_res=2,
    token_features=SimpleNamespace(mask=jnp.ones(2), asym_id=jnp.ones(2)),
    pseudo_beta_info=SimpleNamespace(token_atoms_to_pseudo_beta=None),
  )
  monkeypatch.setattr(af3_model.feat_batch.Batch, 'from_data_dict', lambda _: batch)

  class TinyEvoformer(hk.Module):
    def __init__(self, config, global_config):
      super().__init__(name='evoformer')
      self.config = config

    def __call__(self, *, prev, target_feat, **kwargs):
      def layer(x):
        y = hk.Linear(8)(x)
        capture.emit('attention', {'query': x, 'result': y})
        return y

      pair = capture.layer_stack(2, stage='trunk')(layer)(prev['pair'])
      return {'pair': pair, 'single': prev['single'], 'target_feat': target_feat}

  class TinyConfidence(hk.Module):
    def __init__(self, *args):
      super().__init__(name='confidence_head')

    def __call__(self, *, dense_atom_positions, **kwargs):
      def layer(x):
        capture.emit('attention', {'query': x, 'result': x})
        return x

      return {
        'score': capture.layer_stack(2, stage='confidence')(layer)(
          dense_atom_positions
        ).sum()
      }

  class TinyDistogram:
    def __init__(self, *args):
      pass

    def __call__(self, *args, **kwargs):
      return {}

  monkeypatch.setattr(af3_model.evoformer_network, 'Evoformer', TinyEvoformer)
  monkeypatch.setattr(
    af3_model, 'create_target_feat_embedding', lambda **kwargs: jnp.zeros((2, 8))
  )
  monkeypatch.setattr(af3_model.confidence_head, 'ConfidenceHead', TinyConfidence)
  monkeypatch.setattr(af3_model.distogram_head, 'DistogramHead', TinyDistogram)
  def sample_diffusion(self, *args, **kwargs):
    if trunk_only:
      raise AssertionError('Diffusion must not execute in trunk-only mode')
    return {'atom_positions': jnp.ones((3, 2, 3))}

  monkeypatch.setattr(af3_model.Model, '_sample_diffusion', sample_diffusion)
  config.evoformer.pair_channel = 8
  config.evoformer.seq_channel = 8
  _, rows = run_capture(tmp_path, lambda: af3_model.Model(config)({}))
  trunk = [r['metadata'] for r, _ in rows if r['metadata']['stage'] == 'trunk']
  confidence = [
    r['metadata'] for r, _ in rows if r['metadata']['stage'] == 'confidence'
  ]
  assert len(trunk) == 2 * (num_recycles + 1)
  assert {(r['recycle'], r['layer']) for r in trunk} == {
    (i, j) for i in range(num_recycles + 1) for j in range(2)
  }
  if trunk_only:
    assert not confidence
    return
  assert len(confidence) == 6
  assert {(r['sample'], r['layer']) for r in confidence} == {
    (i, j) for i in range(3) for j in range(2)
  }
  assert all(r['recycle'] is None for r in confidence)


def test_pairformer_selected_blocks_and_recycles(tmp_path, monkeypatch):
  writes = []
  original_write = capture.CaptureSession.write

  def counted_write(self, metadata, *args, **kwargs):
    writes.append(metadata)
    return original_write(self, metadata, *args, **kwargs)

  monkeypatch.setattr(capture.CaptureSession, 'write', counted_write)

  def net(x):
    def layer(x):
      return modules.GridSelfAttention(
        modules.GridSelfAttention.Config(num_head=2), GC, transpose=False, name='grid'
      )(x, jnp.ones(x.shape[:2]))

    stack = capture.layer_stack(48, stage='trunk')(layer)
    with capture.scope(stage='conditioning'):
      capture.emit('identity', {'result': x})

    def recycle(i, x):
      with capture.scope(recycle=i, recycle_count=11):
        return stack(x)

    if hk.running_init():
      return recycle(0, x)
    return hk.fori_loop(0, 11, recycle, x)

  _, rows = run_capture(
    tmp_path, net, jax.random.normal(KEY, (5, 5, 8)),
    config={'stage': ['conditioning', 'trunk'], 'pairformer_blocks': [1, 4, 8, 16],
            'pairformer_recycles': [0, 5, 10]},
  )
  trunk = [r['metadata'] for r, _ in rows if r['metadata']['stage'] == 'trunk']
  assert len(trunk) == 3 * 4 * 3  # Three actual attention chunks per block.
  assert {(r['recycle'], r['layer']) for r in trunk} == {
    (r, l) for r in (0, 5, 10) for l in (0, 3, 7, 15)
  }
  assert all(r['recycle_count'] == 11 and r['layer_count'] == 48 for r in trunk)
  assert len(rows) == len(trunk) + 1
  assert len(writes) == len(rows)  # Unselected blocks never reach the host writer.


def test_runtime_callback_failure(tmp_path, monkeypatch):
  sink = capture.CaptureSession(tmp_path)

  def fail(*args, **kwargs):
    raise OSError('simulated capture disk full')

  def net(x):
    capture.emit('attention', {'query': x, 'result': x})
    return x

  f = hk.transform(net)
  params = f.init(KEY, jnp.ones(2))
  with pytest.raises(Exception, match='simulated capture disk full'):
    with sink.activate():
      invocation = sink.begin('failure', 0)
      with capture.scope(invocation_id=invocation):
        monkeypatch.setattr(np, 'savez', fail)
        jax.block_until_ready(jax.jit(f.apply)(params, KEY, jnp.ones(2)))
      sink.finish(invocation)
  manifest = json.loads((sink.path / 'manifest.json').read_text())
  assert manifest['status'] == 'failed'
  assert manifest['invocations'][0]['status'] == 'failed'


def test_lossless_compressed_capture_and_disk_reserve(tmp_path, monkeypatch):
  import zipfile
  monkeypatch.setenv('AF3_CAPTURE_COMPRESS', '1')
  sink = capture.CaptureSession(tmp_path)
  value = np.arange(24, dtype=np.float32).reshape(4, 6)
  with sink.activate():
    invocation = sink.begin('compressed', 1)
    sink.write({'invocation_id': invocation, 'operator': 'attention'}, {'result': value})
    sink.finish(invocation)
  record, tensors = records(sink)[0]
  np.testing.assert_array_equal(tensors['result'], value)
  with zipfile.ZipFile(sink.path / 'events' / (record['event_id'] + '.npz')) as archive:
    assert all(m.compress_type == zipfile.ZIP_DEFLATED for m in archive.infolist())
  monkeypatch.setenv('AF3_CAPTURE_MIN_FREE_GB', '100000000')
  sink = capture.CaptureSession(tmp_path)
  with pytest.raises(OSError, match='disk reserve'):
    with sink.activate():
      invocation = sink.begin('disk-reserve', 1)
      sink.write({'invocation_id': invocation, 'operator': 'attention'}, {'result': value})
  assert sink.manifest['status'] == 'failed'
  assert sink.manifest['event_count'] == 0


def test_confidence_vmap_keeps_distinct_attention_sites(tmp_path):
  config = modules.PairFormerIteration.Config(num_layer=1)
  config.pair_attention.num_head = 2
  config.single_attention = dt.SelfAttentionConfig(num_head=2)
  config.single_transition = modules.TransitionBlock.Config()
  config.triangle_multiplication_incoming.use_glu_kernel = False
  config.triangle_multiplication_outgoing.use_glu_kernel = False

  def net(pair, single):
    def sample(p, s, index):
      def block(xs):
        p, s = xs
        return modules.PairFormerIteration(config, GC, with_single=True,
            name='confidence_pairformer')(p, jnp.ones(p.shape[:2]), s, jnp.ones(s.shape[:1]))
      with capture.scope(stage='confidence', sample=index, sample_count=2):
        return capture.layer_stack(2, stage='confidence')(block)((p, s))
    return mapping.sharded_map(sample, in_axes=(0, 0, 0))(
        jnp.stack([pair, pair]), jnp.stack([single, single]), jnp.arange(2))

  _, rows = run_capture(tmp_path, net, jax.random.normal(KEY, (3, 3, 8)),
                        jax.random.normal(KEY, (3, 8)))
  attention = [r['metadata'] for r, _ in rows if r['metadata']['operator'] == 'attention']
  assert {m['direction'] for m in attention} == {'starting', 'ending'}
  identities = {(m['module_path'], m['direction'], m['layer'], m['sample'],
                 m['chunk_start'], m['chunk_end']) for m in attention}
  assert len(identities) == len(attention)


def test_capture_singleton_map_preserves_random_keys(tmp_path):
  def net(x):
    def one(value):
      key = hk.next_rng_key()
      noise = jax.random.normal(key, value.shape)
      capture.emit('attention', {'query': value, 'result': value + noise})
      return value + noise
    result = mapping.sharded_map(one)(x)
    return result, hk.next_rng_key()
  _, rows = run_capture(tmp_path, net, jnp.ones((3, 2, 2)))
  assert len(rows) == 3


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_transition_sharding_preserves_parameters_and_values(dtype):
  x = jax.random.normal(KEY, (13, 11, 128)).astype(dtype)

  def transform(chunk):
    def fn(value):
      transition = modules.TransitionBlock(
          modules.TransitionBlock.Config(), GC, name='transition'
      )
      return mapping.sharded_apply(transition, chunk)(value) if chunk else transition(value)
    return hk.transform(fn)

  ordinary, sharded = transform(None), transform(4)
  params = ordinary.init(KEY, x)
  assert_tree_equal(params, sharded.init(KEY, x), exact=True)
  expected = jax.jit(ordinary.apply)(params, KEY, x)
  actual = jax.jit(sharded.apply)(params, KEY, x)
  expected, actual = map(lambda v: np.asarray(v, dtype=np.float32), (expected, actual))
  assert np.isfinite(actual).all()
  assert np.linalg.norm(expected - actual) / np.linalg.norm(expected) < 0.005
