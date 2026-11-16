"""Validate selected trunk archives and independently replay representative operands.

Full archive checks cover every saved element. Numerical checks cover selected
channels/batch entries and complete query/key axes, as reported explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_trunk import dump, now, sha
from alphafold3.model import capture


def readable_name(m, block_base=0):
  stem = f'{m["input_name"]}_seed-{m["seed"]}_pairformer_recycle-{m["recycle"]:02d}_block-{m["layer"] + block_base:02d}'
  if m['operator'] == 'attention':
    stem += f'_pair-attention_{m["direction"]}_chunk-{m["chunk_start"]:04d}-{m["chunk_end"]:04d}'
    return stem if m.get('sampled_i') is None else stem + f'_i-{m["sampled_i"]:04d}'
  if m['operator'] == 'triangle_multiplication':
    return stem + f'_triangle-multiplication_{m["direction"]}'
  assert m['operator'] == 'diffusion_global_attention'
  return stem + '_single-attention'


def archives(out, run):
  expected = json.loads((out / 'expected.json').read_text())
  state = json.loads((out / 'state.json').read_text())
  n = expected['padded_tokens']
  chunk = expected['attention_chunk_size']
  i_step = expected.get('capture_attention_i_step')
  selected_i = expected.get('selected_attention_i')
  wanted_ids = {(r, b) for r in expected['selected_recycles'] for b in expected['selected_layers']}
  index = run / 'events.jsonl'
  rows = [json.loads(line) for line in index.read_text().splitlines()]
  manifest = json.loads((run / 'manifest.json').read_text())
  assert manifest['status'] == 'complete' and manifest['event_count'] == len(rows) == expected['events']
  assert len({row['event_id'] for row in rows}) == expected['events']
  for kind in ('npz', 'json'):
    assert {str(p.relative_to(run)) for p in (run / 'events').glob(f'*.{kind}')} == {row[kind] for row in rows}
  assert not list((run / 'events').glob('*.tmp'))
  counts = Counter()
  identities = Counter()
  chunks = defaultdict(list)
  raw = arrays = compressed = 0
  start = time.monotonic()
  for row in rows:
    descriptor = json.loads((run / row['json']).read_text())
    with zipfile.ZipFile(run / row['npz']) as archive:
      specs = {k: v for k, v in descriptor['tensors'].items() if v is not None}
      assert set(archive.namelist()) == {k + '.npy' for k in specs}
      for name, spec in specs.items():
        with archive.open(name + '.npy') as member:
          version = np.lib.format.read_magic(member)
          assert version in ((1, 0), (2, 0))
          read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                         else np.lib.format.read_array_header_2_0)
          shape, _, dtype = read_header(member)
          assert list(shape) == spec['shape'] and str(dtype) == spec['storage_dtype']
          assert archive.getinfo(name + '.npy').file_size == member.tell() + math.prod(shape) * dtype.itemsize
    record, tensors = capture.load_event(run / row['json'])
    m = record['metadata']
    assert record['event_id'] == row['event_id']
    assert m['stage'] == 'trunk' and (m['recycle'], m['layer']) in wanted_ids
    assert m['recycle_count'] == expected['trunk_passes'] and m['layer_count'] == 48
    assert m['input_name'] == state['input_name'] and m['seed'] == 1 and m['invocation_id'] == 0
    counts[(m['operator'], m['direction'])] += 1
    identities[(m['recycle'], m['layer'], m['operator'], m['direction'])] += 1
    if m['operator'] == 'attention':
      key = (m['recycle'], m['layer'], m['direction'])
      start, end = m['chunk_start'], m['chunk_end']
      assert start % chunk == 0 and end == min(start + chunk, n)
      if i_step is None:
        assert m.get('sampled_i') is None
        chunks[key].append((start, end))
        rows_saved = end - start
      else:
        assert m['i_step'] == i_step and start <= m['sampled_i'] < end
        chunks[key].append(m['sampled_i'])
        rows_saved = 1
      assert m['chunk_total'] == n
      for name in ('query', 'key', 'value', 'result'):
        assert tensors[name].shape == (rows_saved, n, 4, 32)
      assert tensors['mask'].shape == (rows_saved, 1, 1, n)
      assert tensors['bias'].shape == (1, 4, n, n)
    for name, value in tensors.items():
      if value is None:
        continue
      spec = record['tensors'][name]
      assert list(value.shape) == spec['shape'] and str(value.dtype) == spec['dtype']
      # Check in slices to bound temporary boolean memory for large pair arrays.
      flat = value.reshape(-1)
      assert all(np.isfinite(flat[k:k + 1048576]).all() for k in range(0, flat.size, 1048576))
      raw += value.nbytes
      arrays += 1
    compressed += (run / row['npz']).stat().st_size
    print('Checked archive', arrays, 'arrays:', row['npz'], flush=True)
    del tensors
  assert counts == {(c['operator'], c['direction']): c['events'] for c in expected['operators']}, counts
  for recycle, layer in wanted_ids:
    for op, direction in [('triangle_multiplication', 'outgoing'), ('triangle_multiplication', 'incoming'), ('diffusion_global_attention', None)]:
      assert identities[(recycle, layer, op, direction)] == 1
    for direction in ('starting', 'ending'):
      wanted = selected_i if i_step is not None else [(k, min(k + chunk, n)) for k in range(0, n, chunk)]
      assert sorted(chunks[(recycle, layer, direction)]) == wanted
  assert arrays == expected['arrays'] and raw == expected['raw_array_bytes'], (arrays, raw)
  dump(out / 'full_tensor_validation.json', {
      'complete': True, 'events': len(rows), 'arrays': arrays,
      'raw_array_bytes': raw, 'npz_bytes': compressed,
      'all_arrays_finite': True, 'all_crc_and_descriptors_valid': True,
      'all_selected_identities_valid': True, 'attention_capture_coverage_complete': True,
      'capture_attention_i_step': i_step, 'selected_attention_i': selected_i,
      'seconds': time.monotonic() - start,
      'counts': [{'operator': k[0], 'direction': k[1], 'count': v} for k, v in counts.items()]})
  # Rename only after all original archives pass. Keep the original index.
  if not (run / 'filename_map.json').exists():
    mapping = []
    (run / 'events.before_readable_names.jsonl').write_bytes(index.read_bytes())
    (run / 'manifest.before_readable_names.json').write_bytes((run / 'manifest.json').read_bytes())
    for row in rows:
      record = json.loads((run / row['json']).read_text())
      stem = readable_name(record['metadata'], state.get('filename_block_base', 0))
      old = dict(row)
      for kind in ('json', 'npz'):
        src = run / row[kind]
        dest = run / 'events' / f'{stem}.{kind}'
        assert not dest.exists()
        before = src.stat()
        src.rename(dest)
        after = dest.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        row[kind] = str(dest.relative_to(run))
      mapping.append({'old': old, 'new': row})
    tmp = index.with_suffix('.jsonl.tmp')
    tmp.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    tmp.replace(index)
    dump(run / 'filename_map.json', {'complete': True, 'indexing': f'recycle zero-based; block base={state.get("filename_block_base", 0)}',
                                    'all_inodes_sizes_mtimes_preserved': True, 'mapping': mapping})
    manifest['readable_filenames'] = {'mapping': 'filename_map.json', 'zero_based_blocks': state.get('filename_block_base', 0) == 0}
    dump(run / 'manifest.json', manifest)
  assert all((run / row[k]).is_file() for row in rows for k in ('npz', 'json'))
  dump(out / 'coverage_after_readable_names.json', {'complete': True, 'events': expected['events'],
       'index_sha256': sha(index), 'scope': 'All identities, counts and chunks audited before rename; all files preserve inode/size/mtime.'})


def softmax(logits):
  weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
  return weights / weights.sum(axis=-1, keepdims=True)


def error_metrics(reference, actual):
  reference = np.asarray(reference, dtype=np.float32)
  actual = np.asarray(actual, dtype=np.float32)
  difference = reference - actual
  numerator = float(np.sum(difference.astype(np.float64)**2))
  denominator = float(np.sum(actual.astype(np.float64)**2))
  return {'relative_l2': math.sqrt(numerator / max(denominator, 1e-30)),
          'max_abs_error': float(np.max(np.abs(difference))),
          'finite': bool(np.isfinite(reference).all() and np.isfinite(actual).all())}


def numerical(out, run):
  expected = json.loads((out / 'expected.json').read_text())
  expected_groups = 5 * len(expected['selected_recycles']) * len(expected['selected_layers'])
  rows = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
  representatives = {}
  for row in rows:
    record = json.loads((run / row['json']).read_text())
    m = record['metadata']
    representatives.setdefault((m['recycle'], m['layer'], m['operator'], m['direction']), row)
  reports = []
  for key, row in representatives.items():
    record, tensors = capture.load_event(run / row['json'])
    m = record['metadata']
    pieces = []
    if m['operator'] == 'triangle_multiplication':
      # Three full N x N contraction channels; GPU output is BF16.
      for channel in (0, 64, 127):
        a = tensors['a'][channel].astype(np.float32)
        b = tensors['b'][channel].astype(np.float32)
        predicted = a @ b.T if m['direction'] == 'outgoing' else b.T @ a
        actual = tensors['contraction_result'][channel]
        pieces.append(error_metrics(predicted.astype(actual.dtype), actual))
      scope = f'Contraction channels [0,64,127], all {a.shape[0]}x{b.shape[0]} entries; not the gated module output'
    elif m['operator'] == 'attention':
      indices = sorted({0, tensors['query'].shape[0] // 2, tensors['query'].shape[0] - 1})
      for i in indices:
        q, k, v = [tensors[x][i].astype(np.float32).transpose(1, 0, 2) for x in ('query', 'key', 'value')]
        logits = (q @ k.swapaxes(-1, -2)) * np.float32(m['scale'])
        logits += tensors['bias'][0].astype(np.float32)
        logits = np.where(tensors['mask'][i], logits, np.finfo(np.float32).min)
        predicted = (softmax(logits) @ v).transpose(1, 0, 2)
        actual = tensors['result'][i]
        pieces.append(error_metrics(predicted.astype(actual.dtype), actual))
      scope = f'First saved chunk, i offsets {indices}, all heads and all query/key tokens; pre-gate/projection'
    else:
      # Replay single attention one head at a time to limit peak memory.
      q, k, v = [tensors[x] for x in ('query', 'key', 'value')]
      for h in range(q.shape[-2]):
        logits = (q[:, h].astype(np.float32) * np.float32(m['scale'])) @ k[:, h].astype(np.float32).T
        logits += tensors['mask_bias'][0, 0]
        if tensors['pair_logits'] is not None:
          logits += tensors['pair_logits'][h].astype(np.float32)
        weights = softmax(logits).astype(v.dtype).astype(np.float32)
        predicted = weights @ v[:, h].astype(np.float32)
        actual = tensors['result'][:, h]
        pieces.append(error_metrics(predicted.astype(actual.dtype), actual))
      scope = 'All heads and all query/key tokens; pre-gate/projection'
    report = {'recycle': key[0], 'layer': key[1], 'operator': key[2], 'direction': key[3], 'event_id': record['event_id'],
              'chunk_start': m.get('chunk_start'), 'chunk_end': m.get('chunk_end'),
              'sampled_i': m.get('sampled_i'),
              'scope': scope, 'checks': pieces,
              'passed': all(x['finite'] and x['relative_l2'] <= 0.02 for x in pieces)}
    reports.append(report)
    print('Numerical check:', key, report['passed'], max(x['relative_l2'] for x in pieces), flush=True)
    dump(out / 'value_validation.json', {'complete': len(reports) == expected_groups,
         'passed': all(r['passed'] for r in reports), 'relative_l2_tolerance': 0.02,
         'scope': 'Representative numerical checks, with channel/batch coverage recorded per event',
         'results': reports})
    del tensors
  assert len(reports) == expected_groups and all(r['passed'] for r in reports)


def finalize(out):
  expected = json.loads((out / 'expected.json').read_text())
  state = json.loads((out / 'state.json').read_text())
  assert state['status'] == 'inference_complete' and state['process_exit_code'] == 0
  report = json.loads((out / 'full_tensor_validation.json').read_text())
  values = json.loads((out / 'value_validation.json').read_text())
  assert report['complete'] and values['complete'] and values['passed']
  with np.load(out / 'trunk_outputs.npz', allow_pickle=False) as archive:
    assert set(archive.files) == {'pair_embeddings', 'single_embeddings', 'target_feat'}
    for k in archive.files:
      value = archive[k]
      assert list(value.shape) == state['outputs'][k]['shape']
      assert str(value.dtype) == 'float32' and np.isfinite(value).all()
  assert sha(out / 'trunk_outputs.npz') == state['outputs_sha256']
  state.update(status='complete', validated_at=now(), validations={
      'all_events_and_arrays': True, 'attention_capture_coverage_complete': True,
      'events': expected['events'], 'arrays': expected['arrays'],
      'representative_numerical_events': len(values['results']), 'outputs_roundtrip_exact': True},
      captured_raw_array_bytes=report['raw_array_bytes'],
      captured_npz_bytes=report['npz_bytes'],
      numerical_max_relative_l2=max(x['relative_l2'] for r in values['results'] for x in r['checks']))
  dump(out / 'state.json', state)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('output_dir', type=Path)
  parser.add_argument('--phase', choices=('archives', 'numerical', 'finalize'), required=True)
  args = parser.parse_args()
  state = json.loads((args.output_dir / 'state.json').read_text())
  run = Path(state['capture_run'])
  if args.phase == 'archives':
    archives(args.output_dir, run)
  elif args.phase == 'numerical':
    numerical(args.output_dir, run)
  else:
    finalize(args.output_dir)


if __name__ == '__main__':
  main()
