"""Storage-oracle and indexing compatibility checks for the trunk CLI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytest
from capture_trunk import storage_estimate
from validate_trunk_capture import readable_name


@pytest.mark.parametrize('tokens,bucket,events,arrays,raw', [
    (1188, 1280, 69, 420, 23954741760),
    (70, 128, 15, 96, 234064128),
    (2445, 2560, 489, 2940, 117803627520),
])
def test_recycle10_three_blocks_space(tokens, bucket, events, arrays, raw):
  estimate = storage_estimate(tokens, bucket, [10], [1, 16, 32], 10)
  assert (estimate['events'], estimate['arrays'], estimate['raw_array_bytes']) == (events, arrays, raw)
  assert estimate['selected_layers'] == [0, 15, 31]
  assert estimate['trunk_passes'] == 11


def test_previous_demo_cut_estimate_is_preserved():
  e = storage_estimate(1257, 1280, [0], [1], 0)
  assert (e['events'], e['arrays'], e['raw_array_bytes']) == (23, 140, 7984913920)


def test_demo_sampled_space_keeps_compute_chunks():
  e = storage_estimate(2445, 2560, [10], [1, 16, 32], 10, attention_i_step=256)
  assert e['attention_chunk_size'] == 32
  assert e['selected_attention_i'] == list(range(0, 2560, 256))
  assert (e['events'], e['arrays'], e['raw_array_bytes']) == (69, 420, 55636331520)


@pytest.mark.parametrize('base,block', [(0, '00'), (1, '01')])
def test_filename_block_base_is_explicit(base, block):
  m = {'input_name': 'theta', 'seed': 1, 'recycle': 10, 'layer': 0,
       'operator': 'diffusion_global_attention'}
  assert readable_name(m, base) == f'theta_seed-1_pairformer_recycle-10_block-{block}_single-attention'
