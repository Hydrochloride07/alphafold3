"""Checks bias encoding and mask semantics outside the CUDA kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphafold3.model.components import triangle_attention
from mod.triattn import example_attention, sageattention2pp_cuda


@pytest.mark.parametrize(
    ('dtype', 'scale'), ((jnp.bfloat16, 0.25), (jnp.float32, -0.25))
)
def test_bias_mask_and_tail_tiles(monkeypatch, dtype, scale):
  torch = pytest.importorskip('torch')
  calls = []

  def dense_attention(query, key, value, *, sm_scale):
    calls.append((query.shape[0], query.shape[1], key.shape[1]))
    logits = torch.einsum('bqhd,bkhd->bhqk', query.float(), key.float())
    weights = (logits * sm_scale).softmax(dim=-1)
    return torch.einsum('bhqk,bkhd->bqhd', weights, value.float())

  monkeypatch.setattr(
      sageattention2pp_cuda, '_cuda_backend', lambda: (torch, 'cpu', dense_attention)
  )
  rng = np.random.default_rng(0)
  shape = (11, 103, 2, 32)
  q, k, v = (jnp.asarray(rng.integers(-4, 5, shape) / 8, dtype) for _ in range(3))
  bias = jnp.asarray(rng.integers(-8, 9, (1, 2, 103, 103)) / 8, dtype)
  mask = np.zeros((11, 1, 1, 103), dtype=bool)
  mask[:9, ..., ::2] = True
  mask[9, ..., 1::3] = True
  mask = jnp.asarray(mask)

  @jax.jit
  def run(q, k, v, bias, mask):
    return triangle_attention.attention(
        q, k, v, bias=bias, mask=mask, scale=scale, implementation='xla',
        kernel_spec='mod.triattn.sageattention2pp_cuda:attention',
    )

  actual = run(q, k, v, bias, mask)
  expected = example_attention.attention(
      query=q, key=k, value=v, bias=bias, mask=mask, scale=scale
  )
  assert actual.shape == shape
  assert actual.dtype == dtype
  np.testing.assert_allclose(
      np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32),
      atol=5e-4, rtol=0,
  )
  assert sorted(calls) == sorted(
      (batch, width, keys)
      for batch, keys in ((8, 52), (1, 52), (1, 34))
      for width in (96, 7)
  )
