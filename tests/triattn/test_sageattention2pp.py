"""Tests the SageAttention2++ plugin through the triangle attention wrapper."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alphafold3.model.components import triangle_attention
from mod.triattn import example_attention


_KERNEL = 'mod.triattn.sageattention2pp:attention'


def _run_kernel(kernel_spec, scale):
  @jax.jit
  def run(query, key, value, bias, mask):
    return triangle_attention.attention(
        query, key, value, bias=bias, mask=mask, scale=scale,
        implementation='xla', kernel_spec=kernel_spec,
    )

  return run


@pytest.mark.parametrize(
    ('dtype', 'scale'), ((jnp.bfloat16, 0.25), (jnp.float32, -0.25))
)
def test_bias_mask_and_tail_tiles(dtype, scale):
  rng = np.random.default_rng(0)
  length = 131
  shape = (3, length, 2, 17)
  q, k, v = (jnp.asarray(rng.normal(size=shape), dtype) for _ in range(3))
  bias = rng.normal(size=(1, 2, length, length))
  bias[:, :, 0, :128] = -np.inf
  bias = jnp.asarray(bias, dtype)

  mask = np.zeros((3, 1, 1, length), dtype=bool)
  mask[0, ..., ::2] = True
  mask[1, ..., 130] = True  # Two entirely masked key tiles before the valid key.
  mask = jnp.asarray(mask)

  actual = _run_kernel(_KERNEL, scale)(q, k, v, bias, mask)
  expected = example_attention.attention(
      query=q.astype(jnp.float32), key=k.astype(jnp.float32),
      value=v.astype(jnp.float32), bias=bias, mask=mask, scale=scale,
  )
  assert actual.shape == shape
  assert actual.dtype == dtype
  actual, expected = (np.asarray(x, dtype=np.float32) for x in (actual, expected))
  rtol = 4.5e-3 if dtype == jnp.bfloat16 else 1e-3
  np.testing.assert_allclose(actual, expected, atol=4e-4, rtol=rtol)
  mean = np.asarray(v[2], dtype=np.float32).mean(axis=0).astype(dtype)
  np.testing.assert_allclose(actual[2, 0], mean, atol=1e-7, rtol=0)
  np.testing.assert_array_equal(
      actual[2], np.broadcast_to(actual[2, 0], shape[1:])
  )


def test_residual_correction_reduces_error():
  rng = np.random.default_rng(1)
  shape = (1, 37, 1, 8)
  q, k, v = (jnp.asarray(rng.normal(size=shape), jnp.float32) for _ in range(3))
  bias = jnp.asarray(rng.normal(size=(1, 1, 37, 37)), jnp.float32)
  mask = jnp.ones((1, 1, 1, 37), dtype=bool)
  scale = 0.5
  expected = example_attention.attention(
      query=q, key=k, value=v, bias=bias, mask=mask, scale=scale
  )

  corrected = _run_kernel(_KERNEL, scale)(q, k, v, bias, mask)
  uncorrected = _run_kernel(_KERNEL + '_uncorrected', scale)(q, k, v, bias, mask)
  assert np.linalg.norm(corrected - expected) < 0.05 * np.linalg.norm(
      uncorrected - expected
  )
