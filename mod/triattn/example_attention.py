"""Dense JAX example for the AlphaFold 3 triangle attention interface."""

import jax
import jax.numpy as jnp


def attention(
    *,
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    bias: jax.Array,
    mask: jax.Array,
    scale: float,
) -> jax.Array:
  """Computes masked attention in FP32 and returns the input dtype."""
  precision = jax.lax.Precision.HIGHEST
  logits = jnp.einsum(
      'iqhd,ikhd->ihqk',
      query.astype(jnp.float32),
      key.astype(jnp.float32),
      precision=precision,
  )
  logits = logits * scale + bias.astype(jnp.float32)
  logits = jnp.where(mask, logits, -jnp.inf)
  # Equal logits give uniform weights over the key axis on fully masked rows.
  logits = jnp.where(jnp.any(mask, axis=-1, keepdims=True), logits, 0.0)
  weights = jax.nn.softmax(logits, axis=-1)
  output = jnp.einsum(
      'ihqk,ikhd->iqhd', weights, value.astype(jnp.float32), precision=precision
  )
  return output.astype(query.dtype)
