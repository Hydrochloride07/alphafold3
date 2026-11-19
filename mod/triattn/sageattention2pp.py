"""Portable SageAttention2++ (8+8) numerics for biased triangle attention.

Uses per-token INT8 Q/K, E4M3 P/V and FP16 partial sums buffered in FP32.
The default entry point adds one INT8 residual term for Q/K and two E4M3
residual terms for P/V. JAX handles device placement and JIT.
"""

import jax
import jax.numpy as jnp


_QUERY_TILE = 128
_PV_STEP = 32
_KEY_TILE = 2 * _PV_STEP
# 64 * 224 * 4.5 = 64512, within the FP16 finite range.
_PROBABILITY_RANGE = 224.0
_VALUE_RANGE = 4.5


def _quantize(x, *, axis, limit, dtype, components): # 多组残差
  """Returns successively quantized (operand, scale) terms."""
  terms = []
  for _ in range(components):
    scale = jnp.max(jnp.abs(x), axis=axis, keepdims=True) / limit
    scale = jnp.where(scale > 0, scale, 1.0) # >0 是否有浮点问题？是否改成 >eps
    normalized = jnp.clip(x / scale, -limit, limit)
    if dtype == jnp.int8:
      normalized = jnp.rint(normalized)
    quantized = normalized.astype(dtype)
    terms.append((quantized, scale))
    x = x - quantized.astype(jnp.float32) * scale
  return tuple(terms)


def _qk_dot(q_terms, k_terms):
  """Multiplies the INT8 components and dequantizes each INT32 dot."""
  logits = jnp.zeros((q_terms[0][0].shape[0], k_terms[0][0].shape[0]), jnp.float32)
  for q, q_scale in q_terms:
    for k, k_scale in k_terms:
      dot = jnp.dot(q, k.T, preferred_element_type=jnp.int32)
      logits += dot.astype(jnp.float32) * q_scale * k_scale.T
  return logits


def _pv_dot(p_terms, v_terms):
  """Multiplies E4M3 components with two FP16 steps per FP32 buffer update."""
  partial = jnp.zeros((p_terms[0][0].shape[0], v_terms[0][0].shape[1]), jnp.float32)
  precision = jax.lax.Precision.HIGHEST
  for p, p_scale in p_terms:
    p = p.astype(jnp.float32)
    for v, v_scale in v_terms:
      # An E4M3 product is exactly representable in FP32. Use portable dots
      # for each 32-key step and round at the FP16 accumulator boundaries.
      v = v.astype(jnp.float32)
      first = jnp.dot(p[:, :_PV_STEP], v[:_PV_STEP], precision=precision)
      acc = first.astype(jnp.float16).astype(jnp.float32)
      second = jnp.dot(p[:, _PV_STEP:], v[_PV_STEP:], precision=precision)
      acc = (acc + second).astype(jnp.float16).astype(jnp.float32)
      partial += acc * p_scale * v_scale
  return partial


def _attention_head(query, key, value, bias, keep, *, scale, residual):
  """Evaluates one head with tiled QK and online softmax."""
  query_length, head_dim = query.shape
  blocks = keep.size // _KEY_TILE
  query_tile = min(query_length, _QUERY_TILE)
  qk_components = 2 if residual else 1
  pv_components = 3 if residual else 1

  # Subtracting the same K mean shifts every valid logit in a row equally.
  key_mean = jnp.sum(jnp.where(keep[:, None], key, 0.0), axis=0) / keep.sum()
  key = jnp.where(keep[:, None], key - key_mean, 0.0).reshape(blocks, _KEY_TILE, head_dim)
  value = jnp.where(keep[:, None], value, 0.0).reshape(blocks, _KEY_TILE, head_dim)
  key_terms = _quantize(
      key, axis=-1, limit=127.0, dtype=jnp.int8, components=qk_components
  )
  value_terms = _quantize(
      value,
      axis=-2,
      limit=_VALUE_RANGE,
      dtype=jnp.float8_e4m3fn,
      components=pv_components,
  )
  keep = keep.reshape(blocks, _KEY_TILE)

  def query_block(block_index):
    start = block_index * query_tile
    q = jax.lax.dynamic_slice_in_dim(query, start, query_tile)
    q_mean = q.mean(axis=0, keepdims=True)
    q_terms = _quantize(
        q - q_mean,
        axis=-1,
        limit=127.0,
        dtype=jnp.int8,
        components=qk_components,
    )

    def key_block(index, state):
      def update(state):
        row_max, denominator, numerator = state
        k_terms = jax.tree.map(lambda x: x[index], key_terms)
        v_terms = jax.tree.map(lambda x: x[index], value_terms)
        # Restore the Q mean removed before quantization. Add pair bias in
        # logit units, after QK dequantization and before applying the mask.
        mean_correction = jnp.dot(
            q_mean, key[index].T, precision=jax.lax.Precision.HIGHEST
        )
        logits = (_qk_dot(q_terms, k_terms) + mean_correction) * scale
        bias_tile = jax.lax.dynamic_slice(
            bias, (start, index * _KEY_TILE), (query_tile, _KEY_TILE)
        )
        logits = jnp.where(keep[index], logits + bias_tile, -jnp.inf)
        next_max = jnp.maximum(row_max, logits.max(axis=-1, keepdims=True))
        # A bias tile can assign -inf to every valid key for a query.
        # Keep that tile's weights at zero until a finite maximum arrives.
        safe_max = jnp.where(jnp.isneginf(next_max), 0.0, next_max)
        previous_weight = jnp.exp(row_max - safe_max)
        weights = jnp.exp(logits - safe_max)
        p_terms = _quantize(
            weights,
            axis=None,
            limit=_PROBABILITY_RANGE,
            dtype=jnp.float8_e4m3fn,
            components=pv_components,
        )
        return (
            next_max,
            denominator * previous_weight + weights.sum(axis=-1, keepdims=True),
            numerator * previous_weight + _pv_dot(p_terms, v_terms),
        )

      return jax.lax.cond(jnp.any(keep[index]), update, lambda x: x, state)

    initial = (
        jnp.full((query_tile, 1), -jnp.inf, dtype=jnp.float32),
        jnp.zeros((query_tile, 1), jnp.float32),
        jnp.zeros((query_tile, head_dim), jnp.float32),
    )
    _, denominator, numerator = jax.lax.fori_loop(0, blocks, key_block, initial)
    return numerator / denominator

  indices = jnp.arange(query_length // query_tile)
  return jax.lax.map(query_block, indices).reshape(query.shape)


def _attention(*, query, key, value, bias, mask, scale, residual):
  batch, length = query.shape[:2]
  if batch == 0 or length == 0:
    return jnp.empty_like(query)
  query_padding = -length % min(length, _QUERY_TILE)
  key_padding = -length % _KEY_TILE

  def head_major(x, padding):
    return jnp.pad(
        x.astype(jnp.float32).transpose(0, 2, 1, 3),
        ((0, 0), (0, 0), (0, padding), (0, 0)),
    )

  padded_query = head_major(query, query_padding)
  padded_key = head_major(key, key_padding)
  padded_value = head_major(value, key_padding)
  padded_bias = jnp.pad(
      bias[0].astype(jnp.float32), ((0, 0), (0, query_padding), (0, key_padding))
  )
  keep = jnp.pad(mask[:, 0, 0], ((0, 0), (0, key_padding)))

  def batch_row(inputs):
    q, k, v, valid = inputs

    def attend():
      return jax.vmap(
          lambda q, k, v, b: _attention_head(
              q, k, v, b, valid, scale=scale, residual=residual
          )
      )(q, k, v, padded_bias)

    def all_masked():
      # The contract defines this case over the original sequence, including
      # AF3's model padding. Exclude only the extra padding added by this code.
      return jnp.broadcast_to(v[:, :length].mean(axis=1, keepdims=True), q.shape)

    return jax.lax.cond(jnp.any(valid), attend, all_masked)

  result = jax.lax.map(batch_row, (padded_query, padded_key, padded_value, keep))
  return result[:, :, :length].transpose(0, 2, 1, 3).astype(query.dtype)


def attention(*, query, key, value, bias, mask, scale) -> jax.Array:
  """SageAttention2++ with FP32 additive bias and quantized residuals."""
  return _attention(
      query=query, key=key, value=value, bias=bias, mask=mask, scale=scale,
      residual=True,
  )


def attention_uncorrected(*, query, key, value, bias, mask, scale) -> jax.Array:
  """Single-component INT8 and E4M3 ablation."""
  return _attention(
      query=query, key=key, value=value, bias=bias, mask=mask, scale=scale,
      residual=False,
  )
