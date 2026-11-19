"""Triangle attention adapter for the upstream SageAttention 2++ CUDA kernel."""

import functools

import jax
import numpy as np


_MAX_HEAD_DIM = 128
_BATCH_TILE = 8


@functools.cache
def _cuda_backend():
  try:
    import torch
  except ImportError as exc:
    raise RuntimeError('SageAttention 2++ requires CUDA-enabled PyTorch') from exc
  if not torch.cuda.is_available():
    raise RuntimeError('SageAttention 2++ requires a visible CUDA GPU')
  device = torch.device('cuda:0')
  capability = torch.cuda.get_device_capability(device)
  if capability not in ((8, 9), (12, 0)):
    actual = ''.join(map(str, capability))
    raise RuntimeError(
        'SageAttention 2++ v2.2.0 requires sm_89 or sm_120 for this FP8 '
        f'kernel. Found {torch.cuda.get_device_name(device)} (sm_{actual}).'
    )

  try:
    from sageattention import core
  except ImportError as exc:
    raise RuntimeError('Install the SageAttention v2.2.0 CUDA extension') from exc
  if not core.SM89_ENABLED:
    raise RuntimeError('The SageAttention FP8 CUDA extension could not be loaded')

  # The v2.2.0 dispatcher uses per-warp quantization on Blackwell.
  kernel = functools.partial(
      core.sageattn_qk_int8_pv_fp8_cuda,
      tensor_layout='NHD',
      is_causal=False,
      qk_quant_gran='per_thread' if capability == (8, 9) else 'per_warp',
      pv_accum_dtype='fp32+fp16',
  )
  return torch, device, kernel


def _run_attention(query, key, value, bias, mask, *, scale, backend):
  torch, device, kernel = backend
  query_dtype = np.asarray(query).dtype
  batch, length, heads, dim = query.shape
  bias_columns = min(length, _MAX_HEAD_DIM - dim)
  patterns, pattern_ids = np.unique(
      np.asarray(mask)[:, 0, 0, :], axis=0, return_inverse=True
  )
  pattern_ids = pattern_ids.reshape(-1)
  logit_scale = abs(scale) or 1.0

  def to_device(array):
    # torch.tensor cannot directly consume NumPy's ml_dtypes.bfloat16.
    return torch.tensor(np.asarray(array, dtype=np.float32), device=device)

  with torch.inference_mode():
    q, k, v, b = map(to_device, (query, key, value, bias))
    if scale <= 0:
      k = k * (scale / logit_scale)
    result = torch.empty((batch, length, heads, dim), device=device, dtype=v.dtype)
    # Keep Q, K and bias magnitudes. Only the selector needs the reciprocal
    # of the softmax scale applied inside the kernel.
    selectors = torch.eye(bias_columns, device=device, dtype=q.dtype) / logit_scale

    for pattern_idx, keep in enumerate(patterns):
      batch_rows = np.flatnonzero(pattern_ids == pattern_idx)
      keys = torch.as_tensor(np.flatnonzero(keep), device=device)
      # Bound temporary storage for the copies of the tile bias appended to K.
      for group_start in range(0, batch_rows.size, _BATCH_TILE):
        rows = torch.as_tensor(
            batch_rows[group_start : group_start + _BATCH_TILE], device=device
        )
        if keys.numel() == 0:
          result[rows] = v[rows].mean(dim=1, keepdim=True)
          continue

        group_q = q[rows]
        group_k = k[rows[:, None], keys]
        group_v = v[rows[:, None], keys]
        for start in range(0, length, bias_columns):
          stop = min(start + bias_columns, length)
          width = stop - start
          one_hot = selectors[:width, :width][None, :, None, :]
          tile_bias = (
              b[:, :, start:stop, :].index_select(-1, keys).permute(0, 3, 1, 2)
          )
          tile_q = torch.cat(
              (group_q[:, start:stop], one_hot.expand(rows.numel(), -1, heads, -1)),
              dim=-1,
          )
          tile_k = torch.cat(
              (group_k, tile_bias.expand(rows.numel(), -1, -1, -1)), dim=-1
          )
          tile_v = torch.nn.functional.pad(group_v, (0, width))
          output = kernel(
              tile_q.to(torch.bfloat16),
              tile_k.to(torch.bfloat16),
              tile_v.to(torch.bfloat16),
              sm_scale=logit_scale,
          )
          result[rows, start:stop] = output[..., :dim].to(result.dtype)

    return np.asarray(result.cpu().numpy(), dtype=query_dtype)


def attention(*, query, key, value, bias, mask, scale) -> jax.Array:
  """Runs masked, biased triangle attention with the upstream FP8 kernel."""
  if not 0 < query.shape[-1] < _MAX_HEAD_DIM:
    raise ValueError('SageAttention 2++ bias encoding requires 1–127 channels per head')
  backend = _cuda_backend()
  if not query.shape[0] or not query.shape[1]:
    return query
  return jax.pure_callback(
      functools.partial(_run_attention, scale=scale, backend=backend),
      jax.ShapeDtypeStruct(query.shape, query.dtype),
      query,
      key,
      value,
      bias,
      mask,
      vmap_method='sequential',
  )
