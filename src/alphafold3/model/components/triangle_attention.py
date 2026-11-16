"""Dispatch for the gate-free triangle attention kernel."""

import functools
import importlib
import math
from typing import Protocol, cast

import jax
import tokamax


class Kernel(Protocol):
  """JAX callable used in place of the built-in attention kernel."""

  def __call__(
      self,
      *,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      bias: jax.Array,
      mask: jax.Array,
      scale: float,
  ) -> jax.Array: ...


@functools.cache
def load_kernel(spec: str) -> Kernel:
  """Resolves an importable `MODULE:FUNCTION` once per process."""
  module_name, separator, function_name = spec.partition(':')
  if (
      not separator
      or not all(part.isidentifier() for part in module_name.split('.'))
      or not function_name.isidentifier()
  ):
    raise ValueError(f'Expected triangle attention MODULE:FUNCTION, got {spec!r}')
  try:
    kernel = getattr(importlib.import_module(module_name), function_name)
  except Exception as exc:
    raise ValueError(f'Cannot load triangle attention kernel {spec!r}: {exc}') from exc
  if not callable(kernel):
    raise TypeError(f'Triangle attention kernel {spec!r} must be callable')
  return cast(Kernel, kernel)


def attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    bias: jax.Array,
    mask: jax.Array,
    implementation: tokamax.DotProductAttentionImplementation,
    kernel_spec: str | None = None,
    scale: float | None = None,
) -> jax.Array:
  """Returns attention before gating, using the supplied chunk and layout."""
  if kernel_spec is None:
    return tokamax.dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        scale=scale,
        implementation=implementation,
    )

  kernel = load_kernel(kernel_spec)
  output = kernel(
      query=query,
      key=key,
      value=value,
      bias=bias,
      mask=mask,
      scale=1 / math.sqrt(query.shape[-1]) if scale is None else scale,
  )
  if not isinstance(output, jax.Array):
    raise TypeError(
        f'Triangle attention kernel {kernel_spec!r} must return a JAX array'
    )
  if output.shape != query.shape or output.dtype != query.dtype:
    raise ValueError(
        f'Triangle attention kernel {kernel_spec!r} returned {output.shape} '
        f'{output.dtype}, expected {query.shape} {query.dtype}'
    )
  return output
