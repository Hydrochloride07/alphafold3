"""Profile AF3's LayerNorm with synthetic data; no model weights required."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu')
parser.add_argument('--rows', type=int, default=128)
parser.add_argument('--channels', type=int, default=384)
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--output-dir', type=Path, default=Path('.local/profiles/layernorm'))
parser.add_argument('--trace', action='store_true')
args = parser.parse_args()
if min(args.rows, args.channels, args.iterations) < 1:
    parser.error('rows, channels and iterations must be positive')
os.environ['JAX_PLATFORMS'] = 'cuda' if args.backend == 'gpu' else 'cpu'

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from alphafold3.model.components import haiku_modules

device = jax.devices(args.backend)[0]
args.output_dir.mkdir(parents=True, exist_ok=True)
with jax.default_device(device):
    x = jax.random.normal(jax.random.PRNGKey(0), (args.rows, args.channels))
    net = hk.without_apply_rng(hk.transform(
        lambda x: haiku_modules.LayerNorm(name='benchmark_norm')(x)))
    params = net.init(jax.random.PRNGKey(1), x)
    forward = jax.jit(net.apply)
    jax.block_until_ready((params, x))
    start = time.perf_counter()
    y = jax.block_until_ready(forward(params, x))
    first_call = time.perf_counter() - start
    # Correctness check is outside all measured regions.
    host_x = np.asarray(x)
    reference = (host_x - host_x.mean(-1, keepdims=True)) / np.sqrt(
        host_x.var(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(np.asarray(y), reference, rtol=2e-4, atol=2e-4)
    times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        jax.block_until_ready(forward(params, x))
        times.append(time.perf_counter() - start)
    if args.trace:
        with jax.profiler.trace(str(args.output_dir / 'trace'), create_perfetto_trace=True):
            with jax.profiler.TraceAnnotation('af3_layer_norm'):
                for _ in range(args.iterations):
                    jax.block_until_ready(forward(params, x))

report = {
    'module': 'alphafold3.model.components.haiku_modules.LayerNorm',
    'backend': args.backend, 'device': str(device), 'device_kind': device.device_kind,
    'shape': [args.rows, args.channels], 'dtype': str(x.dtype), 'seed': 0,
    'versions': {name: importlib.metadata.version(name)
                 for name in ('alphafold3', 'jax', 'jaxlib', 'dm-haiku')},
    'first_call_seconds_including_compile': first_call,
    'steady_seconds': times, 'median_seconds': statistics.median(times),
    'correctness': 'passed',
}
(args.output_dir / 'timing.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
