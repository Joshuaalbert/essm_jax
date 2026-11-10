"""Compare the same float64 model across source checkouts; run each in a fresh process."""
import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from essm_jax.essm import ExtendedStateSpaceModel

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--size', type=int, default=64)
parser.add_argument('--steps', type=int, default=256)
parser.add_argument('--reference', help='Saved .npz outputs from the comparison checkout')
parser.add_argument('--trace-repetitions', type=int, default=5)
parser.add_argument('--run-repetitions', type=int, default=9)
args = parser.parse_args()
if min(args.size, args.steps, args.trace_repetitions, args.run_repetitions) < 1:
    parser.error('sizes and repetition counts must be positive')
jax.config.update('jax_enable_x64', True)
n, steps = args.size, args.steps
identity = jnp.eye(n, dtype=jnp.float64)
prior = tfp.distributions.MultivariateNormalTriL(jnp.zeros(n), identity)


def transition(z, t, t_next, scale):
    return tfp.distributions.MultivariateNormalTriL(0.8 * z + 0.05 * jnp.sin(z), 0.3 * scale * identity)


def observation(z, t, scale):
    return tfp.distributions.MultivariateNormalTriL(z + 0.1 * jnp.sin(z), 0.6 * identity)


model = ExtendedStateSpaceModel(transition, observation, prior)
observations = jnp.sin(jnp.arange(steps * n, dtype=jnp.float64).reshape((steps, n)) * 0.01)
scale = jnp.array(1., dtype=jnp.float64)
jax.block_until_ready((observations, scale))
report = dict(platform=platform.platform(), jax=jax.__version__, jaxlib=jaxlib.__version__,
              devices=[str(d) for d in jax.devices()], dtype='float64', latent_size=n,
              observation_size=n, steps=steps, trace_repetitions=args.trace_repetitions,
              run_repetitions=args.run_repetitions)
outputs = {}
for name, fn in [('filter', model.forward_filter), ('log_prob', model.log_prob),
                 ('value_and_grad', jax.value_and_grad(model.log_prob, argnums=1))]:
    lowering_times, compile_times = [], []
    for _ in range(args.trace_repetitions):
        jax.clear_caches()
        start = time.perf_counter()
        lowered = jax.jit(fn).lower(observations, scale)
        lowering_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        compiled = lowered.compile()
        compile_times.append(time.perf_counter() - start)
    result = jax.block_until_ready(compiled(observations, scale))
    durations = []
    for _ in range(args.run_repetitions):
        start = time.perf_counter()
        jax.block_until_ready(compiled(observations, scale))
        durations.append(time.perf_counter() - start)
    memory = compiled.memory_analysis()
    report[name] = {}
    for phase, timings in [('lower', lowering_times), ('compile', compile_times), ('run', durations)]:
        report[name][phase] = dict(median_s=float(np.median(timings)),
                                   q25_s=float(np.quantile(timings, 0.25)),
                                   q75_s=float(np.quantile(timings, 0.75)))
    report[name]['compiler_memory'] = {
        field: getattr(memory, field) for field in
        ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')
    }
    Path(args.output + '.' + name + '.hlo').write_text(compiled.as_text())
    for i, leaf in enumerate(jax.tree.leaves(result)):
        outputs[f'{name}_{i}'] = np.asarray(leaf)
    print(name, report[name], flush=True)

# Inspect the scan's outputs before an enclosing jit can eliminate unused history.
program = jax.make_jaxpr(model.log_prob)(observations, scale)
scans = [eq for eq in program.jaxpr.eqns if eq.primitive.name == 'scan']
report['log_prob_scan_output_bytes'] = sum(
    int(np.prod(var.aval.shape)) * np.dtype(var.aval.dtype).itemsize
    for scan in scans for var in scan.outvars
)
if args.reference:
    with np.load(args.reference) as reference:
        assert set(outputs) == set(reference.files), 'Reference output structure differs'
        for name, actual in outputs.items():
            np.testing.assert_allclose(actual, reference[name], rtol=1e-10, atol=1e-10, equal_nan=False)
    report['reference'] = dict(path=args.reference, rtol=1e-10, atol=1e-10, passed=True)
Path(args.output + '.json').write_text(json.dumps(report, indent=2) + '\n')
np.savez(args.output + '.npz', **outputs)
