# Extended State Space Models in JAX

Given a potentially non-linear state space model this allows you to solve the forward and backward inference steps, for
linear space space models this is equivalent to the Kalman and Rauch-Tung-Striebel recursions.

Support for Python 3.10+.

## Development and CI

Use the `essm_py` conda environment for local development:

```bash
conda run -n essm_py pip install -e '.[tests]'
conda run -n essm_py pytest
conda run -n essm_py pytest --cov --cov-config=pyproject.toml --cov-report=term-missing --cov-report=xml --cov-report=html
conda run -n essm_py python -m flake8 src --select=E9,F63,F7,F82
conda run -n essm_py ruff check src
```

The [CI workflow](.github/workflows/unittests.yml) runs on pull requests to `main`
and `develop`, pushes to `main`, and manual dispatch. It tests Python 3.10–3.14
on CPU, requires 80% combined line-and-branch coverage of production code, and
uploads test results and XML/HTML coverage reports for each Python version.
After the test matrix passes, it builds and checks the source distribution and
wheel and uploads them as artifacts.

`pytest` excludes the hardware-dependent timing benchmark by default. Run it
explicitly with `conda run -n essm_py pytest -m benchmark`.

For separate tracing, compilation, and synchronized execution timings, run:

```bash
JAX_PLATFORMS=cpu conda run -n essm_py python benchmarks/filtering.py --output /tmp/essm-benchmark
```

This writes JSON measurements, numerical outputs, and compiled HLO alongside
the output prefix. Use `--size 128 --steps 1000` for the larger case and
`--reference /tmp/baseline.npz` to check values and parameter gradients against
another checkout. Measurements and their limits are recorded in [LEARNINGS.md](LEARNINGS.md).

Read [AGENTS.md](AGENTS.md), [INVARIANTS.md](INVARIANTS.md), and
[LEARNINGS.md](LEARNINGS.md) before making changes. Runtime requirements use
`tfp-nightly` for the JAX substrate and allow NumPy 2 for modern Python versions.
All packaging and dependencies are declared in `pyproject.toml`, including the
`tests` and `examples` extras; no setup script or requirements files are needed.

Releases remain manual. `bash update_pypi.sh` builds and checks distributions in
a fresh temporary directory using `essm_py`; `bash update_pypi.sh --upload` also
uploads those checked files to PyPI using your configured Twine credentials.

## Example

Define transition and observation callbacks and an initial state prior using
`MultivariateNormalLinearOperator` distributions from `tensorflow_probability`.
Pass callback inputs with the keyword-only `transition_args=()` and
`observation_args=()` tuples. Each callback receives its own tuple as a single
final argument and unpacks it as needed, even when the tuple is empty.

This replaces the shared positional `*theta` API. When migrating, add an argument
tuple parameter to each callback and move its inputs into the corresponding
keyword. `incremental_predict` and transition Jacobian methods take only
`transition_args`; `incremental_update` and observation Jacobian methods take
only `observation_args`. Filtering, smoothing, likelihoods, and sampling take
both. The tuples may contain nested JAX pytrees and differentiable arrays.

```python
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from essm_jax.essm import ExtendedStateSpaceModel

tfpd = tfp.distributions


def transition_fn(z, t, t_next, args):
    process_variance, = args
    mean = z + jnp.sin(2 * jnp.pi * t / 10 * z)
    cov = process_variance * jnp.eye(np.size(z))
    return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))


def observation_fn(z, t, args):
    variance_rate, = args
    mean = z
    cov = t * variance_rate * jnp.eye(np.size(z))
    return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))


n = 1

initial_state_prior = tfpd.MultivariateNormalTriL(jnp.zeros(n), jnp.eye(n))

essm = ExtendedStateSpaceModel(
    transition_fn=transition_fn,
    observation_fn=observation_fn,
    initial_state_prior=initial_state_prior,
    materialise_jacobians=False,  # Fast
    more_data_than_params=False  # if observation is bigger than latent we can speed it up.
)

T = 100
transition_args = (0.1,)
observation_args = (0.01,)
samples = essm.sample(
    jax.random.PRNGKey(0), num_time=T,
    transition_args=transition_args, observation_args=observation_args,
)

# Suppose we only observe every 3rd observation
mask = jnp.arange(T) % 3 != 0

# Marginal likelihood, p(x[:]) = prod_t p(x[t] | x[:t-1])
log_prob = essm.log_prob(
    samples.observation, mask=mask,
    transition_args=transition_args, observation_args=observation_args,
)
print(log_prob)

# Filtered latent distribution, p(z[t] | x[:t])
filter_result = essm.forward_filter(
    samples.observation, mask=mask,
    transition_args=transition_args, observation_args=observation_args,
)

# Smoothed latent distribution, p(z[t] | x[:]), i.e. past latents given all future observations
# Including new estimate for prior state p(z[0])
smooth_result, posterior_prior = essm.backward_smooth(
    filter_result, include_prior=True,
    transition_args=transition_args, observation_args=observation_args,
)
print(smooth_result)

# Forward simulate the model
forward_samples = essm.forward_simulate(
    key=jax.random.PRNGKey(0),
    num_time=25,
    filter_result=filter_result,
    transition_args=transition_args,
    observation_args=observation_args,
)

import pylab as plt

plt.plot(samples.t, samples.latent[:, 0], label='latent')
plt.plot(filter_result.t, filter_result.filtered_mean[:, 0], label='filtered latent')
plt.plot(forward_samples.t, forward_samples.latent[:, 0], label='forward_simulated latent')
plt.legend()
plt.show()

plt.plot(samples.t, samples.observation[:, 0], label='observation')
plt.plot(filter_result.t, filter_result.observation_mean[:, 0], label='filtered obs')
plt.plot(forward_samples.t, forward_samples.observation[:, 0], label='forward_simulated obs')
plt.legend()
plt.show()
```

## Online Filtering

Take a look at [examples](./docs/examples) to learn how to do online filtering, for interactive application.

# Change Log

5 November 2024: 1.0.2 released. Added conditional argument passing, dual EKF, better performance in diagonal transition
and observation case. OU example.

14 August 2024: 1.0.1 released. Add incremental API for online filtering. Arbitrary dt.

13 August 2024: Initial release 1.0.0.

## Star History

<a href="https://star-history.com/#joshuaalbert/jaxns&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/essm_jax&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/essm_jax&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=joshuaalbert/essm_jax&type=Date" />
  </picture>
</a>
