# Extended State Space Models in JAX

Given a potentially non-linear state space model this allows you to solve the forward and backward inference steps, for
linear space space models this is equivalent to the Kalman and Rauch-Tung-Striebel recursions.

Support for Python 3.10+.

## Example

All you need to do is define the transition and observation functions, and the initial state prior. These are all in
terms of `MultivariateNormalLinearOperator` distributions from `tensorflow_probability`.

```python
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from essm_jax.essm import ExtendedStateSpaceModel

tfpd = tfp.distributions


def transition_fn(z, t, t_next, *args):
    mean = z + jnp.sin(2 * jnp.pi * t / 10 * z)
    cov = 0.1 * jnp.eye(np.size(z))
    return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))


def observation_fn(z, t, *args):
    mean = z
    cov = t * 0.01 * jnp.eye(np.size(z))
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
samples = essm.sample(jax.random.PRNGKey(0), num_time=T)

# Suppose we only observe every 3rd observation
mask = jnp.arange(T) % 3 != 0

# Marginal likelihood, p(x[:]) = prod_t p(x[t] | x[:t-1])
log_prob = essm.log_prob(samples.observation, mask=mask)
print(log_prob)

# Filtered latent distribution, p(z[t] | x[:t])
filter_result = essm.forward_filter(samples.observation, mask=mask)

# Smoothed latent distribution, p(z[t] | x[:]), i.e. past latents given all future observations
# Including new estimate for prior state p(z[0])
smooth_result, posterior_prior = essm.backward_smooth(filter_result, include_prior=True)
print(smooth_result)

# Forward simulate the model
forward_samples = essm.forward_simulate(
    key=jax.random.PRNGKey(0),
    num_time=25,
    filter_result=filter_result
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

13 August 2024: Initial release 1.0.0.
14 August 2024: 1.0.1 released. Added sparse util. Add incremental API for online filtering. Arbitrary dt.

## Star History

<a href="https://star-history.com/#joshuaalbert/jaxns&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/essm_jax&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=joshuaalbert/essm_jax&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=joshuaalbert/essm_jax&type=Date" />
  </picture>
</a>