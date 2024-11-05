import time

import jax

jax.config.update('jax_enable_x64', True)

import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp

from essm_jax.essm import ExtendedStateSpaceModel, _efficient_add_diag

tfpd = tfp.distributions


def test_extended_state_space_model():
    num_time = 10

    def transition_fn(z, t, t_next, *args):
        mean = 2 * z
        cov = jnp.eye(np.size(z))
        return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))

    def observation_fn(z, t, *args):
        mean = z
        cov = jnp.eye(np.size(z))
        return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))

    initial_state_prior = tfpd.MultivariateNormalTriL(jnp.zeros(2), jnp.eye(2))

    essm = ExtendedStateSpaceModel(
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        initial_state_prior=initial_state_prior
    )

    sample = essm.sample(jax.random.PRNGKey(0), num_time)
    assert sample.latent.shape == (num_time, 2)

    # print(sample)
    with jax.disable_jit():
        filter_result = essm.forward_filter(sample.observation)

    # Assert all finite
    assert jnp.all(jnp.isfinite(filter_result.log_cumulative_marginal_likelihood))
    assert jnp.all(jnp.isfinite(filter_result.filtered_mean))
    assert jnp.all(jnp.isfinite(filter_result.filtered_cov))
    assert jnp.all(jnp.isfinite(filter_result.predicted_mean))
    assert jnp.all(jnp.isfinite(filter_result.predicted_cov))
    assert jnp.all(jnp.isfinite(filter_result.observation_mean))
    assert jnp.all(jnp.isfinite(filter_result.observation_cov))

    # print(filter_result)

    smoothing_result, smoothed_prior = essm.backward_smooth(filter_result, include_prior=True)

    # Assert all finite
    assert jnp.all(jnp.isfinite(smoothing_result.smoothed_mean))
    assert jnp.all(jnp.isfinite(smoothing_result.smoothed_cov))
    assert jnp.all(jnp.isfinite(smoothing_result.smoothed_obs_mean))
    assert jnp.all(jnp.isfinite(smoothing_result.smoothed_obs_cov))

    assert jnp.all(jnp.isfinite(smoothed_prior.mean))
    assert jnp.all(jnp.isfinite(smoothed_prior.covariance))

    print(smoothed_prior)

    # print(smoothing_result)

    transition_matrix = 2 * jnp.eye(2)
    transition_noise = tfpd.MultivariateNormalDiag(
        loc=jnp.zeros(2),
        scale_diag=jnp.ones(2)
    )
    observation_matrix = jnp.eye(2)
    observation_noise = tfpd.MultivariateNormalDiag(
        loc=jnp.zeros(2),
        scale_diag=jnp.ones(2)
    )

    lssm = tfpd.LinearGaussianStateSpaceModel(
        transition_matrix=transition_matrix,
        transition_noise=transition_noise,
        observation_matrix=observation_matrix,
        observation_noise=observation_noise,
        initial_state_prior=initial_state_prior,
        num_timesteps=num_time
    )

    (
        lssm_log_marginal_likelihood,
        lssm_filtered_mean, lssm_filtered_cov,
        lssm_predicted_mean, lssm_predicted_cov,
        lssm_observation_mean, lssm_observation_cov
    ) = lssm.forward_filter(sample.observation)

    def _compare(key):
        sample = essm.sample(key, num_time)
        filter_result = essm.forward_filter(sample.observation)
        (
            lssm_log_marginal_likelihood,
            lssm_filtered_mean, lssm_filtered_cov,
            lssm_predicted_mean, lssm_predicted_cov,
            lssm_observation_mean, lssm_observation_cov
        ) = lssm.forward_filter(sample.observation)
        lssm_diff = lssm_filtered_mean - sample.latent
        essm_diff = filter_result.filtered_mean - sample.latent
        return lssm_diff, essm_diff

    lssm_diff, essm_diff = jax.jit(jax.vmap(_compare))(jax.random.split(jax.random.PRNGKey(0), 100))
    std_lssm = jnp.sqrt(jnp.mean(jnp.square(lssm_diff), axis=0))
    std_essm = jnp.sqrt(jnp.mean(jnp.square(essm_diff), axis=0))

    assert jnp.all(std_essm[0, :] < std_lssm[0, :])

    # print(lssm_filtered_mean)
    # print(filter_result.filtered_mean)
    # print(sample.latent)
    # import pylab as plt
    # plt.plot(sample.t, sample.latent[:, 0], label='latent')
    # plt.plot(sample.t, lssm_filtered_mean[:, 0], label='filtered')
    # plt.plot(sample.t, smoothing_result.smoothed_mean[:, 0], label='smoothed')
    # plt.legend()
    # plt.show()

    # np.testing.assert_allclose(filter_result.predicted_mean, lssm_predicted_mean)
    # np.testing.assert_allclose(filter_result.predicted_cov, lssm_predicted_cov)

    (
        lssm_smoothed_mean, lssm_smoothed_cov
    ) = lssm.posterior_marginals(sample.observation)

    # np.testing.assert_allclose(smoothing_result.smoothed_mean, lssm_smoothed_mean)
    # np.testing.assert_allclose(smoothing_result.smoothed_cov, lssm_smoothed_cov)


def test_jvp_essm():
    def transition_fn(z, t, t_next, *args):
        mean = jnp.sin(2 * z)
        cov = jnp.eye(np.size(z))
        return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))

    def observation_fn(z, t, *args):
        mean = jnp.exp(z)
        cov = jnp.eye(np.size(z))
        return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))

    initial_state_prior = tfpd.MultivariateNormalTriL(jnp.zeros(2), jnp.eye(2))

    essm = ExtendedStateSpaceModel(
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        initial_state_prior=initial_state_prior,
        materialise_jacobians=True
    )

    essm_jvp = ExtendedStateSpaceModel(
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        initial_state_prior=initial_state_prior,
        materialise_jacobians=False
    )

    sample = essm.sample(jax.random.PRNGKey(0), 10)
    filter_result = essm.forward_filter(sample.observation)
    smoothing_result, smoothed_prior = essm.backward_smooth(filter_result, include_prior=True)

    sample_jvp = essm_jvp.sample(jax.random.PRNGKey(0), 10)
    filter_result_jvp = essm_jvp.forward_filter(sample_jvp.observation)
    smoothing_result_jvp, smoothed_prior_jvp = essm_jvp.backward_smooth(filter_result_jvp, include_prior=True)

    assert jnp.allclose(sample.latent, sample_jvp.latent)
    assert jnp.allclose(sample.observation, sample_jvp.observation)

    assert jnp.allclose(filter_result.log_cumulative_marginal_likelihood,
                        filter_result_jvp.log_cumulative_marginal_likelihood)
    assert jnp.allclose(filter_result.filtered_mean, filter_result_jvp.filtered_mean)
    assert jnp.allclose(filter_result.filtered_cov, filter_result_jvp.filtered_cov)
    assert jnp.allclose(filter_result.predicted_mean, filter_result_jvp.predicted_mean)
    assert jnp.allclose(filter_result.predicted_cov, filter_result_jvp.predicted_cov)
    assert jnp.allclose(filter_result.observation_mean, filter_result_jvp.observation_mean)
    assert jnp.allclose(filter_result.observation_cov, filter_result_jvp.observation_cov)

    assert jnp.allclose(smoothing_result.smoothed_mean, smoothing_result_jvp.smoothed_mean)
    assert jnp.allclose(smoothing_result.smoothed_cov, smoothing_result_jvp.smoothed_cov)
    assert jnp.allclose(smoothing_result.smoothed_obs_mean, smoothing_result_jvp.smoothed_obs_mean)
    assert jnp.allclose(smoothing_result.smoothed_obs_cov, smoothing_result_jvp.smoothed_obs_cov)

    assert jnp.allclose(smoothed_prior.mean, smoothed_prior_jvp.mean)
    assert jnp.allclose(smoothed_prior.covariance, smoothed_prior_jvp.covariance)


def test_speed_test_jvp_essm():
    def transition_fn(z, t, t_next, *args):
        mean = jnp.sin(2 * z + t)
        cov = jnp.eye(np.size(z))
        return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))

    def observation_fn(z, t, *args):
        mean = jnp.exp(z) - t
        cov = jnp.eye(np.size(z))
        return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))

    n = 128

    initial_state_prior = tfpd.MultivariateNormalTriL(jnp.zeros(n), jnp.eye(n))

    essm = ExtendedStateSpaceModel(
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        initial_state_prior=initial_state_prior,
        materialise_jacobians=True
    )

    essm_jvp = ExtendedStateSpaceModel(
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        initial_state_prior=initial_state_prior,
        materialise_jacobians=False
    )

    sample = essm.sample(jax.random.PRNGKey(0), 1000)
    filter_fn = jax.jit(
        lambda: essm.forward_filter(sample.observation, marginal_likelihood_only=True)).lower().compile()
    filter_jvp_fn = jax.jit(
        lambda: essm_jvp.forward_filter(sample.observation, marginal_likelihood_only=True)).lower().compile()

    t0 = time.time()
    filter_results = filter_fn()
    filter_results.block_until_ready()
    t1 = time.time()
    dt1 = t1 - t0
    print(f"Time for essm: {t1 - t0}")

    t0 = time.time()
    filter_results_jvp = filter_jvp_fn()
    filter_results_jvp.block_until_ready()
    t1 = time.time()
    dt2 = t1 - t0
    print(f"Time for essm_jvp: {t1 - t0}")

    assert dt2 < dt1


def test_essm_forward_simulation():
    def transition_fn(z, t, t_next, *args):
        mean = z + jnp.sin(2 * jnp.pi * t / 10 * z)
        cov = 0.1 * jnp.eye(np.size(z))
        return tfpd.MultivariateNormalTriL(mean, jnp.linalg.cholesky(cov))

    def observation_fn(z, t, *args):
        mean = z
        cov = 0.01 * jnp.eye(np.size(z))
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

    # Filtered latent distribution, p(z[t] | x[:t])
    filter_result = essm.forward_filter(samples.observation, mask=mask)
    assert np.all(np.isfinite(filter_result.log_cumulative_marginal_likelihood))
    assert np.all(np.isfinite(filter_result.filtered_mean))

    # Marginal likelihood, p(x[:]) = prod_t p(x[t] | x[:t-1])
    log_prob = essm.log_prob(samples.observation, mask=mask)
    assert log_prob == filter_result.log_cumulative_marginal_likelihood[-1]

    # Smoothed latent distribution, p(z[t] | x[:]), i.e. past latents given all future observations
    # Including new estimate for prior state p(z[0])
    smooth_result, posterior_prior = essm.backward_smooth(filter_result, include_prior=True)
    assert np.all(np.isfinite(smooth_result.smoothed_mean))

    # Forward simulate the model
    forward_samples = essm.forward_simulate(
        key=jax.random.PRNGKey(0),
        num_time=25,
        filter_result=filter_result
    )

    try:
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
    except ImportError:
        pass


def test__efficienct_add_scalar_diag():
    A = jnp.array([[1., 2.], [3., 4.]])
    c = 1.
    assert jnp.all(_efficient_add_diag(A, c) == jnp.array([[2., 2.], [3., 5.]]))

    # bigger
    A = jnp.eye(100)
    c = 1.
    assert jnp.all(_efficient_add_diag(A, c) == A + c * jnp.eye(100))


def test_incremental_filtering():
    def transition_fn(z, t, t_next, *args):
        mean = z + z * jnp.sin(2 * jnp.pi * t / 10)
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
    samples = essm.sample(jax.random.PRNGKey(0), 100)

    filter_result = essm.forward_filter(samples.observation)

    filter_state = essm.create_initial_filter_state()

    for i in range(100):
        filter_state = essm.incremental_predict(filter_state)
        filter_state, _ = essm.incremental_update(filter_state, samples.observation[i])
        assert filter_state.t == filter_result.t[i]
        np.testing.assert_allclose(filter_state.log_cumulative_marginal_likelihood,
                                   filter_result.log_cumulative_marginal_likelihood[i], atol=1e-5)
        np.testing.assert_allclose(filter_state.filtered_mean, filter_result.filtered_mean[i], atol=1e-5)
        np.testing.assert_allclose(filter_state.filtered_cov, filter_result.filtered_cov[i], atol=1e-5)
